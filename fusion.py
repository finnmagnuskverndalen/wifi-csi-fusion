"""
WiFi CSI + Camera Fusion
Combines ESP32 CSI presence/position data with MediaPipe camera pose estimation.
Reads UDP from your ESP32 nodes on port 5005, webcam via OpenCV, fuses both.
"""

import socket
import struct
import threading
import time
import json
import argparse
import collections
import math

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

DEFAULT_UDP_PORT    = 5005
DEFAULT_CAMERA_ID   = 0
DEFAULT_WINDOW_W    = 1280
DEFAULT_WINDOW_H    = 480
DEFAULT_CSI_TIMEOUT = 3.0

COL_GREEN  = (0, 255, 120)
COL_CYAN   = (0, 255, 255)
COL_RED    = (0, 60, 255)
COL_YELLOW = (0, 220, 255)
COL_WHITE  = (255, 255, 255)
COL_DARK   = (20, 20, 20)
COL_ORANGE = (0, 165, 255)

MAGIC_CSI  = 0xC5110001
MAGIC_FEAT = 0xC5110006
HEADER_FMT = "<IBIQbBHH"
HEADER_SZ  = struct.calcsize(HEADER_FMT)


def parse_csi_frame(data: bytes) -> dict | None:
    if len(data) < HEADER_SZ:
        return None
    try:
        magic, node_id, seq, ts_ms, rssi, channel, sc_count, data_len = \
            struct.unpack_from(HEADER_FMT, data, 0)
    except struct.error:
        return None

    if magic not in (MAGIC_CSI, MAGIC_FEAT):
        return None

    payload = data[HEADER_SZ: HEADER_SZ + data_len]
    amplitudes = []
    if magic == MAGIC_CSI and len(payload) >= sc_count * 4:
        for i in range(sc_count):
            i_val, q_val = struct.unpack_from("<hh", payload, i * 4)
            amp = math.sqrt(i_val**2 + q_val**2)
            amplitudes.append(amp)

    return {
        "node_id":     node_id,
        "seq":         seq,
        "ts_ms":       ts_ms,
        "rssi":        rssi,
        "channel":     channel,
        "sc_count":    sc_count,
        "amplitudes":  amplitudes,
        "received_at": time.time(),
    }


class CSIReceiver(threading.Thread):
    def __init__(self, port: int, node_positions: dict):
        super().__init__(daemon=True)
        self.port           = port
        self.node_positions = node_positions
        self.nodes: dict    = {}
        self.lock           = threading.Lock()
        self._running       = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        print(f"[CSI] Listening on UDP port {self.port}")
        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                frame = parse_csi_frame(data)
                if frame:
                    with self.lock:
                        nid = frame["node_id"]
                        if nid not in self.nodes:
                            print(f"[CSI] New node: node_id={nid} from {addr[0]}")
                        self.nodes[nid] = {**frame, "ip": addr[0]}
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[CSI] Error: {e}")
        sock.close()

    def stop(self):
        self._running = False

    def get_snapshot(self) -> dict:
        with self.lock:
            return dict(self.nodes)


class CSIMotionEstimator:
    HISTORY = 30

    def __init__(self, node_id: int):
        self.node_id = node_id
        self.history = collections.deque(maxlen=self.HISTORY)

    def update(self, amplitudes: list) -> dict:
        if not amplitudes:
            return {"motion": 0.0, "presence": False, "variance": 0.0}
        mean_amp = float(np.mean(amplitudes))
        self.history.append(mean_amp)

        if len(self.history) < 5:
            return {"motion": 0.0, "presence": False, "variance": 0.0}

        arr      = np.array(self.history)
        variance = float(np.var(arr))
        diff     = float(np.mean(np.abs(np.diff(arr[-10:]))))
        presence = variance > 2.0
        motion   = min(diff / 5.0, 1.0)
        return {
            "motion":   motion,
            "presence": presence,
            "variance": variance,
            "mean_amp": mean_amp,
        }


def render_csi_panel(nodes_snapshot, estimators, timeout, panel_w, panel_h):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (15, 15, 25)

    now = time.time()
    cv2.putText(panel, "WiFi CSI  |  ESP32 Nodes", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_CYAN, 1, cv2.LINE_AA)
    cv2.line(panel, (12, 38), (panel_w - 12, 38), (40, 40, 60), 1)

    max_nodes = 6
    slot_h    = (panel_h - 60) // max_nodes

    for idx in range(1, max_nodes + 1):
        y0   = 50 + (idx - 1) * slot_h
        data = nodes_snapshot.get(idx)
        age  = now - data["received_at"] if data else timeout + 1
        online = data is not None and age < timeout
        status_col = COL_GREEN if online else COL_RED

        cv2.circle(panel, (20, y0 + 12), 6, status_col, -1)
        cv2.putText(panel, f"Node {idx}", (34, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_WHITE, 1, cv2.LINE_AA)

        if not online:
            cv2.putText(panel, "OFFLINE", (panel_w - 80, y0 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_RED, 1, cv2.LINE_AA)
            continue

        est    = estimators.setdefault(idx, CSIMotionEstimator(idx))
        result = est.update(data.get("amplitudes", []))
        rssi   = data.get("rssi", 0)
        sc     = data.get("sc_count", 0)
        motion = result["motion"]
        pres   = result["presence"]

        cv2.putText(panel, f"RSSI:{rssi}dBm  SC:{sc}  Var:{result['variance']:.1f}",
                    (34, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (160, 160, 180), 1, cv2.LINE_AA)

        badge_col = COL_GREEN if pres else (80, 80, 80)
        cv2.putText(panel, "PRESENT" if pres else "EMPTY",
                    (panel_w - 90, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_col, 1, cv2.LINE_AA)

        bar_x  = 34
        bar_y  = y0 + 42
        bar_w  = panel_w - 50
        bar_h  = 8
        filled = int(bar_w * motion)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 60), -1)
        if filled > 0:
            bar_col = COL_YELLOW if motion > 0.5 else COL_GREEN
            cv2.rectangle(panel, (bar_x, bar_y),
                          (bar_x + filled, bar_y + bar_h), bar_col, -1)

        amps = data.get("amplitudes", [])
        if len(amps) > 4:
            spark_y = y0 + 58
            spark_h = 18
            step    = max(1, len(amps) // (panel_w - 50))
            sampled = amps[::step][: panel_w - 50]
            mx      = max(sampled) or 1
            pts = [(bar_x + i, spark_y + spark_h - int(spark_h * v / mx))
                   for i, v in enumerate(sampled)]
            for i in range(len(pts) - 1):
                cv2.line(panel, pts[i], pts[i + 1], COL_CYAN, 1)

    online_count = sum(
        1 for d in nodes_snapshot.values()
        if now - d["received_at"] < timeout
    )
    cv2.putText(panel, f"Nodes online: {online_count} / {len(nodes_snapshot) or 0}",
                (12, panel_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CYAN, 1, cv2.LINE_AA)
    return panel


def render_camera_panel(frame_bgr, pose_results, csi_presence, panel_w, panel_h):
    panel = cv2.resize(frame_bgr, (panel_w, panel_h))

    if pose_results and pose_results.pose_landmarks:
        mp_draw.draw_landmarks(
            panel,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(
                color=(0, 255, 120), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(0, 200, 255), thickness=2),
        )

    cv2.putText(panel, "Camera  +  MediaPipe Pose", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_WHITE, 1, cv2.LINE_AA)

    fuse_col = COL_GREEN if csi_presence else (80, 80, 80)
    fuse_txt = "CSI: PRESENCE DETECTED" if csi_presence else "CSI: NO PRESENCE"
    cv2.putText(panel, fuse_txt, (10, panel_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, fuse_col, 1, cv2.LINE_AA)

    if csi_presence:
        cv2.rectangle(panel, (2, 2), (panel_w - 2, panel_h - 2), COL_GREEN, 2)

    return panel


def main():
    ap = argparse.ArgumentParser(description="WiFi CSI + Camera Fusion")
    ap.add_argument("--port",      type=int,   default=DEFAULT_UDP_PORT)
    ap.add_argument("--camera",    type=int,   default=DEFAULT_CAMERA_ID)
    ap.add_argument("--width",     type=int,   default=DEFAULT_WINDOW_W)
    ap.add_argument("--height",    type=int,   default=DEFAULT_WINDOW_H)
    ap.add_argument("--timeout",   type=float, default=DEFAULT_CSI_TIMEOUT)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--positions", type=str,   default="")
    args = ap.parse_args()

    node_positions = {}
    if args.positions:
        for idx, part in enumerate(args.positions.split(";"), start=1):
            try:
                x, y, z = map(float, part.split(","))
                node_positions[idx] = (x, y, z)
            except ValueError:
                pass

    panel_w = args.width // 2
    panel_h = args.height

    csi_rx = CSIReceiver(port=args.port, node_positions=node_positions)
    csi_rx.start()
    estimators = {}

    cap = None
    if not args.no_camera:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"[WARN] Cannot open camera {args.camera} — running CSI-only")
            cap = None

    pose = mp_pose.Pose(
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) if cap else None

    cv2.namedWindow("WiFi CSI Fusion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WiFi CSI Fusion", args.width, args.height)
    print("[INFO] Press Q to quit, S to save screenshot")
    fps_counter = collections.deque(maxlen=30)

    try:
        while True:
            t0 = time.time()
            snapshot    = csi_rx.get_snapshot()
            now         = time.time()
            csi_present = any(
                now - d["received_at"] < args.timeout
                and estimators.get(nid, CSIMotionEstimator(nid)).history
                and np.var(list(estimators[nid].history)) > 2.0
                for nid, d in snapshot.items()
            )

            csi_panel = render_csi_panel(
                snapshot, estimators, args.timeout, panel_w, panel_h)

            if cap:
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    pose_results = None
                else:
                    rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb)
                cam_panel = render_camera_panel(
                    frame if ok else np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
                    pose_results, csi_present, panel_w, panel_h)
            else:
                cam_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                cv2.putText(cam_panel, "No camera  (--no-camera mode)",
                            (20, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

            divider  = np.full((panel_h, 2, 3), 60, dtype=np.uint8)
            composed = np.hstack([cam_panel, divider, csi_panel])

            fps_counter.append(time.time() - t0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0
            cv2.putText(composed, f"FPS:{fps:.0f}", (args.width - 70, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

            cv2.imshow("WiFi CSI Fusion", composed)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                fname = f"output/screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, composed)
                print(f"[INFO] Saved {fname}")

    finally:
        csi_rx.stop()
        if cap:
            cap.release()
        if pose:
            pose.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
