"""
WiFi CSI Fusion — Hacker Edition
Matrix rain + terminal overlay + ESP32 CSI + MediaPipe pose
"""

import socket, struct, threading, time, argparse, collections, math
import urllib.request, os, random, string

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

# ── Palette ───────────────────────────────────────────────────────────────────
C_BRIGHT   = (0, 255, 70)
C_DIM      = (0, 140, 40)
C_DIMMER   = (0, 60, 20)
C_CYAN     = (255, 220, 0)
C_RED      = (0, 40, 220)
C_WHITE    = (200, 255, 200)
C_YELLOW   = (0, 220, 255)
C_BLACK    = (0, 0, 0)
C_GRID     = (0, 35, 0)

DEFAULT_UDP_PORT    = 5005
DEFAULT_CAMERA_ID   = 0
DEFAULT_WINDOW_W    = 1600
DEFAULT_WINDOW_H    = 600
DEFAULT_CSI_TIMEOUT = 3.0
MODEL_PATH          = "pose_landmarker.task"
MODEL_URL           = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]

MAGIC_CSI  = 0xC5110001
MAGIC_FEAT = 0xC5110006
HEADER_FMT = "<IBIQbBHH"
HEADER_SZ  = struct.calcsize(HEADER_FMT)

# ── Terminal log lines that scroll over the video ─────────────────────────────
FAKE_LOG_POOL = [
    ">> SCANNING RF BAND 2.4GHz...",
    ">> SUBCARRIER LOCK: 56 channels",
    ">> PHASE COHERENCE: OK",
    ">> HAMPEL FILTER: ACTIVE",
    ">> FRESNEL ZONE: COMPUTED",
    ">> MULTISTATIC FUSION: 3 NODES",
    ">> CSI FRAME PARSED [node=1 seq={seq}]",
    ">> CSI FRAME PARSED [node=2 seq={seq}]",
    ">> CSI FRAME PARSED [node=3 seq={seq}]",
    ">> RSSI: {rssi} dBm  CH:{ch}",
    ">> MOTION BAND POWER: {val:.2f}",
    ">> BREATHING RATE: {bpm:.1f} BPM",
    ">> PRESENCE DETECTED — CONFIDENCE {conf:.0f}%",
    ">> KEYPOINT LOCK: 17 joints",
    ">> BODY VEL PROFILE: COMPUTED",
    ">> ATTENTION WEIGHT: {val:.4f}",
    ">> RUVECTOR v2.0.4: ONLINE",
    ">> AMP VARIANCE: {val:.3f}",
    ">> CROSS-VIEW EMBEDDING: OK",
    ">> FIELD MODEL: STABLE",
    ">> COHERENCE GATE: ACCEPT",
    ">> SPOOFING CHECK: PASS",
    ">> UDP RX: {bytes} bytes from 192.168.1.{ip}",
    ">> EDGE DSP CORE 1: RUNNING",
    ">> OTA STATUS: IDLE",
    ">> WASM MODULE: LOADED",
    ">> ADAPTIVE CTRL: state={state}",
]


def rand_log():
    t = random.choice(FAKE_LOG_POOL)
    return t.format(
        seq=random.randint(1000, 9999),
        rssi=random.randint(-80, -50),
        ch=random.choice([1, 6, 11]),
        val=random.uniform(0.001, 99.9),
        bpm=random.uniform(12, 25),
        conf=random.uniform(60, 99),
        bytes=random.randint(32, 276),
        ip=random.randint(40, 250),
        state=random.randint(1, 12),
    )


# ── Matrix rain ───────────────────────────────────────────────────────────────
MATRIX_CHARS = "アイウエオカキクケコサシスセソタチツテトナニヌネノ0123456789ABCDEF><|{}[]"

class MatrixRain:
    def __init__(self, w, h, density=0.04):
        self.w       = w
        self.h       = h
        self.cols    = w // 10
        self.rows    = h // 14
        self.density = density
        self.drops   = [random.randint(0, self.rows) for _ in range(self.cols)]
        self.speeds  = [random.choice([1, 1, 1, 2]) for _ in range(self.cols)]
        self.chars   = [[random.choice(MATRIX_CHARS) for _ in range(self.rows)]
                        for _ in range(self.cols)]
        self.tick    = 0

    def update(self):
        self.tick += 1
        for c in range(self.cols):
            if self.tick % self.speeds[c] == 0:
                self.drops[c] = (self.drops[c] + 1) % (self.rows + 10)
                if random.random() < 0.1:
                    self.chars[c][self.drops[c] % self.rows] = random.choice(MATRIX_CHARS)

    def render(self, canvas, alpha=0.18):
        overlay = np.zeros_like(canvas)
        for c in range(self.cols):
            x = c * 10
            for r in range(self.rows):
                y    = r * 14 + 12
                dist = (self.drops[c] - r) % self.rows
                if dist == 0:
                    col = (200, 255, 200)
                elif dist < 3:
                    col = (0, 220, 60)
                elif dist < 8:
                    col = (0, 130, 30)
                else:
                    col = (0, 40, 10)
                ch = self.chars[c][r]
                cv2.putText(overlay, ch, (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 0.85, col, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1.0, 0, canvas)


# ── Scanline effect ───────────────────────────────────────────────────────────
def apply_scanlines(frame, gap=3, alpha=0.15):
    overlay = frame.copy()
    for y in range(0, frame.shape[0], gap):
        overlay[y] = (overlay[y] * (1 - alpha)).astype(np.uint8)
    return overlay


# ── Green tint for camera panel ───────────────────────────────────────────────
def green_tint(frame, strength=0.55):
    tinted       = frame.copy().astype(np.float32)
    tinted[:,:,0] *= (1 - strength)       # reduce blue
    tinted[:,:,2] *= (1 - strength * 0.8) # reduce red
    tinted[:,:,1]  = np.clip(tinted[:,:,1] * (1 + strength * 0.3), 0, 255)
    return np.clip(tinted, 0, 255).astype(np.uint8)


# ── Terminal overlay on camera ────────────────────────────────────────────────
class TerminalOverlay:
    def __init__(self, max_lines=18):
        self.lines     = collections.deque(maxlen=max_lines)
        self.last_add  = 0
        self.interval  = 0.18

    def update(self):
        if time.time() - self.last_add > self.interval:
            self.lines.append(rand_log())
            self.last_add = time.time()
            self.interval = random.uniform(0.12, 0.45)

    def render(self, canvas):
        h, w = canvas.shape[:2]
        x    = 10
        y0   = h - 14 * len(self.lines) - 10
        for i, line in enumerate(self.lines):
            y      = y0 + i * 14
            bright = (i == len(self.lines) - 1)
            col    = C_BRIGHT if bright else C_DIM
            # subtle shadow
            cv2.putText(canvas, line, (x + 1, y + 1),
                        cv2.FONT_HERSHEY_PLAIN, 0.82, C_BLACK, 1, cv2.LINE_AA)
            cv2.putText(canvas, line, (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.82, col, 1, cv2.LINE_AA)
        # blinking cursor
        if int(time.time() * 2) % 2 == 0:
            last_y = y0 + len(self.lines) * 14
            cv2.putText(canvas, "█", (x, last_y),
                        cv2.FONT_HERSHEY_PLAIN, 0.82, C_BRIGHT, 1, cv2.LINE_AA)


# ── CSI receiver ─────────────────────────────────────────────────────────────
def parse_csi_frame(data):
    if len(data) < HEADER_SZ:
        return None
    try:
        magic, node_id, seq, ts_ms, rssi, channel, sc_count, data_len = \
            struct.unpack_from(HEADER_FMT, data, 0)
    except struct.error:
        return None
    if magic not in (MAGIC_CSI, MAGIC_FEAT):
        return None
    payload    = data[HEADER_SZ: HEADER_SZ + data_len]
    amplitudes = []
    if magic == MAGIC_CSI and len(payload) >= sc_count * 4:
        for i in range(sc_count):
            iv, qv = struct.unpack_from("<hh", payload, i * 4)
            amplitudes.append(math.sqrt(iv**2 + qv**2))
    return {"node_id": node_id, "seq": seq, "ts_ms": ts_ms, "rssi": rssi,
            "channel": channel, "sc_count": sc_count, "amplitudes": amplitudes,
            "received_at": time.time()}


class CSIReceiver(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port; self.nodes = {}
        self.lock = threading.Lock(); self._running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port)); sock.settimeout(0.5)
        print(f"[CSI] Listening on UDP :{self.port}")
        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                frame = parse_csi_frame(data)
                if frame:
                    with self.lock:
                        nid = frame["node_id"]
                        if nid not in self.nodes:
                            print(f"[CSI] Node {nid} online — {addr[0]}")
                        self.nodes[nid] = {**frame, "ip": addr[0]}
            except socket.timeout:
                pass
        sock.close()

    def stop(self): self._running = False
    def get_snapshot(self):
        with self.lock: return dict(self.nodes)


class CSIMotionEstimator:
    def __init__(self, nid):
        self.nid = nid; self.history = collections.deque(maxlen=30)

    def update(self, amps):
        if not amps:
            return {"motion": 0.0, "presence": False, "variance": 0.0}
        m = float(np.mean(amps)); self.history.append(m)
        if len(self.history) < 5:
            return {"motion": 0.0, "presence": False, "variance": 0.0}
        arr = np.array(self.history)
        var = float(np.var(arr))
        dif = float(np.mean(np.abs(np.diff(arr[-10:]))))
        return {"motion": min(dif / 5.0, 1.0), "presence": var > 2.0,
                "variance": var, "mean": m}


# ── CSI panel (hacker style) ──────────────────────────────────────────────────
def render_csi_panel(nodes, estimators, timeout, pw, ph):
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)

    # grid background
    for y in range(0, ph, 20):
        cv2.line(panel, (0, y), (pw, y), C_GRID, 1)
    for x in range(0, pw, 20):
        cv2.line(panel, (x, 0), (x, ph), C_GRID, 1)

    now = time.time()

    # header
    cv2.rectangle(panel, (0, 0), (pw, 44), (0, 20, 0), -1)
    cv2.putText(panel, "[ RUVECTOR v2.0.4 — MULTISTATIC CSI MESH ]",
                (10, 16), cv2.FONT_HERSHEY_PLAIN, 1.0, C_BRIGHT, 1, cv2.LINE_AA)
    ts = time.strftime("%H:%M:%S")
    cv2.putText(panel, f"SYS:{ts}", (pw - 110, 16),
                cv2.FONT_HERSHEY_PLAIN, 0.9, C_DIM, 1, cv2.LINE_AA)
    cv2.putText(panel, f"UDP:5005  NODES:{len(nodes)}  PROTOCOL:ADR-018",
                (10, 34), cv2.FONT_HERSHEY_PLAIN, 0.82, C_DIM, 1, cv2.LINE_AA)
    cv2.line(panel, (0, 44), (pw, 44), C_DIM, 1)

    slot_h = (ph - 90) // 6
    for idx in range(1, 7):
        y0     = 50 + (idx - 1) * slot_h
        data   = nodes.get(idx)
        age    = now - data["received_at"] if data else timeout + 1
        online = data is not None and age < timeout

        # slot border
        col_b = C_DIMMER if not online else (0, 50, 0)
        cv2.rectangle(panel, (6, y0), (pw - 6, y0 + slot_h - 4), col_b, 1)

        # status dot (pulsing)
        pulse = int(abs(math.sin(time.time() * 3 + idx)) * 100)
        dot_c = (0, 100 + pulse, 0) if online else (0, 30, 100 + pulse)
        cv2.circle(panel, (22, y0 + 14), 7, dot_c, -1)
        cv2.circle(panel, (22, y0 + 14), 7, C_DIM, 1)

        node_label = f"NODE_{idx:02d}"
        cv2.putText(panel, node_label, (36, y0 + 18),
                    cv2.FONT_HERSHEY_PLAIN, 1.0,
                    C_BRIGHT if online else C_DIMMER, 1, cv2.LINE_AA)

        if not online:
            cv2.putText(panel, "[ OFFLINE ]", (pw - 120, y0 + 18),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, C_RED, 1, cv2.LINE_AA)
            continue

        est    = estimators.setdefault(idx, CSIMotionEstimator(idx))
        result = est.update(data.get("amplitudes", []))
        rssi   = data.get("rssi", 0)
        sc     = data.get("sc_count", 0)
        motion = result["motion"]
        pres   = result["presence"]
        var    = result["variance"]

        info = f"RSSI:{rssi}dBm  SC:{sc}  VAR:{var:.2f}  IP:{data.get('ip','?')}"
        cv2.putText(panel, info, (36, y0 + 30),
                    cv2.FONT_HERSHEY_PLAIN, 0.78, C_DIM, 1, cv2.LINE_AA)

        # presence badge
        badge = "[ PRESENT ]" if pres else "[  EMPTY  ]"
        badge_col = C_BRIGHT if pres else (0, 80, 0)
        cv2.putText(panel, badge, (pw - 130, y0 + 18),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, badge_col, 1, cv2.LINE_AA)

        # motion bar
        bx = 36; by = y0 + 36; bw = pw - 50; bh = 6
        cv2.rectangle(panel, (bx, by), (bx + bw, by + bh), (0, 25, 0), -1)
        filled = int(bw * motion)
        if filled > 0:
            bar_col = C_YELLOW if motion > 0.6 else C_BRIGHT
            cv2.rectangle(panel, (bx, by), (bx + filled, by + bh), bar_col, -1)
        cv2.putText(panel, f"MOTION {motion*100:.0f}%", (bx, by - 1),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, C_DIMMER, 1, cv2.LINE_AA)

        # amplitude sparkline
        amps = data.get("amplitudes", [])
        if len(amps) > 4:
            sy   = y0 + 50; sh = slot_h - 58
            step = max(1, len(amps) // (bw))
            samp = amps[::step][:bw]
            mx   = max(samp) or 1
            pts  = [(bx + i, sy + sh - int(sh * v / mx)) for i, v in enumerate(samp)]
            for i in range(len(pts) - 1):
                intensity = int(80 + 175 * (samp[i] / mx))
                cv2.line(panel, pts[i], pts[i+1], (0, intensity, 0), 1)

    # bottom status bar
    cv2.rectangle(panel, (0, ph - 28), (pw, ph), (0, 15, 0), -1)
    online_n = sum(1 for d in nodes.values() if now - d["received_at"] < timeout)
    status   = f"  ONLINE:{online_n}/{len(nodes)}  |  FREQ:2.4GHz  |  PORT:5005  |  PROTO:UDP/ADR-018"
    cv2.putText(panel, status, (6, ph - 10),
                cv2.FONT_HERSHEY_PLAIN, 0.82, C_DIM, 1, cv2.LINE_AA)

    return apply_scanlines(panel, gap=2, alpha=0.08)


# ── Camera panel (hacker style) ───────────────────────────────────────────────
def draw_pose(frame, result):
    if not result or not result.pose_landmarks:
        return frame
    h, w = frame.shape[:2]
    for person in result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in person]
        vis = [lm.visibility for lm in person]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts) and vis[a] > 0.5 and vis[b] > 0.5:
                cv2.line(frame, pts[a], pts[b], C_BRIGHT, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            if i < len(vis) and vis[i] > 0.5:
                cv2.circle(frame, (x, y), 5, C_BRIGHT, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 5, (0, 80, 0), 1, cv2.LINE_AA)
    return frame


def render_camera_panel(frame, result, csi_presence, matrix, terminal,
                        panel_w, panel_h):
    panel = cv2.resize(frame, (panel_w, panel_h))
    panel = green_tint(panel)

    # matrix rain behind everything
    matrix.update()
    matrix.render(panel, alpha=0.12)

    # pose skeleton
    panel = draw_pose(panel, result)

    # scanlines
    panel = apply_scanlines(panel, gap=3, alpha=0.12)

    # terminal log overlay
    terminal.update()
    terminal.render(panel)

    # HUD header
    cv2.rectangle(panel, (0, 0), (panel_w, 36), (0, 0, 0, 0), -1)
    cv2.putText(panel, "[ CAMERA + MEDIAPIPE POSE + WiFi CSI FUSION ]",
                (8, 15), cv2.FONT_HERSHEY_PLAIN, 0.95, C_BRIGHT, 1, cv2.LINE_AA)
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(panel, ts, (8, 30),
                cv2.FONT_HERSHEY_PLAIN, 0.82, C_DIM, 1, cv2.LINE_AA)

    # CSI fusion badge
    if csi_presence:
        badge_col = C_BRIGHT
        badge_txt = ">> CSI CONFIRMED: HUMAN DETECTED <<"
        # pulsing border
        pulse = int(abs(math.sin(time.time() * 4)) * 2) + 1
        cv2.rectangle(panel, (pulse, pulse),
                      (panel_w - pulse, panel_h - pulse), C_BRIGHT, pulse)
    else:
        badge_col = C_DIMMER
        badge_txt = ">> CSI: NO PRESENCE DETECTED"

    cv2.putText(panel, badge_txt, (8, panel_h - 8),
                cv2.FONT_HERSHEY_PLAIN, 0.9, badge_col, 1, cv2.LINE_AA)

    # corner brackets
    L = 18
    for (x, y, dx, dy) in [(0,0,1,1),(panel_w-1,0,-1,1),
                            (0,panel_h-1,1,-1),(panel_w-1,panel_h-1,-1,-1)]:
        cv2.line(panel, (x,y), (x + dx*L, y), C_BRIGHT, 2)
        cv2.line(panel, (x,y), (x, y + dy*L), C_BRIGHT, 2)

    return panel


# ── Model download ────────────────────────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading pose model (~7 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved: {MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",      type=int,   default=DEFAULT_UDP_PORT)
    ap.add_argument("--camera",    type=int,   default=DEFAULT_CAMERA_ID)
    ap.add_argument("--width",     type=int,   default=DEFAULT_WINDOW_W)
    ap.add_argument("--height",    type=int,   default=DEFAULT_WINDOW_H)
    ap.add_argument("--timeout",   type=float, default=DEFAULT_CSI_TIMEOUT)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--positions", type=str,   default="")
    args = ap.parse_args()

    panel_w = args.width // 2
    panel_h = args.height

    csi_rx     = CSIReceiver(port=args.port)
    estimators = {}
    csi_rx.start()

    matrix   = MatrixRain(panel_w, panel_h, density=0.03)
    terminal = TerminalOverlay(max_lines=18)

    cap = detector = None
    if not args.no_camera:
        download_model()
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("[WARN] No camera — CSI-only mode")
            cap = None
        else:
            opts = PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=4,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = PoseLandmarker.create_from_options(opts)
            print("[INFO] Pose detector ONLINE")

    cv2.namedWindow("WiFi CSI Fusion", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("WiFi CSI Fusion", args.width, args.height)
    print("[INFO] Q=quit  S=screenshot")

    fps_q = collections.deque(maxlen=30)
    ts_ms = 0

    try:
        while True:
            t0       = time.time()
            snapshot = csi_rx.get_snapshot()
            now      = time.time()

            csi_present = any(
                now - d["received_at"] < args.timeout
                and estimators.get(nid, CSIMotionEstimator(nid)).history
                and np.var(list(estimators[nid].history)) > 2.0
                for nid, d in snapshot.items()
            )

            csi_panel = render_csi_panel(
                snapshot, estimators, args.timeout, panel_w, panel_h)

            result = None
            if cap and detector:
                ok, frame = cap.read()
                if ok:
                    ts_ms += 33
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = detector.detect_for_video(mp_img, ts_ms)
                    cam_panel = render_camera_panel(
                        frame, result, csi_present,
                        matrix, terminal, panel_w, panel_h)
                else:
                    cam_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            else:
                blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                matrix.update(); matrix.render(blank, alpha=0.5)
                terminal.update(); terminal.render(blank)
                cam_panel = blank

            divider  = np.zeros((panel_h, 3, 3), dtype=np.uint8)
            divider[:, 1] = (0, 100, 0)
            composed = np.hstack([cam_panel, divider, csi_panel])

            fps_q.append(time.time() - t0)
            fps = 1.0 / (sum(fps_q) / len(fps_q)) if fps_q else 0
            cv2.putText(composed, f"FPS:{fps:.0f}", (args.width - 80, 20),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, C_DIM, 1, cv2.LINE_AA)

            cv2.imshow("WiFi CSI Fusion", composed)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                fn = f"output/screenshot_{int(time.time())}.png"
                cv2.imwrite(fn, composed)
                print(f"[INFO] Saved {fn}")

    finally:
        csi_rx.stop()
        if cap: cap.release()
        if detector: detector.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown")


if __name__ == "__main__":
    main()
