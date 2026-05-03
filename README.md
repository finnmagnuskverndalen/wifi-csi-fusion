# WiFi CSI Fusion

Real-time fusion of **WiFi Channel State Information (CSI)** from ESP32-S3 nodes with **MediaPipe camera pose estimation**. Inspired by [@sgpcard](https://www.instagram.com/sgpcard/) on Instagram.

Each ESP32-S3 node streams CSI data over UDP. This script captures that data alongside your webcam feed, runs MediaPipe Pose on the camera, and fuses both into a split-screen visualization — CSI presence/motion on the right, camera skeleton overlay on the left.

```
┌─────────────────────────┬──────────────────────────┐
│  Camera + MediaPipe     │  WiFi CSI  |  ESP32 Nodes │
│  Pose skeleton overlay  │  Node 1 ●  RSSI:-67  PRES │
│                         │  Node 2 ●  RSSI:-68  PRES │
│  [CSI: PRESENCE]        │  Node 3 ●  RSSI:-71  PRES │
└─────────────────────────┴──────────────────────────┘
```

---

## Hardware

- 3× ESP32-S3 nodes flashed with [RuView firmware](https://github.com/ruvnet/RuView)
- Laptop/PC with webcam
- All devices on the same WiFi network

Node placement used in this setup:

| Node | Position | Height |
|------|----------|--------|
| 1 | Center wall | 40 cm |
| 2 | Left wall, 2m right of node 1 | 1.0 m |
| 3 | Right wall, 5m left of node 1 | 1.0 m |

Room depth: ~4 meters.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/wifi-csi-fusion.git
cd wifi-csi-fusion

pip install -r requirements.txt
```

> **Note:** Requires Python 3.10+. On Debian/Ubuntu use `pip install --break-system-packages` if needed.

---

## Usage

### Basic — camera + CSI

```bash
python3 fusion.py
```

### With node positions (enables spatial fusion)

```bash
python3 fusion.py --positions "0,0,0.4;2,4,1.0;-5,4,1.0"
```

### CSI only (no webcam)

```bash
python3 fusion.py --no-camera
```

### Custom port / camera

```bash
python3 fusion.py --port 5005 --camera 0
```

### All options

```
--port       UDP port for ESP32 CSI frames     (default: 5005)
--camera     OpenCV camera device index        (default: 0)
--width      Total window width in pixels      (default: 1280)
--height     Window height in pixels           (default: 480)
--timeout    Seconds before node goes offline  (default: 3.0)
--no-camera  CSI-only mode, no webcam
--positions  Node positions: "x,y,z;x,y,z;..."
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `output/` |

---

## ESP32 Setup

Flash the RuView firmware to each ESP32-S3, then provision with your network details:

```bash
# Node 1
python3 provision.py --port /dev/ttyACM0 \
  --ssid "YourWiFi" --password "YourPassword" \
  --target-ip YOUR_LAPTOP_IP --target-port 5005 --node-id 1

# Node 2
python3 provision.py --port /dev/ttyACM0 \
  --ssid "YourWiFi" --password "YourPassword" \
  --target-ip YOUR_LAPTOP_IP --target-port 5005 --node-id 2

# Node 3
python3 provision.py --port /dev/ttyACM0 \
  --ssid "YourWiFi" --password "YourPassword" \
  --target-ip YOUR_LAPTOP_IP --target-port 5005 --node-id 3
```

Find your laptop IP with:

```bash
ip addr show | grep 'inet ' | grep -v 127.0.0.1
```

---

## How It Works

```
ESP32-S3 nodes (x3)
  └─ WiFi CSI captured at 20 Hz
  └─ ADR-018 binary UDP frames → port 5005
        │
        ▼
CSIReceiver thread
  └─ Parses frames per node
  └─ Extracts amplitude per subcarrier
  └─ CSIMotionEstimator: variance → presence/motion
        │
        ▼
fusion.py main loop
  ├─ CSI panel: node status, RSSI, motion bars, amplitude sparkline
  └─ Camera panel: MediaPipe Pose skeleton overlay
        │
        ▼
Side-by-side window (OpenCV)
  └─ Green border when CSI confirms presence
  └─ Press S to screenshot
```

The CSI parser handles the [ADR-018 binary frame format](https://github.com/ruvnet/RuView/blob/main/docs/adr/ADR-018-esp32-dev-implementation.md) used by the RuView firmware — `magic(4) + node_id(1) + seq(4) + timestamp_ms(8) + rssi(1) + channel(1) + subcarrier_count(2) + data_len(2)` followed by I/Q pairs.

---

## Project Structure

```
wifi-csi-fusion/
├── fusion.py          # Main script
├── requirements.txt   # Python dependencies
├── config.json        # Example config / node positions
├── output/            # Screenshots saved here
└── README.md
```

---

## Pushing to GitHub

```bash
cd wifi-csi-fusion
git init
git add .
git commit -m "Initial commit: WiFi CSI + MediaPipe camera fusion"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/wifi-csi-fusion.git
git push -u origin main
```

---

## Credits

- Inspired by [@sgpcard](https://www.instagram.com/sgpcard/) WiFi CSI demo
- ESP32 firmware: [ruvnet/RuView](https://github.com/ruvnet/RuView)
- Pose estimation: [MediaPipe](https://github.com/google/mediapipe)
- Original research: [DensePose From WiFi — CMU 2023](https://arxiv.org/abs/2301.00250)

---

## License

MIT
