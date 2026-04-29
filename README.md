# 🚗 Real-time Driver Monitoring System (DMS)

AI pipeline for real-time driver fatigue and distraction detection using facial landmarks

## 🎯 Features

- **Drowsiness Detection** — Eye Aspect Ratio (EAR)
- **Yawning Detection** — Mouth Aspect Ratio (MAR)  
- **Distraction Detection** — Head Pose Estimation (Yaw/Pitch)
- **Real-time** — 30fps on CPU, no GPU required

## How It Works

```
Webcam Frame
     ↓
MediaPipe FaceMesh (478 landmarks)
     ↓
┌────────────────────────────────┐
│  EAR  │  MAR  │  Head Pose     │
│ (eyes)│(mouth)│ (yaw/pitch)    │
└────────────────────────────────┘
     ↓
Consecutive Frame Counter
     ↓
Alert System
```

**EAR (Eye Aspect Ratio):**
```
EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
Normal: ~0.30 | Drowsy: < 0.20
```

**MAR (Mouth Aspect Ratio):**
```
MAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)
Normal: ~0.10 | Yawning: > 0.60
```

**Head Pose:**
```
solvePnP(3D model points, 2D landmarks, camera matrix)
→ pitch, yaw, roll (degrees)
Distracted: |yaw| > 20°
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Face Detection | MediaPipe FaceMesh (478 landmarks) |
| Landmark Detection | BlazeFace + Regression Network |
| Head Pose | OpenCV solvePnP |
| Real-time Video | OpenCV VideoCapture |
| Language | Python 3.9+ |

## ⚡ Why MediaPipe over YOLO?

YOLO-face provides only 5 keypoints — insufficient for EAR/MAR calculation which requires 6 points per eye and 6 points for mouth. MediaPipe provides 478 3D landmarks and runs at 30fps on CPU without GPU.

## 🚀 Setup

```bash
# Clone repo
git clone https://github.com/Thi-Dzung/driver-monitorning-system.git

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe model
curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## ▶️ Run

```bash
python src/pipeline.py
```

## ⚙️ Configuration

Edit `configs/config.yaml` to tune thresholds:

```yaml
thresholds:
  ear: 0.20    # eye closed threshold
  mar: 0.60    # yawning threshold
  yaw: 20.0    # distraction angle (degrees)

consecutive_frames:
  ear: 20      # ~0.67s at 30fps
  mar: 15      # ~0.50s
  yaw: 20      # ~0.67s
```

## 📁 Project Structure

```
driver-monitoring-system/
├── configs/
│   └── config.yaml
├── notebooks/
│   └── 01_landmark_exploration.ipynb
├── src/
│   ├── config.py
│   ├── metrics.py
│   └── pipeline.py
├── assets/
│   └── demo.gif
├── requirements.txt
└── README.md
```

## 🔮 Future Improvements

- [ ] Per-user EAR/MAR calibration
- [ ] Audio alert system
- [ ] Night driving mode (low light optimization)
- [ ] Time-based thresholds (ms) instead of frame-based counters
      to ensure consistent alert timing regardless of camera FPS