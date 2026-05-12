# Real-time Driver Monitoring System (DMS)

> AI pipeline detecting driver fatigue and distraction in real-time using facial landmarks and machine learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-latest-yellow)

---

## Overview

Real-time system detecting **3 dangerous driver states**:

| State | Signal | Method |
|---|---|---|
| **Drowsy** | Eye closure | EAR + XGBoost |
| **Yawning** | Mouth opening | MAR + LightGBM |
| **Distracted** | Head rotation | Yaw rule-based |

Runs at **30fps on CPU** — no GPU required.

---

## System Architecture

```
Webcam Frame
      ↓
MediaPipe FaceMesh (478 landmarks)
      ↓
Extract EAR + MAR + YAW
      ↓
Sliding Window (30 frames = 1s)
      ↓
┌─────────────────────────────────┐
│  Parallel Feature Extraction    │
│  EAR features → Drowsy Model   │
│  MAR features → Yawn Model     │
└─────────────────────────────────┘
      ↓
EMA Probability Smoothing (α=0.3)
      ↓
Decision Fusion
      ↓
Final State: ALERT / DROWSY / YAWNING / DISTRACTED
```

---

## ML Pipeline

### Drowsy Detection
- **Dataset**: Custom recorded videos (open/closed eyes)
- **Features**: EAR sliding window → `[mean, min, std, ratio]`
- **Threshold**: Data-driven via Otsu's method (EAR = 0.279)
- **Model**: XGBoost (Recall = 1.0 on test set)

### Yawn Detection
- **Dataset**: YawDD Mirror camera (318 videos, 2 subjects)
- **Challenge**: Video-level weak labels → frame-level relabeling
- **Signal Processing Pipeline**:
  1. Per-person P20 baseline normalization
  2. Adaptive threshold: `max(P95, mean + 2×std of no-yawn videos)`
  3. Moving average smoothing (window=5)
  4. Segment extraction with minimum duration filter (0.5s)
- **Features**: Normalized MAR sliding window → `[mean, max, std, ratio]`
- **Model**: LightGBM with `scale_pos_weight=4.5` (Recall = 1.0)

### Distraction Detection
- **Signal**: Head pose yaw angle via `cv2.solvePnP`
- **Method**: Rule-based threshold (|yaw| > 20°)

---

## Project Structure

```
driver-monitoring-system/
├── configs/
│   └── config.yaml              ← thresholds, camera settings
├── data/
│   ├── raw/                     ← NOT committed (too large)
│   │   ├── YawDD/
│   │   └── custom_videos/
│   └── processed/
│       ├── drowsy_features.csv
│       ├── yawn_features.csv
│       └── yawn_features_balanced.csv
├── models/
│   ├── face_landmarker.task     ← NOT committed
│   ├── drowsy_model.pkl
│   └── yawn_model.pkl
├── notebooks/
│   ├── 01_landmark_exploration.ipynb
│   ├── 02_data_understanding.py
│   ├── 03_eda_drowsy.py
│   └── 04_eda_yawn.py
├── src/
│   ├── metrics.py               ← EAR, MAR, Head Pose
│   ├── config.py                ← load config
│   ├── feature_extractor.py     ← sliding window
│   ├── data_pipeline_drowsy.py  ← extract EAR features
│   ├── data_pipeline_yawn.py    ← extract MAR features
│   ├── train_drowsy.py          ← train drowsy model
│   ├── train_yawn.py            ← train yawn model
│   ├── classifier.py            ← load + predict + EMA
│   └── pipeline.py              ← real-time webcam loop
├── assets/
│   └── demo.gif
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/driver-monitoring-system.git
cd driver-monitoring-system

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

### 2. Download MediaPipe Model

```bash
mkdir models
curl -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 3. Run Real-time DMS

```bash
python src/pipeline.py
```

---

## Train Your Own Models

### Drowsy Model

```bash
# 1. Record videos: open_01.mp4, close_01.mp4, ...
# 2. Extract features
python -m src.data_pipeline_drowsy

# 3. Train
python -m src.train_drowsy
```

### Yawn Model

```bash
# 1. Download YawDD dataset → data/raw/YawDD/
# 2. Extract features
python -m src.data_pipeline_yawn

# 3. Balance dataset
python notebooks/balance_yawn_data.py

# 4. Train
python -m src.train_yawn
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
model:
  path: "models/face_landmarker.task"
  num_faces: 1
  detection_confidence: 0.5

camera:
  device_id: 0
  width: 640
  height: 480
  fps: 30

thresholds:
  ear: 0.20
  mar: 0.60
  yaw: 20.0

sliding_window:
  size: 30      # frames (1s at 30fps)
  step: 10
```

---

## Key Technical Decisions

### Why MediaPipe over YOLO-face?
YOLO-face provides only 5 keypoints — insufficient for EAR/MAR calculation requiring 6 points per eye. MediaPipe FaceMesh provides 478 3D landmarks running at 30fps on CPU.

### Why Sliding Window over Frame-level?
Frame-level prediction causes false positives from normal eye blinks (~3 frames). A 30-frame sliding window captures temporal patterns, distinguishing genuine drowsiness (sustained EAR drop) from momentary blinks.

### Why Per-person MAR Normalization?
Different people have different natural mouth sizes. A fixed MAR threshold causes false positives for people with larger mouths. P20 baseline normalization makes the threshold universally applicable.

### Why LightGBM for Yawn vs XGBoost for Drowsy?
XGBoost learned a single `ratio > threshold` rule for drowsy detection (clean binary data). LightGBM distributed importance across all 4 MAR features (mean, max, std, ratio), learning more robust patterns for yawn detection where the boundary is less clear.

---

## Model Performance

| Model | Task | Recall | F1 | CV Recall |
|---|---|---|---|---|
| XGBoost | Drowsiness | 1.000 | 1.000 | 1.000 ± 0.000 |
| LightGBM | Yawning | 1.000 | 1.000 | 0.994 ± 0.011 |

> **Note**: Perfect accuracy reflects clean feature separation in training data. Real-world performance validated via webcam testing.

---

## Limitations & Future Work

- [ ] Per-user EAR/MAR calibration at startup
- [ ] Time-based thresholds (ms) instead of frame-based counters
- [ ] Collect intermediate eye states (half-open) for robustness
- [ ] Audio alert system
- [ ] Night driving mode (low-light optimization)
- [ ] Multi-driver support
- [ ] Ground-truth frame-level annotations to break label circularity

---

## Tech Stack

| Component | Technology |
|---|---|
| Face Landmarks | MediaPipe FaceMesh (478 pts) |
| Head Pose | OpenCV solvePnP |
| Drowsy Model | XGBoost |
| Yawn Model | LightGBM |
| Feature Extraction | Sliding Window + NumPy |
| Real-time Video | OpenCV VideoCapture |
| Language | Python 3.9+ |

---

## References

- Soukupová & Čech (2016) — Real-Time Eye Blink Detection using Facial Landmarks
- YawDD Dataset — Abtahi et al., Yawning Detection Dataset
- MediaPipe FaceMesh — Google Research
- Otsu (1979) — A Threshold Selection Method from Gray-Level Histograms

---
