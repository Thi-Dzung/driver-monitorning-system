"""
pipeline.py — v2
Real-time Driver Monitoring System
Integrated with ML models: drowsy_model + yawn_model
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    calculate_ear, calculate_mar, calculate_head_pose,
    LEFT_EYE, RIGHT_EYE, MOUTH
)
from src.config import load_config
from src.classifier import DMSClassifier

# ── Setup MediaPipe ────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ── Alert colors ───────────────────────────────────────────────
COLORS = {
    'ALERT'      : (0,   255, 0  ),  # green
    'DROWSY'     : (0,   0,   255),  # red
    'YAWNING'    : (0,   165, 255),  # orange
    'DISTRACTED' : (0,   255, 255),  # yellow
    'INITIALIZING': (128, 128, 128), # gray
}


def draw_ui(frame, ear, mar, yaw, pitch, result):
    """Draw metrics + alert on frame."""
    h, w = frame.shape[:2]
    state = result['state']
    color = COLORS.get(state, (255, 255, 255))

    # ── Background panel ───────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (320, 160), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (320, 160), (50, 50, 50), 1)

    # ── Metrics ────────────────────────────────────────────────
    cv2.putText(frame, f"EAR        : {ear:.3f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR        : {mar:.3f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"YAW        : {yaw:+.1f}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Drowsy prob: {result['drowsy_prob']:.3f}",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 100, 255), 1)
    cv2.putText(frame, f"Yawn prob  : {result['yawn_prob']:.3f}",
                (10, 125), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 165, 255), 1)
    cv2.putText(frame, f"State      : {state}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1)

    # ── Alert banner ───────────────────────────────────────────
    if state != 'ALERT' and state != 'INITIALIZING':
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (w//2 - 200, h//2 - 40),
                      (w//2 + 200, h//2 + 40),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, f"⚠ {state}!",
                    (w//2 - 150, h//2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)

    # ── Initializing bar ───────────────────────────────────────
    if state == 'INITIALIZING':
        n_filled = len(result.get('ear_buffer', []))
        progress = n_filled / 30
        cv2.rectangle(frame, (10, h-30), (w-10, h-10),
                      (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h-30),
                      (int(10 + (w-20)*progress), h-10),
                      (0, 255, 0), -1)
        cv2.putText(frame, f"Initializing... {int(progress*100)}%",
                    (w//2-100, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return frame


def run(config_path: str = "configs/config.yaml"):
    # ── Load config ───
    try:
        cfg = load_config(config_path)
        print("Config loaded!")
    except FileNotFoundError as e:
        print(f"Error {e}")
        return

    # ── Setup MediaPipe ──────
    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=cfg["model"]["path"]
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=cfg["model"]["num_faces"],
            min_face_detection_confidence=cfg["model"]["detection_confidence"],
            min_face_presence_confidence=cfg["model"]["presence_confidence"],
            min_tracking_confidence=cfg["model"]["tracking_confidence"],
        )
        print("MediaPipe loaded!")
    except Exception as e:
        print(f"MediaPipe setup failed: {e}")
        return

    # ── Setup Classifier ───────────────────────────────────────
    try:
        classifier = DMSClassifier(window_size=30)
        print("Classifier loaded!")
    except Exception as e:
        print(f"Classifier failed: {e}")
        return

    # ── Setup Camera ───────────────────────────────────────────
    cap = cv2.VideoCapture(cfg["camera"]["device_id"])
    if not cap.isOpened():
        print("Can't open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS,          cfg["camera"]["fps"])
    print("Camera opened!")
    print("DMS running — press Q to quit\n")

    # ── Main Loop ─────
    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Webcam disconnected!")
                    break

                h, w = frame.shape[:2]

                # Convert + detect
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image  = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb
                )
                timestamp_ms = int(time.time() * 1000)

                try:
                    result_mp = landmarker.detect_for_video(
                        mp_image, timestamp_ms
                    )
                except Exception as e:
                    print(f"Detection error: {e}")
                    continue

                ear = mar = yaw = pitch = 0.0

                if result_mp.face_landmarks:
                    lms = result_mp.face_landmarks[0]

                    # Calculate metrics
                    left_ear  = calculate_ear(lms, LEFT_EYE)
                    right_ear = calculate_ear(lms, RIGHT_EYE)
                    ear       = (left_ear + right_ear) / 2.0
                    mar       = calculate_mar(lms, MOUTH)
                    pitch, yaw, roll = calculate_head_pose(lms, w, h)

                    # Update classifier
                    classifier.update(ear=ear, mar=mar)

                    # Get prediction
                    clf_result = classifier.predict(yaw=yaw)
                    clf_result['ear_buffer'] = list(
                        classifier.ear_buffer
                    )

                else:
                    # No face detected
                    clf_result = {
                        'state'      : 'INITIALIZING',
                        'drowsy_prob': 0.0,
                        'yawn_prob'  : 0.0,
                        'is_ready'   : False,
                        'ear_buffer' : []
                    }

                # Draw UI
                frame = draw_ui(frame, ear, mar, yaw, pitch, clf_result)
                cv2.imshow("Driver Monitoring System", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nDMS stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cleanup done.")


if __name__ == "__main__":
    run()