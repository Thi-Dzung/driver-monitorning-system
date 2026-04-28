"""
pipeline.py
Real-time Driver Monitoring System
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

# Import functions từ metrics.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import (
    calculate_ear, calculate_mar, calculate_head_pose,
    LEFT_EYE, RIGHT_EYE, MOUTH
)

# ── Config ─────────────────────────────────────────────────────
MODEL_PATH     = "models/face_landmarker.task"
EAR_THRESHOLD  = 0.20
MAR_THRESHOLD  = 0.60
YAW_THRESHOLD  = 20.0
EAR_CONSEC     = 20   # frames
MAR_CONSEC     = 15
YAW_CONSEC     = 20

# ── Setup MediaPipe ────────────────────────────────────────────
BaseOptions         = mp.tasks.BaseOptions
FaceLandmarker      = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,   # ← VIDEO mode!
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5, #face in frame or not
    min_tracking_confidence=0.5  #is traking or not
)

# ── Counters ───────────────────────────────────────────────────
ear_counter = 0
mar_counter = 0
yaw_counter = 0

# ── Draw UI ────────────────────────────────────────────────────
def draw_ui(frame, ear, mar, yaw, pitch, alerts):
    h, w = frame.shape[:2]

    # Background panel
    cv2.rectangle(frame, (0, 0), (280, 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (280, 130), (50, 50, 50), 1)

    # Metrics
    cv2.putText(frame, f"EAR  : {ear:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR  : {mar:.3f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"YAW  : {yaw:+.1f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"PITCH: {pitch:+.1f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Alerts
    y_alert = 160
    for alert_text, color in alerts:
        cv2.putText(frame, alert_text, (w//2 - 150, y_alert),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y_alert += 50

    return frame

# ── Main Loop ──────────────────────────────────────────────────
def run():
    global ear_counter, mar_counter, yaw_counter

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam!")
        return

    print("✅ DMS đang chạy — nhấn Q để thoát")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            alerts = []

            # Convert BGR → RGB cho MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            # Detect — VIDEO mode cần timestamp
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Default values
            ear = mar = yaw = pitch = 0.0

            if result.face_landmarks:
                lms = result.face_landmarks[0]

                # Tính metrics
                left_ear  = calculate_ear(lms, LEFT_EYE)
                right_ear = calculate_ear(lms, RIGHT_EYE)
                ear       = (left_ear + right_ear) / 2.0
                mar       = calculate_mar(lms, MOUTH)
                pitch, yaw, roll = calculate_head_pose(lms, w, h)

                # ── EAR counter ───────────────────────────
                if ear < EAR_THRESHOLD:
                    ear_counter += 1
                    if ear_counter >= EAR_CONSEC:
                        alerts.append(("DROWSY!", (0, 0, 255)))
                else:
                    ear_counter = 0

                # ── MAR counter ───────────────────────────
                if mar > MAR_THRESHOLD:
                    mar_counter += 1
                    if mar_counter >= MAR_CONSEC:
                        alerts.append(("YAWNING!", (0, 165, 255)))
                else:
                    mar_counter = 0

                # ── YAW counter ───────────────────────────
                if abs(yaw) > YAW_THRESHOLD:
                    yaw_counter += 1
                    if yaw_counter >= YAW_CONSEC:
                        alerts.append(("DISTRACTED!", (0, 255, 255)))
                else:
                    yaw_counter = 0

            else:
                # Không detect được mặt
                alerts.append(("NO FACE", (128, 255, 128)))

            # Vẽ UI
            frame = draw_ui(frame, ear, mar, yaw, pitch, alerts)

            cv2.imshow("Driver Monitoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("DMS stopped.")

if __name__ == "__main__":
    run()