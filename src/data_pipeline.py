"""
data_pipeline.py
Extract facial features from YawDD dataset videos.
Output: data/processed/features.csv
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    calculate_ear, calculate_mar, calculate_head_pose,
    LEFT_EYE, RIGHT_EYE, MOUTH
)

# ── Config ─────────────────────────────────────────────────────
MIRROR_PATH  = r"D:\YawDD_dataset\Mirror\Mirror"
OUTPUT_CSV   = "data/processed/features.csv"
MODEL_PATH   = "models/face_landmarker.task"
SAMPLE_EVERY = 10

# ── Label mapping ──────────────────────────────────────────────
LABEL_MAP = {
    'Normal'         : 'alert',
    'Talking'        : 'alert',
    'Yawning'        : 'yawning',
    'TalkingYawning' : 'yawning',
    'Talkingyawning' : 'yawning',
}

# ── Setup MediaPipe ────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_label(filename: str):
    """Extract và map label từ tên file."""
    parts = filename.replace('.avi', '').split('-')
    raw_label = parts[-1]
    return LABEL_MAP.get(raw_label, None)


def extract_features_from_video(video_path: str) -> list:
    """
    Extract EAR, MAR, yaw, pitch, roll từ 1 video.
    Tạo landmarker mới cho mỗi video để reset timestamp.
    Returns list of feature dicts.
    """
    features = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        parts = video_path.replace('.avi', '').split('-')
        label = parts[-1]
        if not cap.isOpened():
            print(f"  ❌ Cannot open: {video_path}")
            return features

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % SAMPLE_EVERY != 0:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            timestamp_ms = int((frame_idx / fps) * 1000)

            try:
                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
            except Exception:
                frame_idx += 1
                continue

            if result.face_landmarks:
                lms = result.face_landmarks[0]

                left_ear  = calculate_ear(lms, LEFT_EYE)
                right_ear = calculate_ear(lms, RIGHT_EYE)
                ear       = (left_ear + right_ear) / 2.0
                mar       = calculate_mar(lms, MOUTH)
                if label == 'yawning' and mar < 0.8:
                        
                        frame_idx += 1
                        continue

                if label == 'alert' and mar > 0.6:
                    frame_idx += 1
                    continue

                features.append({
                    'ear'  : round(ear, 4),
                    'mar'  : round(mar, 4),
                    'label': label        
                })
                frame_idx += 1

        cap.release()

    return features


def run():
    """Main pipeline — extract features từ toàn bộ dataset."""
    all_rows     = []
    total_videos = 0
    failed_videos = 0

    os.makedirs("data/processed", exist_ok=True)

    print("🚀 Starting feature extraction...")
    print(f"   Source : {MIRROR_PATH}")
    print(f"   Output : {OUTPUT_CSV}")
    print(f"   Sampling: 1 frame every {SAMPLE_EVERY} frames\n")

    start_total = time.time()

    for root, dirs, files in os.walk(MIRROR_PATH):
        for filename in sorted(files):
            if not filename.endswith('.avi'):
                continue

            label = get_label(filename)
            if label is None:
                print(f"  ⚠️ Skipped (unknown label): {filename}")
                continue

            video_path    = os.path.join(root, filename)
            total_videos += 1

            print(f"[{total_videos}] {filename} → {label}")
            start = time.time()

            features = extract_features_from_video(video_path)

            if not features:
                failed_videos += 1
                print(f"  ⚠️ No features extracted")
                continue

            for feat in features:
                feat['label'] = label
                all_rows.append(feat)

            elapsed = time.time() - start
            print(f"  ✅ {len(features)} samples | {elapsed:.1f}s")

    # ── Save CSV ───────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.time() - start_total

    print(f"\n{'='*45}")
    print(f"✅ Feature extraction complete!")
    print(f"   Total videos    : {total_videos}")
    print(f"   Failed videos   : {failed_videos}")
    print(f"   Success rate    : {(total_videos-failed_videos)/total_videos*100:.1f}%")
    print(f"   Total samples   : {len(all_rows)}")
    print(f"   Total time      : {total_elapsed/60:.1f} min")
    print(f"   Output          : {OUTPUT_CSV}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nFeature stats:")
    print(df.drop('label', axis=1).describe().round(4))


if __name__ == "__main__":
    run()