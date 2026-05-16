"""
data_pipeline_yawn.py — v5
Phase 1: Analyze no_yawn videos → compute tau_absolute
Phase 2: Extract windows from all videos
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from src.metrics import calculate_mar, MOUTH

# ── Config ─────────────────────────────────────────────────────
MIRROR_PATH       = r"D:\YawDD_dataset\Mirror\Mirror" # Path of video dataset
OUTPUT_CSV        = "data/processed/yawn_features.csv"
MODEL_PATH        = "models/face_landmarker.task"
WINDOW_SIZE       = 30
STEP_SIZE         = 10
FPS               = 30.0
MIN_YAWN_DURATION = 0.5   # seconds
BASELINE_PCT      = 20    # P20 for per-person baseline
Z_SCORE           = 2.0   # mean + 2*std

# ── Label mapping ──────────────────────────────────────────────
LABEL_MAP = {
    'Normal'         : 'no_yawn',
    'Talking'        : 'no_yawn',
    'Yawning'        : 'yawn',
    'TalkingYawning' : 'yawn',
    'Talkingyawning' : 'yawn',
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


# ── Core functions ─────────────────────────────────────────────
def get_label(filename: str):
    parts     = filename.replace('.avi', '').split('-')
    raw_label = parts[-1]
    return LABEL_MAP.get(raw_label, None)


def extract_mar_sequence(video_path: str) -> tuple:
    """Extract raw MAR values + FPS."""
    mar_sequence = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], FPS

        fps       = cap.get(cv2.CAP_PROP_FPS) or FPS
        total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total}...", end='\r')

            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(
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
                mar = calculate_mar(lms, MOUTH)
                mar_sequence.append(mar)

            frame_idx += 1

        cap.release()

    return mar_sequence, fps


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def smooth_mask(mask: np.ndarray, k: int) -> np.ndarray:
    m        = mask.astype(float)
    kernel   = np.ones(k) / k
    smoothed = np.convolve(m, kernel, mode='same')
    return smoothed > 0.5


def extract_segments(mask: np.ndarray) -> list:
    segments   = []
    in_segment = False
    start      = 0

    for i, val in enumerate(mask):
        if val and not in_segment:
            start      = i
            in_segment = True
        elif not val and in_segment:
            segments.append(np.arange(start, i))
            in_segment = False

    if in_segment:
        segments.append(np.arange(start, len(mask)))

    return segments


def get_normalized_mar(mar_sequence: list) -> tuple:
    """
    Normalize MAR per-person:
    baseline = P20 (robust to yawning outliers)
    r_mar    = mar / baseline
    """
    arr      = np.array(mar_sequence)
    arr      = moving_average(arr, window=5)
    baseline = float(np.percentile(arr, BASELINE_PCT))

    if baseline <= 0.01:
        baseline = 0.01

    r_mar = arr / baseline
    return r_mar, baseline


# ── Phase 1: Compute tau_absolute ─
def compute_tau_absolute(mirror_path: str) -> float:
    """
    Phase 1: Analyze no_yawn videos only.
    For each no_yawn video:
      1. Normalize MAR per-person (P20 baseline)
      2. Compute P95 of normalized MAR

    tau_absolute = mean(P95s) + Z_SCORE * std(P95s)
    → Upper control limit with 95% confidence
    → Any MAR above this is statistically anomalous
      for a non-yawning state
    """
    no_yawn_p95s = []
    video_count  = 0

    print("Phase 1: Analyzing no_yawn videos...")
    print(f"   Computing tau_absolute = mean + {Z_SCORE}*std\n")

    for root, dirs, files in os.walk(mirror_path):
        for filename in sorted(files):
            if not filename.endswith('.avi'):
                continue

            label = get_label(filename)
            if label != 'no_yawn':  # only process with no_yawn label
                continue

            video_path   = os.path.join(root, filename)
            video_count += 1

            print(f"  [{video_count}] {filename}")

            mar_seq, fps = extract_mar_sequence(video_path)
            if not mar_seq:
                continue

            r_mar, baseline = get_normalized_mar(mar_seq)
            p95             = float(np.percentile(r_mar, 95))
            no_yawn_p95s.append(p95)

            print(f"    baseline={baseline:.3f} | P95={p95:.3f}")

    # Compute tau
    p95_mean     = float(np.mean(no_yawn_p95s))
    p95_std      = float(np.std(no_yawn_p95s))
    tau_absolute = p95_mean + Z_SCORE * p95_std

    print(f"\n=== Phase 1 Results ===")
    print(f"  no_yawn videos  : {video_count}")
    print(f"  P95 mean        : {p95_mean:.4f}")
    print(f"  P95 std         : {p95_std:.4f}")
    print(f"  tau_absolute    : {p95_mean:.4f} + "
          f"{Z_SCORE}×{p95_std:.4f} = {tau_absolute:.4f}")
    print(f"  Interpretation  : no_yawn MAR rarely exceeds "
          f"{tau_absolute:.2f}x baseline\n")

    return tau_absolute


# ── Phase 2: Extract windows ───────────────────────────────────
def find_yawn_segments(r_mar: np.ndarray,
                       tau_absolute: float,
                       fps: float) -> np.ndarray:
    """
    Find yawn segments using adaptive threshold:
    tau = max(P95_of_video, tau_absolute)
    """
    tau_p95 = float(np.percentile(r_mar, 95))
    tau     = max(tau_p95, tau_absolute)

    mask     = r_mar > tau
    mask     = smooth_mask(mask, k=10)
    segments = extract_segments(mask)

    min_frames = int(fps * MIN_YAWN_DURATION)
    segments   = [s for s in segments if len(s) > min_frames]

    if not segments:
        return None

    return max(segments, key=lambda s: float(np.mean(r_mar[s])))


def create_windows(r_mar: np.ndarray,
                   yawn_segment,
                   video_label: str) -> list:
    windows  = []
    n_frames = len(r_mar)

    is_yawn = np.zeros(n_frames, dtype=bool)
    if yawn_segment is not None:
        is_yawn[yawn_segment] = True

    for start in range(
        0, n_frames - WINDOW_SIZE + 1, STEP_SIZE
    ):
        window     = r_mar[start : start + WINDOW_SIZE]
        win_mask   = is_yawn[start : start + WINDOW_SIZE]
        arr        = np.array(window)
        yawn_ratio = float(np.sum(win_mask) / WINDOW_SIZE)

        if yawn_ratio >= 0.5:
            window_label = 'yawn'
        elif yawn_ratio <= 0.2:
            window_label = 'no_yawn'
        else:
            continue

        if video_label == 'yawn' and window_label == 'no_yawn':
            continue
        if video_label == 'no_yawn' and window_label == 'yawn':
            continue

        windows.append({
            'mean' : round(float(np.mean(arr)), 4),
            'max'  : round(float(np.max(arr)),  4),
            'std'  : round(float(np.std(arr)),  4),
            'ratio': round(yawn_ratio,           4),
            'label': window_label
        })

    return windows


# ── Main ───────────────────────────────────────────────────────
def run():
    os.makedirs("data/processed", exist_ok=True)

    # ── Phase 1 ────────────────────────────────────────────────
    tau_absolute = compute_tau_absolute(MIRROR_PATH)

    # ── Phase 2 ────────────────────────────────────────────────
    all_rows      = []
    total_videos  = 0
    failed        = 0
    no_segment    = 0

    print(" Phase 2: Extracting features...")
    print(f"   tau_absolute = {tau_absolute:.4f} (data-driven)\n")

    start_total = time.time()

    for root, dirs, files in os.walk(MIRROR_PATH):
        for filename in sorted(files):
            if not filename.endswith('.avi'):
                continue

            label = get_label(filename)
            if label is None:
                continue

            video_path    = os.path.join(root, filename)
            total_videos += 1

            print(f"[{total_videos}] {filename} → {label}")
            start = time.time()

            mar_seq, fps = extract_mar_sequence(video_path)
            if not mar_seq:
                failed += 1
                print(f"  No MAR extracted")
                continue

            r_mar, baseline = get_normalized_mar(mar_seq)
            tau_p95         = float(np.percentile(r_mar, 95))
            tau             = max(tau_p95, tau_absolute)

            yawn_seg = find_yawn_segments(r_mar, tau_absolute, fps)

            if yawn_seg is None and label == 'yawn':
                no_segment += 1
                print(f"  No yawn segment | "
                      f"baseline={baseline:.3f} tau={tau:.3f}")
            elif yawn_seg is not None:
                print(f"  Segment: {len(yawn_seg)} frames "
                      f"({len(yawn_seg)/fps:.1f}s) | "
                      f"tau={tau:.3f}")

            windows = create_windows(r_mar, yawn_seg, label)
            all_rows.extend(windows)

            elapsed = time.time() - start
            print(f"  Windows: {len(windows)} | {elapsed:.1f}s")

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    elapsed_total = time.time() - start_total

    print(f"\n{'='*45}")
    print(f" Finished!")
    print(f"   tau_absolute  : {tau_absolute:.4f}")
    print(f"   Total videos  : {total_videos}")
    print(f"   Failed        : {failed}")
    print(f"   No segment    : {no_segment}")
    print(f"   Total samples : {len(all_rows)}")
    print(f"   Total time    : {elapsed_total/60:.1f} min")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nFeature stats per class:")
    print(df.groupby('label')[['mean','max','std','ratio']]
          .mean().round(4))


if __name__ == "__main__":
    run()
