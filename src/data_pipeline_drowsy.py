
import cv2
import mediapipe as mp
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import calculate_ear, LEFT_EYE, RIGHT_EYE

# ── Config ─────────────────────────────────────────────────────
VIDEO_PATH = r"D:/monitoring_system_project/eyesdata"
MODEL_PATH = "models/face_landmarker.task"

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
    """
    Read label from file name
    open_01.mp4  → 'alert'
    close_01.mp4 → 'drowsy'
    """
    name = filename.lower()
    if name.startswith('open'):
        return 'alert'
    elif name.startswith('close'):
        return 'drowsy'
    return None


def extract_ear_sequence(video_path: str) -> list:
    """
    Load video and compute EAR of each frame
    → Return a list of EAR values over time

    Returns: [ear_frame1, ear_frame2, ...]
    """
    ear_sequence = []

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  Can't open video: {video_path}")
            return ear_sequence

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            # Timestamp tăng dần
            timestamp_ms = int((frame_idx / fps) * 1000)

            try:
                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )
            except Exception:
                frame_idx += 1
                continue

            # Get only frames with detected faces
            if result.face_landmarks:
                lms       = result.face_landmarks[0]
                left_ear  = calculate_ear(lms, LEFT_EYE)
                right_ear = calculate_ear(lms, RIGHT_EYE)
                ear       = (left_ear + right_ear) / 2.0
                ear_sequence.append(ear)

            frame_idx += 1

        cap.release()

    return ear_sequence

def analyze_threshold(video_folder: str):
    """
    Collect all EAR values from open/close videos
    → Compute the threshold based on the midpoint
    """
    open_ears  = []
    close_ears = []

    for filename in sorted(os.listdir(video_folder)):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        label = get_label(filename)
        if label is None:
            continue

        video_path = os.path.join(video_folder, filename)
        print(f"Processing: {filename} → {label}")

        ear_seq = extract_ear_sequence(video_path)
        if not ear_seq:
            continue

        if label == 'alert':
            open_ears.extend(ear_seq)
        elif label == 'drowsy':
            close_ears.extend(ear_seq)

    # ── Stats ──────────────────────────────────────────────────
    open_mean  = np.mean(open_ears)
    close_mean = np.mean(close_ears)

    print(f"\n EAR Distribution")
    print(f"Open  — mean: {open_mean:.4f} "
          f"| std: {np.std(open_ears):.4f} "
          f"| min: {np.min(open_ears):.4f} "
          f"| max: {np.max(open_ears):.4f}")
    print(f"Close — mean: {close_mean:.4f} "
          f"| std: {np.std(close_ears):.4f} "
          f"| min: {np.min(close_ears):.4f} "
          f"| max: {np.max(close_ears):.4f}")

    # ── Cách 1: Midpoint ───────────────────────────────────────
    threshold_mid = (open_mean + close_mean) / 2
    print(f"\nMidpoint threshold : {threshold_mid:.4f}")

    # ── Cách 2: Otsu-like ──────────────────────────────────────
    best_threshold = threshold_mid
    best_score     = 0

    for t in np.arange(0.05, 0.5, 0.01):
        tpr = sum(e < t for e in close_ears) / len(close_ears)
        tnr = sum(e >= t for e in open_ears)  / len(open_ears)
        if tpr + tnr > 0:
            score = 2 * tpr * tnr / (tpr + tnr)
            if score > best_score:
                best_score     = score
                best_threshold = t

    print(f"Optimal threshold  : {best_threshold:.4f} "
          f"(score={best_score:.4f})")

    #  Plot 
    import matplotlib.pyplot as plt
    os.makedirs("assets", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(open_ears,  bins=50, alpha=0.6,
             color='blue', label='Open (alert)',   density=True)
    plt.hist(close_ears, bins=50, alpha=0.6,
             color='red',  label='Close (drowsy)', density=True)
    plt.axvline(x=best_threshold, color='green',
                linestyle='--', linewidth=2,
                label=f'Threshold={threshold_mid:.3f}')
    plt.axvline(x=best_threshold, color='orange',
                linestyle=':', linewidth=2,
                label=f'Midpoint={threshold_mid:.3f}')
    plt.xlabel("EAR Value")
    plt.ylabel("Density")
    plt.title("EAR Distribution: Open vs Close Eyes")
    plt.legend()
    plt.savefig("assets/ear_threshold.png", dpi=100)
    plt.show()
    print("Saved: assets/ear_threshold.png")

    return best_threshold
def create_windows(ear_sequence: list, label: str,
                   ear_threshold: float,
                   window_size: int = 30,
                   step_size: int = 10) -> list:
    """
    Create sliding windows from EAR sequence.
    
    Filter:
      close video: window phải có > 50% frames nhắm mắt
      open video:  window phải có > 80% frames mở mắt
      
    Features of each window:
      mean  = np.mean(window)
      min   = np.min(window)
      std   = np.std(window)
      ratio = % frames EAR < threshold
    """
    windows = []
    ears    = ear_sequence

    for start in range(0, len(ears) - window_size + 1, step_size):
        window = ears[start : start + window_size]

        # ── Tính ratio trước để filter ────────────────────────
        ratio = sum(e < ear_threshold for e in window) / window_size

        # ── Filter window không đúng label ────────────────────
        if label == 'drowsy' and ratio < 0.5:
            # Video close nhưng window này
            # > 50% frames mắt mở → bỏ
            continue

        if label == 'alert' and ratio > 0.2:
            # Video open nhưng window này
            # > 20% frames mắt nhắm → bỏ
            continue

        arr = np.array(window)

        windows.append({
            'mean' : round(float(np.mean(arr)), 4),
            'min'  : round(float(np.min(arr)),  4),
            'std'  : round(float(np.std(arr)),  4),
            'ratio': round(ratio,               4),
            'label': label
        })

    return windows
if __name__ == "__main__":
    import pandas as pd
    os.makedirs("data/processed", exist_ok=True)
    EAR_THRESHOLD = 0.279 
    WINDOW_SIZE   = 30
    STEP_SIZE     = 10
    all_rows = []

    print(" Sliding Window Feature Extraction \n")

    for filename in sorted(os.listdir(VIDEO_PATH)):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        label = get_label(filename)
        if label is None:
            continue

        video_path = os.path.join(VIDEO_PATH, filename)
        print(f"Processing: {filename} → {label}")

        # Extract EAR sequence
        ear_seq = extract_ear_sequence(video_path)
        if not ear_seq:
            print(f" No EAR extracted")
            continue

        print(f"  EAR frames : {len(ear_seq)}")

        # Create windows
        windows = create_windows(
            ear_seq, label,
            EAR_THRESHOLD,
            WINDOW_SIZE,
            STEP_SIZE
        )
        all_rows.extend(windows)
        print(f"  Windows    : {len(windows)}")

    # Save CSV
    df = pd.DataFrame(all_rows)
    df.to_csv("data/processed/drowsy_features.csv", index=False)

    print(f"\n{'='*45}")
    print(f"Finished!")
    print(f"   Total samples : {len(df)}")
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nFeature stats per class:")
    print(df.groupby('label').describe().round(4))
