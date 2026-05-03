import os

mirror_path = r"D:\YawDD_dataset\Mirror\Mirror"

labels = set()
for root, dirs, files in os.walk(mirror_path):
    for file in files:
        if file.endswith('.avi'):
            # Extract label from file name
            # "1-FemaleNoGlasses-Yawning.avi" → "Yawning"
            parts = file.replace('.avi', '').split('-')
            label = parts[-1]
            labels.add(label)
            
print("Labels:", labels)

# couter label
from collections import Counter
label_counts = Counter()
for root, dirs, files in os.walk(mirror_path):
    for file in files:
        if file.endswith('.avi'):
            parts = file.replace('.avi', '').split('-')
            label = parts[-1]
            label_counts[label] += 1

print("\n=== CLASS DISTRIBUTION ===")
for label, count in label_counts.most_common():
    print(f"  {label}: {count} videos")

# === CLASS DISTRIBUTION ===
#   Normal: 105 videos
#   Yawning: 101 videos
#   Talking: 100 videos
#   TalkingYawning: 12 videos
#   Talkingyawning: 1 videos

# === The DMS task uses two classes:
# ALERT = Normal + Talking (driver is attentive)
# YAWNING = Yawning + TalkingYawning (driver is yawning)