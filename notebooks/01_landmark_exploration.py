import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("MediaPipe version:", mp.__version__)

# Setup FaceLandmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
print("Setup OK!")

# Load image using OpenCV 
IMAGE_PATH = "image.jpg"
img_cv = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

# Convert mp.Image to object
mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=img_rgb
)

with FaceLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

# Visualize
img_cv = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# Draw all landmarks in the
for landmark in result.face_landmarks[0]:
    x = int(landmark.x * w)
    y = int(landmark.y * h)
    cv2.circle(img_rgb, (x, y), 1, (0, 255, 0), -1)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title(f"478 Face Landmarks")
plt.axis('off')
plt.show()