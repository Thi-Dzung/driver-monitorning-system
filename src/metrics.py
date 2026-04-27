"""
Facial metrics for Driver Monitoring System.
EAR, MAR, Head Pose calculation.
"""
import cv2
import numpy as np

# ── Index constants ────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  39,  0,   291, 405, 17 ]
NOSE_TIP  = 1
CHIN      = 152
LEFT_EYE_OUTER  = 263
RIGHT_EYE_OUTER = 33
LEFT_MOUTH      = 287
RIGHT_MOUTH     = 57

MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0  ],   # nose tip
    [0.0,  -63.6,  -12.5 ],   # chin
    [-43.3,  32.7,  -26.0],   # left eye
    [43.3,   32.7,  -26.0],   # right eye
    [-28.9, -28.9,  -24.1],   # left mouth
    [28.9,  -28.9,  -24.1],   # right mouth
], dtype=np.float64)

# ── Core functions ───────────────────────────────────────────
def _distance(lm_a, lm_b) -> float:
    """Euclidean distance between two normalized landmarks."""
    return np.sqrt((lm_a.x - lm_b.x)**2 + (lm_a.y - lm_b.y)**2)

def calculate_ear(landmarks, eye_indices: list) -> float:
    """
    Eye Aspect Ratio — detects drowsiness.
    Normal open eye ~ 0.25-0.35
    Closed eye      < 0.20
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    return (_distance(p2, p6) + _distance(p3, p5)) / (2.0 * _distance(p1, p4))

def calculate_mar(landmarks, mouth_indices: list) -> float:
    """
    Mouth Aspect Ratio — detects yawning.
    Mouth closed ~ 0.0-0.3
    Yawning       > 0.6
    """
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in mouth_indices]
    return (_distance(p2, p6) + _distance(p3, p5)) / (2.0 * _distance(p1, p4))

def calculate_head_pose(landmarks, image_w: int, image_h: int):
    """
    Head pose estimation using solvePnP.
    Returns (pitch, yaw, roll) in degrees.
    Yaw  : left/right  — distraction detection
    Pitch: up/down     — nodding detection
    Roll : tilt        — supplementary
    """
    image_points = np.array([
        [landmarks[NOSE_TIP].x      * image_w, landmarks[NOSE_TIP].y      * image_h],
        [landmarks[CHIN].x          * image_w, landmarks[CHIN].y          * image_h],
        [landmarks[LEFT_EYE_OUTER].x * image_w, landmarks[LEFT_EYE_OUTER].y * image_h],
        [landmarks[RIGHT_EYE_OUTER].x * image_w, landmarks[RIGHT_EYE_OUTER].y * image_h],
        [landmarks[LEFT_MOUTH].x    * image_w, landmarks[LEFT_MOUTH].y    * image_h],
        [landmarks[RIGHT_MOUTH].x   * image_w, landmarks[RIGHT_MOUTH].y   * image_h],
    ], dtype=np.float64)

    focal_length  = image_w
    camera_matrix = np.array([
        [focal_length, 0,            image_w / 2],
        [0,            focal_length, image_h / 2],
        [0,            0,            1           ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags= cv2.SOLVEPNP_ITERATIVE
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll