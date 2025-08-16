"""
Optimized gesture detection using MediaPipe.
Minimal implementation focused on core gestures for Figma control.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import time


class GestureType(Enum):
    """Core gesture types for Figma control."""
    SELECT = "select"          # Point gesture
    MOVE = "move"             # Drag gesture
    CREATE_RECT = "rect"      # Rectangle gesture
    CREATE_CIRCLE = "circle"  # Circle gesture
    DELETE = "delete"         # Fist gesture
    UNDO = "undo"            # Swipe left
    ZOOM = "zoom"            # Pinch gesture
    PAN = "pan"              # Palm open
    NONE = "none"


@dataclass
class Gesture:
    """Simple gesture data structure."""
    type: GestureType
    confidence: float
    position: tuple[int, int]
    timestamp: float


class GestureDetector:
    """Optimized gesture detector using MediaPipe."""

    def __init__(self, min_confidence: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Single hand for simplicity
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> List[Gesture]:
        """Detect gestures in frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        gestures = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self._classify_gesture(hand_landmarks, frame.shape)
                if gesture.type != GestureType.NONE:
                    gestures.append(gesture)

                # Draw landmarks for feedback
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return gestures

    def _classify_gesture(self, landmarks, frame_shape) -> Gesture:
        """Classify hand gesture based on landmarks."""
        # Convert landmarks to pixel coordinates
        h, w, _ = frame_shape
        points = []
        for lm in landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))

        # Get key landmarks
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]
        wrist = points[0]

        # Calculate distances
        thumb_index_dist = self._distance(thumb_tip, index_tip)
        index_middle_dist = self._distance(index_tip, middle_tip)

        # Gesture classification logic
        gesture_type = GestureType.NONE
        confidence = 0.0

        # Point gesture (index finger extended)
        if self._is_finger_extended(points, 8) and not self._is_finger_extended(points, 12):
            gesture_type = GestureType.SELECT
            confidence = 0.9

        # Pinch gesture (thumb and index close)
        elif thumb_index_dist < 30:
            gesture_type = GestureType.ZOOM
            confidence = 0.8

        # Fist gesture (all fingers closed)
        elif not any(self._is_finger_extended(points, tip) for tip in [8, 12, 16, 20]):
            gesture_type = GestureType.DELETE
            confidence = 0.85

        # Open palm (all fingers extended)
        elif all(self._is_finger_extended(points, tip) for tip in [8, 12, 16, 20]):
            gesture_type = GestureType.PAN
            confidence = 0.8

        # Use index finger position as gesture position
        position = index_tip

        return Gesture(
            type=gesture_type,
            confidence=confidence,
            position=position,
            timestamp=time.time()
        )

    def _is_finger_extended(self, points: List[tuple], tip_idx: int) -> bool:
        """Check if finger is extended based on landmarks."""
        if tip_idx == 4:  # Thumb
            return points[tip_idx][0] > points[tip_idx - 1][0]
        else:  # Other fingers
            return points[tip_idx][1] < points[tip_idx - 2][1]

    def _distance(self, p1: tuple, p2: tuple) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
