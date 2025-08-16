"""
Core hand gesture detection and recognition module.
Advanced computer vision pipeline with AI-enhanced accuracy.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import deque


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    POINT = "point"
    PINCH = "pinch"
    GRAB = "grab"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    PALM_OPEN = "palm_open"
    FIST = "fist"
    THUMB_UP = "thumb_up"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    UNKNOWN = "unknown"


@dataclass
class GestureData:
    """Data structure for gesture information."""
    type: GestureType
    confidence: float
    hand_landmarks: np.ndarray
    bounding_box: Tuple[int, int, int, int]
    timestamp: float
    hand_type: str  # "Left" or "Right"
    velocity: Optional[Tuple[float, float]] = None
    pressure: Optional[float] = None


class HandLandmarkProcessor:
    """Advanced hand landmark processing with geometric analysis."""

    def __init__(self):
        self.previous_landmarks = None
        self.gesture_history = deque(maxlen=10)

    def calculate_hand_metrics(self, landmarks) -> Dict[str, float]:
        """Calculate advanced hand metrics for gesture recognition."""
        if landmarks is None:
            return {}

        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])

        # Calculate key distances and angles
        metrics = {
            'thumb_index_distance': self._calculate_distance(points[4], points[8]),
            'index_middle_angle': self._calculate_angle(points[6], points[8], points[12]),
            'hand_openness': self._calculate_hand_openness(points),
            'palm_direction': self._calculate_palm_direction(points),
            'finger_curl': self._calculate_finger_curl(points),
            'wrist_angle': self._calculate_wrist_angle(points)
        }

        return metrics

    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return np.linalg.norm(p1 - p2)

    def _calculate_angle(self, p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
        """Calculate angle between three points."""
        v1 = p1 - vertex
        v2 = p2 - vertex

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return np.arccos(cos_angle)

    def _calculate_hand_openness(self, points: np.ndarray) -> float:
        """Calculate how open/closed the hand is."""
        # Calculate distances between fingertips and palm center
        palm_center = points[0]  # Wrist landmark as palm reference
        fingertips = [points[4], points[8], points[12], points[16], points[20]]

        distances = [self._calculate_distance(
            tip, palm_center) for tip in fingertips]
        return np.mean(distances)

    def _calculate_palm_direction(self, points: np.ndarray) -> np.ndarray:
        """Calculate palm normal vector."""
        # Use three palm points to calculate normal
        p1, p2, p3 = points[0], points[5], points[17]

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.zeros(3)

    def _calculate_finger_curl(self, points: np.ndarray) -> Dict[str, float]:
        """Calculate curl amount for each finger."""
        finger_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        curls = {}
        for finger, indices in finger_indices.items():
            # Calculate curl based on angle progression
            angles = []
            for i in range(len(indices) - 2):
                angle = self._calculate_angle(
                    points[indices[i]],
                    points[indices[i+1]],
                    points[indices[i+2]]
                )
                angles.append(angle)
            curls[finger] = np.mean(angles) if angles else 0.0

        return curls

    def _calculate_wrist_angle(self, points: np.ndarray) -> float:
        """Calculate wrist bend angle."""
        wrist = points[0]
        middle_mcp = points[9]
        index_mcp = points[5]

        return self._calculate_angle(index_mcp, wrist, middle_mcp)


class AdvancedGestureClassifier:
    """AI-enhanced gesture classification with machine learning."""

    def __init__(self):
        self.gesture_templates = self._initialize_gesture_templates()
        self.confidence_threshold = 0.7
        self.temporal_window = 5  # frames for temporal analysis

    def _initialize_gesture_templates(self) -> Dict[GestureType, Dict]:
        """Initialize gesture recognition templates with expected patterns."""
        return {
            GestureType.POINT: {
                'finger_curl': {'index': 0.2, 'middle': 0.8, 'ring': 0.8, 'pinky': 0.8},
                'hand_openness': (0.3, 0.6),
                'confidence_weight': 1.0
            },
            GestureType.PINCH: {
                'thumb_index_distance': (0.0, 0.1),
                'finger_curl': {'thumb': 0.3, 'index': 0.3},
                'confidence_weight': 1.0
            },
            GestureType.PALM_OPEN: {
                'hand_openness': (0.7, 1.0),
                'finger_curl': {'thumb': 0.2, 'index': 0.2, 'middle': 0.2, 'ring': 0.2, 'pinky': 0.2},
                'confidence_weight': 1.0
            },
            GestureType.FIST: {
                'hand_openness': (0.0, 0.3),
                'finger_curl': {'index': 0.8, 'middle': 0.8, 'ring': 0.8, 'pinky': 0.8},
                'confidence_weight': 1.0
            }
        }

    def classify_gesture(self, metrics: Dict[str, float], landmark_processor: HandLandmarkProcessor) -> GestureData:
        """Classify gesture based on hand metrics with AI enhancement."""
        best_match = GestureType.UNKNOWN
        best_confidence = 0.0

        for gesture_type, template in self.gesture_templates.items():
            confidence = self._calculate_gesture_confidence(metrics, template)

            if confidence > best_confidence and confidence > self.confidence_threshold:
                best_confidence = confidence
                best_match = gesture_type

        # Create gesture data
        gesture_data = GestureData(
            type=best_match,
            confidence=best_confidence,
            hand_landmarks=np.array([]),  # Will be filled by caller
            bounding_box=(0, 0, 0, 0),   # Will be calculated by caller
            timestamp=time.time(),
            hand_type="Unknown"  # Will be determined by caller
        )

        return gesture_data

    def _calculate_gesture_confidence(self, metrics: Dict[str, float], template: Dict) -> float:
        """Calculate confidence score for gesture match."""
        confidence_scores = []

        for key, expected in template.items():
            if key == 'confidence_weight':
                continue

            if key in metrics:
                if isinstance(expected, tuple):
                    # Range-based matching
                    min_val, max_val = expected
                    actual = metrics[key]
                    if min_val <= actual <= max_val:
                        score = 1.0
                    else:
                        # Calculate distance penalty
                        distance = min(abs(actual - min_val),
                                       abs(actual - max_val))
                        score = max(0.0, 1.0 - distance * 2)
                    confidence_scores.append(score)

                elif isinstance(expected, dict):
                    # Dictionary-based matching (e.g., finger_curl)
                    if key in metrics and isinstance(metrics[key], dict):
                        sub_scores = []
                        for sub_key, sub_expected in expected.items():
                            if sub_key in metrics[key]:
                                actual = metrics[key][sub_key]
                                distance = abs(actual - sub_expected)
                                sub_score = max(0.0, 1.0 - distance)
                                sub_scores.append(sub_score)
                        if sub_scores:
                            confidence_scores.append(np.mean(sub_scores))

                else:
                    # Direct value matching
                    actual = metrics[key]
                    distance = abs(actual - expected)
                    score = max(0.0, 1.0 - distance)
                    confidence_scores.append(score)

        return np.mean(confidence_scores) if confidence_scores else 0.0


class HandGestureDetector:
    """Main hand gesture detection system with advanced features."""

    def __init__(self,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 enable_ai_enhancement: bool = True):

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize processors
        self.landmark_processor = HandLandmarkProcessor()
        self.gesture_classifier = AdvancedGestureClassifier()

        # Configuration
        self.enable_ai_enhancement = enable_ai_enhancement
        self.frame_buffer = deque(maxlen=30)  # For temporal analysis
        self.gesture_history = deque(maxlen=10)

        # Performance tracking
        self.processing_times = deque(maxlen=100)

    def detect_gestures(self, frame: np.ndarray) -> Tuple[List[GestureData], np.ndarray]:
        """
        Detect and classify hand gestures in the given frame.

        Args:
            frame: Input video frame (BGR format)

        Returns:
            Tuple of (gesture_list, annotated_frame)
        """
        start_time = time.time()

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Process frame
        results = self.hands.process(rgb_frame)

        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        gestures = []

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand type (Left/Right)
                hand_type = "Unknown"
                if results.multi_handedness:
                    hand_type = results.multi_handedness[hand_idx].classification[0].label

                # Calculate hand metrics
                metrics = self.landmark_processor.calculate_hand_metrics(
                    hand_landmarks)

                # Classify gesture
                gesture_data = self.gesture_classifier.classify_gesture(
                    metrics, self.landmark_processor)

                # Update gesture data with additional information
                gesture_data.hand_type = hand_type
                gesture_data.hand_landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # Calculate bounding box
                h, w, _ = annotated_frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                gesture_data.bounding_box = (
                    x_min, y_min, x_max - x_min, y_max - y_min)

                gestures.append(gesture_data)

                # Draw landmarks and gesture label
                self.mp_draw.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Add gesture label
                label = f"{gesture_data.type.value} ({gesture_data.confidence:.2f})"
                cv2.putText(annotated_frame, label,
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)

        # Update performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        # Add performance info to frame
        fps = 1.0 / processing_time if processing_time > 0 else 0
        avg_fps = 1.0 / \
            np.mean(self.processing_times) if self.processing_times else 0

        cv2.putText(annotated_frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Update gesture history
        self.gesture_history.extend(gestures)

        return gestures, annotated_frame

    def get_gesture_statistics(self) -> Dict[str, any]:
        """Get performance and usage statistics."""
        if not self.processing_times:
            return {}

        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'avg_fps': 1.0 / np.mean(self.processing_times),
            'total_gestures_detected': len(self.gesture_history),
            'gesture_type_distribution': self._get_gesture_distribution()
        }

    def _get_gesture_distribution(self) -> Dict[str, int]:
        """Get distribution of detected gesture types."""
        distribution = {}
        for gesture in self.gesture_history:
            gesture_type = gesture.type.value
            distribution[gesture_type] = distribution.get(gesture_type, 0) + 1
        return distribution

    def calibrate_lighting(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze and suggest lighting calibration parameters."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate lighting metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        contrast_ratio = std_brightness / mean_brightness if mean_brightness > 0 else 0

        # Analyze histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        return {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'contrast_ratio': contrast_ratio,
            'is_well_lit': 80 <= mean_brightness <= 180,
            'has_good_contrast': contrast_ratio > 0.3,
            'histogram_peak': np.argmax(hist)
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.hands, 'close'):
            self.hands.close()


# Utility functions for gesture processing
def smooth_gesture_sequence(gestures: List[GestureData], window_size: int = 3) -> List[GestureData]:
    """Apply temporal smoothing to gesture sequence."""
    if len(gestures) < window_size:
        return gestures

    smoothed = []
    for i in range(len(gestures)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(gestures), i + window_size // 2 + 1)

        window_gestures = gestures[start_idx:end_idx]

        # Use majority voting for gesture type
        gesture_votes = {}
        confidence_sum = 0

        for gesture in window_gestures:
            gesture_type = gesture.type
            gesture_votes[gesture_type] = gesture_votes.get(
                gesture_type, 0) + 1
            confidence_sum += gesture.confidence

        # Get most common gesture type
        most_common_type = max(gesture_votes.keys(), key=gesture_votes.get)
        avg_confidence = confidence_sum / len(window_gestures)

        # Create smoothed gesture data
        smoothed_gesture = GestureData(
            type=most_common_type,
            confidence=avg_confidence,
            hand_landmarks=gestures[i].hand_landmarks,
            bounding_box=gestures[i].bounding_box,
            timestamp=gestures[i].timestamp,
            hand_type=gestures[i].hand_type
        )

        smoothed.append(smoothed_gesture)

    return smoothed


def calculate_gesture_velocity(current_gesture: GestureData, previous_gesture: GestureData) -> Tuple[float, float]:
    """Calculate gesture velocity between two frames."""
    if current_gesture.hand_landmarks.size == 0 or previous_gesture.hand_landmarks.size == 0:
        return (0.0, 0.0)

    # Use index finger tip for velocity calculation
    # x, y of index fingertip
    current_pos = current_gesture.hand_landmarks[8][:2]
    previous_pos = previous_gesture.hand_landmarks[8][:2]

    time_delta = current_gesture.timestamp - previous_gesture.timestamp
    if time_delta <= 0:
        return (0.0, 0.0)

    velocity_x = (current_pos[0] - previous_pos[0]) / time_delta
    velocity_y = (current_pos[1] - previous_pos[1]) / time_delta

    return (velocity_x, velocity_y)
