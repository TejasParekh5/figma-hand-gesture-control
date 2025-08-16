"""
Gesture mapping system for converting gestures to Figma actions.
Simple, optimized mapping logic.
"""

from typing import Optional, Dict
from .gesture_detector import Gesture, GestureType


class GestureMapper:
    """Maps gestures to Figma actions."""

    def __init__(self):
        # Simple gesture to action mapping
        self.gesture_actions = {
            GestureType.SELECT: "select",
            GestureType.MOVE: "move",
            GestureType.CREATE_RECT: "create_rectangle",
            GestureType.CREATE_CIRCLE: "create_circle",
            GestureType.DELETE: "delete",
            GestureType.UNDO: "undo",
            GestureType.ZOOM: "zoom",
            GestureType.PAN: "pan"
        }

    def map_gesture_to_action(self, gesture: Gesture) -> Optional[str]:
        """Convert gesture to Figma action."""
        if gesture.confidence < 0.7:  # Confidence threshold
            return None

        return self.gesture_actions.get(gesture.type, None)
