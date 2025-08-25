"""
Gesture mapping system for converting gestures to Figma actions.
Simple, optimized mapping logic.
"""

from typing import Optional, Dict


class GestureMapper:
    """Maps gestures to Figma actions."""

    def __init__(self):
        # Simple gesture string to action mapping for ultra-fast detector
        self.gesture_actions = {
            "open_palm": "select",
            "point": "select",
            "grab": "move",
            "ok": "create_rectangle",
            "peace": "create_circle",
            "thumb_down": "delete",
            "rock": "undo",
            "index_point": "pan",
            "thumbs_up": "zoom"
        }

    def map_gesture_to_action(self, gesture_name: str, confidence: float = 1.0) -> Optional[str]:
        """Convert gesture string to Figma action."""
        if confidence < 0.7:  # Confidence threshold
            return None

        return self.gesture_actions.get(gesture_name, None)
