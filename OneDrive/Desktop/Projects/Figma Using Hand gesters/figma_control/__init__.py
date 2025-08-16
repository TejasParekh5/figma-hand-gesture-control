# Figma Hand Gesture Control System
"""
Optimized minimal implementation for browser automation control.
"""

__version__ = "1.0.0"
__author__ = "Figma Gesture Control Team"

from .gesture_detector import GestureDetector, Gesture, GestureType
from .browser_controller import BrowserController
from .gesture_mapper import GestureMapper

__all__ = [
    "GestureDetector",
    "Gesture",
    "GestureType",
    "BrowserController",
    "GestureMapper"
]
