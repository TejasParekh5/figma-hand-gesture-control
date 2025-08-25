# Figma Hand Gesture Control System
"""
Ultra-fast gesture detection for real-time Figma control.
"""

__version__ = "3.0.0"
__author__ = "Figma Gesture Control Team"

from .browser_controller import BrowserController
from .gesture_mapper import GestureMapper

__all__ = [
    "BrowserController",
    "GestureMapper"
]
