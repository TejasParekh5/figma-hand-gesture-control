# Configuration for Figma Gesture Control
# Gesture detection settings - Improved for accuracy and speed
GESTURE_CONFIDENCE_THRESHOLD = 0.5  # Balanced threshold for accuracy
CAMERA_INDEX = 0
FRAME_WIDTH = 640  # Increased for better gesture recognition
FRAME_HEIGHT = 480  # Increased for better gesture recognition

# Performance settings
USE_IMPROVED_DETECTOR = True  # Use new multi-method detector
USE_GPU = True  # Enable GPU acceleration if available
YOLO_MODEL_PATH = "yolo11n.pt"  # Latest YOLO v11 nano model
DETECTION_FPS = 20  # Increased for smoother detection
GESTURE_COOLDOWN = 0.05  # Reduced for more responsive detection

# Multi-detection settings
ENABLE_MEDIAPIPE_FALLBACK = True  # Use MediaPipe as fallback
ENABLE_COLOR_DETECTION = True  # Use color detection as last resort
MIN_HAND_AREA = 5000  # Minimum hand area for color detection

# Browser settings
DEFAULT_BROWSER = "chrome"
BROWSER_HEADLESS = False
FIGMA_URL = "https://www.figma.com"

# Gesture mapping settings
GESTURE_TIMEOUT = 2.0  # seconds
MIN_GESTURE_DURATION = 0.1  # seconds

# Display settings
SHOW_LANDMARKS = True
SHOW_STATUS_PANEL = True
