#!/usr/bin/env python3
"""
Quick setup script for Figma Hand Gesture Control
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_camera():
    """Check if camera is available."""
    print("📹 Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available!")
            cap.release()
            return True
        else:
            print("❌ Camera not found or in use by another application")
            return False
    except ImportError:
        print("❌ OpenCV not installed")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Figma Hand Gesture Control")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Check camera
    if not check_camera():
        print("⚠️  Camera issues detected. The app may not work properly.")

    print("\n🎉 Setup complete!")
    print("\nTo run the application:")
    print("  python main.py")
    print("\nTo run with options:")
    print("  python main.py --browser chrome")
    print("  python main.py --headless")


if __name__ == "__main__":
    main()
