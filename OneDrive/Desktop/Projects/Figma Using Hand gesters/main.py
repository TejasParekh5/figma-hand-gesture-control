#!/usr/bin/env python3
"""
Figma Hand Gesture Control - Main Application
Minimal, optimized implementation for controlling Figma through hand gestures.
"""

import cv2
import asyncio
import argparse
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from figma_control.gesture_detector import GestureDetector
from figma_control.browser_controller import BrowserController
from figma_control.gesture_mapper import GestureMapper


class FigmaGestureApp:
    """Main application orchestrating gesture detection and Figma control."""

    def __init__(self, browser_type: str = "chrome", headless: bool = False):
        self.console = Console()
        self.detector = GestureDetector()
        self.browser = BrowserController(browser_type, headless)
        self.mapper = GestureMapper()
        self.running = False

    async def initialize(self):
        """Initialize all components."""
        try:
            await self.browser.initialize()
            await self.browser.navigate_to_figma()
            self.console.print("✅ Figma opened successfully", style="green")
            return True
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            return False

    async def run(self):
        """Main application loop."""
        if not await self.initialize():
            return

        self.running = True
        cap = cv2.VideoCapture(0)

        try:
            with Live(self.get_status_panel(), refresh_per_second=10) as live:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Detect gestures
                    gestures = self.detector.detect(frame)

                    # Process each gesture
                    for gesture in gestures:
                        action = self.mapper.map_gesture_to_action(gesture)
                        if action:
                            await self.browser.execute_action(action, gesture)

                    # Update status display
                    live.update(self.get_status_panel(gestures))

                    # Display video feed
                    cv2.imshow('Gesture Control', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            self.console.print("\n🛑 Shutting down...", style="yellow")
        finally:
            await self.cleanup(cap)

    def get_status_panel(self, gestures=None):
        """Create status display panel."""
        status = "🟢 Running" if self.running else "🔴 Stopped"
        gesture_info = ""
        if gestures:
            gesture_info = f"\nDetected: {[g.type.value for g in gestures]}"

        content = f"Status: {status}{gesture_info}\n\nPress 'q' to quit"
        return Panel(content, title="Figma Gesture Control", border_style="blue")

    async def cleanup(self, cap):
        """Clean up resources."""
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        await self.browser.close()


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Figma Hand Gesture Control")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge"],
                        default="chrome", help="Browser to use")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode")

    args = parser.parse_args()

    app = FigmaGestureApp(args.browser, args.headless)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
