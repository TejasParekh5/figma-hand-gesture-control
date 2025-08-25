#!/usr/bin/env python3
"""
Figma Hand Gesture Control - Main Application
Ultra-fast gesture detection for real-time Figma control.
"""

import cv2
import asyncio
import argparse
import time
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

import config
from ultra_fast_detector import UltraFastGestureDetector
from figma_control.browser_controller import BrowserController
from figma_control.gesture_mapper import GestureMapper


class FigmaGestureApp:
    """Main application orchestrating gesture detection and Figma control."""

    def __init__(self, browser_type: str = "chrome", headless: bool = False):
        self.console = Console()
        self.detector = UltraFastGestureDetector(
            target_fps=config.DETECTION_FPS,
            process_every_n_frames=1,
            detection_resolution=(160, 120)
        )
        self.browser = BrowserController(browser_type, headless)
        self.mapper = GestureMapper()
        self.running = False

        # Performance optimization
        self.target_fps = config.DETECTION_FPS
        self.frame_time = 1.0 / self.target_fps
        self.last_frame_time = 0

    async def initialize(self):
        """Initialize all components."""
        try:
            # Start the ultra-fast detector
            self.detector.start_processing()
            self.console.print("⚡ Ultra-fast detector started", style="green")

            await self.browser.initialize()
            await self.browser.navigate_to_figma()
            self.console.print("✅ Figma opened successfully", style="green")
            return True
        except Exception as e:
            self.console.print(f"❌ Initialization failed: {e}", style="red")
            return False

    async def run(self):
        """Main application loop with performance optimization."""
        if not await self.initialize():
            return

        self.running = True
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        try:
            self.console.print(
                "🚀 Starting optimized gesture detection...", style="green")
            self.console.print(
                f"📹 Camera: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} @ {self.target_fps}FPS", style="blue")
            self.console.print(
                "👋 Show your hand gestures to control Figma!", style="yellow")
            self.console.print("Press 'q' to quit", style="red")

            with Live(self.get_status_panel(), refresh_per_second=5) as live:
                while self.running:
                    current_time = time.time()

                    # Frame rate limiting for consistent performance
                    if current_time - self.last_frame_time < self.frame_time:
                        # Small sleep to prevent CPU overload
                        await asyncio.sleep(0.001)
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        continue

                    self.last_frame_time = current_time

                    # Ultra-fast gesture detection
                    results = self.detector.detect_gestures_threaded(frame)

                    # Process detected gestures
                    if results:
                        await self._process_gestures(results)

                    # Update status display
                    live.update(self.get_status_panel(results))

                    # Show video feed with detections
                    if results:
                        frame = self.detector.draw_results_fast(frame, results)

                    cv2.imshow('Figma Gesture Control (Optimized)', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            self.console.print("\n🛑 Shutting down...", style="yellow")
        finally:
            await self.cleanup(cap)

    async def _process_gestures(self, results):
        """Process detected gestures and execute Figma actions."""
        current_time = time.time()

        # Apply gesture cooldown
        if not hasattr(self, 'last_gesture_time'):
            self.last_gesture_time = 0

        if current_time - self.last_gesture_time < config.GESTURE_COOLDOWN:
            return

        for result in results:
            if result.confidence > config.GESTURE_CONFIDENCE_THRESHOLD:
                # Map gesture to Figma action
                action = self.mapper.map_gesture_to_action(result.gesture)

                if action:
                    try:
                        # Execute action in Figma
                        await self.browser.execute_action(action)
                        self.last_gesture_time = current_time

                        self.console.print(
                            f"✅ Executed: {result.gesture} → {action}", style="green")

                    except Exception as e:
                        self.console.print(
                            f"⚠️ Action failed: {e}", style="yellow")

                # Only process one gesture at a time for stability
                break

    def get_status_panel(self, results=None):
        """Create status display panel."""
        status = "🟢 Running" if self.running else "🔴 Stopped"

        # Create status table
        table = Table.grid(padding=1)
        table.add_column(style="cyan", justify="right")
        table.add_column(style="white")

        table.add_row("Status:", status)
        table.add_row("FPS:", f"{self.target_fps}")

        if results:
            gestures = [r.gesture for r in results]
            table.add_row("Detected:", ", ".join(gestures))

        # Get detector stats if available
        if hasattr(self.detector, 'get_performance_stats'):
            stats = self.detector.get_performance_stats()
            table.add_row("Detection FPS:",
                          f"{stats.get('effective_fps', 0):.1f}")

        return Panel(
            table,
            title="⚡ Ultra-Fast Figma Gesture Control",
            border_style="green"
        )

    async def cleanup(self, cap):
        """Clean up resources."""
        self.running = False

        # Stop the detector
        if self.detector:
            self.detector.stop_processing()

        cap.release()
        cv2.destroyAllWindows()
        await self.browser.close()

        self.console.print("🧹 Cleanup completed", style="blue")


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Figma Hand Gesture Control")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge", "brave"],
                        default="chrome", help="Browser to use")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode")

    args = parser.parse_args()

    app = FigmaGestureApp(args.browser, args.headless)
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())
