#!/usr/bin/env python3
"""
Simplified Figma Hand Gesture Control Demo
Ultra-fast gesture detection without external dependencies.
"""

import cv2
import asyncio
import argparse
import time
import sys
from typing import Optional, Dict, List
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Import our optimized detector
from ultra_fast_detector import UltraFastGestureDetector


class SimpleFigmaDemo:
    """Simplified demo without browser integration."""

    def __init__(self):
        """Initialize the demo."""
        self.console = Console()
        self.running = False

        # Performance tracking
        self.frame_count = 0
        self.gesture_count = 0
        self.start_time = 0

        # Gesture action mapping
        self.gesture_actions = {
            "fist": "Select Tool",
            "open_palm": "Hand Tool (Pan)",
            "point": "Create Rectangle",
            "peace": "Create Circle",
            "unknown": "No Action"
        }

        # Recent gestures for display
        self.recent_gestures = []

        self.console.print("🎮 Figma Gesture Control Demo", style="bold green")

    async def run_demo(self, duration: int = 30):
        """Run the gesture detection demo."""
        # Initialize detector
        self.console.print(
            "⚡ Initializing ultra-fast gesture detector...", style="yellow")
        detector = UltraFastGestureDetector(
            target_fps=30,
            process_every_n_frames=1,
            detection_resolution=(160, 120)
        )
        detector.start_processing()

        # Open camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            self.console.print("❌ Cannot open camera", style="red")
            return

        self.running = True
        self.start_time = time.time()

        try:
            with Live(self._get_status_panel(detector), refresh_per_second=10) as live:
                self.console.print(
                    f"🎥 Starting {duration}s gesture detection demo...", style="green")
                self.console.print(
                    "👋 Show your hand gestures!", style="yellow")
                self.console.print("🤜 Fist = Select Tool", style="cyan")
                self.console.print("✋ Open Palm = Hand Tool", style="cyan")
                self.console.print("👉 Point = Create Rectangle", style="cyan")
                self.console.print("✌️ Peace = Create Circle", style="cyan")

                while self.running and time.time() - self.start_time < duration:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    self.frame_count += 1

                    # Ultra-fast gesture detection
                    results = detector.detect_gestures_threaded(frame)

                    # Process gestures
                    if results:
                        self._process_demo_gestures(results)

                    # Update live display
                    live.update(self._get_status_panel(detector))

                    # Maintain target FPS
                    await asyncio.sleep(1/30)

        except KeyboardInterrupt:
            self.console.print("🛑 Demo stopped by user", style="yellow")

        finally:
            detector.stop_processing()
            cap.release()
            await self._show_final_results(detector)

    def _process_demo_gestures(self, results):
        """Process detected gestures for demo display."""
        for result in results:
            if result.confidence > 0.5:  # Confidence threshold
                action = self.gesture_actions.get(result.gesture, "Unknown")

                # Add to recent gestures (keep last 5)
                gesture_info = {
                    "gesture": result.gesture,
                    "action": action,
                    "confidence": result.confidence,
                    "time": time.time()
                }

                self.recent_gestures.append(gesture_info)
                if len(self.recent_gestures) > 5:
                    self.recent_gestures.pop(0)

                self.gesture_count += 1

                # Only process one gesture at a time
                break

    def _get_status_panel(self, detector) -> Panel:
        """Create status panel for live display."""
        runtime = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / runtime if runtime > 0 else 0

        # Main stats table
        main_table = Table.grid(padding=1)
        main_table.add_column(style="cyan", justify="right")
        main_table.add_column(style="white")

        main_table.add_row("Runtime:", f"{runtime:.1f}s")
        main_table.add_row("Camera FPS:", f"{fps:.1f}")
        main_table.add_row("Total Gestures:", str(self.gesture_count))

        # Detector stats
        if detector:
            stats = detector.get_performance_stats()
            main_table.add_row(
                "Detection FPS:", f"{stats['effective_fps']:.1f}")
            main_table.add_row("Processed Frames:",
                               str(stats['frames_processed']))

        # Recent gestures table
        gesture_table = Table(title="Recent Gestures")
        gesture_table.add_column("Gesture", style="green")
        gesture_table.add_column("Figma Action", style="blue")
        gesture_table.add_column("Confidence", style="yellow")

        for gesture_info in self.recent_gestures[-3:]:  # Show last 3
            gesture_table.add_row(
                gesture_info["gesture"].title(),
                gesture_info["action"],
                f"{gesture_info['confidence']:.2f}"
            )

        # Combine panels
        main_panel = Panel(
            main_table, title="📊 Performance Stats", border_style="green")
        gesture_panel = Panel(gesture_table, border_style="blue")

        # Create combined layout
        combined_table = Table.grid()
        combined_table.add_row(main_panel)
        combined_table.add_row(gesture_panel)

        return Panel(
            combined_table,
            title="🎯 Figma Gesture Control Demo",
            subtitle="⚡ Ultra-Fast Detection",
            border_style="green"
        )

    async def _show_final_results(self, detector):
        """Show final demo results."""
        runtime = time.time() - self.start_time
        fps = self.frame_count / runtime
        stats = detector.get_performance_stats()

        self.console.print(f"\n📊 Demo Results:", style="bold green")

        results_table = Table(title="Final Performance Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")

        results_table.add_row("Runtime", f"{runtime:.1f}s")
        results_table.add_row("Camera FPS", f"{fps:.1f}")
        results_table.add_row("Detection FPS", f"{stats['effective_fps']:.1f}")
        results_table.add_row("Total Gestures", str(self.gesture_count))
        results_table.add_row("Processed Frames",
                              str(stats['frames_processed']))
        results_table.add_row("Gestures per Minute",
                              f"{self.gesture_count * 60 / runtime:.1f}")

        self.console.print(results_table)

        # Performance assessment
        self.console.print(f"\n🎯 Performance Assessment:", style="bold blue")

        if fps >= 25:
            self.console.print(
                "🚀 Excellent camera performance!", style="green")
        elif fps >= 15:
            self.console.print("✅ Good camera performance", style="green")
        else:
            self.console.print(
                "⚠️ Camera performance could be better", style="yellow")

        if stats['effective_fps'] >= 20:
            self.console.print(
                "🚀 Excellent detection performance!", style="green")
        elif stats['effective_fps'] >= 10:
            self.console.print("✅ Good detection performance", style="green")
        else:
            self.console.print(
                "⚠️ Detection performance could be better", style="yellow")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Figma Hand Gesture Control Demo")
    parser.add_argument("--duration", type=int, default=30,
                        help="Demo duration in seconds (default: 30)")

    args = parser.parse_args()

    console = Console()

    try:
        demo = SimpleFigmaDemo()
        await demo.run_demo(duration=args.duration)

    except Exception as e:
        console.print(f"❌ Demo error: {e}", style="red")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
