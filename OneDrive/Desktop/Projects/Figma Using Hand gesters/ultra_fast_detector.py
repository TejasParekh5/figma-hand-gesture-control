"""
Ultra-fast gesture detection optimized for real-time performance.
Focus on speed over accuracy with multiple optimization strategies.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from threading import Thread
import queue
from rich.console import Console

console = Console()


@dataclass
class FastGestureResult:
    """Simplified gesture result for speed."""
    gesture: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float = 0.0


class UltraFastGestureDetector:
    """
    Ultra-fast gesture detector optimized for real-time performance:
    1. Multi-threaded processing
    2. Frame skipping
    3. Simplified gesture classification
    4. Optimized OpenCV operations
    5. Reduced resolution processing
    """

    def __init__(self,
                 target_fps: int = 30,
                 process_every_n_frames: int = 2,
                 detection_resolution: Tuple[int, int] = (160, 120)):
        """
        Initialize ultra-fast detector.

        Args:
            target_fps: Target frames per second
            process_every_n_frames: Process every N frames for speed
            detection_resolution: Resolution for gesture detection
        """
        self.target_fps = target_fps
        self.process_every_n_frames = process_every_n_frames
        self.detection_resolution = detection_resolution

        # Frame processing
        self.frame_count = 0
        self.last_results = []
        self.result_cache_time = 0.0

        # Multi-threading
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing_thread = None
        self.running = False

        # Optimized parameters
        self.min_contour_area = 2000
        self.max_contours = 5

        # Skin detection parameters (optimized for speed)
        self.skin_lower = np.array([0, 48, 80], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)

        # Morphological operations kernel
        self.morph_kernel = np.ones((3, 3), np.uint8)

        # Gesture classification cache
        self.gesture_cache = {}

        # Performance tracking
        self.stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "avg_processing_time": 0.0,
            "gesture_counts": {}
        }

        console.print(
            "⚡ Ultra-Fast Gesture Detector initialized", style="green")
        console.print(f"🎯 Target FPS: {target_fps}", style="blue")
        console.print(
            f"📐 Detection resolution: {detection_resolution}", style="blue")

    def start_processing(self):
        """Start background processing thread."""
        if not self.running:
            self.running = True
            self.processing_thread = Thread(
                target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            console.print("🚀 Background processing started", style="green")

    def stop_processing(self):
        """Stop background processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        console.print("🛑 Background processing stopped", style="yellow")

    def detect_gestures_fast(self, frame: np.ndarray) -> List[FastGestureResult]:
        """
        Fast gesture detection with optimizations.

        Args:
            frame: Input video frame

        Returns:
            List of detected gestures
        """
        start_time = time.time()

        # Frame skipping for speed
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            self.stats["frames_skipped"] += 1
            # Return cached results
            if time.time() - self.result_cache_time < 0.2:  # Use cache for 200ms
                return self.last_results
            return []

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, self.detection_resolution)

        # Detect hands using optimized color detection
        results = self._detect_hands_fast(small_frame, frame.shape[:2])

        # Cache results
        self.last_results = results
        self.result_cache_time = start_time

        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time)

        return results

    def detect_gestures_threaded(self, frame: np.ndarray) -> List[FastGestureResult]:
        """
        Threaded gesture detection for maximum performance.

        Args:
            frame: Input video frame

        Returns:
            List of detected gestures
        """
        # Add frame to processing queue (non-blocking)
        try:
            self.frame_queue.put_nowait((frame.copy(), time.time()))
        except queue.Full:
            pass  # Skip frame if queue is full

        # Get results from processing queue (non-blocking)
        try:
            results, timestamp = self.result_queue.get_nowait()
            # Only return results if they're recent
            if time.time() - timestamp < 0.1:  # 100ms freshness
                return results
        except queue.Empty:
            pass

        # Return last cached results or empty
        return self.last_results if time.time() - self.result_cache_time < 0.2 else []

    def _processing_loop(self):
        """Background processing loop for threaded detection."""
        while self.running:
            try:
                # Get frame from queue with timeout
                frame, frame_time = self.frame_queue.get(timeout=0.1)

                # Process frame
                start_time = time.time()

                # Resize for speed
                small_frame = cv2.resize(frame, self.detection_resolution)

                # Detect gestures
                results = self._detect_hands_fast(small_frame, frame.shape[:2])

                # Add results to output queue
                try:
                    self.result_queue.put_nowait((results, start_time))
                except queue.Full:
                    # Remove old result and add new one
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait((results, start_time))
                    except queue.Empty:
                        pass

                # Update statistics
                processing_time = time.time() - start_time
                self._update_stats(processing_time)

            except queue.Empty:
                continue
            except Exception as e:
                console.print(f"⚠️ Processing error: {e}", style="yellow")

    def _detect_hands_fast(self, small_frame: np.ndarray, original_size: Tuple[int, int]) -> List[FastGestureResult]:
        """
        Fast hand detection using optimized color segmentation.

        Args:
            small_frame: Resized frame for processing
            original_size: Original frame size for coordinate scaling

        Returns:
            List of detected gestures
        """
        results = []

        # Convert to HSV
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # Create skin mask with optimized parameters
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        # Fast morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                self.morph_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                self.morph_kernel, iterations=2)

        # Find contours (limit number for speed)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea,
                          reverse=True)[:self.max_contours]

        # Scale factors for coordinate conversion
        scale_x = original_size[1] / self.detection_resolution[0]
        scale_y = original_size[0] / self.detection_resolution[1]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Scale back to original size
                x_orig = int(x * scale_x)
                y_orig = int(y * scale_y)
                w_orig = int(w * scale_x)
                h_orig = int(h * scale_y)

                # Fast gesture classification
                gesture = self._classify_gesture_fast(contour, area)

                # Create result
                result = FastGestureResult(
                    gesture=gesture,
                    # Simple confidence based on area
                    confidence=min(0.9, area / 10000),
                    bbox=(x_orig, y_orig, x_orig + w_orig, y_orig + h_orig),
                    timestamp=time.time()
                )
                results.append(result)

        return results

    def _classify_gesture_fast(self, contour: np.ndarray, area: float) -> str:
        """
        Ultra-fast gesture classification using simple geometric properties.

        Args:
            contour: Hand contour
            area: Contour area

        Returns:
            Detected gesture name
        """
        # Use cached result if available
        contour_hash = hash(contour.tobytes())
        if contour_hash in self.gesture_cache:
            return self.gesture_cache[contour_hash]

        try:
            # Calculate simple properties
            perimeter = cv2.arcLength(contour, True)

            # Convex hull for solidity calculation
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            # Calculate solidity (area/hull_area)
            solidity = area / hull_area if hull_area > 0 else 0

            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1

            # Simple gesture classification rules
            if solidity > 0.9 and aspect_ratio < 1.5:
                gesture = "fist"
            elif solidity > 0.75:
                gesture = "open_palm"
            elif aspect_ratio > 2.0:
                gesture = "point"
            elif solidity < 0.6:
                gesture = "peace"
            else:
                gesture = "open_palm"

            # Cache result (limit cache size)
            if len(self.gesture_cache) < 100:
                self.gesture_cache[contour_hash] = gesture

            return gesture

        except Exception:
            return "unknown"

    def _update_stats(self, processing_time: float):
        """Update performance statistics."""
        self.stats["frames_processed"] += 1

        # Update average processing time
        count = self.stats["frames_processed"]
        current_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (
            current_avg * (count - 1) + processing_time) / count

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        total_frames = self.stats["frames_processed"] + \
            self.stats["frames_skipped"]
        skip_rate = (self.stats["frames_skipped"] /
                     total_frames * 100) if total_frames > 0 else 0
        fps = 1.0 / \
            self.stats["avg_processing_time"] if self.stats["avg_processing_time"] > 0 else 0

        return {
            **self.stats,
            "skip_rate": skip_rate,
            "effective_fps": fps,
            "total_frames": total_frames
        }

    def draw_results_fast(self, frame: np.ndarray, results: List[FastGestureResult]) -> np.ndarray:
        """
        Fast drawing of results with minimal operations.

        Args:
            frame: Input frame
            results: Detection results

        Returns:
            Frame with drawn results
        """
        for result in results:
            x1, y1, x2, y2 = result.bbox

            # Simple rectangle (faster than thick lines)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Simple text
            cv2.putText(frame, result.gesture, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def __del__(self):
        """Cleanup."""
        self.stop_processing()


def test_ultra_fast_detector():
    """Test the ultra-fast gesture detector."""
    console.print("⚡ Testing Ultra-Fast Gesture Detector", style="bold green")

    # Initialize detector
    detector = UltraFastGestureDetector(
        target_fps=30,
        process_every_n_frames=1,  # Process every frame for testing
        detection_resolution=(160, 120)
    )

    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("❌ Cannot open camera", style="red")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    console.print("📹 Testing fast detection (100 frames)...", style="blue")

    # Test both modes
    test_results = {}

    # Test 1: Direct detection
    console.print("🔄 Testing direct detection...", style="cyan")
    start_time = time.time()
    frame_count = 0
    detection_count = 0

    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_gestures_fast(frame)
        detection_count += len(results)
        frame_count += 1

    direct_time = time.time() - start_time
    direct_fps = frame_count / direct_time

    test_results["direct"] = {
        "fps": direct_fps,
        "detections": detection_count,
        "time": direct_time
    }

    # Test 2: Threaded detection
    console.print("🧵 Testing threaded detection...", style="cyan")
    detector.start_processing()

    start_time = time.time()
    frame_count = 0
    detection_count = 0

    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_gestures_threaded(frame)
        detection_count += len(results)
        frame_count += 1

    threaded_time = time.time() - start_time
    threaded_fps = frame_count / threaded_time

    test_results["threaded"] = {
        "fps": threaded_fps,
        "detections": detection_count,
        "time": threaded_time
    }

    detector.stop_processing()
    cap.release()

    # Display results
    console.print("\n📊 Performance Comparison", style="bold green")

    from rich.table import Table
    table = Table(title="Ultra-Fast Detector Results")
    table.add_column("Method", style="cyan")
    table.add_column("FPS", style="green")
    table.add_column("Detections", style="yellow")
    table.add_column("Time (s)", style="blue")

    table.add_row("Direct", f"{direct_fps:.1f}", str(
        test_results["direct"]["detections"]), f"{direct_time:.2f}")
    table.add_row("Threaded", f"{threaded_fps:.1f}", str(
        test_results["threaded"]["detections"]), f"{threaded_time:.2f}")

    console.print(table)

    # Performance assessment
    best_fps = max(direct_fps, threaded_fps)

    console.print(
        f"\n🎯 Best Performance: {best_fps:.1f} FPS", style="bold green")

    if best_fps >= 25:
        console.print(
            "🚀 Excellent! Real-time performance achieved", style="green")
    elif best_fps >= 15:
        console.print("✅ Good performance for real-time use", style="green")
    elif best_fps >= 10:
        console.print("⚠️ Acceptable for some applications", style="yellow")
    else:
        console.print("❌ Too slow for real-time use", style="red")

    # Show detector statistics
    stats = detector.get_performance_stats()
    console.print(f"\n📈 Processing Stats:", style="blue")
    console.print(f"Processed frames: {stats['frames_processed']}")
    console.print(
        f"Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
    console.print(f"Effective FPS: {stats['effective_fps']:.1f}")


if __name__ == "__main__":
    test_ultra_fast_detector()
