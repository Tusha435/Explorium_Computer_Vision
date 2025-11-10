"""
Video processor for object detection on static and live videos.
"""
import cv2
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm


class VideoProcessor:
    """
    Process videos with object detection models.
    Supports both static video files and live camera feeds.
    """

    def __init__(self, detector, output_dir='output'):
        """
        Initialize video processor.

        Args:
            detector: Object detector instance
            output_dir (str): Directory to save output videos
        """
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_time': 0.0,
            'avg_fps': 0.0
        }

    def process_video(self, video_path, output_path=None, show_display=True,
                     target_classes=None, save_stats=True):
        """
        Process a static video file.

        Args:
            video_path (str): Path to input video
            output_path (str): Path to output video (optional)
            show_display (bool): Whether to display video during processing
            target_classes (list): List of class IDs to detect (None for all)
            save_stats (bool): Whether to save statistics

        Returns:
            dict: Processing statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing Video: {video_path}")
        print(f"Model: {self.detector.model_name}")
        print(f"{'='*60}\n")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video Info: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Reset statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_time': 0.0,
            'avg_fps': 0.0,
            'detections_per_frame': []
        }

        frame_count = 0
        start_time = time.time()

        try:
            with tqdm(total=total_frames, desc="Processing") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Detect objects
                    detections, inference_time = self.detector.detect_with_timing(frame)

                    # Filter by target classes if specified
                    if target_classes is not None:
                        detections = [d for d in detections if d['class_id'] in target_classes]

                    # Update statistics
                    self.stats['total_frames'] += 1
                    self.stats['total_detections'] += len(detections)
                    self.stats['total_time'] += inference_time
                    self.stats['detections_per_frame'].append(len(detections))

                    # Visualize
                    annotated_frame = self.visualize_detections(
                        frame, detections, inference_time
                    )

                    # Write frame
                    if writer:
                        writer.write(annotated_frame)

                    # Display
                    if show_display:
                        display_frame = annotated_frame
                        if width > 1280:
                            scale = 1280 / width
                            new_size = (1280, int(height * scale))
                            display_frame = cv2.resize(annotated_frame, new_size)

                        cv2.imshow(f'{self.detector.model_name} - Detection', display_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\n\nProcessing interrupted by user")
                            break
                        elif key == ord(' '):
                            cv2.waitKey(0)  # Pause

                    pbar.update(1)

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        # Calculate final statistics
        elapsed_time = time.time() - start_time
        self.stats['avg_fps'] = self.stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0

        # Print statistics
        self._print_stats()

        # Save statistics
        if save_stats and output_path:
            stats_path = Path(output_path).with_suffix('.txt')
            self._save_stats(stats_path)

        return self.stats

    def process_live(self, camera_index=0, target_classes=None, record_output=None):
        """
        Process live camera feed.

        Args:
            camera_index (int): Camera index (0 for default camera)
            target_classes (list): List of class IDs to detect (None for all)
            record_output (str): Path to save recorded video (optional)

        Returns:
            dict: Processing statistics
        """
        print(f"\n{'='*60}")
        print(f"Starting Live Detection")
        print(f"Model: {self.detector.model_name}")
        print(f"Camera: {camera_index}")
        print(f"{'='*60}\n")
        print("Press 'q' to quit, 'p' to pause, 'r' to toggle recording")

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_index}")

        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30  # Default FPS for recording

        print(f"Camera resolution: {width}x{height}")

        # Setup video writer if recording
        writer = None
        is_recording = record_output is not None
        if record_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(record_output, fourcc, fps, (width, height))

        # Reset statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_time': 0.0,
            'avg_fps': 0.0
        }

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                frame_count += 1

                # Detect objects
                detections, inference_time = self.detector.detect_with_timing(frame)

                # Filter by target classes if specified
                if target_classes is not None:
                    detections = [d for d in detections if d['class_id'] in target_classes]

                # Update statistics
                self.stats['total_frames'] += 1
                self.stats['total_detections'] += len(detections)
                self.stats['total_time'] += inference_time

                # Visualize
                annotated_frame = self.visualize_detections(
                    frame, detections, inference_time, show_fps=True
                )

                # Add recording indicator
                if is_recording:
                    cv2.circle(annotated_frame, (width - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "REC", (width - 70, 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Write frame if recording
                if writer and is_recording:
                    writer.write(annotated_frame)

                # Display
                cv2.imshow(f'{self.detector.model_name} - Live Detection', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nStopping live detection...")
                    break
                elif key == ord('p'):
                    print("Paused. Press any key to continue...")
                    cv2.waitKey(0)
                elif key == ord('r'):
                    is_recording = not is_recording
                    status = "started" if is_recording else "stopped"
                    print(f"Recording {status}")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        # Calculate final statistics
        elapsed_time = time.time() - start_time
        self.stats['avg_fps'] = self.stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0

        # Print statistics
        self._print_stats()

        return self.stats

    def visualize_detections(self, frame, detections, inference_time, show_fps=False):
        """
        Visualize detections on frame.

        Args:
            frame (np.ndarray): Input frame
            detections (list): List of detections
            inference_time (float): Inference time in seconds
            show_fps (bool): Whether to show FPS

        Returns:
            np.ndarray: Annotated frame
        """
        from ..utils.coco_classes import get_class_color

        annotated = frame.copy()

        # Draw detections
        for det in detections:
            bbox = det['bbox'].astype(int)
            class_name = det['class_name']
            confidence = det['confidence']
            class_id = det['class_id']

            # Get color
            color = get_class_color(class_id)

            # Draw box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Background for label
            cv2.rectangle(annotated,
                        (bbox[0], bbox[1] - label_size[1] - 10),
                        (bbox[0] + label_size[0], bbox[1]),
                        color, -1)

            # Label text
            cv2.putText(annotated, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw info panel
        info_y = 30
        cv2.putText(annotated, f"Model: {self.detector.model_name}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info_y += 30
        cv2.putText(annotated, f"Detections: {len(detections)}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if show_fps:
            info_y += 30
            fps = 1.0 / inference_time if inference_time > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f}",
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info_y += 30
        cv2.putText(annotated, f"Time: {inference_time*1000:.1f}ms",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated

    def _print_stats(self):
        """Print processing statistics."""
        print(f"\n{'='*60}")
        print("Processing Statistics")
        print(f"{'='*60}")
        print(f"Model: {self.detector.model_name}")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"Avg Detections/Frame: {self.stats['total_detections']/self.stats['total_frames']:.2f}")
        print(f"Total Time: {self.stats['total_time']:.2f}s")
        print(f"Avg Inference Time: {(self.stats['total_time']/self.stats['total_frames'])*1000:.2f}ms")
        print(f"Avg FPS: {self.stats['avg_fps']:.2f}")
        print(f"{'='*60}\n")

    def _save_stats(self, filepath):
        """Save statistics to file."""
        with open(filepath, 'w') as f:
            f.write(f"Model: {self.detector.model_name}\n")
            f.write(f"Total Frames: {self.stats['total_frames']}\n")
            f.write(f"Total Detections: {self.stats['total_detections']}\n")
            f.write(f"Avg Detections/Frame: {self.stats['total_detections']/self.stats['total_frames']:.2f}\n")
            f.write(f"Total Time: {self.stats['total_time']:.2f}s\n")
            f.write(f"Avg Inference Time: {(self.stats['total_time']/self.stats['total_frames'])*1000:.2f}ms\n")
            f.write(f"Avg FPS: {self.stats['avg_fps']:.2f}\n")

        print(f"Statistics saved to: {filepath}")
