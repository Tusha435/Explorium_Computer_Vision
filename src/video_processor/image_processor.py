"""
Image processor for object detection on static images.
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ImageProcessor:
    """
    Process images with object detection models.
    """

    def __init__(self, detector, output_dir='output'):
        """
        Initialize image processor.

        Args:
            detector: Object detector instance
            output_dir (str): Directory to save output images
        """
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path, output_path=None, show_display=True,
                     target_classes=None, save_stats=True):
        """
        Process a static image file.

        Args:
            image_path (str): Path to input image
            output_path (str): Path to output image (optional)
            show_display (bool): Whether to display image
            target_classes (list): List of class IDs to detect (None for all)
            save_stats (bool): Whether to save statistics

        Returns:
            dict: Processing statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing Image: {image_path}")
        print(f"Model: {self.detector.model_name}")
        print(f"{'='*60}\n")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        height, width = image.shape[:2]
        print(f"Image Info: {width}x{height}")

        # Detect objects
        print("Running detection...")
        detections, inference_time = self.detector.detect_with_timing(image)

        # Filter by target classes if specified
        if target_classes is not None:
            detections = [d for d in detections if d['class_id'] in target_classes]

        # Statistics
        stats = {
            'image_path': str(image_path),
            'model': self.detector.model_name,
            'total_detections': len(detections),
            'inference_time_ms': inference_time * 1000,
            'detections_by_class': self._count_by_class(detections)
        }

        # Visualize
        annotated_image = self.visualize_detections(image, detections, inference_time)

        # Save output
        if output_path is None:
            output_path = self.output_dir / f"{Path(image_path).stem}_{self.detector.model_name}_output.png"

        cv2.imwrite(str(output_path), annotated_image)
        print(f"\n✓ Output saved to: {output_path}")

        # Display
        if show_display:
            self.display_image(annotated_image)

        # Print statistics
        self._print_stats(stats)

        # Save statistics
        if save_stats:
            stats_path = Path(output_path).with_suffix('.txt')
            self._save_stats(stats, stats_path)

        return stats

    def process_batch(self, image_paths, output_dir=None, target_classes=None):
        """
        Process multiple images.

        Args:
            image_paths (list): List of image paths
            output_dir (str): Output directory for all images
            target_classes (list): Target class IDs

        Returns:
            list: List of statistics for each image
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        all_stats = []

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING: {len(image_paths)} images")
        print(f"{'='*60}\n")

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {image_path}")

            try:
                stats = self.process_image(
                    image_path,
                    show_display=False,
                    target_classes=target_classes
                )
                all_stats.append(stats)
            except Exception as e:
                print(f"✗ Failed to process {image_path}: {e}")
                all_stats.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })

        # Summary
        self._print_batch_summary(all_stats)

        return all_stats

    def visualize_detections(self, image, detections, inference_time):
        """
        Visualize detections on image.

        Args:
            image (np.ndarray): Input image
            detections (list): List of detections
            inference_time (float): Inference time in seconds

        Returns:
            np.ndarray: Annotated image
        """
        from ..utils.coco_classes import get_class_color

        annotated = image.copy()
        height, width = image.shape[:2]

        # Draw detections
        for det in detections:
            bbox = det['bbox'].astype(int)
            class_name = det['class_name']
            confidence = det['confidence']
            class_id = det['class_id']

            # Get color
            color = get_class_color(class_id)

            # Draw box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

            # Background for label
            y_label = max(bbox[1] - 10, label_size[1] + 10)
            cv2.rectangle(annotated,
                        (bbox[0], y_label - label_size[1] - 10),
                        (bbox[0] + label_size[0] + 10, y_label),
                        color, -1)

            # Label text
            cv2.putText(annotated, label,
                       (bbox[0] + 5, y_label - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw info panel with semi-transparent background
        panel_height = 120
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)

        # Draw info text
        info_y = 30
        cv2.putText(annotated, f"Model: {self.detector.model_name}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        info_y += 35
        cv2.putText(annotated, f"Detections: {len(detections)}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        info_y += 35
        cv2.putText(annotated, f"Inference Time: {inference_time*1000:.1f}ms",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return annotated

    def display_image(self, image, window_name='Detection Result'):
        """
        Display image in a window.

        Args:
            image (np.ndarray): Image to display
            window_name (str): Window name
        """
        # Resize if too large
        height, width = image.shape[:2]
        max_width = 1280
        max_height = 720

        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        cv2.imshow(window_name, image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _count_by_class(self, detections):
        """Count detections by class."""
        counts = {}
        for det in detections:
            class_name = det['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

    def _print_stats(self, stats):
        """Print processing statistics."""
        print(f"\n{'='*60}")
        print("Detection Statistics")
        print(f"{'='*60}")
        print(f"Model: {stats['model']}")
        print(f"Total Detections: {stats['total_detections']}")
        print(f"Inference Time: {stats['inference_time_ms']:.2f}ms")

        if stats['detections_by_class']:
            print("\nDetections by Class:")
            for class_name, count in sorted(stats['detections_by_class'].items()):
                print(f"  {class_name}: {count}")

        print(f"{'='*60}\n")

    def _print_batch_summary(self, all_stats):
        """Print batch processing summary."""
        successful = [s for s in all_stats if 'error' not in s]
        failed = [s for s in all_stats if 'error' in s]

        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Images: {len(all_stats)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if successful:
            total_detections = sum(s['total_detections'] for s in successful)
            avg_inference_time = sum(s['inference_time_ms'] for s in successful) / len(successful)

            print(f"\nTotal Detections: {total_detections}")
            print(f"Avg Inference Time: {avg_inference_time:.2f}ms")

            # Aggregate class counts
            all_class_counts = {}
            for s in successful:
                for class_name, count in s['detections_by_class'].items():
                    all_class_counts[class_name] = all_class_counts.get(class_name, 0) + count

            if all_class_counts:
                print("\nTotal Detections by Class:")
                for class_name, count in sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {count}")

        if failed:
            print("\nFailed Images:")
            for s in failed:
                print(f"  {s['image_path']}: {s['error']}")

        print(f"{'='*60}\n")

    def _save_stats(self, stats, filepath):
        """Save statistics to file."""
        with open(filepath, 'w') as f:
            f.write(f"Model: {stats['model']}\n")
            f.write(f"Image: {stats['image_path']}\n")
            f.write(f"Total Detections: {stats['total_detections']}\n")
            f.write(f"Inference Time: {stats['inference_time_ms']:.2f}ms\n")

            if stats['detections_by_class']:
                f.write("\nDetections by Class:\n")
                for class_name, count in sorted(stats['detections_by_class'].items()):
                    f.write(f"  {class_name}: {count}\n")

        print(f"Statistics saved to: {filepath}")
