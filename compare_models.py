"""
Compare multiple computer vision detection models on the same video.
"""
import argparse
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.models import *
from src.video_processor import VideoProcessor
from src.utils.coco_classes import TARGET_CLASSES


class ModelComparison:
    """Compare multiple object detection models."""

    def __init__(self, output_dir='comparison_results'):
        """
        Initialize model comparison.

        Args:
            output_dir (str): Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.results = {}

    def add_model(self, name, detector):
        """
        Add a model to comparison.

        Args:
            name (str): Model identifier
            detector: Detector instance
        """
        self.models[name] = detector
        print(f"Added model: {name}")

    def run_comparison(self, video_path, target_classes=None, save_outputs=True):
        """
        Run comparison on all models.

        Args:
            video_path (str): Path to test video
            target_classes (list): Target class IDs (None for all)
            save_outputs (bool): Whether to save output videos

        Returns:
            dict: Comparison results
        """
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON")
        print(f"{'='*80}")
        print(f"Video: {video_path}")
        print(f"Models: {list(self.models.keys())}")
        print(f"{'='*80}\n")

        for model_name, detector in self.models.items():
            print(f"\n{'='*80}")
            print(f"Testing: {model_name}")
            print(f"{'='*80}")

            try:
                # Load model
                if not detector.is_loaded:
                    detector.load_model()

                # Setup output path
                output_path = None
                if save_outputs:
                    output_path = str(self.output_dir / f"{model_name}_output.mp4")

                # Process video
                processor = VideoProcessor(detector, output_dir=str(self.output_dir))
                stats = processor.process_video(
                    video_path,
                    output_path=output_path,
                    show_display=False,
                    target_classes=target_classes
                )

                # Store results
                self.results[model_name] = {
                    'stats': stats,
                    'model_info': detector.get_model_info()
                }

                print(f"✓ {model_name} completed successfully")

            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                self.results[model_name] = {
                    'stats': None,
                    'model_info': None,
                    'error': str(e)
                }

        # Generate comparison report
        self.generate_report()

        return self.results

    def generate_report(self):
        """Generate comparison report with visualizations."""
        print(f"\n{'='*80}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*80}\n")

        # Create comparison dataframe
        comparison_data = []

        for model_name, result in self.results.items():
            if result['stats'] is None:
                continue

            stats = result['stats']
            comparison_data.append({
                'Model': model_name,
                'Total Frames': stats['total_frames'],
                'Total Detections': stats['total_detections'],
                'Avg Detections/Frame': stats['total_detections'] / stats['total_frames'],
                'Total Time (s)': stats['total_time'],
                'Avg Inference Time (ms)': (stats['total_time'] / stats['total_frames']) * 1000,
                'Avg FPS': stats['avg_fps']
            })

        df = pd.DataFrame(comparison_data)

        # Save to CSV
        csv_path = self.output_dir / 'comparison_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Results saved to: {csv_path}")

        # Print comparison table
        print("\n" + "="*80)
        print("COMPARISON RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")

        # Create visualizations
        self.create_visualizations(df)

        # Save JSON report
        json_path = self.output_dir / 'comparison_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ Detailed results saved to: {json_path}")

    def create_visualizations(self, df):
        """
        Create comparison visualizations.

        Args:
            df (pd.DataFrame): Comparison dataframe
        """
        # Set style
        sns.set_style('whitegrid')
        sns.set_palette('husl')

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Object Detection Models Comparison', fontsize=16, fontweight='bold')

        # 1. Average FPS comparison
        ax1 = axes[0, 0]
        df_sorted = df.sort_values('Avg FPS', ascending=False)
        ax1.barh(df_sorted['Model'], df_sorted['Avg FPS'])
        ax1.set_xlabel('FPS (Higher is Better)')
        ax1.set_title('Average FPS Comparison')
        ax1.grid(axis='x', alpha=0.3)

        # 2. Average Inference Time comparison
        ax2 = axes[0, 1]
        df_sorted = df.sort_values('Avg Inference Time (ms)')
        ax2.barh(df_sorted['Model'], df_sorted['Avg Inference Time (ms)'])
        ax2.set_xlabel('Inference Time (ms) (Lower is Better)')
        ax2.set_title('Average Inference Time Comparison')
        ax2.grid(axis='x', alpha=0.3)

        # 3. Total Detections comparison
        ax3 = axes[1, 0]
        df_sorted = df.sort_values('Total Detections', ascending=False)
        ax3.barh(df_sorted['Model'], df_sorted['Total Detections'])
        ax3.set_xlabel('Total Detections')
        ax3.set_title('Total Detections Comparison')
        ax3.grid(axis='x', alpha=0.3)

        # 4. Avg Detections per Frame
        ax4 = axes[1, 1]
        df_sorted = df.sort_values('Avg Detections/Frame', ascending=False)
        ax4.barh(df_sorted['Model'], df_sorted['Avg Detections/Frame'])
        ax4.set_xlabel('Detections per Frame')
        ax4.set_title('Average Detections per Frame')
        ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save figure
        plot_path = self.output_dir / 'comparison_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plots saved to: {plot_path}")

        plt.close()

        # Create speedaccuracy plot
        self.create_speed_accuracy_plot(df)

    def create_speed_accuracy_plot(self, df):
        """
        Create speed vs accuracy scatter plot.

        Args:
            df (pd.DataFrame): Comparison dataframe
        """
        plt.figure(figsize=(10, 8))

        # Use detections per frame as proxy for accuracy
        plt.scatter(df['Avg FPS'], df['Avg Detections/Frame'], s=200, alpha=0.6)

        for idx, row in df.iterrows():
            plt.annotate(row['Model'],
                        (row['Avg FPS'], row['Avg Detections/Frame']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')

        plt.xlabel('Average FPS (Speed)', fontsize=12)
        plt.ylabel('Avg Detections/Frame (Proxy for Recall)', fontsize=12)
        plt.title('Speed vs Detection Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plot_path = self.output_dir / 'speed_vs_performance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Speed vs performance plot saved to: {plot_path}")

        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Object Detection Models')
    parser.add_argument('--video', type=str, required=True, help='Path to test video')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['ssd', 'faster-rcnn', 'yolov8n', 'yolov11n', 'detr'],
                       help='Models to compare')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--target-classes', type=int, nargs='+',
                       help='Target class IDs (e.g., 0 1 2 3 for person, bicycle, car, motorcycle)')
    parser.add_argument('--no-save-outputs', action='store_true',
                       help='Do not save individual output videos')

    args = parser.parse_args()

    # Initialize comparison
    comparison = ModelComparison(output_dir=args.output_dir)

    # Add models based on arguments
    model_registry = {
        'ssd': lambda: SSDMobileNetV2(confidence_threshold=args.confidence),
        'ssd-v1': lambda: SSDMobileNetV1(confidence_threshold=args.confidence),
        'faster-rcnn': lambda: FasterRCNNResNet50(confidence_threshold=args.confidence),
        'faster-rcnn-101': lambda: FasterRCNNResNet101(confidence_threshold=args.confidence),
        'yolov8n': lambda: YOLOv8Nano(confidence_threshold=args.confidence),
        'yolov8s': lambda: YOLOv8Small(confidence_threshold=args.confidence),
        'yolov8m': lambda: YOLOv8Medium(confidence_threshold=args.confidence),
        'yolov11n': lambda: YOLOv11Nano(confidence_threshold=args.confidence),
        'yolov11s': lambda: YOLOv11Small(confidence_threshold=args.confidence),
        'yolov11m': lambda: YOLOv11Medium(confidence_threshold=args.confidence),
        'detr': lambda: DETRResNet50(confidence_threshold=args.confidence),
        'detr-101': lambda: DETRResNet101(confidence_threshold=args.confidence),
    }

    for model_name in args.models:
        if model_name in model_registry:
            try:
                detector = model_registry[model_name]()
                comparison.add_model(model_name, detector)
            except Exception as e:
                print(f"Failed to add {model_name}: {e}")
        else:
            print(f"Unknown model: {model_name}")
            print(f"Available models: {list(model_registry.keys())}")

    # Run comparison
    comparison.run_comparison(
        video_path=args.video,
        target_classes=args.target_classes,
        save_outputs=not args.no_save_outputs
    )

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
