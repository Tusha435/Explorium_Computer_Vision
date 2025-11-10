"""
Single model object detection on video, images, or live camera.
"""
import argparse
from pathlib import Path
from src.models import *
from src.video_processor import VideoProcessor, ImageProcessor
from src.utils.video_downloader import VideoDownloader


def main():
    parser = argparse.ArgumentParser(description='Object Detection with Various Models')

    # Input options
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--image-dir', type=str, help='Path to directory containing images')
    parser.add_argument('--camera', type=int, help='Camera index for live detection')
    parser.add_argument('--download-sample', action='store_true',
                       help='Download and use sample video')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic test video')

    # Model selection
    parser.add_argument('--model', type=str, default='yolov8n',
                       choices=[
                           'ssd', 'ssd-v1',
                           'faster-rcnn', 'faster-rcnn-101',
                           'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                           'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x',
                           'detr', 'detr-101'
                       ],
                       help='Detection model to use')

    # Detection parameters
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--target-classes', type=int, nargs='+',
                       help='Target class IDs (e.g., 0 1 2 3 for person, bicycle, car, motorcycle)')

    # Output options
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video during processing')
    parser.add_argument('--record', type=str, help='Record live camera output to file')

    args = parser.parse_args()

    # Validate input
    if not any([args.video, args.image, args.image_dir, args.camera is not None,
                args.download_sample, args.generate_synthetic]):
        parser.error("Must specify --video, --image, --image-dir, --camera, --download-sample, or --generate-synthetic")

    print(f"\n{'='*80}")
    print("OBJECT DETECTION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"{'='*80}\n")

    # Create model
    model_registry = {
        'ssd': lambda: SSDMobileNetV2(confidence_threshold=args.confidence),
        'ssd-v1': lambda: SSDMobileNetV1(confidence_threshold=args.confidence),
        'faster-rcnn': lambda: FasterRCNNResNet50(confidence_threshold=args.confidence),
        'faster-rcnn-101': lambda: FasterRCNNResNet101(confidence_threshold=args.confidence),
        'yolov8n': lambda: YOLOv8Nano(confidence_threshold=args.confidence),
        'yolov8s': lambda: YOLOv8Small(confidence_threshold=args.confidence),
        'yolov8m': lambda: YOLOv8Medium(confidence_threshold=args.confidence),
        'yolov8l': lambda: YOLOv8Large(confidence_threshold=args.confidence),
        'yolov8x': lambda: YOLOv8XLarge(confidence_threshold=args.confidence),
        'yolov11n': lambda: YOLOv11Nano(confidence_threshold=args.confidence),
        'yolov11s': lambda: YOLOv11Small(confidence_threshold=args.confidence),
        'yolov11m': lambda: YOLOv11Medium(confidence_threshold=args.confidence),
        'yolov11l': lambda: YOLOv11Large(confidence_threshold=args.confidence),
        'yolov11x': lambda: YOLOv11XLarge(confidence_threshold=args.confidence),
        'detr': lambda: DETRResNet50(confidence_threshold=args.confidence),
        'detr-101': lambda: DETRResNet101(confidence_threshold=args.confidence),
    }

    detector = model_registry[args.model]()

    # Load model
    print("Loading model...")
    detector.load_model()
    print()

    # Handle input source
    if args.download_sample or args.generate_synthetic:
        downloader = VideoDownloader()

        if args.download_sample:
            print("Generating synthetic video for testing...")
            video_path = downloader.generate_synthetic_video()
        else:
            print("Generating synthetic video...")
            video_path = downloader.generate_synthetic_video()

        args.video = video_path

    # Process based on input type
    if args.image or args.image_dir:
        # Image processing
        image_processor = ImageProcessor(detector, output_dir=args.output_dir)

        if args.image_dir:
            # Batch process directory
            image_dir = Path(args.image_dir)
            image_files = []

            # Collect all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(image_dir.glob(ext))

            if not image_files:
                print(f"✗ No images found in {image_dir}")
                return

            print(f"Found {len(image_files)} images in {image_dir}")

            image_processor.process_batch(
                image_paths=image_files,
                target_classes=args.target_classes
            )
        else:
            # Single image
            image_processor.process_image(
                image_path=args.image,
                output_path=args.output,
                show_display=not args.no_display,
                target_classes=args.target_classes
            )

    elif args.camera is not None:
        # Live camera
        video_processor = VideoProcessor(detector, output_dir=args.output_dir)
        video_processor.process_live(
            camera_index=args.camera,
            target_classes=args.target_classes,
            record_output=args.record
        )
    else:
        # Static video
        video_processor = VideoProcessor(detector, output_dir=args.output_dir)
        video_processor.process_video(
            video_path=args.video,
            output_path=args.output,
            show_display=not args.no_display,
            target_classes=args.target_classes
        )

    print(f"\n{'='*80}")
    print("DETECTION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
