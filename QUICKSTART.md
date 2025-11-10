# Quick Start Guide

Get started with Computer Vision Detection Models Comparison in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python -c "import tensorflow as tf; import torch; print('✓ Installation successful!')"

# 3. Download sample media (RECOMMENDED!)
python download_samples.py
```

This downloads **10 videos + 10 images** with traffic scenes for immediate testing!

## Test with Sample Media (Recommended!)

### Download Samples
```bash
# Download all (10 videos + 10 images)
python download_samples.py

# List what's available
python download_samples.py --list
```

### Test on Sample Image
```bash
# Detect on a sample image
python detect.py --image sample_images/street_traffic_cars.jpg --model yolov8n
```

### Test on Sample Video
```bash
# Detect on a sample video
python detect.py --video sample_videos/traffic_highway_busy.mp4 --model yolov11n
```

### Batch Process All Sample Images
```bash
# Process all 10 sample images at once
python detect.py --image-dir sample_images/ --model yolov8s
```

## Test with Synthetic Video (No download needed!)

### Test Single Model
```bash
# Generate synthetic video and detect with YOLOv8
python detect.py --generate-synthetic --model yolov8n
```

### Compare Models
```bash
# Generate synthetic video
python -c "from src.utils.video_downloader import VideoDownloader; VideoDownloader().generate_synthetic_video()"

# Compare multiple models
python compare_models.py --video sample_videos/synthetic_traffic.mp4 --models yolov8n yolov11n ssd
```

## Test with Webcam

```bash
# Live detection with YOLOv11
python detect.py --camera 0 --model yolov11n
```

Press 'q' to quit, 'p' to pause

## Test with Your Own Video

```bash
# Single model
python detect.py --video your_video.mp4 --model yolov8s --output result.mp4

# Compare models
python compare_models.py --video your_video.mp4 --models yolov8n yolov11n faster-rcnn
```

## Available Models

| Model Name | Speed | Accuracy | Command |
|------------|-------|----------|---------|
| yolov11n | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | `--model yolov11n` |
| yolov8n | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | `--model yolov8n` |
| ssd | ⚡⚡⚡ | ⭐⭐⭐ | `--model ssd` |
| faster-rcnn | ⚡⚡ | ⭐⭐⭐⭐ | `--model faster-rcnn` |
| detr | ⚡ | ⭐⭐⭐⭐ | `--model detr` |

## Filter Specific Objects

Detect only specific objects (person, car, motorcycle, bicycle):

```bash
python detect.py --video input.mp4 --model yolov8n --target-classes 0 1 2 3
```

## Troubleshooting

### GPU not detected
```bash
# Check CUDA
nvidia-smi

# Verify TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Verify PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Model download issues
Models are automatically downloaded on first use. If you have issues:

```bash
# Pre-download YOLOv8
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Pre-download YOLOv11
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## Next Steps

1. Read full [README.md](README.md) for advanced usage
2. Check comparison results in `comparison_results/`
3. Experiment with different confidence thresholds
4. Try different model sizes (n, s, m, l, x)

---

Need help? Open an issue on GitHub!
