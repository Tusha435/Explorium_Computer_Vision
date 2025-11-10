# Computer Vision Detection Models Comparison Framework

A comprehensive framework for comparing different computer vision object detection techniques including **SSD, Faster R-CNN, YOLOv8, YOLOv11, and DETR** on the same video datasets. Supports COCO dataset classes (person, bicycle, car, motorcycle, etc.) with both static video and live camera processing.

## 🌟 Features

### Multiple Detection Models
- **SSD (Single Shot MultiBox Detector)**
  - SSD MobileNet V1
  - SSD MobileNet V2 ✓ Fast, lightweight

- **Faster R-CNN (Region-based CNN)**
  - ResNet-50 backbone
  - ResNet-101 backbone
  - Inception ResNet V2 backbone

- **YOLOv8 (You Only Look Once v8)**
  - Nano (n) - Fastest
  - Small (s) - Balanced
  - Medium (m) - Good accuracy
  - Large (l) - High accuracy
  - X-Large (x) - Best accuracy

- **YOLOv11 (Latest YOLO)**
  - All sizes: n, s, m, l, x
  - State-of-the-art performance

- **DETR (Detection Transformer)**
  - ResNet-50 backbone
  - ResNet-101 backbone
  - Transformer-based detection

### Video Processing
- ✅ **Static Video Files** (.mp4, .avi, .mov, etc.)
- ✅ **Live Camera Feed** (webcam, USB camera)
- ✅ **Sample Video Generator** (synthetic test videos)
- ✅ **Video Downloader** (download test videos)

### Detection Capabilities
- **80 COCO Classes** including:
  - Person
  - Bicycle
  - Car
  - Motorcycle
  - Bus
  - Truck
  - And 74 more classes

### Comparison & Benchmarking
- Side-by-side model comparison
- Performance metrics (FPS, inference time)
- Detection accuracy comparison
- Automated visualization generation
- CSV and JSON export

## 🚀 Installation

### Requirements
- Python 3.8+
- CUDA 11.2 (for GPU support)
- NVIDIA GPU (optional, for faster processing)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Explorium_Computer_Vision.git
cd Explorium_Computer_Vision
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify TensorFlow GPU support** (optional):
```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## 📊 Usage

### 1. Single Model Detection

#### Detect on Video File
```bash
python detect.py --video path/to/video.mp4 --model yolov8n --output output.mp4
```

#### Live Camera Detection
```bash
python detect.py --camera 0 --model yolov11n
```

#### Use Synthetic Test Video
```bash
python detect.py --generate-synthetic --model detr
```

#### Filter Specific Classes (Person, Bicycle, Car, Motorcycle)
```bash
python detect.py --video input.mp4 --model faster-rcnn --target-classes 0 1 2 3
```

### 2. Compare Multiple Models

#### Compare Default Models
```bash
python compare_models.py --video test_video.mp4
```

#### Compare Specific Models
```bash
python compare_models.py --video test_video.mp4 \
    --models yolov8n yolov11n ssd faster-rcnn detr
```

#### Compare with Specific Classes
```bash
python compare_models.py --video traffic.mp4 \
    --models yolov8s yolov11s \
    --target-classes 0 1 2 3 5 7  # person, bicycle, car, motorcycle, bus, truck
```

### 3. Advanced Usage

#### High Confidence Threshold
```bash
python detect.py --video input.mp4 --model yolov8m --confidence 0.7
```

#### No Display (Faster Processing)
```bash
python detect.py --video input.mp4 --model ssd --no-display --output result.mp4
```

#### Record Live Camera
```bash
python detect.py --camera 0 --model yolov11s --record live_recording.mp4
```

## 📝 Model Details

### SSD (Single Shot MultiBox Detector)
- **Framework**: TensorFlow 2.12
- **Speed**: ⚡⚡⚡ Fast
- **Accuracy**: ⭐⭐⭐ Good
- **Best For**: Real-time applications, embedded systems

### Faster R-CNN
- **Framework**: TensorFlow 2.12
- **Speed**: ⚡⚡ Moderate
- **Accuracy**: ⭐⭐⭐⭐ Very Good
- **Best For**: High accuracy requirements

### YOLOv8
- **Framework**: PyTorch (Ultralytics)
- **Speed**: ⚡⚡⚡⚡ Very Fast
- **Accuracy**: ⭐⭐⭐⭐ Very Good
- **Best For**: Real-time detection, balanced performance

### YOLOv11
- **Framework**: PyTorch (Ultralytics)
- **Speed**: ⚡⚡⚡⚡ Very Fast
- **Accuracy**: ⭐⭐⭐⭐⭐ Excellent
- **Best For**: State-of-the-art performance

### DETR (Detection Transformer)
- **Framework**: PyTorch (Hugging Face)
- **Speed**: ⚡ Slow
- **Accuracy**: ⭐⭐⭐⭐ Very Good
- **Best For**: Research, transformer-based detection

## 📈 Comparison Outputs

After running `compare_models.py`, you'll get:

### 1. Console Output
```
================================================================
COMPARISON RESULTS
================================================================
Model          Total Frames  Avg FPS  Avg Inference Time (ms)
----------------------------------------------------------------
YOLOv11n       300          45.2      22.1
YOLOv8n        300          42.8      23.4
SSD-V2         300          38.5      26.0
Faster-RCNN    300          15.2      65.8
DETR           300          8.3       120.4
================================================================
```

### 2. CSV Report
`comparison_results/comparison_results.csv`:
- Model names
- Total frames processed
- Total detections
- Average detections per frame
- Average FPS
- Average inference time

### 3. Visualization Plots
`comparison_results/comparison_plots.png`:
- Average FPS comparison
- Inference time comparison
- Total detections comparison
- Speed vs performance scatter plot

### 4. JSON Report
`comparison_results/comparison_results.json`:
- Detailed statistics for each model
- Model configuration
- Error logs (if any)

## 🎯 COCO Classes Reference

### Vehicle Classes
```python
1: 'bicycle'
2: 'car'
3: 'motorcycle'
5: 'bus'
7: 'truck'
```

### Common Classes
```python
0: 'person'
1: 'bicycle'
2: 'car'
3: 'motorcycle'
9: 'traffic light'
11: 'stop sign'
```

[See complete list in src/utils/coco_classes.py]

## 🖥️ System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- CPU: Any modern processor

### Recommended for GPU Acceleration
- NVIDIA GPU with CUDA 11.2 support
- 16GB RAM
- TensorFlow 2.12 with GPU support
- PyTorch with CUDA support

### Tested Configurations
- ✅ NVIDIA RTX 3060 + CUDA 11.2
- ✅ NVIDIA GTX 1660 + CUDA 11.2
- ✅ CPU-only (slower but functional)

## 📂 Project Structure

```
Explorium_Computer_Vision/
├── src/
│   ├── models/
│   │   ├── base_detector.py       # Base detector interface
│   │   ├── ssd_detector.py        # SSD implementation (TF 2.12)
│   │   ├── faster_rcnn_detector.py # Faster R-CNN (TF 2.12)
│   │   ├── yolov8_detector.py     # YOLOv8 (Ultralytics)
│   │   ├── yolov11_detector.py    # YOLOv11 (Ultralytics)
│   │   └── detr_detector.py       # DETR (Transformers)
│   ├── video_processor/
│   │   └── processor.py           # Video processing pipeline
│   └── utils/
│       ├── coco_classes.py        # COCO class definitions
│       └── video_downloader.py    # Video download utilities
├── detect.py                      # Single model detection
├── compare_models.py              # Multi-model comparison
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🎥 Sample Videos

### Generate Synthetic Video
```bash
python -c "from src.utils.video_downloader import VideoDownloader; VideoDownloader().generate_synthetic_video()"
```

### Use with Detection
```bash
python detect.py --generate-synthetic --model yolov8n
```

## 🐛 Troubleshooting

### TensorFlow GPU Issues
```bash
# Check CUDA version
nvidia-smi

# Verify TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install specific TensorFlow version for CUDA 11.2
pip install tensorflow==2.12.0
```

### PyTorch GPU Issues
```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Model Download Issues
```bash
# Manually download YOLOv8 models
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Manually download YOLOv11 models
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

## 🔧 Configuration

### Customize Confidence Thresholds
All models support custom confidence thresholds:
```python
from src.models import YOLOv8Nano

detector = YOLOv8Nano(confidence_threshold=0.7)  # Higher threshold
```

### Customize Target Classes
Filter specific classes:
```python
from src.utils.coco_classes import TARGET_CLASSES

# Detect only person, bicycle, car, motorcycle
target_classes = [0, 1, 2, 3]
```

## 📊 Performance Benchmarks

| Model | Speed (FPS) | Accuracy | GPU Memory | Best Use Case |
|-------|------------|----------|------------|---------------|
| YOLOv11n | 45-55 | ⭐⭐⭐⭐⭐ | 2GB | Real-time, high accuracy |
| YOLOv8n | 40-50 | ⭐⭐⭐⭐ | 2GB | Real-time, balanced |
| SSD MobileNet | 35-45 | ⭐⭐⭐ | 1GB | Embedded, mobile |
| Faster R-CNN | 10-20 | ⭐⭐⭐⭐ | 4GB | High accuracy |
| DETR | 5-10 | ⭐⭐⭐⭐ | 6GB | Research, transformers |

*Benchmarks on NVIDIA RTX 3060, 1080p video

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more detection models (EfficientDet, RetinaNet, etc.)
- Implement custom dataset training
- Add tracking capabilities (DeepSORT, ByteTrack)
- Mobile optimization (TFLite, ONNX)

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- TensorFlow Team for TF 2.12 and TensorFlow Hub
- Ultralytics for YOLOv8 and YOLOv11
- Hugging Face for DETR implementation
- COCO Dataset for class definitions

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Detecting! 🎯**
