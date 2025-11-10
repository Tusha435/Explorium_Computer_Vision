"""
YOLOv8 detector using Ultralytics.
"""
import numpy as np
from ultralytics import YOLO
from .base_detector import BaseDetector
from ..utils.coco_classes import get_class_name


class YOLOv8Detector(BaseDetector):
    """
    YOLOv8 detector using Ultralytics library.
    """

    def __init__(self, model_size='n', confidence_threshold=0.5, device='auto'):
        """
        Initialize YOLOv8 detector.

        Args:
            model_size (str): Model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold (float): Confidence threshold
            device (str): Device ('cuda', 'cpu', or 'auto')
        """
        model_name = f'YOLOv8{model_size}'
        super().__init__(model_name, confidence_threshold)

        self.model_size = model_size
        self.device = device
        self.model_path = f'yolov8{model_size}.pt'

    def load_model(self):
        """Load YOLOv8 model."""
        print(f"Loading {self.model_name}...")

        try:
            self.model = YOLO(self.model_path)

            # Set device
            if self.device == 'auto':
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.model.to(self.device)
            self.is_loaded = True
            print(f"✓ {self.model_name} loaded successfully on {self.device}!")
        except Exception as e:
            print(f"✗ Failed to load {self.model_name}: {e}")
            raise

    def detect(self, image):
        """
        Detect objects in image.

        Args:
            image (np.ndarray): Input image (BGR)

        Returns:
            list: Detections
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Run inference
        results = self.model(image, conf=self.confidence_threshold, verbose=False)[0]

        # Extract detections
        detections = []
        boxes = results.boxes

        for i in range(len(boxes)):
            box = boxes[i]

            # Get bounding box coordinates
            bbox = box.xyxy[0].cpu().numpy()

            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detection = {
                'bbox': bbox,
                'class_id': class_id,
                'class_name': get_class_name(class_id),
                'confidence': confidence
            }
            detections.append(detection)

        return detections


class YOLOv8Nano(YOLOv8Detector):
    """YOLOv8 Nano - Fastest, smallest model."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('n', confidence_threshold, device)


class YOLOv8Small(YOLOv8Detector):
    """YOLOv8 Small - Balanced speed and accuracy."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('s', confidence_threshold, device)


class YOLOv8Medium(YOLOv8Detector):
    """YOLOv8 Medium - Good accuracy."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('m', confidence_threshold, device)


class YOLOv8Large(YOLOv8Detector):
    """YOLOv8 Large - High accuracy."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('l', confidence_threshold, device)


class YOLOv8XLarge(YOLOv8Detector):
    """YOLOv8 X-Large - Highest accuracy, slowest."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('x', confidence_threshold, device)
