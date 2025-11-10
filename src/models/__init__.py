"""
Object detection models package.
"""
from .base_detector import BaseDetector
from .ssd_detector import SSDDetector, SSDMobileNetV1, SSDMobileNetV2
from .faster_rcnn_detector import (
    FasterRCNNDetector,
    FasterRCNNResNet50,
    FasterRCNNResNet101,
    FasterRCNNInceptionResNetV2
)
from .yolov8_detector import (
    YOLOv8Detector,
    YOLOv8Nano,
    YOLOv8Small,
    YOLOv8Medium,
    YOLOv8Large,
    YOLOv8XLarge
)
from .yolov11_detector import (
    YOLOv11Detector,
    YOLOv11Nano,
    YOLOv11Small,
    YOLOv11Medium,
    YOLOv11Large,
    YOLOv11XLarge
)
from .detr_detector import DETRDetector, DETRResNet50, DETRResNet101

__all__ = [
    'BaseDetector',
    'SSDDetector',
    'SSDMobileNetV1',
    'SSDMobileNetV2',
    'FasterRCNNDetector',
    'FasterRCNNResNet50',
    'FasterRCNNResNet101',
    'FasterRCNNInceptionResNetV2',
    'YOLOv8Detector',
    'YOLOv8Nano',
    'YOLOv8Small',
    'YOLOv8Medium',
    'YOLOv8Large',
    'YOLOv8XLarge',
    'YOLOv11Detector',
    'YOLOv11Nano',
    'YOLOv11Small',
    'YOLOv11Medium',
    'YOLOv11Large',
    'YOLOv11XLarge',
    'DETRDetector',
    'DETRResNet50',
    'DETRResNet101',
]
