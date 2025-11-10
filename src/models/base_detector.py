"""
Base detector interface for all object detection models.
"""
from abc import ABC, abstractmethod
import time
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for all object detectors.
    All detector implementations must inherit from this class.
    """

    def __init__(self, model_name, confidence_threshold=0.5):
        """
        Initialize the detector.

        Args:
            model_name (str): Name of the model
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False

        # Performance metrics
        self.total_inference_time = 0.0
        self.total_frames = 0

    @abstractmethod
    def load_model(self):
        """Load the detection model."""
        pass

    @abstractmethod
    def detect(self, image):
        """
        Perform detection on an image.

        Args:
            image (np.ndarray): Input image in BGR format

        Returns:
            list: List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - class_id: int
                - class_name: str
                - confidence: float
        """
        pass

    def preprocess(self, image):
        """
        Preprocess image for detection (can be overridden).

        Args:
            image (np.ndarray): Input image

        Returns:
            np.ndarray: Preprocessed image
        """
        return image

    def postprocess(self, detections):
        """
        Postprocess detections (can be overridden).

        Args:
            detections (list): Raw detections

        Returns:
            list: Processed detections
        """
        return detections

    def detect_with_timing(self, image):
        """
        Perform detection with timing measurement.

        Args:
            image (np.ndarray): Input image

        Returns:
            tuple: (detections, inference_time)
        """
        start_time = time.time()
        detections = self.detect(image)
        inference_time = time.time() - start_time

        self.total_inference_time += inference_time
        self.total_frames += 1

        return detections, inference_time

    def get_average_fps(self):
        """Get average FPS across all processed frames."""
        if self.total_frames == 0:
            return 0.0
        avg_time = self.total_inference_time / self.total_frames
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_average_inference_time(self):
        """Get average inference time in milliseconds."""
        if self.total_frames == 0:
            return 0.0
        return (self.total_inference_time / self.total_frames) * 1000

    def reset_metrics(self):
        """Reset performance metrics."""
        self.total_inference_time = 0.0
        self.total_frames = 0

    def get_model_info(self):
        """
        Get model information.

        Returns:
            dict: Model information
        """
        return {
            'name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'is_loaded': self.is_loaded,
            'total_frames_processed': self.total_frames,
            'average_fps': self.get_average_fps(),
            'average_inference_time_ms': self.get_average_inference_time()
        }

    def __str__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, conf={self.confidence_threshold})"

    def __repr__(self):
        return self.__str__()
