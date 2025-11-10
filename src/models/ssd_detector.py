"""
SSD MobileNet detector using TensorFlow 2.12.
Uses TensorFlow Hub pre-trained models.
"""
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from .base_detector import BaseDetector
from ..utils.coco_classes import get_class_name, get_class_color


class SSDDetector(BaseDetector):
    """
    SSD MobileNet V2 detector using TensorFlow Hub.
    """

    def __init__(self, model_variant='mobilenet_v2', confidence_threshold=0.5):
        """
        Initialize SSD detector.

        Args:
            model_variant (str): Model variant ('mobilenet_v2', 'mobilenet_v1')
            confidence_threshold (float): Confidence threshold
        """
        model_name = f'SSD-{model_variant}'
        super().__init__(model_name, confidence_threshold)

        self.model_variant = model_variant
        self.model_url = self._get_model_url()

    def _get_model_url(self):
        """Get TensorFlow Hub model URL."""
        if self.model_variant == 'mobilenet_v2':
            return "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        elif self.model_variant == 'mobilenet_v1':
            return "https://tfhub.dev/tensorflow/ssd_mobilenet_v1/1"
        else:
            raise ValueError(f"Unknown model variant: {self.model_variant}")

    def load_model(self):
        """Load SSD model from TensorFlow Hub."""
        print(f"Loading {self.model_name} from TensorFlow Hub...")
        print(f"URL: {self.model_url}")

        try:
            self.model = hub.load(self.model_url)
            self.is_loaded = True
            print(f"✓ {self.model_name} loaded successfully!")
        except Exception as e:
            print(f"✗ Failed to load {self.model_name}: {e}")
            raise

    def preprocess(self, image):
        """
        Preprocess image for SSD.

        Args:
            image (np.ndarray): BGR image

        Returns:
            tf.Tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)

        # Add batch dimension
        input_tensor = tf.expand_dims(input_tensor, 0)

        return input_tensor

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

        # Preprocess
        input_tensor = self.preprocess(image)

        # Run detection
        detector_output = self.model(input_tensor)

        # Extract results
        boxes = detector_output['detection_boxes'][0].numpy()
        classes = detector_output['detection_classes'][0].numpy().astype(int)
        scores = detector_output['detection_scores'][0].numpy()

        # Convert to expected format
        height, width = image.shape[:2]
        detections = []

        for i in range(len(scores)):
            if scores[i] >= self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(xmin * width)
                y1 = int(ymin * height)
                x2 = int(xmax * width)
                y2 = int(ymax * height)

                # COCO class IDs are 1-indexed in TF models, convert to 0-indexed
                class_id = classes[i] - 1

                detection = {
                    'bbox': np.array([x1, y1, x2, y2]),
                    'class_id': class_id,
                    'class_name': get_class_name(class_id),
                    'confidence': float(scores[i])
                }
                detections.append(detection)

        return detections


class SSDMobileNetV1(SSDDetector):
    """SSD MobileNet V1 detector."""

    def __init__(self, confidence_threshold=0.5):
        super().__init__('mobilenet_v1', confidence_threshold)


class SSDMobileNetV2(SSDDetector):
    """SSD MobileNet V2 detector."""

    def __init__(self, confidence_threshold=0.5):
        super().__init__('mobilenet_v2', confidence_threshold)
