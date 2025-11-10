"""
DETR (Detection Transformer) detector using Hugging Face Transformers.
"""
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from .base_detector import BaseDetector
from ..utils.coco_classes import get_class_name


class DETRDetector(BaseDetector):
    """
    DETR (Detection Transformer) detector using Hugging Face.
    """

    def __init__(self, model_variant='resnet50', confidence_threshold=0.5, device='auto'):
        """
        Initialize DETR detector.

        Args:
            model_variant (str): Model variant ('resnet50', 'resnet101')
            confidence_threshold (float): Confidence threshold
            device (str): Device ('cuda', 'cpu', or 'auto')
        """
        model_name = f'DETR-{model_variant}'
        super().__init__(model_name, confidence_threshold)

        self.model_variant = model_variant
        self.device = device
        self.processor = None

        # Model name for Hugging Face
        if model_variant == 'resnet50':
            self.hf_model_name = 'facebook/detr-resnet-50'
        elif model_variant == 'resnet101':
            self.hf_model_name = 'facebook/detr-resnet-101'
        else:
            raise ValueError(f"Unknown model variant: {model_variant}")

    def load_model(self):
        """Load DETR model from Hugging Face."""
        print(f"Loading {self.model_name} from Hugging Face...")

        try:
            # Set device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Load processor and model
            self.processor = DetrImageProcessor.from_pretrained(self.hf_model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.hf_model_name)
            self.model.to(self.device)
            self.model.eval()

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

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Preprocess
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]

        # Convert to expected format
        detections = []
        height, width = image.shape[:2]

        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        ):
            # DETR outputs [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = box

            # Ensure coordinates are within image bounds
            x1 = max(0, min(int(x1), width))
            y1 = max(0, min(int(y1), height))
            x2 = max(0, min(int(x2), width))
            y2 = max(0, min(int(y2), height))

            # DETR uses COCO labels (0-indexed)
            class_id = int(label)

            detection = {
                'bbox': np.array([x1, y1, x2, y2]),
                'class_id': class_id,
                'class_name': get_class_name(class_id),
                'confidence': float(score)
            }
            detections.append(detection)

        return detections


class DETRResNet50(DETRDetector):
    """DETR with ResNet-50 backbone."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('resnet50', confidence_threshold, device)


class DETRResNet101(DETRDetector):
    """DETR with ResNet-101 backbone."""

    def __init__(self, confidence_threshold=0.5, device='auto'):
        super().__init__('resnet101', confidence_threshold, device)
