"""
Vehicle detector using YOLOv8 for motorcycle detection.
"""
import torch
import numpy as np
from ultralytics import YOLO
import cv2


class VehicleDetector:
    """
    Motorcycle and vehicle detector using YOLOv8.
    """

    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.3, device=None):
        """
        Initialize the detector.

        Args:
            model_name (str): YOLOv8 model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            conf_threshold (float): Confidence threshold for detections
            device (str): Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.conf_threshold = conf_threshold

        # Auto-select device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing YOLOv8 detector on {self.device}...")
        self.model = YOLO(model_name)
        self.model.to(self.device)

        # COCO class IDs for vehicles
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = {
            'motorcycle': 3,
            'bicycle': 1,
            'car': 2,
            'bus': 5,
            'truck': 7
        }

        # Focus on motorcycles and bicycles for this application
        self.target_classes = [3, 1]  # motorcycle, bicycle

    def detect(self, frame, target_classes=None):
        """
        Detect vehicles in a frame.

        Args:
            frame (np.ndarray): Input image (BGR format)
            target_classes (list): List of class IDs to detect (None for default)

        Returns:
            tuple: (detections, raw_results)
                detections: list of dicts with keys: 'bbox', 'confidence', 'class_id', 'class_name'
                raw_results: raw YOLOv8 results object
        """
        if target_classes is None:
            target_classes = self.target_classes

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []

        # Extract detections
        boxes = results.boxes
        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Filter by target classes
            if class_id in target_classes:
                # Get bounding box coordinates [x1, y1, x2, y2]
                bbox = box.xyxy[0].cpu().numpy()

                detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': results.names[class_id]
                }
                detections.append(detection)

        return detections, results

    def detect_motorcycles(self, frame):
        """
        Detect only motorcycles in a frame.

        Args:
            frame (np.ndarray): Input image (BGR format)

        Returns:
            list: List of detection dictionaries
        """
        detections, _ = self.detect(frame, target_classes=[self.vehicle_classes['motorcycle']])
        return detections

    def visualize_detections(self, frame, detections, show_info=True):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame (np.ndarray): Input image (BGR format)
            detections (list): List of detection dictionaries
            show_info (bool): Whether to show class name and confidence

        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()

        for det in detections:
            bbox = det['bbox'].astype(int)
            confidence = det['confidence']
            class_name = det['class_name']

            # Draw bounding box
            color = self._get_color(det['class_id'])
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            if show_info:
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame,
                            (bbox[0], bbox[1] - label_size[1] - 10),
                            (bbox[0] + label_size[0], bbox[1]),
                            color, -1)
                cv2.putText(annotated_frame, label,
                          (bbox[0], bbox[1] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated_frame

    @staticmethod
    def _get_color(class_id):
        """Get a unique color for each class."""
        colors = {
            1: (255, 0, 0),    # bicycle - blue
            2: (0, 255, 0),    # car - green
            3: (0, 0, 255),    # motorcycle - red
            5: (255, 255, 0),  # bus - cyan
            7: (255, 0, 255)   # truck - magenta
        }
        return colors.get(class_id, (128, 128, 128))

    def get_bbox_array(self, detections):
        """
        Convert detections to numpy array of bounding boxes.

        Args:
            detections (list): List of detection dictionaries

        Returns:
            np.ndarray: Array of bounding boxes [N x 4] (x1, y1, x2, y2)
        """
        if not detections:
            return np.empty((0, 4))

        bboxes = np.array([det['bbox'] for det in detections])
        return bboxes

    def get_confidences(self, detections):
        """
        Get confidence scores from detections.

        Args:
            detections (list): List of detection dictionaries

        Returns:
            np.ndarray: Array of confidence scores
        """
        if not detections:
            return np.empty(0)

        return np.array([det['confidence'] for det in detections])
