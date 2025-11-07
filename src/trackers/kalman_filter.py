"""
Kalman Filter implementation for vehicle tracking.
Tracks position and velocity in 2D space.
"""
import numpy as np
from filterpy.kalman import KalmanFilter as KF


class KalmanFilter:
    """
    Kalman Filter for tracking bounding boxes in image space.
    State vector: [x, y, vx, vy, w, h]
    where (x, y) is the center position, (vx, vy) is velocity,
    and (w, h) is the bounding box width and height.
    """

    def __init__(self):
        """Initialize Kalman filter with 6D state space."""
        # State: [x, y, vx, vy, w, h]
        # Measurement: [x, y, w, h]
        self.kf = KF(dim_x=6, dim_z=4)

        # State transition matrix (assumes constant velocity model)
        # x_new = x + vx * dt, y_new = y + vy * dt
        self.kf.F = np.array([
            [1, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 1, 0],  # w = w
            [0, 0, 0, 0, 0, 1]   # h = h
        ])

        # Measurement function (we observe x, y, w, h)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement noise covariance
        self.kf.R *= 10.0

        # Process noise covariance
        self.kf.Q[2:4, 2:4] *= 0.01  # Low process noise for velocity
        self.kf.Q[4:, 4:] *= 0.01    # Low process noise for size

        # Initial state covariance
        self.kf.P[2:4, 2:4] *= 1000.0  # High uncertainty in velocity initially
        self.kf.P *= 10.0

    def predict(self):
        """
        Predict the next state using the motion model.

        Returns:
            np.ndarray: Predicted state [x, y, vx, vy, w, h]
        """
        self.kf.predict()
        return self.kf.x

    def update(self, bbox):
        """
        Update the state with a new measurement.

        Args:
            bbox (np.ndarray): Bounding box [x, y, w, h]
        """
        self.kf.update(bbox)

    def get_state(self):
        """
        Get the current state estimate.

        Returns:
            np.ndarray: Current state [x, y, vx, vy, w, h]
        """
        return self.kf.x

    def get_bbox(self):
        """
        Get the current bounding box prediction.

        Returns:
            np.ndarray: Bounding box [x, y, w, h]
        """
        return self.kf.x[[0, 1, 4, 5]]


class KalmanBoxTracker:
    """
    Tracks a single object using Kalman filter.
    Converts between different bbox formats.
    """

    count = 0  # Global track ID counter

    def __init__(self, bbox):
        """
        Initialize tracker with initial bounding box.

        Args:
            bbox (np.ndarray): Initial bounding box [x1, y1, x2, y2]
        """
        self.kf = KalmanFilter()
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []

        # Initialize with first detection
        bbox_center = self._convert_bbox_to_z(bbox)
        self.kf.kf.x[:4] = bbox_center
        self.kf.kf.x[4:] = [bbox[2] - bbox[0], bbox[3] - bbox[1]]

    def update(self, bbox):
        """
        Update the tracker with a new detection.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2]
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history.append(bbox)

        bbox_center = self._convert_bbox_to_z(bbox)
        self.kf.update(bbox_center)

    def predict(self):
        """
        Predict the next bounding box location.

        Returns:
            np.ndarray: Predicted bounding box [x1, y1, x2, y2]
        """
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        bbox = self._convert_x_to_bbox(self.kf.get_state())
        return bbox

    def get_state(self):
        """
        Get the current bounding box estimate.

        Returns:
            np.ndarray: Current bounding box [x1, y1, x2, y2]
        """
        return self._convert_x_to_bbox(self.kf.get_state())

    def get_velocity(self):
        """
        Get the current velocity estimate.

        Returns:
            np.ndarray: Velocity [vx, vy]
        """
        state = self.kf.get_state()
        return state[2:4]

    @staticmethod
    def _convert_bbox_to_z(bbox):
        """
        Convert [x1, y1, x2, y2] to [cx, cy, w, h] format.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2]

        Returns:
            np.ndarray: Bounding box [cx, cy, w, h]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        return np.array([cx, cy, w, h])

    @staticmethod
    def _convert_x_to_bbox(x):
        """
        Convert [cx, cy, vx, vy, w, h] to [x1, y1, x2, y2] format.

        Args:
            x (np.ndarray): State vector [cx, cy, vx, vy, w, h]

        Returns:
            np.ndarray: Bounding box [x1, y1, x2, y2]
        """
        cx, cy = x[0], x[1]
        w, h = x[4], x[5]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        return np.array([x1, y1, x2, y2])
