"""
Lane detection and deviation analysis for vehicle tracking.
"""
import numpy as np
import cv2


class LaneAnalyzer:
    """
    Analyzes vehicle positions relative to lanes and detects deviations.
    """

    def __init__(self, frame_width, frame_height, num_lanes=2):
        """
        Initialize lane analyzer.

        Args:
            frame_width (int): Width of video frame
            frame_height (int): Height of video frame
            num_lanes (int): Number of lanes to consider
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_lanes = num_lanes

        # Define lane boundaries (can be customized or detected)
        self.lane_boundaries = self._init_lane_boundaries()

    def _init_lane_boundaries(self):
        """
        Initialize default lane boundaries.
        Divides the frame into equal vertical sections.

        Returns:
            list: Lane boundaries [(x_left, x_right), ...]
        """
        lane_width = self.frame_width / self.num_lanes
        boundaries = []

        for i in range(self.num_lanes):
            x_left = i * lane_width
            x_right = (i + 1) * lane_width
            boundaries.append((x_left, x_right))

        return boundaries

    def set_custom_lanes(self, lane_boundaries):
        """
        Set custom lane boundaries.

        Args:
            lane_boundaries (list): List of tuples [(x_left, x_right), ...]
        """
        self.lane_boundaries = lane_boundaries
        self.num_lanes = len(lane_boundaries)

    def detect_lanes_from_image(self, frame):
        """
        Detect lane markings from image using edge detection.
        (Basic implementation - can be enhanced with deep learning)

        Args:
            frame (np.ndarray): Input frame (BGR)

        Returns:
            list: Detected lane boundaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Region of interest (bottom half of frame)
        roi_mask = np.zeros_like(edges)
        roi_mask[self.frame_height // 2:, :] = 1
        edges = edges * roi_mask

        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                               minLineLength=100, maxLineGap=50)

        if lines is None:
            return self.lane_boundaries

        # Process detected lines (simplified)
        # In production, use more sophisticated lane detection
        return self.lane_boundaries

    def get_vehicle_lane(self, bbox):
        """
        Determine which lane a vehicle is in.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2]

        Returns:
            int: Lane index (0-based), or -1 if outside lanes
        """
        # Get vehicle center x-coordinate
        cx = (bbox[0] + bbox[2]) / 2.0

        # Find which lane the vehicle is in
        for i, (x_left, x_right) in enumerate(self.lane_boundaries):
            if x_left <= cx < x_right:
                return i

        return -1  # Outside defined lanes

    def compute_lane_deviation(self, bbox, lane_id):
        """
        Compute deviation from lane center.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2]
            lane_id (int): Lane index

        Returns:
            float: Deviation from lane center (positive = right, negative = left)
        """
        if lane_id < 0 or lane_id >= self.num_lanes:
            return 0.0

        # Get vehicle center
        cx = (bbox[0] + bbox[2]) / 2.0

        # Get lane center
        x_left, x_right = self.lane_boundaries[lane_id]
        lane_center = (x_left + x_right) / 2.0

        # Compute deviation
        deviation = cx - lane_center

        return deviation

    def analyze_trajectory_lane_stability(self, trajectory):
        """
        Analyze lane stability across a trajectory.

        Args:
            trajectory (np.ndarray): Trajectory bboxes [N x 4]

        Returns:
            dict: Analysis results containing:
                - lanes: Lane IDs for each position
                - deviations: Deviation from lane center
                - lane_changes: Number of lane changes
                - stability_score: Lane keeping stability (0-1)
                - current_lane: Most recent lane
        """
        if len(trajectory) == 0:
            return {
                'lanes': [],
                'deviations': [],
                'lane_changes': 0,
                'stability_score': 1.0,
                'current_lane': -1
            }

        lanes = []
        deviations = []

        for bbox in trajectory:
            lane_id = self.get_vehicle_lane(bbox)
            lanes.append(lane_id)

            if lane_id >= 0:
                deviation = self.compute_lane_deviation(bbox, lane_id)
                deviations.append(deviation)
            else:
                deviations.append(0.0)

        lanes = np.array(lanes)
        deviations = np.array(deviations)

        # Count lane changes
        lane_changes = 0
        if len(lanes) > 1:
            lane_changes = np.sum(lanes[1:] != lanes[:-1])

        # Compute stability score
        if len(deviations) > 0:
            deviation_std = np.std(deviations)
            stability_score = 1.0 / (1.0 + deviation_std / 50.0)  # Normalize by 50 pixels
        else:
            stability_score = 1.0

        current_lane = lanes[-1] if len(lanes) > 0 else -1

        return {
            'lanes': lanes,
            'deviations': deviations,
            'lane_changes': int(lane_changes),
            'stability_score': stability_score,
            'current_lane': int(current_lane)
        }

    def detect_lane_departure(self, bbox, lane_id, threshold=50):
        """
        Detect if vehicle is departing from lane.

        Args:
            bbox (np.ndarray): Bounding box [x1, y1, x2, y2]
            lane_id (int): Current lane ID
            threshold (float): Deviation threshold in pixels

        Returns:
            dict: Contains:
                - is_departing: Boolean
                - deviation: Deviation value
                - direction: 'left' or 'right'
        """
        if lane_id < 0:
            return {
                'is_departing': False,
                'deviation': 0.0,
                'direction': 'none'
            }

        deviation = self.compute_lane_deviation(bbox, lane_id)
        is_departing = abs(deviation) > threshold

        if deviation > threshold:
            direction = 'right'
        elif deviation < -threshold:
            direction = 'left'
        else:
            direction = 'none'

        return {
            'is_departing': is_departing,
            'deviation': deviation,
            'direction': direction
        }

    def visualize_lanes(self, frame):
        """
        Draw lane boundaries on frame.

        Args:
            frame (np.ndarray): Input frame (BGR)

        Returns:
            np.ndarray: Frame with lane boundaries drawn
        """
        vis_frame = frame.copy()

        for i, (x_left, x_right) in enumerate(self.lane_boundaries):
            # Draw left boundary
            cv2.line(vis_frame,
                    (int(x_left), 0),
                    (int(x_left), self.frame_height),
                    (0, 255, 0), 2)

            # Draw right boundary
            cv2.line(vis_frame,
                    (int(x_right), 0),
                    (int(x_right), self.frame_height),
                    (0, 255, 0), 2)

            # Draw lane center (dashed)
            center_x = int((x_left + x_right) / 2)
            for y in range(0, self.frame_height, 40):
                cv2.line(vis_frame,
                        (center_x, y),
                        (center_x, y + 20),
                        (255, 255, 0), 1)

            # Lane number
            cv2.putText(vis_frame, f"Lane {i}",
                       (int(x_left) + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return vis_frame
