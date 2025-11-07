"""
Trajectory analysis for vehicle movement.
Computes speed, direction, curvature, and movement patterns.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter


class TrajectoryAnalyzer:
    """
    Analyzes vehicle trajectories to extract movement features.
    """

    def __init__(self, fps=30, pixels_per_meter=50):
        """
        Initialize trajectory analyzer.

        Args:
            fps (int): Frame rate of the video
            pixels_per_meter (float): Conversion factor from pixels to meters
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.time_per_frame = 1.0 / fps

    def compute_speed(self, trajectory, smooth=True):
        """
        Compute speed along trajectory.

        Args:
            trajectory (np.ndarray): Trajectory points [N x 4] (x1, y1, x2, y2)
            smooth (bool): Whether to apply smoothing

        Returns:
            np.ndarray: Speed values in meters per second [N-1]
        """
        if len(trajectory) < 2:
            return np.array([0.0])

        # Get center points
        centers = self._get_centers(trajectory)

        # Compute displacements
        displacements = np.diff(centers, axis=0)

        # Compute distances in pixels
        distances_px = np.linalg.norm(displacements, axis=1)

        # Convert to meters
        distances_m = distances_px / self.pixels_per_meter

        # Compute speed (meters per second)
        speeds = distances_m / self.time_per_frame

        # Apply smoothing if requested
        if smooth and len(speeds) > 5:
            window_length = min(5, len(speeds) if len(speeds) % 2 == 1 else len(speeds) - 1)
            speeds = savgol_filter(speeds, window_length, 2)

        return speeds

    def compute_direction(self, trajectory):
        """
        Compute movement direction (angle) along trajectory.

        Args:
            trajectory (np.ndarray): Trajectory points [N x 4] (x1, y1, x2, y2)

        Returns:
            np.ndarray: Direction angles in degrees [N-1]
        """
        if len(trajectory) < 2:
            return np.array([0.0])

        # Get center points
        centers = self._get_centers(trajectory)

        # Compute direction vectors
        directions = np.diff(centers, axis=0)

        # Compute angles (in degrees, 0 = right, 90 = up)
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        angles_deg = np.degrees(angles)

        return angles_deg

    def compute_curvature(self, trajectory, window_size=5):
        """
        Compute path curvature along trajectory.

        Args:
            trajectory (np.ndarray): Trajectory points [N x 4] (x1, y1, x2, y2)
            window_size (int): Window size for local curvature computation

        Returns:
            np.ndarray: Curvature values (1/radius) [N]
        """
        if len(trajectory) < 3:
            return np.zeros(len(trajectory))

        # Get center points
        centers = self._get_centers(trajectory)

        curvatures = []

        for i in range(len(centers)):
            # Get local window
            start = max(0, i - window_size // 2)
            end = min(len(centers), i + window_size // 2 + 1)

            if end - start < 3:
                curvatures.append(0.0)
                continue

            local_points = centers[start:end]

            # Fit a circle to local points and compute curvature
            curvature = self._compute_local_curvature(local_points)
            curvatures.append(curvature)

        return np.array(curvatures)

    def compute_acceleration(self, speeds):
        """
        Compute acceleration from speed values.

        Args:
            speeds (np.ndarray): Speed values [N]

        Returns:
            np.ndarray: Acceleration values in m/s^2 [N-1]
        """
        if len(speeds) < 2:
            return np.array([0.0])

        # Compute acceleration
        accelerations = np.diff(speeds) / self.time_per_frame

        return accelerations

    def analyze_trajectory(self, trajectory):
        """
        Perform complete trajectory analysis.

        Args:
            trajectory (np.ndarray): Trajectory points [N x 4] (x1, y1, x2, y2)

        Returns:
            dict: Analysis results containing:
                - speeds: Speed values (m/s)
                - directions: Direction angles (degrees)
                - curvatures: Curvature values (1/m)
                - accelerations: Acceleration values (m/s^2)
                - avg_speed: Average speed (m/s)
                - max_speed: Maximum speed (m/s)
                - total_distance: Total distance traveled (m)
                - path_smoothness: Path smoothness score (0-1, higher is smoother)
        """
        if len(trajectory) < 2:
            return {
                'speeds': np.array([0.0]),
                'directions': np.array([0.0]),
                'curvatures': np.array([0.0]),
                'accelerations': np.array([0.0]),
                'avg_speed': 0.0,
                'max_speed': 0.0,
                'total_distance': 0.0,
                'path_smoothness': 1.0
            }

        # Compute features
        speeds = self.compute_speed(trajectory)
        directions = self.compute_direction(trajectory)
        curvatures = self.compute_curvature(trajectory)
        accelerations = self.compute_acceleration(speeds)

        # Compute aggregate statistics
        avg_speed = np.mean(speeds)
        max_speed = np.max(speeds)
        total_distance = np.sum(speeds) * self.time_per_frame

        # Compute path smoothness (inverse of curvature variance)
        curvature_std = np.std(curvatures)
        path_smoothness = 1.0 / (1.0 + curvature_std)

        return {
            'speeds': speeds,
            'directions': directions,
            'curvatures': curvatures,
            'accelerations': accelerations,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'total_distance': total_distance,
            'path_smoothness': path_smoothness
        }

    def detect_erratic_movement(self, trajectory, threshold=2.0):
        """
        Detect erratic or unstable movement patterns.

        Args:
            trajectory (np.ndarray): Trajectory points [N x 4] (x1, y1, x2, y2)
            threshold (float): Threshold for erratic movement detection

        Returns:
            dict: Contains:
                - is_erratic: Boolean indicating if movement is erratic
                - erratic_score: Score indicating level of erratic behavior
                - reason: Description of why movement is erratic
        """
        if len(trajectory) < 5:
            return {
                'is_erratic': False,
                'erratic_score': 0.0,
                'reason': 'Insufficient trajectory data'
            }

        analysis = self.analyze_trajectory(trajectory)

        # Check for erratic patterns
        erratic_indicators = []
        erratic_score = 0.0

        # 1. High variation in direction
        direction_changes = np.abs(np.diff(analysis['directions']))
        avg_direction_change = np.mean(direction_changes)

        if avg_direction_change > 30:  # More than 30 degrees average change
            erratic_indicators.append('high direction variation')
            erratic_score += avg_direction_change / 30.0

        # 2. High curvature variance
        curvature_std = np.std(analysis['curvatures'])
        if curvature_std > 0.05:
            erratic_indicators.append('unstable path curvature')
            erratic_score += curvature_std * 20

        # 3. High acceleration variance
        if len(analysis['accelerations']) > 0:
            accel_std = np.std(analysis['accelerations'])
            if accel_std > 2.0:  # m/s^2
                erratic_indicators.append('unstable acceleration')
                erratic_score += accel_std / 2.0

        # 4. Sudden speed changes
        if len(analysis['speeds']) > 1:
            speed_changes = np.abs(np.diff(analysis['speeds']))
            max_speed_change = np.max(speed_changes)
            if max_speed_change > 5.0:  # m/s sudden change
                erratic_indicators.append('sudden speed changes')
                erratic_score += max_speed_change / 5.0

        is_erratic = erratic_score > threshold

        reason = ', '.join(erratic_indicators) if erratic_indicators else 'stable movement'

        return {
            'is_erratic': is_erratic,
            'erratic_score': erratic_score,
            'reason': reason
        }

    @staticmethod
    def _get_centers(bboxes):
        """
        Get center points from bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes [N x 4] (x1, y1, x2, y2)

        Returns:
            np.ndarray: Center points [N x 2] (cx, cy)
        """
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        return np.column_stack([cx, cy])

    @staticmethod
    def _compute_local_curvature(points):
        """
        Compute curvature from local points using Menger curvature.

        Args:
            points (np.ndarray): Local trajectory points [N x 2]

        Returns:
            float: Curvature value
        """
        if len(points) < 3:
            return 0.0

        # Use three consecutive points to compute curvature
        # Using Menger curvature formula
        mid_idx = len(points) // 2
        if mid_idx > 0 and mid_idx < len(points) - 1:
            p1 = points[mid_idx - 1]
            p2 = points[mid_idx]
            p3 = points[mid_idx + 1]

            # Compute area of triangle
            area = 0.5 * np.abs(
                p1[0] * (p2[1] - p3[1]) +
                p2[0] * (p3[1] - p1[1]) +
                p3[0] * (p1[1] - p2[1])
            )

            # Compute side lengths
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)

            # Menger curvature
            denominator = a * b * c
            if denominator < 1e-6:
                return 0.0

            curvature = 4 * area / denominator
            return curvature

        return 0.0
