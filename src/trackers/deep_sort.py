"""
Deep SORT tracker implementation for multi-object tracking.
Uses Kalman filters for motion prediction and Hungarian algorithm for data association.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanBoxTracker


class Track:
    """
    Represents a single tracked object with full trajectory history.
    """

    def __init__(self, track_id, bbox, frame_id):
        """
        Initialize a new track.

        Args:
            track_id (int): Unique track identifier
            bbox (np.ndarray): Initial bounding box [x1, y1, x2, y2]
            frame_id (int): Frame number when track was created
        """
        self.id = track_id
        self.kalman_tracker = KalmanBoxTracker(bbox)

        # Track state
        self.is_confirmed = False
        self.is_deleted = False
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Trajectory history
        self.trajectory = [bbox]
        self.velocities = []
        self.frame_ids = [frame_id]

        # Track quality
        self.max_age = 30  # Maximum frames to keep track without updates
        self.min_hits = 3  # Minimum hits before confirming track

    def predict(self):
        """Predict next state using Kalman filter."""
        predicted_bbox = self.kalman_tracker.predict()
        self.age += 1
        self.time_since_update += 1
        return predicted_bbox

    def update(self, bbox, frame_id):
        """
        Update track with new detection.

        Args:
            bbox (np.ndarray): New bounding box [x1, y1, x2, y2]
            frame_id (int): Current frame number
        """
        self.kalman_tracker.update(bbox)
        self.hits += 1
        self.time_since_update = 0

        # Update trajectory
        self.trajectory.append(bbox)
        self.frame_ids.append(frame_id)

        # Calculate velocity
        velocity = self.kalman_tracker.get_velocity()
        self.velocities.append(velocity)

        # Confirm track if it has enough hits
        if self.hits >= self.min_hits:
            self.is_confirmed = True

    def mark_missed(self):
        """Mark track as missed in current frame."""
        if self.time_since_update > self.max_age:
            self.is_deleted = True

    def get_current_bbox(self):
        """Get current bounding box estimate."""
        return self.kalman_tracker.get_state()

    def get_current_velocity(self):
        """Get current velocity estimate."""
        return self.kalman_tracker.get_velocity()

    def get_trajectory(self):
        """Get full trajectory history."""
        return np.array(self.trajectory)

    def get_velocities(self):
        """Get velocity history."""
        if not self.velocities:
            return np.array([])
        return np.array(self.velocities)


class DeepSORT:
    """
    Deep SORT tracker for multi-object tracking.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize Deep SORT tracker.

        Args:
            max_age (int): Maximum frames to keep track without updates
            min_hits (int): Minimum hits before confirming track
            iou_threshold (float): IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections (np.ndarray): Detections array [N x 4] (x1, y1, x2, y2)

        Returns:
            list: List of active Track objects
        """
        self.frame_count += 1

        # Predict new locations for existing tracks
        predicted_bboxes = []
        for track in self.tracks:
            bbox = track.predict()
            predicted_bboxes.append(bbox)

        # Match detections to existing tracks
        if len(predicted_bboxes) > 0 and len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
                detections, predicted_bboxes
            )

            # Update matched tracks
            for det_idx, trk_idx in matched:
                self.tracks[trk_idx].update(detections[det_idx], self.frame_count)

            # Mark unmatched tracks as missed
            for trk_idx in unmatched_trks:
                self.tracks[trk_idx].mark_missed()

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_new_track(detections[det_idx])
        elif len(detections) > 0:
            # No existing tracks, create new ones
            for detection in detections:
                self._create_new_track(detection)
        else:
            # No detections, mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # Return only confirmed tracks
        return [t for t in self.tracks if t.is_confirmed]

    def _create_new_track(self, bbox):
        """Create a new track from detection."""
        track = Track(self.track_id_counter, bbox, self.frame_count)
        track.max_age = self.max_age
        track.min_hits = self.min_hits
        self.tracks.append(track)
        self.track_id_counter += 1

    def _associate_detections_to_tracks(self, detections, predicted_bboxes):
        """
        Associate detections to tracked objects using IOU.

        Args:
            detections (np.ndarray): Detection bounding boxes [N x 4]
            predicted_bboxes (list): Predicted bounding boxes from tracks [M x 4]

        Returns:
            tuple: (matched_indices, unmatched_detections, unmatched_tracks)
        """
        if len(predicted_bboxes) == 0:
            return [], list(range(len(detections))), []

        # Compute IOU cost matrix
        iou_matrix = self._compute_iou_matrix(detections, predicted_bboxes)

        # Convert IOU to cost (1 - IOU)
        cost_matrix = 1 - iou_matrix

        # Use Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter matches by IOU threshold
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(predicted_bboxes)))

        for det_idx, trk_idx in zip(row_ind, col_ind):
            if iou_matrix[det_idx, trk_idx] >= self.iou_threshold:
                matched_indices.append((det_idx, trk_idx))
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(trk_idx)

        return matched_indices, unmatched_detections, unmatched_tracks

    @staticmethod
    def _compute_iou_matrix(bboxes1, bboxes2):
        """
        Compute IOU matrix between two sets of bounding boxes.

        Args:
            bboxes1 (np.ndarray): First set of bboxes [N x 4]
            bboxes2 (list or np.ndarray): Second set of bboxes [M x 4]

        Returns:
            np.ndarray: IOU matrix [N x M]
        """
        bboxes2 = np.array(bboxes2)

        # Compute intersection
        x1_max = np.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0])
        y1_max = np.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1])
        x2_min = np.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2])
        y2_min = np.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3])

        intersection = np.maximum(0, x2_min - x1_max) * np.maximum(0, y2_min - y1_max)

        # Compute areas
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

        # Compute union
        union = area1[:, None] + area2 - intersection

        # Compute IOU
        iou = intersection / (union + 1e-6)

        return iou

    def get_all_tracks(self):
        """Get all tracks (including unconfirmed)."""
        return self.tracks

    def get_confirmed_tracks(self):
        """Get only confirmed tracks."""
        return [t for t in self.tracks if t.is_confirmed]
