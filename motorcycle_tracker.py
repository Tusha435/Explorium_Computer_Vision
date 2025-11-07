"""
Main motorcycle tracking and safety analysis system.
Integrates detection, tracking, trajectory analysis, and safety assessment.
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
import json

from src.models.detector import VehicleDetector
from src.trackers.deep_sort import DeepSORT
from src.analyzers.safety_analyzer import SafetyAnalyzer


class MotorcycleTracker:
    """
    Complete motorcycle tracking and safety analysis system.
    """

    def __init__(self, model_name='yolov8n.pt', fps=30, pixels_per_meter=50):
        """
        Initialize the tracker.

        Args:
            model_name (str): YOLOv8 model name
            fps (int): Video frame rate
            pixels_per_meter (float): Pixel to meter conversion factor
        """
        print("Initializing Motorcycle Tracking System...")

        # Initialize components
        self.detector = VehicleDetector(model_name=model_name, conf_threshold=0.3)
        self.tracker = DeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter

        # Safety analyzer (will be initialized with frame shape)
        self.safety_analyzer = None

        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'safety_incidents': 0
        }

        # Results storage
        self.track_histories = {}
        self.safety_reports = []

    def process_video(self, video_path, output_path=None, visualize=True, save_report=True):
        """
        Process a video file for motorcycle tracking and safety analysis.

        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            visualize (bool): Whether to show visualization
            save_report (bool): Whether to save analysis report

        Returns:
            dict: Processing results and statistics
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

        # Initialize safety analyzer with frame dimensions
        self.safety_analyzer = SafetyAnalyzer(
            fps=fps,
            pixels_per_meter=self.pixels_per_meter,
            frame_shape=(frame_height, frame_width)
        )

        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.stats['total_frames'] = frame_count

                # Process frame
                annotated_frame = self.process_frame(frame, frame_count)

                # Write frame
                if writer:
                    writer.write(annotated_frame)

                # Display
                if visualize:
                    # Resize for display if too large
                    display_frame = annotated_frame
                    if frame_width > 1280:
                        scale = 1280 / frame_width
                        new_width = 1280
                        new_height = int(frame_height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))

                    cv2.imshow('Motorcycle Safety Tracking', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing interrupted by user")
                        break
                    elif key == ord(' '):
                        # Pause on space
                        cv2.waitKey(0)

                # Progress update
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)")

        finally:
            cap.release()
            if writer:
                writer.release()
            if visualize:
                cv2.destroyAllWindows()

        # Generate final report
        if save_report:
            self._save_analysis_report(video_path, output_path)

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Total detections: {self.stats['total_detections']}")
        print(f"Total tracks: {self.stats['total_tracks']}")
        print(f"Safety incidents: {self.stats['safety_incidents']}")
        print("="*60)

        return self.stats

    def process_frame(self, frame, frame_id):
        """
        Process a single frame.

        Args:
            frame (np.ndarray): Input frame
            frame_id (int): Frame number

        Returns:
            np.ndarray: Annotated frame
        """
        # Detect motorcycles
        detections = self.detector.detect_motorcycles(frame)
        self.stats['total_detections'] += len(detections)

        # Convert to numpy array for tracker
        if detections:
            detection_bboxes = np.array([det['bbox'] for det in detections])
        else:
            detection_bboxes = np.empty((0, 4))

        # Update tracker
        active_tracks = self.tracker.update(detection_bboxes)

        # Update track count
        self.stats['total_tracks'] = self.tracker.track_id_counter

        # Analyze each track for safety
        safety_results = []
        for track in active_tracks:
            # Perform safety analysis
            safety_analysis = self.safety_analyzer.analyze_rider_behavior(track)
            safety_results.append(safety_analysis)

            # Check for movement intention (newbie detection)
            if len(track.trajectory) >= 5 and len(track.trajectory) <= 20:
                movement_analysis = self.safety_analyzer.detect_intention_to_move(track)
                safety_analysis['movement_intention'] = movement_analysis

            # Store track history
            if track.id not in self.track_histories:
                self.track_histories[track.id] = []
            self.track_histories[track.id].append({
                'frame': frame_id,
                'bbox': track.get_current_bbox().tolist(),
                'velocity': track.get_current_velocity().tolist(),
                'safety_status': safety_analysis['safety_status'],
                'risk_score': safety_analysis['risk_score']
            })

            # Count safety incidents
            if safety_analysis['risk_score'] > 50:
                self.stats['safety_incidents'] += 1

            # Store significant safety reports
            if safety_analysis['accident_prediction']['is_high_risk']:
                self.safety_reports.append({
                    'frame': frame_id,
                    'track_id': track.id,
                    'analysis': safety_analysis
                })

        # Visualize results
        annotated_frame = self._visualize_tracking(frame, active_tracks, safety_results)

        return annotated_frame

    def _visualize_tracking(self, frame, tracks, safety_results):
        """
        Visualize tracking and safety analysis.

        Args:
            frame (np.ndarray): Input frame
            tracks (list): List of Track objects
            safety_results (list): List of safety analysis results

        Returns:
            np.ndarray: Annotated frame
        """
        vis_frame = frame.copy()

        # Draw lane markers
        if self.safety_analyzer:
            vis_frame = self.safety_analyzer.lane_analyzer.visualize_lanes(vis_frame)

        # Draw each track
        for track, safety in zip(tracks, safety_results):
            bbox = track.get_current_bbox().astype(int)
            velocity = track.get_current_velocity()
            risk_score = safety['risk_score']
            safety_status = safety['safety_status']

            # Color based on safety status
            if safety_status == 'SAFE':
                color = (0, 255, 0)  # Green
            elif safety_status == 'CAUTION':
                color = (0, 255, 255)  # Yellow
            elif safety_status == 'WARNING':
                color = (0, 165, 255)  # Orange
            else:  # DANGER
                color = (0, 0, 255)  # Red

            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

            # Draw trajectory
            trajectory = track.get_trajectory()
            if len(trajectory) > 1:
                points = []
                for traj_bbox in trajectory[-20:]:  # Last 20 points
                    cx = int((traj_bbox[0] + traj_bbox[2]) / 2)
                    cy = int((traj_bbox[1] + traj_bbox[3]) / 2)
                    points.append([cx, cy])

                points = np.array(points, dtype=np.int32)
                cv2.polylines(vis_frame, [points], False, color, 2)

            # Draw info box
            info_y = bbox[1] - 10
            line_height = 20

            # Track ID and status
            label = f"ID:{track.id} {safety_status}"
            cv2.putText(vis_frame, label, (bbox[0], info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Risk score
            info_y -= line_height
            risk_label = f"Risk: {risk_score:.1f}%"
            cv2.putText(vis_frame, risk_label, (bbox[0], info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Speed
            speed = np.linalg.norm(velocity) / self.fps * self.pixels_per_meter
            info_y -= line_height
            speed_label = f"Speed: {speed:.1f} m/s"
            cv2.putText(vis_frame, speed_label, (bbox[0], info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display warnings
            if safety['warnings']:
                info_y = bbox[3] + 20
                for i, warning in enumerate(safety['warnings'][:3]):  # Max 3 warnings
                    cv2.putText(vis_frame, warning[:50], (bbox[0], info_y + i * line_height),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Draw movement intention warning for newbies
            if 'movement_intention' in safety:
                movement = safety['movement_intention']
                if movement['should_warn']:
                    warning_text = movement['safety_advice'][:60]
                    # Draw warning box at top of vehicle
                    text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(vis_frame,
                                (bbox[0] - 5, bbox[1] - 50),
                                (bbox[0] + text_size[0] + 5, bbox[1] - 25),
                                (0, 0, 255), -1)
                    cv2.putText(vis_frame, warning_text,
                               (bbox[0], bbox[1] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw global statistics
        stats_y = 30
        cv2.putText(vis_frame, f"Frame: {self.stats['total_frames']}",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Active Tracks: {len(tracks)}",
                   (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Total Detections: {self.stats['total_detections']}",
                   (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis_frame

    def _save_analysis_report(self, video_path, output_path):
        """
        Save comprehensive analysis report to JSON.

        Args:
            video_path (str): Input video path
            output_path (str): Output video path
        """
        report_path = Path(video_path).stem + '_analysis_report.json'

        report = {
            'video_info': {
                'input_video': str(video_path),
                'output_video': str(output_path) if output_path else None,
                'total_frames': self.stats['total_frames']
            },
            'statistics': self.stats,
            'safety_incidents': self.safety_reports,
            'track_summaries': {}
        }

        # Generate summary for each track
        for track_id, history in self.track_histories.items():
            if len(history) < 5:
                continue

            risk_scores = [h['risk_score'] for h in history]
            avg_risk = np.mean(risk_scores)
            max_risk = np.max(risk_scores)

            report['track_summaries'][str(track_id)] = {
                'duration_frames': len(history),
                'average_risk_score': float(avg_risk),
                'max_risk_score': float(max_risk),
                'final_status': history[-1]['safety_status']
            }

        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nAnalysis report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Motorcycle Tracking and Safety Analysis')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, help='Path to output video (optional)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--pixels-per-meter', type=float, default=50,
                       help='Pixel to meter conversion factor')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--no-report', action='store_true', help='Disable report generation')

    args = parser.parse_args()

    # Initialize tracker
    tracker = MotorcycleTracker(
        model_name=args.model,
        fps=args.fps,
        pixels_per_meter=args.pixels_per_meter
    )

    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        visualize=not args.no_viz,
        save_report=not args.no_report
    )


if __name__ == '__main__':
    main()
