"""
Example usage of the Motorcycle Tracking and Safety Analysis system.
Demonstrates how to use the system with custom configurations.
"""
import cv2
import numpy as np
from motorcycle_tracker import MotorcycleTracker


def example_basic_tracking():
    """Basic example: Track motorcycles in a video."""
    print("Example 1: Basic Motorcycle Tracking")
    print("-" * 60)

    # Initialize tracker
    tracker = MotorcycleTracker(
        model_name='yolov8n.pt',  # Fast model for real-time
        fps=30,
        pixels_per_meter=50
    )

    # Process video
    stats = tracker.process_video(
        video_path='path/to/your/video.mp4',
        output_path='output_basic.mp4',
        visualize=True,
        save_report=True
    )

    print(f"\nProcessing complete! Stats: {stats}")


def example_webcam_tracking():
    """Example: Real-time tracking from webcam."""
    print("Example 2: Real-time Webcam Tracking")
    print("-" * 60)

    # Initialize tracker
    tracker = MotorcycleTracker(model_name='yolov8n.pt', fps=30)

    # Open webcam
    cap = cv2.VideoCapture(0)

    frame_count = 0
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process frame
        annotated_frame = tracker.process_frame(frame, frame_count)

        # Display
        cv2.imshow('Motorcycle Safety Tracking - Webcam', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTracked {frame_count} frames")


def example_custom_analysis():
    """Example: Custom safety analysis with specific thresholds."""
    print("Example 3: Custom Safety Analysis")
    print("-" * 60)

    from src.models.detector import VehicleDetector
    from src.trackers.deep_sort import DeepSORT
    from src.analyzers.safety_analyzer import SafetyAnalyzer

    # Initialize components with custom settings
    detector = VehicleDetector(model_name='yolov8s.pt', conf_threshold=0.4)
    tracker = DeepSORT(max_age=50, min_hits=5, iou_threshold=0.25)

    # Initialize safety analyzer
    safety_analyzer = SafetyAnalyzer(fps=30, pixels_per_meter=50, frame_shape=(720, 1280))

    # Customize safety thresholds
    safety_analyzer.thresholds['max_safe_speed'] = 12.0  # Stricter speed limit
    safety_analyzer.thresholds['max_safe_acceleration'] = 2.5  # Stricter acceleration

    print("Custom thresholds set:")
    print(f"  Max safe speed: {safety_analyzer.thresholds['max_safe_speed']} m/s")
    print(f"  Max safe acceleration: {safety_analyzer.thresholds['max_safe_acceleration']} m/s²")

    # Now you can use these components for processing
    # (Similar to the main tracker but with custom settings)


def example_trajectory_analysis():
    """Example: Detailed trajectory analysis for a single track."""
    print("Example 4: Detailed Trajectory Analysis")
    print("-" * 60)

    from src.analyzers.trajectory_analyzer import TrajectoryAnalyzer

    # Create sample trajectory (replace with real data)
    # Format: [x1, y1, x2, y2] bounding boxes over time
    sample_trajectory = np.array([
        [100, 200, 150, 280],
        [110, 200, 160, 280],
        [125, 198, 175, 278],
        [145, 195, 195, 275],
        [170, 193, 220, 273],
        [200, 190, 250, 270],
        [235, 188, 285, 268],
        [275, 185, 325, 265],
    ])

    # Initialize analyzer
    analyzer = TrajectoryAnalyzer(fps=30, pixels_per_meter=50)

    # Perform analysis
    analysis = analyzer.analyze_trajectory(sample_trajectory)

    print("\nTrajectory Analysis Results:")
    print(f"  Average speed: {analysis['avg_speed']:.2f} m/s")
    print(f"  Max speed: {analysis['max_speed']:.2f} m/s")
    print(f"  Total distance: {analysis['total_distance']:.2f} m")
    print(f"  Path smoothness: {analysis['path_smoothness']:.2f}")

    # Detect erratic movement
    erratic = analyzer.detect_erratic_movement(sample_trajectory)
    print(f"\nErratic movement detected: {erratic['is_erratic']}")
    print(f"  Erratic score: {erratic['erratic_score']:.2f}")
    print(f"  Reason: {erratic['reason']}")


def example_lane_analysis():
    """Example: Lane deviation analysis."""
    print("Example 5: Lane Deviation Analysis")
    print("-" * 60)

    from src.analyzers.lane_analyzer import LaneAnalyzer

    # Initialize lane analyzer
    lane_analyzer = LaneAnalyzer(frame_width=1280, frame_height=720, num_lanes=2)

    # Sample vehicle position
    vehicle_bbox = np.array([300, 400, 350, 480])

    # Get lane
    lane_id = lane_analyzer.get_vehicle_lane(vehicle_bbox)
    print(f"Vehicle is in lane: {lane_id}")

    # Check deviation
    deviation = lane_analyzer.compute_lane_deviation(vehicle_bbox, lane_id)
    print(f"Deviation from lane center: {deviation:.1f} pixels")

    # Check for lane departure
    departure = lane_analyzer.detect_lane_departure(vehicle_bbox, lane_id, threshold=50)
    print(f"\nLane departure detected: {departure['is_departing']}")
    print(f"  Direction: {departure['direction']}")


def example_newbie_safety():
    """Example: Newbie rider safety analysis."""
    print("Example 6: Newbie Rider Safety Analysis")
    print("-" * 60)

    from src.trackers.deep_sort import Track
    from src.analyzers.safety_analyzer import SafetyAnalyzer

    # Create a simulated track (replace with real track object)
    # Simulating a newbie's shaky start
    initial_bbox = np.array([200, 300, 250, 380])
    track = Track(track_id=1, bbox=initial_bbox, frame_id=0)

    # Simulate trajectory with jerky movements
    jerky_trajectory = [
        np.array([200, 300, 250, 380]),
        np.array([210, 298, 260, 378]),  # Small movement
        np.array([230, 295, 280, 375]),  # Sudden acceleration
        np.array([260, 293, 310, 373]),  # Continuing fast
        np.array([270, 292, 320, 372]),  # Slowing down
        np.array([275, 291, 325, 371]),  # Almost stopped
    ]

    # Update track with trajectory
    for i, bbox in enumerate(jerky_trajectory[1:], 1):
        track.update(bbox, i)

    # Initialize safety analyzer
    safety_analyzer = SafetyAnalyzer(fps=30, pixels_per_meter=50, frame_shape=(720, 1280))

    # Analyze movement intention
    movement_analysis = safety_analyzer.detect_intention_to_move(track)

    print("\nNewbie Movement Analysis:")
    print(f"  Attempting movement: {movement_analysis['is_attempting_movement']}")
    print(f"  Movement quality: {movement_analysis['movement_quality']}")
    print(f"  Safety advice: {movement_analysis['safety_advice']}")
    print(f"  Predicted issues: {movement_analysis['predicted_issues']}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MOTORCYCLE TRACKING SYSTEM - EXAMPLES")
    print("="*60 + "\n")

    # Run non-video examples
    example_custom_analysis()
    print("\n")

    example_trajectory_analysis()
    print("\n")

    example_lane_analysis()
    print("\n")

    example_newbie_safety()
    print("\n")

    print("="*60)
    print("NOTE: To run video examples, update the video paths")
    print("      and uncomment the desired example in main()")
    print("="*60)

    # Uncomment to run video examples (update paths first):
    # example_basic_tracking()
    # example_webcam_tracking()


if __name__ == '__main__':
    main()
