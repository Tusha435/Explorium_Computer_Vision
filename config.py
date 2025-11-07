"""
Configuration file for the Motorcycle Tracking System.
Modify these settings to customize the system behavior.
"""

# Detection Configuration
DETECTION_CONFIG = {
    'model_name': 'yolov8n.pt',  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    'confidence_threshold': 0.3,
    'device': None,  # 'cuda', 'cpu', or None for auto-detect
}

# Tracking Configuration
TRACKING_CONFIG = {
    'max_age': 30,  # Maximum frames to keep track without updates
    'min_hits': 3,  # Minimum hits before confirming track
    'iou_threshold': 0.3,  # IOU threshold for matching
}

# Video Configuration
VIDEO_CONFIG = {
    'fps': 30,
    'pixels_per_meter': 50,  # Calibration parameter
}

# Lane Configuration
LANE_CONFIG = {
    'num_lanes': 2,
    'auto_detect': False,  # Set to True to auto-detect lanes
    'custom_boundaries': None,  # [(x_left, x_right), ...] or None
}

# Safety Thresholds
SAFETY_THRESHOLDS = {
    'max_safe_speed': 15.0,  # m/s (~54 km/h)
    'max_safe_acceleration': 3.0,  # m/s²
    'max_safe_curvature': 0.1,  # 1/m
    'max_direction_change': 45.0,  # degrees per frame
    'lane_deviation_threshold': 60,  # pixels
    'erratic_movement_threshold': 2.0,
}

# Risk Scoring Weights
RISK_WEIGHTS = {
    'speed': 0.25,
    'acceleration': 0.20,
    'erratic_movement': 0.25,
    'lane_deviation': 0.15,
    'sudden_direction_change': 0.15,
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'show_lanes': True,
    'show_trajectory': True,
    'trajectory_length': 20,  # Number of points to display
    'show_warnings': True,
    'max_warnings_per_track': 3,
    'font_scale': 0.6,
    'line_thickness': 2,
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_video': True,
    'save_report': True,
    'report_format': 'json',  # 'json' or 'csv'
    'verbose': True,
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_gpu': True,
    'batch_size': 1,
    'num_threads': 4,
}
