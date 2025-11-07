"""
Safety analyzer for detecting dangerous movements and predicting potential accidents.
Specifically designed to help newbie riders avoid accidents by detecting risky behaviors.
"""
import numpy as np
from .trajectory_analyzer import TrajectoryAnalyzer
from .lane_analyzer import LaneAnalyzer


class SafetyAnalyzer:
    """
    Analyzes riding behavior to detect potentially dangerous situations
    and predict when a newbie rider might lose control.
    """

    def __init__(self, fps=30, pixels_per_meter=50, frame_shape=(720, 1280)):
        """
        Initialize safety analyzer.

        Args:
            fps (int): Frame rate
            pixels_per_meter (float): Pixel to meter conversion
            frame_shape (tuple): (height, width) of video frame
        """
        self.trajectory_analyzer = TrajectoryAnalyzer(fps, pixels_per_meter)
        self.lane_analyzer = LaneAnalyzer(frame_shape[1], frame_shape[0])

        # Safety thresholds
        self.thresholds = {
            'max_safe_speed': 15.0,  # m/s (~54 km/h)
            'max_safe_acceleration': 3.0,  # m/s^2
            'max_safe_curvature': 0.1,  # 1/m
            'max_direction_change': 45.0,  # degrees per frame
            'lane_deviation_threshold': 60,  # pixels
            'erratic_movement_threshold': 2.0,
        }

        # Risk scoring weights
        self.risk_weights = {
            'speed': 0.25,
            'acceleration': 0.20,
            'erratic_movement': 0.25,
            'lane_deviation': 0.15,
            'sudden_direction_change': 0.15
        }

    def analyze_rider_behavior(self, track):
        """
        Comprehensive analysis of rider behavior for safety assessment.

        Args:
            track: Track object with trajectory history

        Returns:
            dict: Safety analysis results
        """
        trajectory = track.get_trajectory()

        if len(trajectory) < 3:
            return self._default_safety_report()

        # 1. Trajectory analysis
        traj_analysis = self.trajectory_analyzer.analyze_trajectory(trajectory)

        # 2. Erratic movement detection
        erratic_analysis = self.trajectory_analyzer.detect_erratic_movement(trajectory)

        # 3. Lane stability analysis
        lane_analysis = self.lane_analyzer.analyze_trajectory_lane_stability(trajectory)

        # 4. Detect specific dangerous patterns
        danger_patterns = self._detect_danger_patterns(trajectory, traj_analysis)

        # 5. Compute overall risk score
        risk_score = self._compute_risk_score(traj_analysis, erratic_analysis, lane_analysis)

        # 6. Generate safety warnings
        warnings = self._generate_warnings(traj_analysis, erratic_analysis,
                                          lane_analysis, danger_patterns, risk_score)

        # 7. Predict accident likelihood
        accident_prediction = self._predict_accident_likelihood(
            trajectory, traj_analysis, erratic_analysis, lane_analysis, risk_score
        )

        return {
            'track_id': track.id,
            'trajectory_analysis': traj_analysis,
            'erratic_movement': erratic_analysis,
            'lane_analysis': lane_analysis,
            'danger_patterns': danger_patterns,
            'risk_score': risk_score,
            'warnings': warnings,
            'accident_prediction': accident_prediction,
            'safety_status': self._get_safety_status(risk_score)
        }

    def _detect_danger_patterns(self, trajectory, traj_analysis):
        """
        Detect specific dangerous riding patterns.

        Args:
            trajectory (np.ndarray): Trajectory history
            traj_analysis (dict): Trajectory analysis results

        Returns:
            dict: Detected danger patterns
        """
        patterns = {
            'excessive_speed': False,
            'hard_braking': False,
            'sudden_acceleration': False,
            'sharp_turn': False,
            'weaving': False,
            'unstable_start': False,
            'loss_of_control_signs': False
        }

        # Check excessive speed
        if traj_analysis['max_speed'] > self.thresholds['max_safe_speed']:
            patterns['excessive_speed'] = True

        # Check hard braking (sudden deceleration)
        if len(traj_analysis['accelerations']) > 0:
            min_accel = np.min(traj_analysis['accelerations'])
            if min_accel < -self.thresholds['max_safe_acceleration']:
                patterns['hard_braking'] = True

            # Check sudden acceleration
            max_accel = np.max(traj_analysis['accelerations'])
            if max_accel > self.thresholds['max_safe_acceleration']:
                patterns['sudden_acceleration'] = True

        # Check sharp turns (high curvature)
        if len(traj_analysis['curvatures']) > 0:
            max_curvature = np.max(np.abs(traj_analysis['curvatures']))
            if max_curvature > self.thresholds['max_safe_curvature']:
                patterns['sharp_turn'] = True

        # Check weaving (frequent direction changes)
        if len(traj_analysis['directions']) > 2:
            direction_changes = np.abs(np.diff(traj_analysis['directions']))
            # Normalize to handle angle wrapping
            direction_changes = np.minimum(direction_changes, 360 - direction_changes)

            frequent_changes = np.sum(direction_changes > 20) > len(direction_changes) * 0.3
            if frequent_changes:
                patterns['weaving'] = True

        # Check unstable start (initial frames show erratic behavior)
        if len(trajectory) >= 10:
            initial_trajectory = trajectory[:10]
            initial_analysis = self.trajectory_analyzer.detect_erratic_movement(initial_trajectory)
            if initial_analysis['is_erratic']:
                patterns['unstable_start'] = True

        # Check loss of control signs (combination of factors)
        loss_of_control_indicators = [
            patterns['weaving'],
            patterns['sharp_turn'],
            patterns['hard_braking'],
            traj_analysis['path_smoothness'] < 0.3
        ]
        if sum(loss_of_control_indicators) >= 2:
            patterns['loss_of_control_signs'] = True

        return patterns

    def _compute_risk_score(self, traj_analysis, erratic_analysis, lane_analysis):
        """
        Compute overall risk score (0-100).

        Args:
            traj_analysis (dict): Trajectory analysis
            erratic_analysis (dict): Erratic movement analysis
            lane_analysis (dict): Lane analysis

        Returns:
            float: Risk score (0-100, higher is more risky)
        """
        risk_components = {}

        # Speed risk
        speed_risk = min(100, (traj_analysis['max_speed'] / self.thresholds['max_safe_speed']) * 100)
        risk_components['speed'] = speed_risk

        # Acceleration risk
        if len(traj_analysis['accelerations']) > 0:
            max_accel = np.max(np.abs(traj_analysis['accelerations']))
            accel_risk = min(100, (max_accel / self.thresholds['max_safe_acceleration']) * 100)
        else:
            accel_risk = 0
        risk_components['acceleration'] = accel_risk

        # Erratic movement risk
        erratic_risk = min(100, (erratic_analysis['erratic_score'] / self.thresholds['erratic_movement_threshold']) * 100)
        risk_components['erratic_movement'] = erratic_risk

        # Lane deviation risk
        lane_stability_risk = (1.0 - lane_analysis['stability_score']) * 100
        risk_components['lane_deviation'] = lane_stability_risk

        # Direction change risk
        if len(traj_analysis['directions']) > 1:
            direction_changes = np.abs(np.diff(traj_analysis['directions']))
            direction_changes = np.minimum(direction_changes, 360 - direction_changes)
            max_direction_change = np.max(direction_changes)
            direction_risk = min(100, (max_direction_change / self.thresholds['max_direction_change']) * 100)
        else:
            direction_risk = 0
        risk_components['sudden_direction_change'] = direction_risk

        # Weighted sum
        total_risk = sum(risk_components[key] * self.risk_weights[key]
                        for key in risk_components.keys())

        return min(100, total_risk)

    def _generate_warnings(self, traj_analysis, erratic_analysis, lane_analysis,
                          danger_patterns, risk_score):
        """
        Generate safety warnings based on analysis.

        Args:
            traj_analysis (dict): Trajectory analysis
            erratic_analysis (dict): Erratic movement analysis
            lane_analysis (dict): Lane analysis
            danger_patterns (dict): Detected danger patterns
            risk_score (float): Overall risk score

        Returns:
            list: List of warning messages
        """
        warnings = []

        # Critical warnings (high priority)
        if danger_patterns['loss_of_control_signs']:
            warnings.append("CRITICAL: Signs of losing control detected! Reduce speed and stabilize.")

        if danger_patterns['excessive_speed']:
            warnings.append(f"WARNING: Excessive speed ({traj_analysis['max_speed']:.1f} m/s). Slow down!")

        if danger_patterns['hard_braking']:
            warnings.append("WARNING: Hard braking detected. Anticipate stops earlier.")

        # Important warnings
        if danger_patterns['sharp_turn']:
            warnings.append("CAUTION: Sharp turn detected. Reduce speed before turning.")

        if danger_patterns['weaving']:
            warnings.append("CAUTION: Weaving movement detected. Keep steady control.")

        if danger_patterns['unstable_start']:
            warnings.append("NOTICE: Unstable starting behavior. Practice smooth acceleration.")

        if erratic_analysis['is_erratic']:
            warnings.append(f"CAUTION: Erratic movement - {erratic_analysis['reason']}")

        if lane_analysis['stability_score'] < 0.5:
            warnings.append("CAUTION: Poor lane keeping. Stay centered in lane.")

        # Risk-based warnings
        if risk_score > 70:
            warnings.append("DANGER: High risk level! Exercise extreme caution.")
        elif risk_score > 50:
            warnings.append("WARNING: Moderate risk detected. Be more careful.")

        return warnings

    def _predict_accident_likelihood(self, trajectory, traj_analysis, erratic_analysis,
                                    lane_analysis, risk_score):
        """
        Predict the likelihood of an accident in the near future.

        Args:
            trajectory (np.ndarray): Trajectory history
            traj_analysis (dict): Trajectory analysis
            erratic_analysis (dict): Erratic movement analysis
            lane_analysis (dict): Lane analysis
            risk_score (float): Overall risk score

        Returns:
            dict: Accident prediction results
        """
        # Factors that increase accident likelihood
        accident_factors = []
        likelihood_score = 0.0

        # 1. High risk score
        if risk_score > 70:
            accident_factors.append('very high risk score')
            likelihood_score += 30

        # 2. Erratic movement
        if erratic_analysis['is_erratic']:
            accident_factors.append('erratic movement pattern')
            likelihood_score += 20

        # 3. Poor lane keeping
        if lane_analysis['stability_score'] < 0.4:
            accident_factors.append('poor lane control')
            likelihood_score += 15

        # 4. High speed with instability
        if traj_analysis['avg_speed'] > 10 and traj_analysis['path_smoothness'] < 0.4:
            accident_factors.append('high speed with unstable path')
            likelihood_score += 20

        # 5. Recent sharp maneuvers
        if len(trajectory) >= 5:
            recent_traj = trajectory[-5:]
            recent_analysis = self.trajectory_analyzer.analyze_trajectory(recent_traj)
            if np.max(np.abs(recent_analysis['curvatures'])) > 0.08:
                accident_factors.append('recent sharp maneuver')
                likelihood_score += 15

        # Determine risk level
        if likelihood_score > 60:
            risk_level = 'CRITICAL'
            recommendation = "STOP or SLOW DOWN IMMEDIATELY! High accident risk detected."
        elif likelihood_score > 40:
            risk_level = 'HIGH'
            recommendation = "Reduce speed significantly and stabilize your riding."
        elif likelihood_score > 20:
            risk_level = 'MODERATE'
            recommendation = "Exercise caution and improve control."
        else:
            risk_level = 'LOW'
            recommendation = "Continue riding safely."

        return {
            'likelihood_score': min(100, likelihood_score),
            'risk_level': risk_level,
            'contributing_factors': accident_factors,
            'recommendation': recommendation,
            'is_high_risk': likelihood_score > 40
        }

    def detect_intention_to_move(self, track, min_trajectory_length=5):
        """
        Detect when a newbie rider intends to move their vehicle.
        This helps predict potentially dangerous starting movements.

        Args:
            track: Track object
            min_trajectory_length (int): Minimum trajectory length to analyze

        Returns:
            dict: Movement intention analysis
        """
        trajectory = track.get_trajectory()

        if len(trajectory) < min_trajectory_length:
            return {
                'is_attempting_movement': False,
                'movement_quality': 'unknown',
                'safety_advice': 'Insufficient data for analysis',
                'predicted_issues': []
            }

        # Analyze initial movement pattern
        traj_analysis = self.trajectory_analyzer.analyze_trajectory(trajectory)

        predicted_issues = []
        movement_quality = 'good'

        # Check for common newbie mistakes during starting

        # 1. Too fast acceleration
        if len(traj_analysis['accelerations']) > 0:
            initial_accel = traj_analysis['accelerations'][0] if len(traj_analysis['accelerations']) > 0 else 0
            if initial_accel > 4.0:  # Very high initial acceleration
                predicted_issues.append('Excessive initial acceleration - risk of losing balance')
                movement_quality = 'poor'

        # 2. Jerky start (high acceleration variance early on)
        if len(traj_analysis['accelerations']) >= 3:
            early_accel = traj_analysis['accelerations'][:3]
            if np.std(early_accel) > 2.0:
                predicted_issues.append('Jerky start - practice smoother throttle control')
                movement_quality = 'poor' if movement_quality == 'poor' else 'fair'

        # 3. Immediate direction change (trying to turn while starting)
        if len(traj_analysis['directions']) >= 2:
            initial_direction_change = abs(traj_analysis['directions'][1] - traj_analysis['directions'][0])
            if initial_direction_change > 30:
                predicted_issues.append('Turning while starting - stabilize first before turning')
                movement_quality = 'poor'

        # 4. Wobbling (high curvature variance in first few frames)
        if len(traj_analysis['curvatures']) >= 5:
            early_curvature = traj_analysis['curvatures'][:5]
            if np.std(early_curvature) > 0.05:
                predicted_issues.append('Wobbling detected - keep handlebars steady')
                movement_quality = 'poor' if movement_quality == 'poor' else 'fair'

        # Generate safety advice
        if movement_quality == 'poor':
            safety_advice = "DANGEROUS START! " + "; ".join(predicted_issues)
        elif movement_quality == 'fair':
            safety_advice = "CAUTION: " + "; ".join(predicted_issues)
        else:
            safety_advice = "Good starting technique. Maintain steady control."

        return {
            'is_attempting_movement': True,
            'movement_quality': movement_quality,
            'safety_advice': safety_advice,
            'predicted_issues': predicted_issues,
            'should_warn': movement_quality in ['poor', 'fair']
        }

    def _get_safety_status(self, risk_score):
        """
        Get safety status category based on risk score.

        Args:
            risk_score (float): Risk score (0-100)

        Returns:
            str: Safety status
        """
        if risk_score < 20:
            return 'SAFE'
        elif risk_score < 40:
            return 'CAUTION'
        elif risk_score < 70:
            return 'WARNING'
        else:
            return 'DANGER'

    def _default_safety_report(self):
        """Return default safety report when insufficient data."""
        return {
            'track_id': -1,
            'trajectory_analysis': {},
            'erratic_movement': {'is_erratic': False, 'erratic_score': 0.0, 'reason': 'insufficient data'},
            'lane_analysis': {},
            'danger_patterns': {},
            'risk_score': 0.0,
            'warnings': [],
            'accident_prediction': {
                'likelihood_score': 0,
                'risk_level': 'UNKNOWN',
                'contributing_factors': [],
                'recommendation': 'Gathering data...',
                'is_high_risk': False
            },
            'safety_status': 'UNKNOWN'
        }
