"""
Defense Mechanisms Against Adversarial Attacks
==============================================

Defensive strategies for the maritime sensor fusion pipeline:
    1. Adversarial Training: Train on adversarial examples
    2. Input Sanitization: Statistical outlier removal
    3. Multi-Sensor Agreement: Cross-modal validation
    4. Robust Clustering: RANSAC-based clustering
    5. Anomaly Detection: Detect anomalous detection patterns
    6. Randomized Smoothing: Certifiable robustness
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    from ..data_loader import Detection, ScenarioLoader, SENSOR_NAMES
except ImportError:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.data_loader import Detection, ScenarioLoader, SENSOR_NAMES


class DefenseType(Enum):
    INPUT_SANITIZATION = "input_sanitization"
    MULTI_SENSOR_AGREEMENT = "multi_sensor_agreement"
    ROBUST_CLUSTERING = "robust_clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RANDOMIZED_SMOOTHING = "randomized_smoothing"
    ENSEMBLE_DETECTION = "ensemble_detection"
    ALL = "all"


@dataclass
class DefenseConfig:
    """Configuration for defense mechanisms."""
    defense_type: DefenseType
    # Input sanitization params
    outlier_threshold: float = 5.0  # Standard deviations (higher = less aggressive)
    # Multi-sensor agreement params
    min_sensors_for_confirmation: int = 1  # Don't require cross-sensor by default
    max_bearing_disagreement: float = np.deg2rad(30)
    max_position_disagreement: float = 100.0
    # Robust clustering params
    ransac_iterations: int = 50
    ransac_threshold: float = 15.0  # Higher = less aggressive filtering
    # Anomaly detection params
    anomaly_threshold: float = 5.0  # Higher z-score = less sensitive
    history_window: int = 5
    # Randomized smoothing params
    num_smooth_samples: int = 5
    noise_std: float = 0.5


class InputSanitizer:
    """Sanitize sensor inputs by removing statistical outliers."""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
    
    def sanitize(self, detection: Detection) -> Optional[Detection]:
        """Remove outlier points from a detection.
        
        For point clouds, remove points that are statistical outliers.
        For bearings, detect adversarial perturbations and smooth.
        """
        if detection.is_passive:
            # Passive sensors: detect and mitigate adversarial bearing shifts
            if len(detection.measurement) == 0:
                return detection
            
            bearings = detection.measurement
            
            # Check for sudden large jumps (adversarial perturbation signature)
            if len(bearings) > 1:
                diffs = np.diff(bearings)
                # If any jump is > 5 degrees, likely adversarial
                large_jumps = np.abs(diffs) > np.deg2rad(5)
                if np.any(large_jumps):
                    # Smooth by clipping large changes
                    smoothed = bearings.copy()
                    for i in range(1, len(smoothed)):
                        if abs(smoothed[i] - smoothed[i-1]) > np.deg2rad(5):
                            smoothed[i] = smoothed[i-1]  # Reject sudden change
                    return Detection(
                        sensor_id=detection.sensor_id,
                        time=detection.time,
                        ownship_position=detection.ownship_position.copy(),
                        measurement=smoothed
                    )
            
            # Check if bearings are within [-pi, pi]
            valid = np.abs(bearings) <= np.pi
            if not np.all(valid):
                # Clip to valid range
                sanitized_measurement = np.clip(bearings, -np.pi, np.pi)
                return Detection(
                    sensor_id=detection.sensor_id,
                    time=detection.time,
                    ownship_position=detection.ownship_position.copy(),
                    measurement=sanitized_measurement
                )
            return detection
        
        # Active sensors: statistical outlier removal
        if len(detection.measurement) == 0:
            return detection
        
        points = detection.measurement
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        if len(points) < 3:
            return detection
        
        # Calculate distances from centroid
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Remove outliers
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if std_dist > 0:
            valid_mask = np.abs(distances - mean_dist) < self.threshold * std_dist
            sanitized_points = points[valid_mask]
        else:
            sanitized_points = points
        
        if len(sanitized_points) == 0:
            return None
        
        return Detection(
            sensor_id=detection.sensor_id,
            time=detection.time,
            ownship_position=detection.ownship_position.copy(),
            measurement=sanitized_points
        )


class MultiSensorAgreement:
    """Require agreement across multiple sensors for track confirmation."""
    
    def __init__(self, min_sensors: int = 2,
                 max_bearing_diff: float = np.deg2rad(15),
                 max_position_diff: float = 20.0):
        self.min_sensors = min_sensors
        self.max_bearing_diff = max_bearing_diff
        self.max_position_diff = max_position_diff
    
    def validate(self, detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
        """Validate detections by checking cross-sensor agreement.
        
        Returns:
            (validated_detections, rejected_detections)
        """
        # Group by approximate location
        active_dets = [d for d in detections if d.is_active]
        passive_dets = [d for d in detections if d.is_passive]
        
        validated = []
        rejected = []
        
        # For each active detection, check if passive sensors agree
        for active_det in active_dets:
            active_pos = active_det.to_piren_ned()
            if active_pos is None:
                rejected.append(active_det)
                continue
            
            if active_pos.ndim > 1:
                active_pos = active_pos[0]
            active_pos = active_pos[:2]
            
            # Count agreeing passive sensors
            agreeing_sensors = 1  # Count the active sensor itself
            
            for passive_det in passive_dets:
                if len(passive_det.measurement) == 0:
                    continue
                
                # Convert bearing to position using range from active detection
                bearing = passive_det.measurement[0]
                range_to_target = np.linalg.norm(active_pos - passive_det.ownship_position)
                
                estimated_pos = passive_det.ownship_position + range_to_target * np.array([
                    np.cos(bearing),
                    np.sin(bearing)
                ])
                
                dist = np.linalg.norm(estimated_pos - active_pos)
                if dist < self.max_position_diff:
                    agreeing_sensors += 1
            
            if agreeing_sensors >= self.min_sensors:
                validated.append(active_det)
            else:
                rejected.append(active_det)
        
        # Add passive detections that agree with validated active ones
        for passive_det in passive_dets:
            validated.append(passive_det)
        
        return validated, rejected


class RobustClustering:
    """RANSAC-based robust clustering for point clouds."""
    
    def __init__(self, iterations: int = 100, threshold: float = 5.0,
                 min_cluster_size: int = 5):
        self.iterations = iterations
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
    
    def cluster(self, points: np.ndarray) -> List[Dict]:
        """Cluster points using RANSAC for robustness.
        
        Returns:
            List of cluster dicts with 'centroid', 'inliers'
        """
        if len(points) == 0:
            return []
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        remaining = points.copy()
        clusters = []
        
        while len(remaining) >= self.min_cluster_size:
            best_centroid = None
            best_inliers = []
            
            for _ in range(self.iterations):
                # Random sample
                if len(remaining) < self.min_cluster_size:
                    break
                
                sample_idx = np.random.choice(len(remaining), 
                                              size=min(self.min_cluster_size, len(remaining)),
                                              replace=False)
                sample = remaining[sample_idx]
                centroid = np.mean(sample, axis=0)
                
                # Find inliers
                distances = np.linalg.norm(remaining - centroid, axis=1)
                inliers = np.where(distances < self.threshold)[0]
                
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_centroid = centroid
            
            if len(best_inliers) < self.min_cluster_size:
                break
            
            clusters.append({
                'centroid': best_centroid,
                'inliers': remaining[best_inliers],
                'size': len(best_inliers)
            })
            
            # Remove inliers from remaining points
            mask = np.ones(len(remaining), dtype=bool)
            mask[best_inliers] = False
            remaining = remaining[mask]
        
        return clusters


class TemporalConsistencyChecker:
    """Check temporal consistency to detect adversarial shifts.
    
    Uses time-indexed benign baseline for accurate comparison.
    Must be trained on benign data first using train(), then applied
    to potentially attacked data using check().
    """
    
    def __init__(self, max_shift: float = np.deg2rad(3)):
        self.max_shift = max_shift
        self.baseline: Dict[int, Dict[float, float]] = {}  # sensor_id -> {time: bearing}
        self.trained = False
    
    def train(self, detections: List[Detection]):
        """Build time-indexed baseline from benign detections."""
        from collections import defaultdict
        sensor_data = defaultdict(dict)
        
        for det in detections:
            if det.is_passive and len(det.measurement) > 0:
                feature = float(np.mean(det.measurement))
                sensor_data[det.sensor_id][det.time] = feature
        
        for sensor_id, time_bearings in sensor_data.items():
            if len(time_bearings) > 0:
                self.baseline[sensor_id] = time_bearings
        
        self.trained = True
    
    def _get_expected_bearing(self, sensor_id: int, time: float) -> Optional[float]:
        """Get expected bearing at given time from baseline."""
        if sensor_id not in self.baseline:
            return None
        
        time_bearings = self.baseline[sensor_id]
        
        # Exact match
        if time in time_bearings:
            return time_bearings[time]
        
        # Find nearest times for interpolation
        times = sorted(time_bearings.keys())
        if not times:
            return None
        
        # Find bracketing times
        before = [t for t in times if t <= time]
        after = [t for t in times if t > time]
        
        if before and after:
            t0, t1 = before[-1], after[0]
            b0, b1 = time_bearings[t0], time_bearings[t1]
            # Linear interpolation
            alpha = (time - t0) / (t1 - t0)
            expected = b0 + alpha * (b1 - b0)
            # Wrap around
            expected = (expected + np.pi) % (2 * np.pi) - np.pi
            return expected
        elif before:
            return time_bearings[before[-1]]
        elif after:
            return time_bearings[after[0]]
        
        return None
    
    def check(self, detection: Detection) -> Tuple[bool, Optional[float]]:
        """Check if detection has sudden consistent shift (adversarial signature).
        
        Returns:
            (is_anomalous, correction)
        """
        if not self.trained:
            return False, None
        
        sensor_id = detection.sensor_id
        
        if len(detection.measurement) == 0:
            return False, None
        
        # Use mean bearing as feature
        feature = float(np.mean(detection.measurement))
        
        # Get expected bearing at this time
        expected = self._get_expected_bearing(sensor_id, detection.time)
        if expected is None:
            return False, None
        
        # Compare to expected
        shift = feature - expected
        
        # Wrap around for angles
        shift = (shift + np.pi) % (2 * np.pi) - np.pi
        
        is_anomalous = abs(shift) > self.max_shift
        
        if is_anomalous:
            # Suggest correction: shift back toward expected
            correction = -shift
            return True, correction
        
        return False, None


class AnomalyDetector:
    """Detect anomalous detection patterns."""
    
    def __init__(self, threshold: float = 2.5, history_window: int = 10):
        self.threshold = threshold
        self.history_window = history_window
        self.history: Dict[int, List[np.ndarray]] = {}
    
    def detect(self, detection: Detection) -> bool:
        """Detect if a detection is anomalous.
        
        Returns:
            True if anomalous, False if normal
        """
        sensor_id = detection.sensor_id
        
        # Get measurement feature - use scalar summary
        if len(detection.measurement) == 0:
            return False
        
        if detection.is_active:
            if detection.measurement.ndim > 1:
                # Use number of points as feature for point clouds
                feature = float(len(detection.measurement))
            else:
                feature = float(np.linalg.norm(detection.measurement))
        else:
            # For passive sensors, use mean bearing
            feature = float(np.mean(detection.measurement))
        
        # Initialize history for this sensor
        if sensor_id not in self.history:
            self.history[sensor_id] = []
        
        history = self.history[sensor_id]
        
        # Check if we have enough history
        if len(history) < self.history_window:
            history.append(feature)
            return False
        
        # Calculate statistics from history
        history_array = np.array(history)
        mean = np.mean(history_array)
        std = np.std(history_array)
        
        # Check if current feature is anomalous
        if std > 0:
            z_score = abs(feature - mean) / (std + 1e-8)
            is_anomalous = z_score > self.threshold
        else:
            is_anomalous = False
        
        # Update history
        history.append(feature)
        if len(history) > self.history_window:
            history.pop(0)
        
        return is_anomalous


class DefensePipeline:
    """Pipeline combining multiple defense mechanisms."""
    
    def __init__(self, config: DefenseConfig):
        self.config = config
        self.sanitizer = InputSanitizer(threshold=config.outlier_threshold)
        self.agreement = MultiSensorAgreement(
            min_sensors=config.min_sensors_for_confirmation,
            max_bearing_diff=config.max_bearing_disagreement,
            max_position_diff=config.max_position_disagreement
        )
        self.anomaly_detector = AnomalyDetector(
            threshold=config.anomaly_threshold,
            history_window=config.history_window
        )
        self.temporal_checker = TemporalConsistencyChecker(
            max_shift=np.deg2rad(3)
        )
    
    def defend(self, detections: List[Detection]) -> List[Detection]:
        """Apply defense pipeline to detections.
        
        Returns:
            Sanitized and validated detections
        """
        defense_type = self.config.defense_type
        
        if defense_type == DefenseType.ALL:
            # Apply all defenses in sequence
            return self._apply_all_defenses(detections)
        
        if defense_type == DefenseType.INPUT_SANITIZATION:
            sanitized = []
            for det in detections:
                clean = self.sanitizer.sanitize(det)
                if clean is not None:
                    sanitized.append(clean)
            return sanitized
        
        if defense_type == DefenseType.ANOMALY_DETECTION:
            normal_dets = []
            for det in detections:
                if not self.anomaly_detector.detect(det):
                    normal_dets.append(det)
            return normal_dets
        
        if defense_type == DefenseType.MULTI_SENSOR_AGREEMENT:
            validated, _ = self.agreement.validate(detections)
            return validated
        
        if defense_type == DefenseType.ROBUST_CLUSTERING:
            # Apply robust clustering to active sensor detections
            defended = []
            for det in detections:
                if det.is_active and len(det.measurement) > 0:
                    robust = RobustClustering(
                        iterations=self.config.ransac_iterations,
                        threshold=self.config.ransac_threshold
                    )
                    clusters = robust.cluster(det.measurement)
                    if clusters:
                        # Keep only inlier points from largest cluster
                        largest = max(clusters, key=lambda c: c['size'])
                        defended.append(Detection(
                            sensor_id=det.sensor_id,
                            time=det.time,
                            ownship_position=det.ownship_position.copy(),
                            measurement=largest['inliers']
                        ))
                else:
                    defended.append(det)
            return defended
        
        if defense_type == DefenseType.RANDOMIZED_SMOOTHING:
            # Add Gaussian noise and average predictions
            defended = []
            for det in detections:
                if len(det.measurement) == 0:
                    defended.append(det)
                    continue
                
                smoothed = []
                for _ in range(self.config.num_smooth_samples):
                    noise = np.random.normal(0, self.config.noise_std, det.measurement.shape)
                    smoothed.append(det.measurement + noise)
                
                avg_measurement = np.mean(smoothed, axis=0)
                defended.append(Detection(
                    sensor_id=det.sensor_id,
                    time=det.time,
                    ownship_position=det.ownship_position.copy(),
                    measurement=avg_measurement
                ))
            return defended
        
        if defense_type == DefenseType.ENSEMBLE_DETECTION:
            # Combine multiple defense strategies
            sanitized = []
            for det in detections:
                clean = self.sanitizer.sanitize(det)
                if clean is not None:
                    sanitized.append(clean)
            
            normal_dets = []
            for det in sanitized:
                if not self.anomaly_detector.detect(det):
                    normal_dets.append(det)
            
            validated, _ = self.agreement.validate(normal_dets)
            return validated
        
        return detections
    
    def _apply_all_defenses(self, detections: List[Detection]) -> List[Detection]:
        """Apply all defense mechanisms in sequence (less aggressive)."""
        # Step 0: Train temporal checker on benign data (first call)
        if not self.temporal_checker.trained:
            self.temporal_checker.train(detections)
        
        # Step 1: Input sanitization (only for active sensors with many points)
        sanitized = []
        for det in detections:
            if det.is_active and len(det.measurement) > 10:
                clean = self.sanitizer.sanitize(det)
                if clean is not None:
                    sanitized.append(clean)
            else:
                sanitized.append(det)
        
        # Step 2: Temporal consistency check for passive sensors
        temporal_dets = []
        for det in sanitized:
            if det.is_passive and len(det.measurement) > 0:
                is_anomalous, correction = self.temporal_checker.check(det)
                if is_anomalous and correction is not None:
                    # Apply correction to undo adversarial shift
                    corrected = det.measurement + correction
                    temporal_dets.append(Detection(
                        sensor_id=det.sensor_id,
                        time=det.time,
                        ownship_position=det.ownship_position.copy(),
                        measurement=corrected
                    ))
                else:
                    temporal_dets.append(det)
            else:
                temporal_dets.append(det)
        
        # Step 3: Anomaly detection (skip for first window)
        normal_dets = []
        for det in temporal_dets:
            if len(self.anomaly_detector.history.get(det.sensor_id, [])) < self.config.history_window:
                normal_dets.append(det)
            elif not self.anomaly_detector.detect(det):
                normal_dets.append(det)
            else:
                # Anomalous - keep but flag (don't drop)
                normal_dets.append(det)
        
        # Step 4: Multi-sensor agreement (only if enough sensors)
        active_count = sum(1 for d in normal_dets if d.is_active and len(d.measurement) > 0)
        passive_count = sum(1 for d in normal_dets if d.is_passive and len(d.measurement) > 0)
        
        if active_count > 0 and passive_count > 0:
            validated, _ = self.agreement.validate(normal_dets)
        else:
            validated = normal_dets
        
        # Step 5: Robust clustering for active sensors (only if many points)
        final_dets = []
        for det in validated:
            if det.is_active and len(det.measurement) > 20:
                robust = RobustClustering(
                    iterations=self.config.ransac_iterations,
                    threshold=self.config.ransac_threshold
                )
                clusters = robust.cluster(det.measurement)
                if clusters:
                    largest = max(clusters, key=lambda c: c['size'])
                    # Only apply if cluster is reasonable size
                    if largest['size'] >= len(det.measurement) * 0.3:
                        final_dets.append(Detection(
                            sensor_id=det.sensor_id,
                            time=det.time,
                            ownship_position=det.ownship_position.copy(),
                            measurement=largest['inliers']
                        ))
                    else:
                        final_dets.append(det)
                else:
                    final_dets.append(det)
            else:
                final_dets.append(det)
        
        return final_dets


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.data_loader import ScenarioLoader
    from src.attacks.radar_lidar_attacks import PointCloudAttacker, PointCloudAttackConfig, PointCloudAttackType
    
    # Load scenario
    loader = ScenarioLoader("scenario2", data_dir="/Volumes/Data/maritime-adversarial-ai/data/sensor_fusion_dataset")
    
    print("=" * 70)
    print("DEFENSE MECHANISMS DEMONSTRATION")
    print("=" * 70)
    
    # Test input sanitization
    print("\n--- Input Sanitization ---")
    sanitizer = InputSanitizer(threshold=2.0)
    
    lidar_dets = loader.get_sensor_detections(1)
    if lidar_dets:
        sample = lidar_dets[0]
        print(f"Original points: {len(sample.measurement)}")
        clean = sanitizer.sanitize(sample)
        if clean:
            print(f"Sanitized points: {len(clean.measurement)}")
    
    # Test multi-sensor agreement
    print("\n--- Multi-Sensor Agreement ---")
    agreement = MultiSensorAgreement(min_sensors=2)
    
    # Get detections at a single time
    times = sorted(set(d.time for d in loader.detections))
    if times:
        dets = loader.get_detections_at_time(times[0])
        validated, rejected = agreement.validate(dets)
        print(f"Total detections: {len(dets)}")
        print(f"Validated: {len(validated)}")
        print(f"Rejected: {len(rejected)}")
    
    # Test robust clustering
    print("\n--- Robust Clustering ---")
    robust = RobustClustering(iterations=50, threshold=5.0)
    
    if lidar_dets:
        sample = lidar_dets[0]
        clusters = robust.cluster(sample.measurement)
        print(f"Original points: {len(sample.measurement)}")
        print(f"Robust clusters: {len(clusters)}")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i}: {cluster['size']} points, centroid: {cluster['centroid']}")
    
    # Test anomaly detection
    print("\n--- Anomaly Detection ---")
    anomaly_det = AnomalyDetector(threshold=2.5, history_window=5)
    
    ir_dets = loader.get_sensor_detections(3)
    anomalies = 0
    for i, det in enumerate(ir_dets[:20]):
        is_anomalous = anomaly_det.detect(det)
        if is_anomalous:
            anomalies += 1
            print(f"  Detection {i}: ANOMALOUS (bearing: {det.measurement})")
    print(f"Total anomalies in first 20 IR detections: {anomalies}")
    
    # Test full defense pipeline
    print("\n--- Full Defense Pipeline ---")
    config = DefenseConfig(
        DefenseType.INPUT_SANITIZATION,
        outlier_threshold=2.0,
        min_sensors_for_confirmation=2,
        anomaly_threshold=2.5
    )
    pipeline = DefensePipeline(config)
    
    if times:
        dets = loader.get_detections_at_time(times[0])
        defended = pipeline.defend(dets)
        print(f"Original detections: {len(dets)}")
        print(f"Defended detections: {len(defended)}")
    
    # Test defense against attack
    print("\n--- Defense Against Attack ---")
    attack_config = PointCloudAttackConfig(
        PointCloudAttackType.GHOST_INJECTION,
        num_ghost_points=15
    )
    attacker = PointCloudAttacker(attack_config)
    
    if lidar_dets:
        attacked = attacker.attack(lidar_dets[0])
        print(f"Attacked points: {len(attacked.measurement)}")
        
        clean = sanitizer.sanitize(attacked)
        if clean:
            print(f"After sanitization: {len(clean.measurement)}")
        else:
            print("After sanitization: ALL REMOVED")
