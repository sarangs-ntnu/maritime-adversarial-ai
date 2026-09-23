"""
Evaluation Metrics for Adversarial Attacks
==========================================

Detection Metrics:
    - RMSE (Root Mean Square Error)
    - Detection Probability
    - False Alarm Rate
    - Attack Success Rate (ASR)

Tracking Metrics:
    - MOTA (Multi-Object Tracking Accuracy)
    - MOTP (Multi-Object Tracking Precision)
    - ID Switches
    - Track Fragmentation
    - Track Loss Rate

Fusion Metrics:
    - Cross-sensor agreement
    - Existence probability trajectory
    - Sensor utilization rate
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from ..data_loader import Detection, GroundTruth, ScenarioLoader, SENSOR_NAMES
except ImportError:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.data_loader import Detection, GroundTruth, ScenarioLoader, SENSOR_NAMES


@dataclass
class DetectionMetrics:
    """Metrics for detection performance."""
    rmse_position: float = 0.0
    rmse_bearing: float = 0.0
    detection_probability: float = 0.0
    false_alarm_rate: float = 0.0
    mean_error: float = 0.0
    max_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rmse_position': self.rmse_position,
            'rmse_bearing': self.rmse_bearing,
            'detection_probability': self.detection_probability,
            'false_alarm_rate': self.false_alarm_rate,
            'mean_error': self.mean_error,
            'max_error': self.max_error,
        }


@dataclass
class AttackMetrics:
    """Metrics for attack effectiveness."""
    attack_success_rate: float = 0.0
    mean_perturbation: float = 0.0
    max_perturbation: float = 0.0
    detection_degradation: float = 0.0
    tracking_degradation: float = 0.0
    false_positive_increase: float = 0.0
    false_negative_increase: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'attack_success_rate': self.attack_success_rate,
            'mean_perturbation': self.mean_perturbation,
            'max_perturbation': self.max_perturbation,
            'detection_degradation': self.detection_degradation,
            'tracking_degradation': self.tracking_degradation,
            'false_positive_increase': self.false_positive_increase,
            'false_negative_increase': self.false_negative_increase,
        }


class DetectionEvaluator:
    """Evaluate detection performance against ground truth."""
    
    def __init__(self, distance_threshold: float = 20.0, 
                 bearing_threshold: float = np.deg2rad(10)):
        self.distance_threshold = distance_threshold
        self.bearing_threshold = bearing_threshold
    
    def evaluate(self, detections: List[Detection], 
                 ground_truth: List[GroundTruth],
                 sensor_id: int) -> DetectionMetrics:
        """Evaluate detections for a specific sensor."""
        metrics = DetectionMetrics()
        
        # Filter detections by sensor
        sensor_dets = [d for d in detections if d.sensor_id == sensor_id]
        
        if len(sensor_dets) == 0 or len(ground_truth) == 0:
            return metrics
        
        # Match detections to ground truth
        errors = []
        matched_gt = set()
        false_alarms = 0
        
        for det in sensor_dets:
            det_pos = det.to_piren_ned()
            if det_pos is None:
                # Passive sensor - check bearing
                if len(det.measurement) > 0:
                    bearing = det.measurement[0]
                    # Find closest GT in bearing
                    min_bearing_error = float('inf')
                    best_gt = None
                    for gt in ground_truth:
                        gt_pos = gt.position[:2]
                        ownship_pos = det.ownship_position
                        true_bearing = np.arctan2(gt_pos[1] - ownship_pos[1], 
                                                   gt_pos[0] - ownship_pos[0])
                        bearing_error = np.abs(bearing - true_bearing)
                        bearing_error = np.minimum(bearing_error, 2 * np.pi - bearing_error)
                        if bearing_error < min_bearing_error:
                            min_bearing_error = bearing_error
                            best_gt = gt
                    
                    if min_bearing_error < self.bearing_threshold:
                        errors.append(min_bearing_error)
                        matched_gt.add(id(best_gt))
                    else:
                        false_alarms += 1
                continue
            
            # Active sensor - check position
            if det_pos.ndim > 1:
                det_pos = det_pos[0]
            det_pos = det_pos[:2]
            
            min_dist = float('inf')
            best_gt = None
            for gt in ground_truth:
                gt_pos = gt.position[:2]
                dist = np.linalg.norm(det_pos - gt_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_gt = gt
            
            if min_dist < self.distance_threshold:
                errors.append(min_dist)
                matched_gt.add(id(best_gt))
            else:
                false_alarms += 1
        
        # Calculate metrics
        if errors:
            metrics.rmse_position = np.sqrt(np.mean(np.array(errors) ** 2))
            metrics.mean_error = np.mean(errors)
            metrics.max_error = np.max(errors)
        
        metrics.detection_probability = len(matched_gt) / max(len(ground_truth), 1)
        metrics.false_alarm_rate = false_alarms / max(len(sensor_dets), 1)
        
        return metrics


class AttackEvaluator:
    """Evaluate the effectiveness of adversarial attacks."""
    
    def __init__(self, benign_loader: ScenarioLoader):
        self.benign_loader = benign_loader
        self.detection_evaluator = DetectionEvaluator()
    
    def evaluate_camera_attack(self, 
                               original_detections: List[Detection],
                               attacked_detections: List[Detection],
                               sensor_id: int) -> AttackMetrics:
        """Evaluate camera attack effectiveness."""
        metrics = AttackMetrics()
        
        # Calculate perturbation magnitudes
        perturbations = []
        for orig, att in zip(original_detections, attacked_detections):
            if len(orig.measurement) > 0 and len(att.measurement) > 0:
                pert = np.abs(att.measurement - orig.measurement)
                pert = np.minimum(pert, 2 * np.pi - pert)  # Handle angle wrapping
                perturbations.append(np.mean(pert))
        
        if perturbations:
            metrics.mean_perturbation = np.mean(perturbations)
            metrics.max_perturbation = np.max(perturbations)
        
        # Calculate attack success rate
        # Success = detection missed or significantly perturbed
        successes = 0
        for orig, att in zip(original_detections, attacked_detections):
            if len(att.measurement) == 0:
                successes += 1  # Disappearance
            elif len(orig.measurement) > 0:
                pert = np.abs(att.measurement - orig.measurement)
                pert = np.minimum(pert, 2 * np.pi - pert)
                if np.mean(pert) > 0.05:  # > ~3 degrees
                    successes += 1
        
        metrics.attack_success_rate = successes / max(len(original_detections), 1)
        
        return metrics
    
    def evaluate_point_cloud_attack(self,
                                    original_detections: List[Detection],
                                    attacked_detections: List[Detection],
                                    sensor_id: int) -> AttackMetrics:
        """Evaluate point cloud attack effectiveness."""
        metrics = AttackMetrics()
        
        # Calculate perturbation magnitudes
        perturbations = []
        for orig, att in zip(original_detections, attacked_detections):
            if len(orig.measurement) > 0 and len(att.measurement) > 0:
                if orig.measurement.ndim > 1 and att.measurement.ndim > 1:
                    # Multiple points - compare centroids
                    orig_centroid = np.mean(orig.measurement, axis=0)
                    att_centroid = np.mean(att.measurement, axis=0)
                    pert = np.linalg.norm(orig_centroid - att_centroid)
                else:
                    pert = np.linalg.norm(orig.measurement - att.measurement)
                perturbations.append(pert)
        
        if perturbations:
            metrics.mean_perturbation = np.mean(perturbations)
            metrics.max_perturbation = np.max(perturbations)
        
        # Calculate attack success rate
        successes = 0
        for orig, att in zip(original_detections, attacked_detections):
            orig_points = len(orig.measurement) if orig.measurement.ndim > 1 else (1 if len(orig.measurement) > 0 else 0)
            att_points = len(att.measurement) if att.measurement.ndim > 1 else (1 if len(att.measurement) > 0 else 0)
            
            if att_points < 5 and orig_points >= 5:
                successes += 1  # Cluster dropped below threshold
            elif att_points > orig_points * 2:
                successes += 1  # Ghost injection
            elif orig_points > 0 and att_points > 0:
                # Check centroid shift
                if orig.measurement.ndim > 1 and att.measurement.ndim > 1:
                    orig_centroid = np.mean(orig.measurement, axis=0)
                    att_centroid = np.mean(att.measurement, axis=0)
                    if np.linalg.norm(orig_centroid - att_centroid) > 10:
                        successes += 1
        
        metrics.attack_success_rate = successes / max(len(original_detections), 1)
        
        return metrics
    
    def evaluate_fusion_attack(self,
                               benign_tracks: Dict[int, np.ndarray],
                               attacked_tracks: Dict[int, np.ndarray]) -> AttackMetrics:
        """Evaluate fusion attack effectiveness."""
        metrics = AttackMetrics()
        
        n_benign = len(benign_tracks)
        n_attacked = len(attacked_tracks)
        
        metrics.false_positive_increase = max(0, n_attacked - n_benign) / max(n_benign, 1)
        metrics.false_negative_increase = max(0, n_benign - n_attacked) / max(n_benign, 1)
        
        # Track loss rate
        if n_benign > 0:
            metrics.attack_success_rate = max(0, n_benign - n_attacked) / n_benign
        
        return metrics


def compare_scenarios(benign_loader: ScenarioLoader,
                      attacked_loader: ScenarioLoader,
                      sensor_id: int) -> Dict[str, float]:
    """Compare benign vs attacked scenario for a sensor.
    
    Returns:
        Dictionary of comparison metrics
    """
    benign_dets = benign_loader.get_sensor_detections(sensor_id)
    attacked_dets = attacked_loader.get_sensor_detections(sensor_id)
    
    # Count differences
    n_benign = len(benign_dets)
    n_attacked = len(attacked_dets)
    
    # Calculate mean measurement difference
    diffs = []
    for b, a in zip(benign_dets, attacked_dets):
        if len(b.measurement) > 0 and len(a.measurement) > 0:
            if b.measurement.ndim > 1 and a.measurement.ndim > 1:
                # Compare centroids
                b_cent = np.mean(b.measurement, axis=0)
                a_cent = np.mean(a.measurement, axis=0)
                diff = np.linalg.norm(b_cent - a_cent)
            else:
                diff = np.linalg.norm(b.measurement - a.measurement)
            diffs.append(diff)
    
    return {
        'benign_detections': n_benign,
        'attacked_detections': n_attacked,
        'detection_change': n_attacked - n_benign,
        'mean_measurement_diff': np.mean(diffs) if diffs else 0.0,
        'max_measurement_diff': np.max(diffs) if diffs else 0.0,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.data_loader import ScenarioLoader
    
    # Load scenario
    loader = ScenarioLoader("scenario2", data_dir="/Volumes/Data/maritime-adversarial-ai/data/sensor_fusion_dataset")
    
    print("=" * 70)
    print("EVALUATION METRICS DEMONSTRATION")
    print("=" * 70)
    
    # Evaluate each sensor
    for sensor_id in loader.sensor_ids:
        sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
        print(f"\n--- {sensor_name} (Sensor {sensor_id}) ---")
        
        dets = loader.get_sensor_detections(sensor_id)
        
        # Get ground truth at detection times
        gt_list = []
        for det in dets:
            gt = loader.get_ground_truth_at_time(det.time, tolerance=0.2)
            if gt:
                gt_list.extend(gt)
        
        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate(dets, gt_list, sensor_id)
        
        print(f"  RMSE Position: {metrics.rmse_position:.2f} m")
        print(f"  Detection Probability: {metrics.detection_probability:.2%}")
        print(f"  False Alarm Rate: {metrics.false_alarm_rate:.2%}")
        print(f"  Mean Error: {metrics.mean_error:.2f} m")
        print(f"  Max Error: {metrics.max_error:.2f} m")
