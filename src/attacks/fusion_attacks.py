"""
Adversarial Attacks on Fusion Layer (JIPDA Tracker)
===================================================

Attacks targeting the multi-sensor JIPDA tracker itself.
Exploits the sensor fusion logic, data association, and track management.

Key vulnerabilities:
    - JIPDA assumes independent sensor errors
    - Existence probability drops when no measurements associated
    - Track initialization requires active sensors
    - Validation gates can be exploited

Attack Types:
    - Existence Suppression: Prevent all sensors from detecting target
    - Association Confusion: Push measurements outside validation gates
    - Cross-Sensor Consistency: Coordinate correlated failures
    - False Track Injection: Create confirmed tracks from ghost measurements
    - Track Merge Manipulation: Force incorrect track merging
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
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


class FusionAttackType(Enum):
    EXISTENCE_SUPPRESSION = "existence_suppression"
    ASSOCIATION_CONFUSION = "association_confusion"
    CROSS_SENSOR_CONSISTENCY = "cross_sensor_consistency"
    FALSE_TRACK_INJECTION = "false_track_injection"
    TRACK_MERGE_MANIPULATION = "track_merge_manipulation"
    SENSOR_DOS = "sensor_dos"


@dataclass
class FusionAttackConfig:
    """Configuration for fusion-layer adversarial attacks."""
    attack_type: FusionAttackType
    # Existence suppression params
    suppression_duration: float = 5.0  # Seconds to suppress detections
    # Association confusion params
    gate_inflation: float = 2.0  # Factor to inflate measurement covariance
    # Cross-sensor params
    sensor_subset: Optional[List[int]] = None  # Which sensors to attack
    # False track params
    false_track_position: Optional[np.ndarray] = None
    false_track_duration: float = 3.0
    # Track merge params
    merge_target_ids: Optional[List[int]] = None
    # Sensor DoS params
    dos_sensors: Optional[List[int]] = None
    dos_probability: float = 0.5


class JIPDASimulator:
    """Simplified JIPDA tracker for attack simulation.
    
    Models key JIPDA components:
        - State prediction (constant velocity model)
        - Validation gating
        - Data association (probabilistic)
        - Existence probability update
        - Track management
    """
    
    def __init__(self, 
                 process_noise: float = 0.5,
                 measurement_noise_active: float = 8.0,
                 measurement_noise_passive: float = 0.1,
                 detection_prob: float = 0.9,
                 false_alarm_rate: float = 0.1,
                 gate_threshold: float = 25.0,
                 existence_confirm: float = 0.35,
                 existence_terminate: float = 0.03):
        self.process_noise = process_noise
        self.measurement_noise_active = measurement_noise_active
        self.measurement_noise_passive = measurement_noise_passive
        self.detection_prob = detection_prob
        self.false_alarm_rate = false_alarm_rate
        self.gate_threshold = gate_threshold
        self.existence_confirm = existence_confirm
        self.existence_terminate = existence_terminate
        
        self.tracks: Dict[int, Dict] = {}  # track_id -> track state
        self.next_track_id = 1
    
    def predict(self, dt: float):
        """Predict track states forward by dt seconds."""
        for track_id, track in self.tracks.items():
            # Constant velocity prediction
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            track['state'] = F @ track['state']
            track['covariance'] = F @ track['covariance'] @ F.T + self.process_noise * np.eye(4)
            
            # Predicted existence probability decays slightly
            track['existence'] *= 0.99
    
    def update(self, detections: List[Detection], dt: float):
        """Update tracks with new detections."""
        self.predict(dt)
        
        # Group detections by sensor
        detections_by_sensor: Dict[int, List[Detection]] = {}
        for det in detections:
            detections_by_sensor.setdefault(det.sensor_id, []).append(det)
        
        # Track which detections have been associated
        associated_dets = set()
        
        # For each track, try to associate with best detection
        for track_id, track in list(self.tracks.items()):
            best_det = None
            best_distance = float('inf')
            
            for sensor_id, sensor_dets in detections_by_sensor.items():
                for i, det in enumerate(sensor_dets):
                    det_key = (det.time, det.sensor_id, i)
                    if det_key in associated_dets:
                        continue
                    
                    # Convert detection to measurement in state space
                    if det.is_active:
                        z = det.to_piren_ned()
                        if z is None:
                            continue
                        if z.ndim > 1:
                            z = z[0]
                        z = z[:2]
                        R = self.measurement_noise_active * np.eye(2)
                        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                    else:
                        if len(det.measurement) == 0:
                            continue
                        bearing = det.measurement[0]
                        range_to_target = np.linalg.norm(track['state'][:2])
                        z = np.array([
                            track['state'][0] + range_to_target * np.cos(bearing),
                            track['state'][1] + range_to_target * np.sin(bearing)
                        ])
                        R = self.measurement_noise_passive * np.eye(2)
                        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                    
                    # Innovation
                    y = z - H @ track['state']
                    S = H @ track['covariance'] @ H.T + R
                    
                    # Mahalanobis distance
                    try:
                        d = y.T @ np.linalg.inv(S) @ y
                    except np.linalg.LinAlgError:
                        continue
                    
                    # Gate test
                    if d < self.gate_threshold and d < best_distance:
                        best_distance = d
                        best_det = (det, z, R, H, det_key)
            
            if best_det is not None:
                det, z, R, H, det_key = best_det
                # Association successful
                y = z - H @ track['state']
                S = H @ track['covariance'] @ H.T + R
                K = track['covariance'] @ H.T @ np.linalg.inv(S)
                track['state'] = track['state'] + K @ y
                track['covariance'] = (np.eye(4) - K @ H) @ track['covariance']
                track['existence'] = min(1.0, track['existence'] + 0.15)
                associated_dets.add(det_key)
            else:
                # No association - existence probability drops slowly
                track['existence'] *= 0.95
        
        # Initialize new tracks from unassociated active sensor detections
        for sensor_id, sensor_dets in detections_by_sensor.items():
            if SENSOR_NAMES.get(sensor_id) in ["Lidar", "Radar"]:
                for i, det in enumerate(sensor_dets):
                    det_key = (det.time, det.sensor_id, i)
                    if det_key in associated_dets:
                        continue
                    
                    z = det.to_piren_ned()
                    if z is not None:
                        if z.ndim > 1:
                            z = z[0]
                        z = z[:2]
                        
                        # Check if close to existing track
                        too_close = False
                        for track in self.tracks.values():
                            dist = np.linalg.norm(z - track['state'][:2])
                            if dist < 15.0:  # 15m threshold
                                too_close = True
                                break
                        
                        if not too_close:
                            # Initialize new tentative track
                            self.tracks[self.next_track_id] = {
                                'state': np.array([z[0], z[1], 0, 0]),
                                'covariance': 100 * np.eye(4),
                                'existence': 0.5,
                                'confirmed': False
                            }
                            self.next_track_id += 1
        
        # Remove terminated tracks
        terminated = [tid for tid, track in self.tracks.items() 
                     if track['existence'] < self.existence_terminate]
        for tid in terminated:
            del self.tracks[tid]
        
        # Confirm tracks with high existence
        for track in self.tracks.values():
            if track['existence'] >= self.existence_confirm:
                track['confirmed'] = True
    
    def get_track_states(self) -> Dict[int, np.ndarray]:
        """Get current track positions."""
        return {tid: track['state'][:2] for tid, track in self.tracks.items()}
    
    def get_confirmed_tracks(self) -> Dict[int, Dict]:
        """Get confirmed tracks only."""
        return {tid: track for tid, track in self.tracks.items() if track['confirmed']}
    
    def track(self, detections: List[Detection]) -> Dict[int, Dict]:
        """Process all detections and return final tracks.
        
        Args:
            detections: List of all detections (sorted by time)
            
        Returns:
            Dictionary of confirmed tracks
        """
        if not detections:
            return {}
        
        # Sort by time
        sorted_dets = sorted(detections, key=lambda d: d.time)
        
        # Group by time
        times = sorted(set(d.time for d in sorted_dets))
        
        prev_time = times[0] if times else 0
        
        for t in times:
            dt = t - prev_time
            dets_at_t = [d for d in sorted_dets if d.time == t]
            self.update(dets_at_t, dt)
            prev_time = t
        
        return self.get_confirmed_tracks()


class FusionAttacker:
    """Adversarial attack generator targeting the JIPDA fusion layer."""
    
    def __init__(self, config: FusionAttackConfig):
        self.config = config
    
    def attack(self, detections: List[Detection]) -> List[Detection]:
        """Apply fusion-layer attack to a list of detections.
        
        Simple wrapper that applies attack per-detection.
        """
        if self.config.attack_type == FusionAttackType.SENSOR_DOS:
            # Remove detections from targeted sensors
            if self.config.dos_sensors:
                return [d for d in detections if d.sensor_id not in self.config.dos_sensors]
            else:
                # Randomly drop detections
                return [d for d in detections if np.random.random() > self.config.dos_probability]
        
        elif self.config.attack_type == FusionAttackType.FALSE_TRACK_INJECTION:
            # Add fake detections
            fake_dets = []
            for _ in range(3):
                if detections:
                    ref = detections[0]
                    fake_pos = ref.ownship_position + np.random.randn(2) * 50
                    fake_dets.append(Detection(
                        sensor_id=1,  # Lidar
                        time=ref.time,
                        ownship_position=ref.ownship_position.copy(),
                        measurement=fake_pos
                    ))
            return detections + fake_dets
        
        elif self.config.attack_type == FusionAttackType.EXISTENCE_SUPPRESSION:
            # Aggressively suppress detections to break association and track loss
            perturbed = []
            for det in detections:
                if len(det.measurement) == 0:
                    perturbed.append(det)
                    continue
                
                if det.is_active:
                    # Move active sensor detections far outside gate (>500m)
                    if det.measurement.ndim > 1:
                        p = det.measurement.copy()
                        for i in range(len(p)):
                            p[i, :2] += np.array([2000, 2000])  # Far away
                    else:
                        p = det.measurement.copy()
                        p[:2] += np.array([2000, 2000])
                    perturbed.append(Detection(
                        sensor_id=det.sensor_id,
                        time=det.time,
                        ownship_position=det.ownship_position.copy(),
                        measurement=p
                    ))
                else:
                    # For passive sensors, shift bearing by 90 degrees
                    perturbed.append(Detection(
                        sensor_id=det.sensor_id,
                        time=det.time,
                        ownship_position=det.ownship_position.copy(),
                        measurement=det.measurement + np.pi / 2
                    ))
            return perturbed
        
        # Default: no attack
        return detections
    
    def attack_scenario(self, loader: ScenarioLoader,
                        target_track_id: Optional[int] = None) -> Dict[float, List[Detection]]:
        """Apply fusion-layer attack to entire scenario.
        
        Returns:
            Dict mapping time -> list of perturbed detections
        """
        attacked_detections: Dict[float, List[Detection]] = {}
        
        # Get all unique times
        all_times = sorted(set(d.time for d in loader.detections))
        
        if self.config.attack_type == FusionAttackType.EXISTENCE_SUPPRESSION:
            attacked_detections = self._existence_suppression_attack(loader, all_times, target_track_id)
        elif self.config.attack_type == FusionAttackType.ASSOCIATION_CONFUSION:
            attacked_detections = self._association_confusion_attack(loader, all_times)
        elif self.config.attack_type == FusionAttackType.CROSS_SENSOR_CONSISTENCY:
            attacked_detections = self._cross_sensor_consistency_attack(loader, all_times)
        elif self.config.attack_type == FusionAttackType.FALSE_TRACK_INJECTION:
            attacked_detections = self._false_track_injection_attack(loader, all_times)
        elif self.config.attack_type == FusionAttackType.SENSOR_DOS:
            attacked_detections = self._sensor_dos_attack(loader, all_times)
        else:
            # No attack - return original
            for t in all_times:
                attacked_detections[t] = loader.get_detections_at_time(t)
        
        return attacked_detections
    
    def _existence_suppression_attack(self, loader: ScenarioLoader,
                                      all_times: List[float],
                                      target_track_id: Optional[int]) -> Dict[float, List[Detection]]:
        """Suppress all detections of a target to drive existence probability to 0.
        
        Strategy: For a duration of time, remove or perturb all detections
        that would associate with the target track.
        """
        attacked_detections: Dict[float, List[Detection]] = {}
        
        # Determine target position trajectory
        if target_track_id is not None:
            gt_times, gt_positions = loader.get_target_trajectory(target_track_id)
        else:
            # Default to first target
            gt_times, gt_positions = loader.get_target_trajectory(loader.target_ids[0])
        
        # Find start time for suppression
        if len(all_times) > 0:
            start_time = all_times[len(all_times) // 3]  # Start 1/3 into scenario
            end_time = start_time + self.config.suppression_duration
        else:
            start_time = 0
            end_time = 0
        
        for t in all_times:
            dets = loader.get_detections_at_time(t)
            
            if start_time <= t <= end_time:
                # During suppression window, perturb all detections
                perturbed_dets = []
                for det in dets:
                    # Check if detection is near target
                    det_pos = det.to_piren_ned()
                    if det_pos is not None:
                        # Find closest ground truth position
                        gt_idx = np.argmin(np.abs(gt_times - t))
                        gt_pos = gt_positions[gt_idx]
                        
                        if det_pos.ndim > 1:
                            det_pos = det_pos[0]
                        det_pos = det_pos[:2]
                        
                        dist = np.linalg.norm(det_pos - gt_pos)
                        
                        if dist < 50.0:  # Within 50m of target
                            # Suppress this detection
                            if det.is_active:
                                # Move detection far away
                                if det.measurement.ndim > 1:
                                    # Multiple sub-detections (radar)
                                    perturbed_measurement = det.measurement.copy()
                                    for i in range(len(perturbed_measurement)):
                                        perturbed_measurement[i, :2] += np.array([1000, 1000])
                                else:
                                    perturbed_measurement = det.measurement.copy()
                                    perturbed_measurement[:2] += np.array([1000, 1000])
                                
                                perturbed = Detection(
                                    sensor_id=det.sensor_id,
                                    time=det.time,
                                    ownship_position=det.ownship_position.copy(),
                                    measurement=perturbed_measurement
                                )
                            else:
                                # For passive, shift bearing by large amount
                                if len(det.measurement) > 0:
                                    perturbed = Detection(
                                        sensor_id=det.sensor_id,
                                        time=det.time,
                                        ownship_position=det.ownship_position.copy(),
                                        measurement=det.measurement + np.pi / 2
                                    )
                                else:
                                    perturbed = det
                            perturbed_dets.append(perturbed)
                        else:
                            perturbed_dets.append(det)
                    else:
                        perturbed_dets.append(det)
                attacked_detections[t] = perturbed_dets
            else:
                attacked_detections[t] = dets
        
        return attacked_detections
    
    def _association_confusion_attack(self, loader: ScenarioLoader,
                                      all_times: List[float]) -> Dict[float, List[Detection]]:
        """Push measurements outside validation gates.
        
        Strategy: Inflate measurement noise or shift measurements
        so they fall outside the track's validation gate.
        """
        attacked_detections: Dict[float, List[Detection]] = {}
        
        for t in all_times:
            dets = loader.get_detections_at_time(t)
            perturbed_dets = []
            
            for det in dets:
                if det.is_active:
                    # Shift active sensor measurements by gate_inflation * sigma
                    shift = self.config.gate_inflation * 5.0  # 5m shift
                    direction = np.random.randn(2)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    
                    if det.measurement.ndim == 1:
                        perturbed_measurement = det.measurement[:2] + shift * direction
                        if len(det.measurement) > 2:
                            perturbed_measurement = np.append(perturbed_measurement, det.measurement[2:])
                    else:
                        perturbed_measurement = det.measurement.copy()
                        for i in range(len(perturbed_measurement)):
                            perturbed_measurement[i, :2] += shift * direction
                    
                    perturbed = Detection(
                        sensor_id=det.sensor_id,
                        time=det.time,
                        ownship_position=det.ownship_position.copy(),
                        measurement=perturbed_measurement
                    )
                else:
                    # For passive sensors, shift bearing
                    if len(det.measurement) > 0:
                        perturbed = Detection(
                            sensor_id=det.sensor_id,
                            time=det.time,
                            ownship_position=det.ownship_position.copy(),
                            measurement=det.measurement + 0.3  # ~17 degrees
                        )
                    else:
                        perturbed = det
                
                perturbed_dets.append(perturbed)
            
            attacked_detections[t] = perturbed_dets
        
        return attacked_detections
    
    def _cross_sensor_consistency_attack(self, loader: ScenarioLoader,
                                         all_times: List[float]) -> Dict[float, List[Detection]]:
        """Coordinate failures across multiple sensors.
        
        Strategy: Attack a subset of sensors simultaneously to create
        correlated errors that break the independence assumption of JIPDA.
        """
        attacked_detections: Dict[float, List[Detection]] = {}
        
        sensors_to_attack = self.config.sensor_subset or [1, 2, 3, 4]
        
        for t in all_times:
            dets = loader.get_detections_at_time(t)
            perturbed_dets = []
            
            for det in dets:
                if det.sensor_id in sensors_to_attack:
                    # Apply consistent perturbation direction
                    # This creates correlated errors across sensors
                    consistent_shift = np.array([10.0, 10.0])  # Same shift for all
                    
                    if det.is_active:
                        if det.measurement.ndim == 1:
                            perturbed_measurement = det.measurement[:2] + consistent_shift
                            if len(det.measurement) > 2:
                                perturbed_measurement = np.append(perturbed_measurement, det.measurement[2:])
                        else:
                            perturbed_measurement = det.measurement.copy()
                            for i in range(len(perturbed_measurement)):
                                perturbed_measurement[i, :2] += consistent_shift
                        
                        perturbed = Detection(
                            sensor_id=det.sensor_id,
                            time=det.time,
                            ownship_position=det.ownship_position.copy(),
                            measurement=perturbed_measurement
                        )
                    else:
                        if len(det.measurement) > 0:
                            perturbed = Detection(
                                sensor_id=det.sensor_id,
                                time=det.time,
                                ownship_position=det.ownship_position.copy(),
                                measurement=det.measurement + 0.2
                            )
                        else:
                            perturbed = det
                    perturbed_dets.append(perturbed)
                else:
                    perturbed_dets.append(det)
            
            attacked_detections[t] = perturbed_dets
        
        return attacked_detections
    
    def _false_track_injection_attack(self, loader: ScenarioLoader,
                                      all_times: List[float]) -> Dict[float, List[Detection]]:
        """Inject consistent ghost measurements across multiple sensors.
        
        Strategy: Create fake detections at a specific location that
        persist across time, causing the tracker to initialize and
        confirm a false track.
        """
        attacked_detections: Dict[float, List[Detection]] = {}
        
        # Determine false track position
        if self.config.false_track_position is not None:
            false_pos = self.config.false_track_position
        else:
            # Place at a plausible location (offset from ownship)
            false_pos = np.array([50.0, 50.0])
        
        # Duration for false track
        if len(all_times) > 0:
            start_time = all_times[len(all_times) // 2]
            end_time = start_time + self.config.false_track_duration
        else:
            start_time = 0
            end_time = 0
        
        for t in all_times:
            dets = loader.get_detections_at_time(t)
            perturbed_dets = list(dets)
            
            if start_time <= t <= end_time:
                # Inject fake detections from active sensors
                for sensor_id in [1, 2]:  # Lidar and Radar
                    # Convert false position to ownship NED
                    ownship_pos = dets[0].ownship_position if dets else np.array([0, 0])
                    false_ownship = false_pos - ownship_pos
                    
                    fake_det = Detection(
                        sensor_id=sensor_id,
                        time=t,
                        ownship_position=ownship_pos.copy(),
                        measurement=false_ownship
                    )
                    perturbed_dets.append(fake_det)
            
            attacked_detections[t] = perturbed_dets
        
        return attacked_detections
    
    def _sensor_dos_attack(self, loader: ScenarioLoader,
                           all_times: List[float]) -> Dict[float, List[Detection]]:
        """Denial of Service on specific sensors.
        
        Strategy: Randomly drop detections from targeted sensors.
        """
        attacked_detections: Dict[float, List[Detection]] = {}
        
        sensors_to_dos = self.config.dos_sensors or [1, 2]
        
        for t in all_times:
            dets = loader.get_detections_at_time(t)
            perturbed_dets = []
            
            for det in dets:
                if det.sensor_id in sensors_to_dos:
                    # Randomly drop with probability
                    if np.random.random() > self.config.dos_probability:
                        perturbed_dets.append(det)
                else:
                    perturbed_dets.append(det)
            
            attacked_detections[t] = perturbed_dets
        
        return attacked_detections


def evaluate_fusion_attack(benign_tracker: JIPDASimulator,
                           attacked_tracker: JIPDASimulator,
                           ground_truth: Dict[int, List[Tuple[float, np.ndarray]]]) -> Dict[str, float]:
    """Evaluate fusion attack effectiveness.
    
    Metrics:
        - Track loss rate
        - False track rate
        - Position RMSE
        - Track fragmentation
    """
    # Compare track states
    benign_tracks = benign_tracker.get_confirmed_tracks()
    attacked_tracks = attacked_tracker.get_confirmed_tracks()
    
    # Count tracks
    n_benign = len(benign_tracks)
    n_attacked = len(attacked_tracks)
    
    # Calculate position errors for existing tracks
    errors = []
    for tid in benign_tracks:
        if tid in attacked_tracks:
            # Track survived - compare positions
            # (Would need full trajectory history for proper comparison)
            pass
    
    return {
        'benign_tracks': n_benign,
        'attacked_tracks': n_attacked,
        'track_change': n_attacked - n_benign,
        'track_loss_rate': max(0, n_benign - n_attacked) / max(n_benign, 1),
        'false_track_rate': max(0, n_attacked - n_benign) / max(n_attacked, 1),
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
    print("FUSION LAYER ADVERSARIAL ATTACK DEMONSTRATION")
    print("=" * 70)
    
    # Initialize trackers
    benign_tracker = JIPDASimulator()
    
    # Run benign tracking
    print("\n--- Benign Tracking ---")
    all_times = sorted(set(d.time for d in loader.detections))
    for i in range(min(100, len(all_times) - 1)):
        t = all_times[i]
        dt = all_times[i + 1] - t if i + 1 < len(all_times) else 0.1
        dets = loader.get_detections_at_time(t)
        benign_tracker.update(dets, dt)
    
    print(f"Confirmed tracks: {len(benign_tracker.get_confirmed_tracks())}")
    print(f"Total tracks: {len(benign_tracker.tracks)}")
    
    # Test existence suppression attack
    print("\n--- Existence Suppression Attack ---")
    config = FusionAttackConfig(
        FusionAttackType.EXISTENCE_SUPPRESSION,
        suppression_duration=10.0
    )
    attacker = FusionAttacker(config)
    attacked_dets = attacker.attack_scenario(loader, target_track_id=1)
    
    attacked_tracker = JIPDASimulator()
    for i in range(min(100, len(all_times) - 1)):
        t = all_times[i]
        dt = all_times[i + 1] - t if i + 1 < len(all_times) else 0.1
        dets = attacked_dets.get(t, [])
        attacked_tracker.update(dets, dt)
    
    print(f"Confirmed tracks: {len(attacked_tracker.get_confirmed_tracks())}")
    print(f"Total tracks: {len(attacked_tracker.tracks)}")
    
    # Evaluate
    metrics = evaluate_fusion_attack(benign_tracker, attacked_tracker, {})
    print(f"\nAttack Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test false track injection
    print("\n--- False Track Injection Attack ---")
    config = FusionAttackConfig(
        FusionAttackType.FALSE_TRACK_INJECTION,
        false_track_position=np.array([100.0, 100.0]),
        false_track_duration=5.0
    )
    attacker = FusionAttacker(config)
    attacked_dets = attacker.attack_scenario(loader)
    
    false_tracker = JIPDASimulator()
    for i in range(min(100, len(all_times) - 1)):
        t = all_times[i]
        dt = all_times[i + 1] - t if i + 1 < len(all_times) else 0.1
        dets = attacked_dets.get(t, [])
        false_tracker.update(dets, dt)
    
    print(f"Confirmed tracks: {len(false_tracker.get_confirmed_tracks())}")
    print(f"Total tracks: {len(false_tracker.tracks)}")
