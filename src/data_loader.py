"""
Data Loader for Autoferry Sensor Fusion Dataset
===============================================

Loads detections and ground truth from JSON files.
Provides unified access to all sensor modalities.

Sensor IDs:
    1: Lidar (active, 2D position [N, E])
    2: Radar (active, multiple 2D positions [[N,E,D], ...])
    3: IR Camera (passive, bearing scalar)
    4: EO Camera (passive, multiple bearings [b1, b2, ...])

Coordinate Frames:
    - Ownship NED: vessel-fixed (measurements)
    - Piren NED: world-fixed, origin at LLA [63.4389, 10.3991, 39.923]
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


# Piren NED origin (Lat, Lon, Alt)
PIREN_LLA = np.array([63.4389029083, 10.39908278, 39.923])

SENSOR_NAMES = {1: "Lidar", 2: "Radar", 3: "IR", 4: "EO"}
SENSOR_TYPES = {1: "active", 2: "active", 3: "passive", 4: "passive"}


@dataclass
class Detection:
    """Single detection/measurement from a sensor."""
    sensor_id: int
    time: float
    ownship_position: np.ndarray  # [N, E] in Piren NED
    measurement: np.ndarray       # sensor-specific
    
    @property
    def sensor_name(self) -> str:
        return SENSOR_NAMES.get(self.sensor_id, "Unknown")
    
    @property
    def is_active(self) -> bool:
        return SENSOR_TYPES.get(self.sensor_id, "") == "active"
    
    @property
    def is_passive(self) -> bool:
        return SENSOR_TYPES.get(self.sensor_id, "") == "passive"
    
    def to_piren_ned(self) -> Optional[np.ndarray]:
        """Convert measurement to Piren NED frame."""
        if self.is_active:
            # Active sensors: measurement is position in ownship NED
            # Add ownship position to get Piren NED
            if self.measurement.ndim == 1 and len(self.measurement) >= 2:
                return self.measurement[:2] + self.ownship_position
            elif self.measurement.ndim == 2:
                # Multiple sub-detections (radar)
                return self.measurement[:, :2] + self.ownship_position
        else:
            # Passive sensors: bearing only, cannot convert without range
            return None
        return None


@dataclass
class GroundTruth:
    """Ground truth target state."""
    target_id: int
    time: float
    position: np.ndarray  # [N, E, D] in Piren NED
    
    def to_ownship_ned(self, ownship_pos: np.ndarray) -> np.ndarray:
        """Convert to ownship NED frame."""
        return self.position[:2] - ownship_pos


class ScenarioLoader:
    """Load and manage a single scenario's data."""
    
    def __init__(self, scenario_name: str, data_dir: str = None):
        if data_dir is None:
            # Default to project root's data directory
            script_dir = Path(__file__).parent.resolve()
            data_dir = script_dir.parent / "data" / "sensor_fusion_dataset"
        self.scenario_name = scenario_name
        self.data_dir = Path(data_dir)
        self.scenario_dir = self.data_dir / scenario_name
        if not self.scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {self.scenario_dir}")
        
        self.detections: List[Detection] = []
        self.ground_truth: List[List[GroundTruth]] = []
        self._detections_by_sensor: Dict[int, List[Detection]] = defaultdict(list)
        self._gt_by_target: Dict[int, List[GroundTruth]] = defaultdict(list)
        
        self._load()
    
    def _load(self):
        """Load detections and ground truth from JSON files."""
        det_file = self.scenario_dir / f"{self.scenario_name}_detections.json"
        gt_file = self.scenario_dir / f"{self.scenario_name}_groundTruth.json"
        
        # Load detections
        with open(det_file, 'r') as f:
            det_data = json.load(f)
        
        for det in det_data:
            sensor_id = det['sensorID']
            measurement = det['measurement']
            
            # Normalize measurement to numpy array
            if isinstance(measurement, (int, float)):
                measurement = np.array([measurement])
            elif isinstance(measurement, list):
                if len(measurement) > 0 and isinstance(measurement[0], list):
                    measurement = np.array(measurement)
                else:
                    measurement = np.array(measurement)
            else:
                measurement = np.array([])
            
            detection = Detection(
                sensor_id=sensor_id,
                time=det['time'],
                ownship_position=np.array(det['ownshipPosition']),
                measurement=measurement
            )
            self.detections.append(detection)
            self._detections_by_sensor[sensor_id].append(detection)
        
        # Load ground truth
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)
        
        for timestep in gt_data:
            gt_step = []
            for entry in timestep:
                gt = GroundTruth(
                    target_id=entry['targetID'],
                    time=entry['time'],
                    position=np.array(entry['position'])
                )
                gt_step.append(gt)
                self._gt_by_target[gt.target_id].append(gt)
            self.ground_truth.append(gt_step)
    
    def get_sensor_detections(self, sensor_id: int) -> List[Detection]:
        """Get all detections from a specific sensor."""
        return self._detections_by_sensor.get(sensor_id, [])
    
    def get_target_ground_truth(self, target_id: int) -> List[GroundTruth]:
        """Get all ground truth entries for a specific target."""
        return self._gt_by_target.get(target_id, [])
    
    def get_all_times(self) -> np.ndarray:
        """Get sorted unique timestamps across all detections."""
        times = sorted(set(d.time for d in self.detections))
        return np.array(times)
    
    def get_detections_at_time(self, time: float, tolerance: float = 0.1) -> List[Detection]:
        """Get all detections within tolerance of a given time."""
        return [d for d in self.detections if abs(d.time - time) <= tolerance]
    
    def get_ground_truth_at_time(self, time: float, tolerance: float = 0.1) -> List[GroundTruth]:
        """Get ground truth at a given time."""
        for timestep in self.ground_truth:
            if timestep and abs(timestep[0].time - time) <= tolerance:
                return timestep
        return []
    
    def get_target_trajectory(self, target_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get time and position trajectory for a target.
        
        Returns:
            times: (N,) array of timestamps
            positions: (N, 2) array of [N, E] positions
        """
        gt_list = self.get_target_ground_truth(target_id)
        times = np.array([gt.time for gt in gt_list])
        positions = np.array([gt.position[:2] for gt in gt_list])
        return times, positions
    
    def get_ownship_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get ownship trajectory from detection timestamps.
        
        Returns:
            times: (N,) array of timestamps
            positions: (N, 2) array of [N, E] positions
        """
        seen_times = set()
        times = []
        positions = []
        for d in self.detections:
            if d.time not in seen_times:
                seen_times.add(d.time)
                times.append(d.time)
                positions.append(d.ownship_position)
        return np.array(times), np.array(positions)
    
    @property
    def num_detections(self) -> int:
        return len(self.detections)
    
    @property
    def num_timesteps(self) -> int:
        return len(self.ground_truth)
    
    @property
    def target_ids(self) -> List[int]:
        return sorted(self._gt_by_target.keys())
    
    @property
    def sensor_ids(self) -> List[int]:
        return sorted(self._detections_by_sensor.keys())
    
    def summary(self) -> str:
        """Return a text summary of the scenario."""
        lines = [
            f"Scenario: {self.scenario_name}",
            f"  Total detections: {self.num_detections}",
            f"  Total timesteps: {self.num_timesteps}",
            f"  Targets: {self.target_ids}",
            f"  Sensors: {self.sensor_ids}",
        ]
        for sid in self.sensor_ids:
            dets = self.get_sensor_detections(sid)
            lines.append(f"    Sensor {sid} ({SENSOR_NAMES[sid]}): {len(dets)} scans")
        return "\n".join(lines)


def load_all_scenarios(data_dir: str = None) -> Dict[str, ScenarioLoader]:
    """Load all available scenarios."""
    if data_dir is None:
        script_dir = Path(__file__).parent.resolve()
        data_dir = script_dir.parent / "data" / "sensor_fusion_dataset"
    data_path = Path(data_dir)
    scenarios = {}
    for scenario_dir in sorted(data_path.glob("scenario*")):
        if scenario_dir.is_dir():
            name = scenario_dir.name
            try:
                scenarios[name] = ScenarioLoader(name, data_dir)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
    return scenarios


if __name__ == "__main__":
    # Quick test
    loader = ScenarioLoader("scenario2")
    print(loader.summary())
    print("\n--- Target 1 trajectory (first 5 points) ---")
    times, positions = loader.get_target_trajectory(1)
    print(f"Times: {times[:5]}")
    print(f"Positions:\n{positions[:5]}")
