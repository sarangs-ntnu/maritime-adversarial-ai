"""Helper functions for notebooks to work with ScenarioLoader API."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from data_loader import ScenarioLoader, Detection, GroundTruth
from typing import Dict, List, Optional


def load_scenario(scenario_name: str, data_dir: str = None):
    """Load a scenario and return a ScenarioLoader instance."""
    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent / 'data' / 'sensor_fusion_dataset')
    return ScenarioLoader(scenario_name, data_dir)


def detections_to_dataframe(detections: List[Detection]) -> pd.DataFrame:
    """Convert list of Detection objects to pandas DataFrame."""
    rows = []
    for det in detections:
        row = {
            'time': det.time,
            'sensor_id': det.sensor_id,
            'sensor_name': det.sensor_name,
        }
        
        # Add position if available (active sensors)
        piren = det.to_piren_ned()
        if piren is not None:
            if piren.ndim == 1:
                row['x_piren'] = piren[1]  # East
                row['y_piren'] = piren[0]  # North
            elif piren.ndim == 2:
                # Multiple sub-detections (radar) - take first
                row['x_piren'] = piren[0, 1]
                row['y_piren'] = piren[0, 0]
        else:
            # Passive sensor - bearing only
            row['x_piren'] = np.nan
            row['y_piren'] = np.nan
        
        # Add measurement
        if det.is_passive:
            if det.measurement.ndim == 0:
                row['bearing'] = float(det.measurement)
            else:
                row['bearing'] = float(det.measurement[0]) if len(det.measurement) > 0 else np.nan
        else:
            row['bearing'] = np.nan
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def ground_truth_to_dataframe(gt_list: List[GroundTruth]) -> pd.DataFrame:
    """Convert list of GroundTruth objects to pandas DataFrame."""
    rows = []
    for gt in gt_list:
        rows.append({
            'time': gt.time,
            'target_id': gt.target_id,
            'x_piren': gt.position[1],  # East
            'y_piren': gt.position[0],  # North
        })
    return pd.DataFrame(rows)


def get_all_detections(loader: ScenarioLoader) -> Dict[int, pd.DataFrame]:
    """Get all detections as DataFrames keyed by sensor_id."""
    result = {}
    for sensor_id in loader.sensor_ids:
        dets = loader.get_sensor_detections(sensor_id)
        result[sensor_id] = detections_to_dataframe(dets)
    return result


def get_ground_truth(loader: ScenarioLoader) -> Dict[int, pd.DataFrame]:
    """Get ground truth as DataFrames keyed by target_id."""
    result = {}
    for target_id in loader.target_ids:
        gt = loader.get_target_ground_truth(target_id)
        result[target_id] = ground_truth_to_dataframe(gt)
    return result


def get_ownship(loader: ScenarioLoader) -> pd.DataFrame:
    """Get ownship trajectory as DataFrame."""
    ownship = loader.get_ownship_trajectory()
    rows = []
    for state in ownship:
        rows.append({
            'time': state.time,
            'x_piren': state.position[1],  # East
            'y_piren': state.position[0],  # North
            'heading': state.heading,
            'speed': state.speed,
        })
    return pd.DataFrame(rows)


def compute_sensor_metrics(detections_df: pd.DataFrame, ground_truth_dict: Dict[int, pd.DataFrame],
                           sensor_id: int, max_distance: float = 50.0) -> Dict:
    """Compute detection metrics for a sensor."""
    if len(detections_df) == 0:
        return {'detection_probability': 0.0, 'false_alarm_rate': 0.0, 'rmse': float('inf')}
    
    # Count detections
    n_detections = len(detections_df)
    
    # Estimate detection probability (detections / expected based on GT)
    total_gt_points = sum(len(gt) for gt in ground_truth_dict.values())
    det_prob = min(1.0, n_detections / max(1, total_gt_points))
    
    # Estimate false alarm rate
    fa_rate = 0.0
    if sensor_id in [1, 2]:
        # Active sensors - compare to ground truth positions
        for _, det in detections_df.iterrows():
            if pd.isna(det['x_piren']):
                continue
            min_dist = float('inf')
            for gt_df in ground_truth_dict.values():
                dists = np.sqrt((gt_df['x_piren'] - det['x_piren'])**2 +
                               (gt_df['y_piren'] - det['y_piren'])**2)
                if len(dists) > 0:
                    min_dist = min(min_dist, dists.min())
            if min_dist > max_distance:
                fa_rate += 1
        fa_rate = fa_rate / max(1, n_detections)
    
    # RMSE for active sensors
    rmse = float('inf')
    if sensor_id in [1, 2] and not detections_df['x_piren'].isna().all():
        errors = []
        for _, det in detections_df.iterrows():
            if pd.isna(det['x_piren']):
                continue
            for gt_df in ground_truth_dict.values():
                dists = np.sqrt((gt_df['x_piren'] - det['x_piren'])**2 +
                               (gt_df['y_piren'] - det['y_piren'])**2)
                if len(dists) > 0:
                    errors.append(dists.min())
        if errors:
            rmse = np.sqrt(np.mean(np.array(errors)**2))
    
    return {
        'detection_probability': det_prob,
        'false_alarm_rate': fa_rate,
        'rmse': rmse,
        'n_detections': n_detections
    }


# ---------------------------------------------------------------------------
# DataFrame-to-Detection helpers
# ---------------------------------------------------------------------------

def _df_to_detection(row: pd.Series, sensor_id: int) -> Detection:
    """Convert a DataFrame row back to a Detection object."""
    if sensor_id in [3, 4]:
        meas = np.array([row['bearing']]) if not pd.isna(row.get('bearing', np.nan)) else np.array([])
    else:
        meas = np.array([[row['y_piren'], row['x_piren']]]) if not pd.isna(row.get('x_piren', np.nan)) else np.array([])
    return Detection(
        sensor_id=sensor_id,
        time=row['time'],
        ownship_position=np.array([0.0, 0.0]),
        measurement=meas
    )


def _detection_to_series(det: Detection) -> Dict:
    """Convert a Detection to a dict for DataFrame construction."""
    row = {
        'time': det.time,
        'sensor_id': det.sensor_id,
        'sensor_name': det.sensor_name,
    }
    piren = det.to_piren_ned()
    if piren is not None:
        if piren.ndim == 1:
            row['x_piren'] = piren[1]
            row['y_piren'] = piren[0]
        elif piren.ndim == 2:
            row['x_piren'] = piren[0, 1]
            row['y_piren'] = piren[0, 0]
    else:
        row['x_piren'] = np.nan
        row['y_piren'] = np.nan
    
    if det.is_passive:
        if det.measurement.ndim == 0:
            row['bearing'] = float(det.measurement)
        else:
            row['bearing'] = float(det.measurement[0]) if len(det.measurement) > 0 else np.nan
    else:
        row['bearing'] = np.nan
    return row


# ---------------------------------------------------------------------------
# Notebook-friendly wrappers
# ---------------------------------------------------------------------------

class CameraAttackerDF:
    """Wrapper for CameraAdversarialAttacker that works with DataFrames."""
    
    def __init__(self, epsilon: float = 0.05, num_steps: int = 10):
        from attacks.camera_attacks import CameraAdversarialAttacker, AttackConfig
        self.epsilon = epsilon
        self.num_steps = num_steps
        # Store config for creating attacker per-attack-type
        self._base_config = AttackConfig
        self._attacker_cls = CameraAdversarialAttacker
    
    def attack_detections(self, df: pd.DataFrame, attack_type, sensor_id: int = 3) -> pd.DataFrame:
        """Apply camera attack to a DataFrame of detections."""
        from attacks.camera_attacks import AttackConfig
        config = AttackConfig(attack_type=attack_type, epsilon=self.epsilon,
                              num_steps=self.num_steps)
        attacker = self._attacker_cls(config)
        
        attacked_rows = []
        for _, row in df.iterrows():
            det = _df_to_detection(row, sensor_id)
            attacked_det = attacker.attack(det)
            attacked_rows.append(_detection_to_series(attacked_det))
        
        return pd.DataFrame(attacked_rows)


class PointCloudAttackerDF:
    """Wrapper for PointCloudAttacker that works with DataFrames."""
    
    def __init__(self, epsilon: float = 5.0):
        from attacks.radar_lidar_attacks import PointCloudAttacker, PointCloudAttackConfig
        self.epsilon = epsilon
        self._config_cls = PointCloudAttackConfig
        self._attacker_cls = PointCloudAttacker
    
    def attack_detections(self, df: pd.DataFrame, attack_type, sensor_id: int = 1) -> pd.DataFrame:
        """Apply point cloud attack to a DataFrame of detections."""
        from attacks.radar_lidar_attacks import PointCloudAttackConfig
        config = PointCloudAttackConfig(attack_type=attack_type, epsilon=self.epsilon)
        attacker = self._attacker_cls(config)
        
        attacked_rows = []
        for _, row in df.iterrows():
            det = _df_to_detection(row, sensor_id)
            attacked_det = attacker.attack(det)
            attacked_rows.append(_detection_to_series(attacked_det))
        
        return pd.DataFrame(attacked_rows)


class FusionAttackerDF:
    """Wrapper for FusionAttacker that works with DataFrames."""
    
    def __init__(self):
        from attacks.fusion_attacks import FusionAttacker, FusionAttackConfig
        self._config_cls = FusionAttackConfig
        self._attacker_cls = FusionAttacker
    
    def attack_scenario(self, detections_dict: Dict[int, pd.DataFrame], attack_type,
                        ground_truth_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply fusion attack to a dict of DataFrames."""
        from attacks.fusion_attacks import FusionAttackConfig
        config = FusionAttackConfig(attack_type=attack_type)
        attacker = self._attacker_cls(config)
        
        # Convert DataFrames to flat Detection list
        all_dets = []
        for sid, df in detections_dict.items():
            for _, row in df.iterrows():
                all_dets.append(_df_to_detection(row, sid))
        
        # Apply attack
        attacked = attacker.attack(all_dets)
        
        # Convert back to DataFrames by sensor
        result = {sid: [] for sid in detections_dict.keys()}
        for det in attacked:
            result[det.sensor_id].append(_detection_to_series(det))
        
        df_result = {}
        for sid in detections_dict.keys():
            df_result[sid] = pd.DataFrame(result[sid]) if result[sid] else detections_dict[sid].iloc[0:0]
        return df_result


class DefensePipelineDF:
    """Wrapper for DefensePipeline that works with DataFrames."""
    
    def __init__(self):
        from defenses.defense_mechanisms import DefensePipeline, DefenseConfig, DefenseType
        config = DefenseConfig(defense_type=DefenseType.ALL)
        self.pipeline = DefensePipeline(config)
    
    def defend_detections(self, detections_dict: Dict[int, pd.DataFrame], defense_type,
                          ground_truth_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply defense to a dict of DataFrames."""
        from defenses.defense_mechanisms import DefenseConfig, DefenseType, DefensePipeline
        config = DefenseConfig(defense_type=defense_type)
        self.pipeline = DefensePipeline(config)
        
        # Convert DataFrames to flat Detection list
        all_dets = []
        for sid, df in detections_dict.items():
            for _, row in df.iterrows():
                all_dets.append(_df_to_detection(row, sid))
        
        # Apply defense
        defended = self.pipeline.defend(all_dets)
        
        # Convert back to DataFrames by sensor
        result = {sid: [] for sid in detections_dict.keys()}
        for det in defended:
            result[det.sensor_id].append(_detection_to_series(det))
        
        df_result = {}
        for sid in detections_dict.keys():
            df_result[sid] = pd.DataFrame(result[sid]) if result[sid] else detections_dict[sid].iloc[0:0]
        return df_result


class CertifiedDefenseDF:
    """Wrapper for CertifiedDefense that works with DataFrames."""
    
    def __init__(self, certified_radius: float = 0.05):
        from defenses.defense_mechanisms import CertifiedDefense
        self.cert = CertifiedDefense(num_samples=50, noise_std=certified_radius, confidence=0.99)
        self.radius = certified_radius
    
    def defend(self, detections_dict: Dict[int, pd.DataFrame],
               ground_truth_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply certified defense to DataFrames."""
        result = {}
        for sid, df in detections_dict.items():
            defended_rows = []
            for _, row in df.iterrows():
                det = _df_to_detection(row, sid)
                defended_det, _ = self.cert.smooth_bearing(det)
                defended_rows.append(_detection_to_series(defended_det))
            result[sid] = pd.DataFrame(defended_rows) if defended_rows else df.iloc[0:0]
        return result


class AdversarialTrainingDF:
    """Wrapper for AdversarialTraining that works with DataFrames."""
    
    def __init__(self, augmentation_ratio: float = 0.3):
        from defenses.defense_mechanisms import AdversarialTraining
        self.adv_train = AdversarialTraining(augmentation_ratio=augmentation_ratio)
        self.augmentation_ratio = augmentation_ratio
    
    def train(self, benign_dict: Dict[int, pd.DataFrame], attacked_dict: Dict[int, pd.DataFrame]):
        """Train on DataFrames."""
        benign_dets = []
        for sid, df in benign_dict.items():
            for _, row in df.iterrows():
                benign_dets.append(_df_to_detection(row, sid))
        
        attacked_dets = []
        for sid, df in attacked_dict.items():
            for _, row in df.iterrows():
                attacked_dets.append(_df_to_detection(row, sid))
        
        self.adv_train.train(benign_dets, attacked_dets)
    
    def defend(self, detections_dict: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Apply adversarial training defense to DataFrames."""
        result = {}
        for sid, df in detections_dict.items():
            defended_rows = []
            for _, row in df.iterrows():
                det = _df_to_detection(row, sid)
                defended_det = self.adv_train.defend(det)
                defended_rows.append(_detection_to_series(defended_det))
            result[sid] = pd.DataFrame(defended_rows) if defended_rows else df.iloc[0:0]
        return result


class MaritimeEOTDF:
    """Wrapper for MaritimeEOT that works with DataFrames."""
    
    def __init__(self, wave_height: float = 1.0, rain_rate: float = 0.0,
                 fog_visibility: float = 10000.0, sun_glint_angle: float = 90.0):
        from attacks.physical_eot import MaritimeEOT, MaritimeEnvironmentParams
        env = MaritimeEnvironmentParams(
            wave_height=wave_height,
            rain_rate=rain_rate,
            fog_visibility=fog_visibility
        )
        self.model = MaritimeEOT(env_params=env)
    
    def transform_detection(self, det_dict: Dict[str, float], perturbation: float) -> float:
        """Evaluate realizability for a single detection dict."""
        from attacks.physical_eot import MaritimeTransformationType
        det = Detection(
            sensor_id=3,
            time=det_dict.get('time', 0),
            ownship_position=np.array([det_dict.get('y_piren', 0), det_dict.get('x_piren', 0), 0]),
            measurement=np.array([perturbation])
        )
        transformed = self.model.transform_detection(det, MaritimeTransformationType.COMBINED, n_samples=10)
        # Return a realizability score based on how much perturbation survived
        if len(transformed.measurement) > 0:
            survived = abs(float(transformed.measurement[0]))
            original = abs(perturbation)
            if original > 1e-10:
                return min(1.0, survived / original)
        return 0.5
