"""Helper functions for notebooks to work with ScenarioLoader API."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
from data_loader import ScenarioLoader, Detection, GroundTruth
from typing import Dict, List


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
    """Get all ground truth as DataFrames keyed by target_id."""
    result = {}
    for target_id in loader.target_ids:
        gt = loader.get_target_ground_truth(target_id)
        result[target_id] = ground_truth_to_dataframe(gt)
    return result


def get_ownship(loader: ScenarioLoader) -> pd.DataFrame:
    """Get ownship trajectory as DataFrame."""
    times, positions = loader.get_ownship_trajectory()
    return pd.DataFrame({
        'time': times,
        'x_piren': positions[:, 1],  # East
        'y_piren': positions[:, 0],  # North
    })


def compute_sensor_metrics(detections_df: pd.DataFrame, ground_truth_dict: Dict[int, pd.DataFrame],
                           sensor_id: int, distance_threshold: float = 20.0) -> Dict[str, float]:
    """Compute detection metrics from DataFrames for a specific sensor."""
    df = detections_df.copy()
    
    # Collect all ground truth points
    all_gt = []
    for tid, gt_df in ground_truth_dict.items():
        all_gt.append(gt_df[['time', 'x_piren', 'y_piren']].copy())
    
    if len(all_gt) == 0 or len(df) == 0:
        return {'detection_probability': 0.0, 'false_alarm_rate': 0.0, 'rmse': 0.0}
    
    gt_combined = pd.concat(all_gt, ignore_index=True)
    
    # For each detection, find closest ground truth in time and space
    matched_gt = set()
    errors = []
    false_alarms = 0
    
    for _, det in df.iterrows():
        if pd.isna(det['x_piren']) or pd.isna(det['y_piren']):
            # Passive sensor - skip for now (would need bearing matching)
            continue
        
        # Find closest GT in time
        time_diffs = np.abs(gt_combined['time'] - det['time'])
        nearby_gt = gt_combined[time_diffs <= 1.0]  # Within 1 second
        
        if len(nearby_gt) == 0:
            false_alarms += 1
            continue
        
        # Find closest in space
        dists = np.sqrt((nearby_gt['x_piren'] - det['x_piren'])**2 + 
                        (nearby_gt['y_piren'] - det['y_piren'])**2)
        min_dist = dists.min()
        min_idx = dists.idxmin()
        
        if min_dist < distance_threshold:
            errors.append(min_dist)
            matched_gt.add(min_idx)
        else:
            false_alarms += 1
    
    total_gt = len(gt_combined)
    det_prob = len(matched_gt) / max(total_gt, 1)
    far = false_alarms / max(len(df), 1)
    rmse = np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0
    
    return {
        'detection_probability': det_prob,
        'false_alarm_rate': far,
        'rmse': rmse,
        'mean_error': np.mean(errors) if errors else 0.0,
        'max_error': np.max(errors) if errors else 0.0,
    }
