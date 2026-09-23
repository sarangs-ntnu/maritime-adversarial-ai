"""
Visualization utilities for maritime sensor fusion data and adversarial attacks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    from ..data_loader import ScenarioLoader, Detection, GroundTruth, SENSOR_NAMES
except ImportError:
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.data_loader import ScenarioLoader, Detection, GroundTruth, SENSOR_NAMES


def plot_scenario_overview(loader: ScenarioLoader, save_path: Optional[str] = None):
    """Plot full scenario: ownship trajectory, target trajectories, and sensor coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Piren NED frame (world-fixed)
    ax = axes[0]
    ax.set_title(f"{loader.scenario_name} — Piren NED Frame (World-Fixed)")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Ownship trajectory
    own_t, own_pos = loader.get_ownship_trajectory()
    ax.plot(own_pos[:, 1], own_pos[:, 0], 'r-', linewidth=1.5, label='Ownship (milliAmpere)', alpha=0.7)
    ax.scatter(own_pos[0, 1], own_pos[0, 0], c='red', s=100, marker='s', zorder=5, label='Start')
    
    # Target trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, tid in enumerate(loader.target_ids):
        t, pos = loader.get_target_trajectory(tid)
        ax.plot(pos[:, 1], pos[:, 0], '-', color=colors[i], linewidth=2, label=f'Target {tid}')
        ax.scatter(pos[0, 1], pos[0, 0], c=[colors[i]], s=80, marker='o', zorder=5)
        ax.scatter(pos[-1, 1], pos[-1, 0], c=[colors[i]], s=80, marker='x', zorder=5)
    
    ax.legend(loc='upper right', fontsize=8)
    
    # Right: Ownship NED frame (vessel-fixed)
    ax = axes[1]
    ax.set_title(f"{loader.scenario_name} — Ownship NED Frame (Vessel-Fixed)")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Ownship at origin
    ax.scatter(0, 0, c='red', s=150, marker='s', zorder=5, label='Ownship')
    
    # Target trajectories in ownship frame
    for i, tid in enumerate(loader.target_ids):
        gt_list = loader.get_target_ground_truth(tid)
        ownship_positions = []
        relative_positions = []
        for gt in gt_list:
            # Find closest ownship position
            closest_det = min(loader.detections, key=lambda d: abs(d.time - gt.time))
            ownship_pos = closest_det.ownship_position
            rel_pos = gt.position[:2] - ownship_pos
            ownship_positions.append(ownship_pos)
            relative_positions.append(rel_pos)
        
        rel_pos = np.array(relative_positions)
        ax.plot(rel_pos[:, 1], rel_pos[:, 0], '-', color=colors[i], linewidth=2, label=f'Target {tid} (relative)')
        ax.scatter(rel_pos[0, 1], rel_pos[0, 0], c=[colors[i]], s=80, marker='o', zorder=5)
        ax.scatter(rel_pos[-1, 1], rel_pos[-1, 0], c=[colors[i]], s=80, marker='x', zorder=5)
    
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sensor_detections(loader: ScenarioLoader, sensor_id: int, 
                           target_id: Optional[int] = None,
                           save_path: Optional[str] = None):
    """Plot detections from a specific sensor overlaid with ground truth."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
    ax.set_title(f"{loader.scenario_name} — {sensor_name} Detections")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Ownship trajectory
    own_t, own_pos = loader.get_ownship_trajectory()
    ax.plot(own_pos[:, 1], own_pos[:, 0], 'r-', linewidth=1, alpha=0.5, label='Ownship')
    
    # Ground truth for specified target or all targets
    target_ids = [target_id] if target_id is not None else loader.target_ids
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, tid in enumerate(target_ids):
        t, pos = loader.get_target_trajectory(tid)
        ax.plot(pos[:, 1], pos[:, 0], '--', color=colors[i], linewidth=2, 
                alpha=0.7, label=f'Target {tid} GT')
    
    # Sensor detections
    dets = loader.get_sensor_detections(sensor_id)
    det_positions = []
    for det in dets:
        piren_pos = det.to_piren_ned()
        if piren_pos is not None:
            if piren_pos.ndim == 1:
                det_positions.append(piren_pos)
            else:
                for p in piren_pos:
                    det_positions.append(p)
    
    if det_positions:
        det_positions = np.array(det_positions)
        ax.scatter(det_positions[:, 1], det_positions[:, 0], 
                  c='blue', s=10, alpha=0.4, label=f'{sensor_name} detections')
    
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_all_sensors(loader: ScenarioLoader, save_path: Optional[str] = None):
    """Plot detections from all sensors in subplots."""
    sensor_ids = loader.sensor_ids
    n_sensors = len(sensor_ids)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, sensor_id in enumerate(sensor_ids):
        ax = axes[idx]
        sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
        ax.set_title(f"{sensor_name} (ID: {sensor_id})")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Ownship
        own_t, own_pos = loader.get_ownship_trajectory()
        ax.plot(own_pos[:, 1], own_pos[:, 0], 'r-', linewidth=1, alpha=0.5)
        
        # Ground truth
        for i, tid in enumerate(loader.target_ids):
            t, pos = loader.get_target_trajectory(tid)
            ax.plot(pos[:, 1], pos[:, 0], '--', color=colors[i], linewidth=2, alpha=0.7)
        
        # Detections
        dets = loader.get_sensor_detections(sensor_id)
        det_positions = []
        for det in dets:
            piren_pos = det.to_piren_ned()
            if piren_pos is not None:
                if piren_pos.ndim == 1:
                    det_positions.append(piren_pos)
                else:
                    for p in piren_pos:
                        det_positions.append(p)
        
        if det_positions:
            det_positions = np.array(det_positions)
            ax.scatter(det_positions[:, 1], det_positions[:, 0], 
                      c='blue', s=8, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_sensors, 4):
        axes[idx].axis('off')
    
    plt.suptitle(f"{loader.scenario_name} — All Sensor Detections", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_attack_comparison(benign_loader: ScenarioLoader, 
                           attacked_loader: ScenarioLoader,
                           sensor_id: int,
                           save_path: Optional[str] = None):
    """Compare benign vs attacked detections for a sensor."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
    
    for ax, loader, title in [(axes[0], benign_loader, "Benign"), 
                               (axes[1], attacked_loader, "Attacked")]:
        ax.set_title(f"{title} — {sensor_name}")
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Ground truth
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, tid in enumerate(loader.target_ids):
            t, pos = loader.get_target_trajectory(tid)
            ax.plot(pos[:, 1], pos[:, 0], '--', color=colors[i], linewidth=2, 
                   alpha=0.7, label=f'Target {tid}')
        
        # Detections
        dets = loader.get_sensor_detections(sensor_id)
        det_positions = []
        for det in dets:
            piren_pos = det.to_piren_ned()
            if piren_pos is not None:
                if piren_pos.ndim == 1:
                    det_positions.append(piren_pos)
                else:
                    for p in piren_pos:
                        det_positions.append(p)
        
        if det_positions:
            det_positions = np.array(det_positions)
            ax.scatter(det_positions[:, 1], det_positions[:, 0], 
                      c='blue', s=10, alpha=0.4, label='Detections')
        
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_detection_timeline(loader: ScenarioLoader, save_path: Optional[str] = None):
    """Plot detection availability over time for each sensor."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sensor_ids = loader.sensor_ids
    colors = {1: 'green', 2: 'blue', 3: 'orange', 4: 'purple'}
    
    for sensor_id in sensor_ids:
        dets = loader.get_sensor_detections(sensor_id)
        times = [d.time for d in dets]
        y = [sensor_id] * len(times)
        ax.scatter(times, y, c=colors.get(sensor_id, 'gray'), 
                  s=20, alpha=0.6, label=SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}"))
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Sensor ID")
    ax.set_title(f"{loader.scenario_name} — Detection Timeline")
    ax.set_yticks(sensor_ids)
    ax.set_yticklabels([SENSOR_NAMES.get(sid, f"Sensor {sid}") for sid in sensor_ids])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_attack_impact_bearings(benign_dets: List[Detection],
                                attacked_dets: List[Detection],
                                sensor_id: int,
                                save_path: Optional[str] = None):
    """Plot bearing values over time: benign vs attacked."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
    
    # Benign bearings
    benign_times = [d.time for d in benign_dets if d.sensor_id == sensor_id]
    benign_bearings = [np.rad2deg(np.mean(d.measurement)) for d in benign_dets if d.sensor_id == sensor_id]
    
    # Attacked bearings
    attacked_times = [d.time for d in attacked_dets if d.sensor_id == sensor_id]
    attacked_bearings = [np.rad2deg(np.mean(d.measurement)) for d in attacked_dets if d.sensor_id == sensor_id]
    
    # Plot benign
    axes[0].plot(benign_times, benign_bearings, 'b-', linewidth=1, alpha=0.7, label='Benign')
    axes[0].set_ylabel("Bearing [°]")
    axes[0].set_title(f"{sensor_name} — Benign Bearings")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot attacked
    axes[1].plot(attacked_times, attacked_bearings, 'r-', linewidth=1, alpha=0.7, label='Attacked')
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Bearing [°]")
    axes[1].set_title(f"{sensor_name} — Attacked Bearings")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_defense_recovery(benign_dets: List[Detection],
                          attacked_dets: List[Detection],
                          defended_dets: List[Detection],
                          sensor_id: int,
                          save_path: Optional[str] = None):
    """Plot bearing values: benign vs attacked vs defended."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sensor_name = SENSOR_NAMES.get(sensor_id, f"Sensor {sensor_id}")
    
    # Filter by sensor
    b_dets = [d for d in benign_dets if d.sensor_id == sensor_id]
    a_dets = [d for d in attacked_dets if d.sensor_id == sensor_id]
    d_dets = [d for d in defended_dets if d.sensor_id == sensor_id]
    
    # Plot
    if b_dets:
        times = [d.time for d in b_dets]
        bearings = [np.rad2deg(np.mean(d.measurement)) for d in b_dets]
        ax.plot(times, bearings, 'b-', linewidth=1.5, alpha=0.8, label='Benign')
    
    if a_dets:
        times = [d.time for d in a_dets]
        bearings = [np.rad2deg(np.mean(d.measurement)) for d in a_dets]
        ax.plot(times, bearings, 'r--', linewidth=1.5, alpha=0.8, label='Attacked')
    
    if d_dets:
        times = [d.time for d in d_dets]
        bearings = [np.rad2deg(np.mean(d.measurement)) for d in d_dets]
        ax.plot(times, bearings, 'g:', linewidth=2, alpha=0.9, label='Defended')
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bearing [°]")
    ax.set_title(f"{sensor_name} — Attack Impact & Defense Recovery")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict],
                            save_path: Optional[str] = None):
    """Plot bar chart comparing metrics across conditions.
    
    Args:
        metrics_dict: {condition: {sensor: {metric: value}}}
            e.g., {'Benign': {'IR': {'det_prob': 0.42}},
                   'Attacked': {'IR': {'det_prob': 0.04}},
                   'Defended': {'IR': {'det_prob': 0.41}}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metric_names = ['detection_probability', 'false_alarm_rate', 'rmse']
    metric_labels = ['Detection Probability', 'False Alarm Rate', 'RMSE (m)']
    
    conditions = list(metrics_dict.keys())
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    
    for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        
        # Get all sensors
        sensors = set()
        for cond_data in metrics_dict.values():
            sensors.update(cond_data.keys())
        sensors = sorted(sensors)
        
        x = np.arange(len(sensors))
        width = 0.25
        
        for i, condition in enumerate(conditions):
            values = []
            for sensor in sensors:
                val = metrics_dict[condition].get(sensor, {}).get(metric, 0)
                values.append(val)
            ax.bar(x + i * width, values, width, label=condition, color=colors[i % len(colors)])
        
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x + width * (len(conditions) - 1) / 2)
        ax.set_xticklabels(sensors, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Attack Impact & Defense Recovery Metrics", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_certified_radius(certified_results: List[Tuple[float, float]],
                          save_path: Optional[str] = None):
    """Plot certified radius distribution.
    
    Args:
        certified_results: List of (certified_radius, attack_epsilon) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    radii = [r for r, _ in certified_results]
    epsilons = [e for _, e in certified_results]
    
    # Scatter plot
    ax.scatter(epsilons, radii, c='blue', alpha=0.5, s=20)
    
    # Diagonal line (radius = epsilon)
    max_val = max(max(radii), max(epsilons))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Certified boundary')
    
    # Fill certified region
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val], 
                    alpha=0.1, color='green', label='Certified robust')
    
    ax.set_xlabel("Attack Epsilon [rad]")
    ax.set_ylabel("Certified Radius [rad]")
    ax.set_title("Certified Robustness: Radius vs Attack Strength")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    import sys
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.data_loader import ScenarioLoader
    
    loader = ScenarioLoader("scenario2", data_dir="/Volumes/Data/maritime-adversarial-ai/data/sensor_fusion_dataset")
    
    # Generate all plots
    results_dir = Path("/Volumes/Data/maritime-adversarial-ai/results")
    results_dir.mkdir(exist_ok=True)
    
    print("Generating overview plot...")
    plot_scenario_overview(loader, save_path=results_dir / "scenario2_overview.png")
    
    print("Generating all sensors plot...")
    plot_all_sensors(loader, save_path=results_dir / "scenario2_all_sensors.png")
    
    print("Generating detection timeline...")
    plot_detection_timeline(loader, save_path=results_dir / "scenario2_timeline.png")
    
    print(f"Plots saved to {results_dir}")
    plt.close('all')
