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
