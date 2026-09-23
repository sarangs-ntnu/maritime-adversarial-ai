"""
Adversarial Attacks on Radar/Lidar Branch (Active Sensors)
==========================================================

Attacks on point cloud data from radar and lidar sensors.
Targets the clustering algorithm (single-link hierarchical clustering)
and the land masking pipeline.

Key vulnerabilities from the paper:
    - Clustering uses Euclidean distance threshold T
    - Minimum cluster size of 5 points
    - Land returns filtered by occupancy grid

Attack Types:
    - Cluster Splitting: Insert sparse points to break density-based grouping
    - Cluster Merging: Add points between targets to force single cluster
    - Ghost Injection: Add fake point clusters to create false tracks
    - Point Suppression: Remove critical points to drop cluster below threshold
    - Land Mask Poisoning: Corrupt land map to let shore returns through
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


class PointCloudAttackType(Enum):
    CLUSTER_SPLIT = "cluster_split"
    CLUSTER_MERGE = "cluster_merge"
    GHOST_INJECTION = "ghost_injection"
    POINT_SUPPRESSION = "point_suppression"
    NOISE_FLOOR = "noise_floor"
    RANDOM_PERTURBATION = "random_perturbation"


@dataclass
class PointCloudAttackConfig:
    """Configuration for point cloud adversarial attacks."""
    attack_type: PointCloudAttackType
    epsilon: float = 5.0          # Max point perturbation (meters)
    num_ghost_points: int = 10    # Points to inject for ghost clusters
    ghost_spread: float = 3.0     # Spread of ghost cluster (meters)
    split_distance: float = 8.0   # Distance to insert splitting points
    merge_distance: float = 5.0   # Distance between targets to merge
    suppression_fraction: float = 0.3  # Fraction of points to remove
    noise_std: float = 2.0        # Standard deviation of Gaussian noise
    target_cluster_size: int = 5  # Minimum cluster size threshold


class PointCloudAttacker:
    """Adversarial attack generator for radar/lidar point clouds."""
    
    def __init__(self, config: PointCloudAttackConfig):
        self.config = config
    
    def attack(self, detection: Detection,
               target_positions: Optional[List[np.ndarray]] = None) -> Detection:
        """Apply adversarial perturbation to a radar/lidar detection.
        
        Args:
            detection: Original detection with point cloud measurement
            target_positions: Known or estimated target positions for targeted attacks
        
        Returns:
            Perturbed detection
        """
        if detection.sensor_id not in [1, 2]:
            raise ValueError(f"Point cloud attacks only for sensors 1 (Lidar) and 2 (Radar), got {detection.sensor_id}")
        
        original_measurement = detection.measurement.copy()
        
        if self.config.attack_type == PointCloudAttackType.CLUSTER_SPLIT:
            perturbed = self._cluster_split_attack(original_measurement, target_positions)
        elif self.config.attack_type == PointCloudAttackType.CLUSTER_MERGE:
            perturbed = self._cluster_merge_attack(original_measurement, target_positions)
        elif self.config.attack_type == PointCloudAttackType.GHOST_INJECTION:
            perturbed = self._ghost_injection_attack(original_measurement)
        elif self.config.attack_type == PointCloudAttackType.POINT_SUPPRESSION:
            perturbed = self._point_suppression_attack(original_measurement)
        elif self.config.attack_type == PointCloudAttackType.NOISE_FLOOR:
            perturbed = self._noise_floor_attack(original_measurement)
        elif self.config.attack_type == PointCloudAttackType.RANDOM_PERTURBATION:
            perturbed = self._random_perturbation_attack(original_measurement)
        else:
            perturbed = original_measurement
        
        perturbed_detection = Detection(
            sensor_id=detection.sensor_id,
            time=detection.time,
            ownship_position=detection.ownship_position.copy(),
            measurement=perturbed
        )
        
        return perturbed_detection
    
    def _cluster_split_attack(self, measurement: np.ndarray,
                              target_positions: Optional[List[np.ndarray]]) -> np.ndarray:
        """Split a single target cluster into multiple sub-threshold clusters.
        
        Strategy: Insert sparse points around the target to prevent
        density-based grouping. Points are placed at distance > T from
        existing points but within sensor range.
        """
        if len(measurement) == 0:
            return measurement
        
        # Handle both single point and multiple points
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        # Calculate cluster centroid
        centroid = np.mean(points, axis=0)
        
        # Generate splitting points: placed between centroid and edges
        # at distance > T from existing points
        n_split = max(3, len(points) // 3)
        split_points = []
        
        for _ in range(n_split):
            # Random direction from centroid (match centroid dimension)
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            if len(centroid) > 2:
                # Pad with zeros to match centroid dimension
                direction = np.pad(direction, (0, len(centroid) - 2), mode='constant')
            
            # Place at distance > T from centroid but < 2*T
            distance = self.config.split_distance * np.random.uniform(1.0, 1.8)
            split_point = centroid + distance * direction
            split_points.append(split_point)
        
        split_points = np.array(split_points)
        
        # Combine original points with splitting points
        perturbed = np.vstack([points, split_points])
        
        return perturbed
    
    def _cluster_merge_attack(self, measurement: np.ndarray,
                              target_positions: Optional[List[np.ndarray]]) -> np.ndarray:
        """Merge two nearby target clusters into one.
        
        Strategy: Add points between two targets to create a bridge
        that the clustering algorithm will group together.
        """
        if len(measurement) == 0:
            return measurement
        
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        # If we have target positions, create bridge between them
        if target_positions is not None and len(target_positions) >= 2:
            pos1 = target_positions[0][:2]  # Use only x,y
            pos2 = target_positions[1][:2]
            
            # Generate bridge points between targets
            n_bridge = 5
            bridge_points = []
            for i in range(1, n_bridge + 1):
                t = i / (n_bridge + 1)
                bridge_point = (1 - t) * pos1 + t * pos2
                # Add small noise
                bridge_point += np.random.randn(2) * 0.5
                if points.shape[1] > 2:
                    bridge_point = np.append(bridge_point, 0)
                bridge_points.append(bridge_point)
            
            bridge_points = np.array(bridge_points)
            perturbed = np.vstack([points, bridge_points])
        else:
            # Without target positions, add points in random directions
            # from existing points
            n_bridge = 5
            bridge_points = []
            for _ in range(n_bridge):
                # Pick random existing point
                idx = np.random.randint(len(points))
                base_point = points[idx]
                
                # Add point at merge_distance
                direction = np.random.randn(len(base_point))
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                bridge_point = base_point + self.config.merge_distance * direction
                bridge_points.append(bridge_point)
            
            bridge_points = np.array(bridge_points)
            perturbed = np.vstack([points, bridge_points])
        
        return perturbed
    
    def _ghost_injection_attack(self, measurement: np.ndarray) -> np.ndarray:
        """Inject fake point clusters to create false tracks.
        
        Strategy: Create a dense cluster of points at a location
        where no target exists. The clustering algorithm will
        produce a centroid that the tracker will initialize as a new track.
        """
        if len(measurement) == 0:
            points = np.empty((0, 2))
        elif measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        # Determine ghost cluster location
        # Place it at a plausible distance from existing points
        if len(points) > 0:
            centroid = np.mean(points, axis=0)
            ndim = len(centroid)
            # Offset by random direction
            offset_distance = 50 + np.random.uniform(0, 100)  # 50-150m away
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            if ndim > 2:
                # Pad direction with zeros for extra dimensions
                direction = np.pad(direction, (0, ndim - 2), mode='constant')
            elif ndim == 1:
                direction = np.array([1.0])
            ghost_center = centroid + offset_distance * direction
        else:
            ghost_center = np.random.randn(2) * 100
        
        # Generate ghost cluster points (dense cluster)
        ghost_points = []
        for _ in range(self.config.num_ghost_points):
            point = ghost_center + np.random.randn(len(ghost_center)) * self.config.ghost_spread
            ghost_points.append(point)
        
        ghost_points = np.array(ghost_points)
        
        # Combine with original points
        if len(points) == 0:
            perturbed = ghost_points
        else:
            perturbed = np.vstack([points, ghost_points])
        
        return perturbed
    
    def _point_suppression_attack(self, measurement: np.ndarray) -> np.ndarray:
        """Remove critical points to drop cluster below minimum size.
        
        Strategy: Randomly remove points from the cluster until
        the remaining count is below the minimum threshold (5).
        """
        if len(measurement) == 0:
            return measurement
        
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        n_points = len(points)
        if n_points <= self.config.target_cluster_size:
            # Already below threshold, remove all
            return np.empty((0, points.shape[1]))
        
        # Calculate how many points to remove
        n_remove = int(n_points * self.config.suppression_fraction)
        n_remove = max(n_remove, n_points - self.config.target_cluster_size + 1)
        
        # Remove points (random selection)
        keep_indices = np.random.choice(n_points, size=n_points - n_remove, replace=False)
        perturbed = points[keep_indices]
        
        return perturbed
    
    def _noise_floor_attack(self, measurement: np.ndarray) -> np.ndarray:
        """Raise noise floor to push true returns below detection threshold.
        
        Strategy: Add Gaussian noise to all points, increasing the
        variance and making it harder for the clustering algorithm
        to find coherent clusters.
        """
        if len(measurement) == 0:
            return measurement
        
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        # Add Gaussian noise
        noise = np.random.randn(*points.shape) * self.config.noise_std
        perturbed = points + noise
        
        return perturbed
    
    def _random_perturbation_attack(self, measurement: np.ndarray) -> np.ndarray:
        """Apply random perturbation within epsilon ball to each point."""
        if len(measurement) == 0:
            return measurement
        
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        # Random perturbation for each point
        perturbations = np.random.uniform(-self.config.epsilon, self.config.epsilon, points.shape)
        perturbed = points + perturbations
        
        return perturbed


class ClusteringSimulator:
    """Simulate the clustering algorithm from the paper.
    
    Single-link hierarchical clustering with:
        - Euclidean distance threshold T
        - Minimum cluster size of 5 points
        - k-d tree optimization (not implemented here for simplicity)
    """
    
    def __init__(self, distance_threshold: float = 5.0, min_cluster_size: int = 5):
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
    
    def cluster(self, points: np.ndarray) -> List[Dict]:
        """Cluster points and return cluster centroids.
        
        Returns:
            List of dicts with 'centroid', 'size', 'points'
        """
        if len(points) == 0:
            return []
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        n_points = len(points)
        assigned = np.zeros(n_points, dtype=bool)
        clusters = []
        
        for i in range(n_points):
            if assigned[i]:
                continue
            
            # Start new cluster with point i
            cluster_indices = [i]
            assigned[i] = True
            
            # Grow cluster: add points within threshold of any cluster member
            changed = True
            while changed:
                changed = False
                for j in range(n_points):
                    if assigned[j]:
                        continue
                    
                    # Check if point j is within threshold of any cluster member
                    for idx in cluster_indices:
                        dist = np.linalg.norm(points[j] - points[idx])
                        if dist <= self.distance_threshold:
                            cluster_indices.append(j)
                            assigned[j] = True
                            changed = True
                            break
            
            # Only keep clusters above minimum size
            if len(cluster_indices) >= self.min_cluster_size:
                cluster_points = points[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                clusters.append({
                    'centroid': centroid,
                    'size': len(cluster_indices),
                    'points': cluster_points,
                    'indices': cluster_indices
                })
        
        return clusters
    
    def evaluate_attack(self, original_points: np.ndarray,
                        attacked_points: np.ndarray) -> Dict[str, any]:
        """Evaluate how the attack affected clustering.
        
        Returns:
            Dict with clustering metrics before and after attack
        """
        orig_clusters = self.cluster(original_points)
        att_clusters = self.cluster(attacked_points)
        
        return {
            'original_clusters': len(orig_clusters),
            'attacked_clusters': len(att_clusters),
            'original_total_points': len(original_points) if original_points.ndim > 1 else (1 if len(original_points) > 0 else 0),
            'attacked_total_points': len(attacked_points) if attacked_points.ndim > 1 else (1 if len(attacked_points) > 0 else 0),
            'cluster_change': len(att_clusters) - len(orig_clusters),
            'original_centroids': [c['centroid'] for c in orig_clusters],
            'attacked_centroids': [c['centroid'] for c in att_clusters],
        }


def apply_point_cloud_attacks_to_scenario(
    loader: ScenarioLoader,
    sensor_id: int,
    config: PointCloudAttackConfig,
    attack_fraction: float = 1.0,
    use_ground_truth: bool = True
) -> List[Detection]:
    """Apply point cloud attacks to a fraction of detections in a scenario.
    
    Args:
        loader: ScenarioLoader with original data
        sensor_id: Sensor to attack (1=Lidar, 2=Radar)
        config: Attack configuration
        attack_fraction: Fraction of detections to attack
        use_ground_truth: Whether to use ground truth for targeted attacks
    
    Returns:
        List of all detections (attacked + benign)
    """
    attacker = PointCloudAttacker(config)
    
    # Get sensor detections
    sensor_dets = loader.get_sensor_detections(sensor_id)
    
    # Select which detections to attack
    n_attack = int(len(sensor_dets) * attack_fraction)
    attack_indices = np.random.choice(len(sensor_dets), size=n_attack, replace=False)
    attack_set = set(attack_indices)
    
    perturbed_detections = []
    for i, det in enumerate(sensor_dets):
        if i in attack_set:
            # Get target positions for targeted attacks
            target_positions = None
            if use_ground_truth:
                gt = loader.get_ground_truth_at_time(det.time, tolerance=0.2)
                if gt:
                    target_positions = [g.position[:2] for g in gt]
            
            perturbed = attacker.attack(det, target_positions)
            perturbed_detections.append(perturbed)
        else:
            perturbed_detections.append(det)
    
    return perturbed_detections


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
    print("RADAR/LIDAR POINT CLOUD ADVERSARIAL ATTACK DEMONSTRATION")
    print("=" * 70)
    
    # Test on Lidar (sensor 1)
    lidar_dets = loader.get_sensor_detections(1)
    print(f"\nLidar detections: {len(lidar_dets)}")
    
    # Find a detection with multiple points
    sample_det = None
    for det in lidar_dets:
        if det.measurement.ndim > 1 and len(det.measurement) > 1:
            sample_det = det
            break
    
    if sample_det is None:
        sample_det = lidar_dets[0]
    
    print(f"Sample measurement shape: {sample_det.measurement.shape}")
    print(f"Sample measurement:\n{sample_det.measurement[:5]}")
    
    # Initialize clustering simulator
    simulator = ClusteringSimulator(distance_threshold=5.0, min_cluster_size=5)
    
    attack_configs = [
        PointCloudAttackConfig(PointCloudAttackType.CLUSTER_SPLIT, split_distance=8.0),
        PointCloudAttackConfig(PointCloudAttackType.GHOST_INJECTION, num_ghost_points=10),
        PointCloudAttackConfig(PointCloudAttackType.POINT_SUPPRESSION, suppression_fraction=0.5),
        PointCloudAttackConfig(PointCloudAttackType.NOISE_FLOOR, noise_std=3.0),
        PointCloudAttackConfig(PointCloudAttackType.RANDOM_PERTURBATION, epsilon=5.0),
    ]
    
    for config in attack_configs:
        print(f"\n--- {config.attack_type.value.upper()} Attack ---")
        attacker = PointCloudAttacker(config)
        
        gt = loader.get_ground_truth_at_time(sample_det.time, tolerance=0.2)
        target_positions = [g.position[:2] for g in gt] if gt else None
        
        attacked = attacker.attack(sample_det, target_positions)
        
        print(f"  Original points: {len(sample_det.measurement)}")
        print(f"  Attacked points: {len(attacked.measurement)}")
        
        # Evaluate clustering effect
        eval_result = simulator.evaluate_attack(sample_det.measurement, attacked.measurement)
        print(f"  Original clusters: {eval_result['original_clusters']}")
        print(f"  Attacked clusters: {eval_result['attacked_clusters']}")
        print(f"  Cluster change: {eval_result['cluster_change']:+d}")
    
    # Test on Radar (sensor 2)
    print("\n" + "=" * 70)
    print("RADAR ATTACKS")
    print("=" * 70)
    
    radar_dets = loader.get_sensor_detections(2)
    print(f"Radar detections: {len(radar_dets)}")
    
    if radar_dets:
        sample_radar = radar_dets[0]
        print(f"Sample measurement shape: {sample_radar.measurement.shape}")
        print(f"Sample measurement:\n{sample_radar.measurement[:3]}")
        
        config = PointCloudAttackConfig(PointCloudAttackType.CLUSTER_MERGE, merge_distance=5.0)
        attacker = PointCloudAttacker(config)
        
        gt = loader.get_ground_truth_at_time(sample_radar.time, tolerance=0.2)
        target_positions = [g.position[:2] for g in gt] if gt else None
        
        attacked = attacker.attack(sample_radar, target_positions)
        print(f"\nMerge Attack:")
        print(f"  Original points: {len(sample_radar.measurement)}")
        print(f"  Attacked points: {len(attacked.measurement)}")
    
    # Full scenario evaluation
    print("\n" + "=" * 70)
    print("FULL SCENARIO ATTACK EVALUATION")
    print("=" * 70)
    
    config = PointCloudAttackConfig(PointCloudAttackType.GHOST_INJECTION, num_ghost_points=15)
    attacked_dets = apply_point_cloud_attacks_to_scenario(loader, 1, config, attack_fraction=1.0)
    
    total_ghost_points = sum(len(d.measurement) for d in attacked_dets)
    original_points = sum(len(d.measurement) for d in lidar_dets)
    
    print(f"\nGhost Injection on Lidar:")
    print(f"  Original total points: {original_points}")
    print(f"  Attacked total points: {total_ghost_points}")
    print(f"  Point increase: {total_ghost_points - original_points} ({(total_ghost_points/original_points - 1)*100:.1f}%)")
