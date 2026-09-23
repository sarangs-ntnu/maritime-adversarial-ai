"""
End-to-End Adversarial AI Pipeline
==================================

Complete pipeline for evaluating adversarial attacks on the maritime
sensor fusion system:
    1. Load scenario data
    2. Apply attacks (camera, radar/lidar, fusion)
    3. Run tracking (JIPDA simulator)
    4. Evaluate metrics (benign vs attacked)
    5. Apply defenses
    6. Generate visualizations
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from data_loader import ScenarioLoader, SENSOR_NAMES
from attacks.camera_attacks import CameraAdversarialAttacker, AttackConfig as CameraAttackConfig, AttackType as CameraAttackType
from attacks.radar_lidar_attacks import PointCloudAttacker, PointCloudAttackConfig, PointCloudAttackType
from attacks.fusion_attacks import FusionAttacker, FusionAttackConfig, FusionAttackType, JIPDASimulator
from evaluation.metrics import DetectionEvaluator, AttackEvaluator, DetectionMetrics
from defenses.defense_mechanisms import DefensePipeline, DefenseConfig, DefenseType


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    scenario_name: str
    data_dir: str
    output_dir: str
    
    # Attack configurations
    camera_attack_type: Optional[CameraAttackType] = None
    camera_epsilon: float = 0.1
    
    pointcloud_attack_type: Optional[PointCloudAttackType] = None
    pointcloud_epsilon: float = 2.0
    
    fusion_attack_type: Optional[FusionAttackType] = None
    
    # Defense configuration
    use_defense: bool = False
    defense_type: DefenseType = DefenseType.INPUT_SANITIZATION
    
    # Evaluation
    compute_tracking: bool = True


class AdversarialPipeline:
    """End-to-end pipeline for adversarial evaluation."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = ScenarioLoader(config.scenario_name, config.data_dir)
        
        # Initialize trackers
        self.benign_tracker = JIPDASimulator()
        self.attacked_tracker = JIPDASimulator()
        
        # Initialize evaluators
        self.detection_eval = DetectionEvaluator()
        self.attack_eval = AttackEvaluator(self.loader)
        
        # Initialize defense
        self.defense = None
        if config.use_defense:
            defense_cfg = DefenseConfig(config.defense_type)
            self.defense = DefensePipeline(defense_cfg)
        
        # Results storage
        self.results = {}
    
    def run_benign(self) -> Dict:
        """Run benign (no attack) pipeline."""
        print("Running benign pipeline...")
        
        detections = self.loader.detections
        
        # Apply defense if enabled
        if self.defense:
            detections = self.defense.defend(detections)
        
        # Run tracker
        tracks = self.benign_tracker.track(detections)
        
        # Evaluate per-timestep
        metrics = self._evaluate_per_timestep(detections)
        
        self.results['benign'] = {
            'tracks': {tid: {'state': t['state'].tolist()} for tid, t in tracks.items()},
            'metrics': metrics,
            'num_detections': len(detections)
        }
        
        return self.results['benign']
    
    def _evaluate_per_timestep(self, detections: List) -> Dict:
        """Evaluate detections per-timestep against ground truth."""
        from collections import defaultdict
        
        # Group detections by time
        dets_by_time = defaultdict(list)
        for d in detections:
            dets_by_time[d.time].append(d)
        
        # Group ground truth by time
        gt_by_time = {}
        for i, timestep in enumerate(self.loader.ground_truth):
            if timestep:
                # Use first detection time as reference for this timestep
                ref_time = self.loader.detections[i].time if i < len(self.loader.detections) else None
                if ref_time:
                    gt_by_time[ref_time] = timestep
        
        # Aggregate metrics across timesteps
        all_metrics = defaultdict(lambda: {'errors': [], 'matched': 0, 'total_gt': 0, 'false_alarms': 0, 'total_dets': 0})
        
        for t in dets_by_time:
            gt_at_t = gt_by_time.get(t, [])
            if not gt_at_t:
                continue
            
            for sensor_id in [1, 2, 3, 4]:
                sensor_dets = [d for d in dets_by_time[t] if d.sensor_id == sensor_id]
                if not sensor_dets:
                    continue
                
                m = self.detection_eval.evaluate(sensor_dets, gt_at_t, sensor_id)
                key = SENSOR_NAMES[sensor_id]
                
                if m.rmse_position > 0:
                    all_metrics[key]['errors'].append(m.rmse_position)
                all_metrics[key]['matched'] += int(m.detection_probability * len(gt_at_t))
                all_metrics[key]['total_gt'] += len(gt_at_t)
                all_metrics[key]['false_alarms'] += int(m.false_alarm_rate * len(sensor_dets))
                all_metrics[key]['total_dets'] += len(sensor_dets)
        
        # Compute final metrics
        result = {}
        for sensor_name in SENSOR_NAMES.values():
            am = all_metrics[sensor_name]
            dm = DetectionMetrics()
            if am['errors']:
                dm.rmse_position = np.mean(am['errors'])
            dm.detection_probability = am['matched'] / max(am['total_gt'], 1)
            dm.false_alarm_rate = am['false_alarms'] / max(am['total_dets'], 1)
            result[sensor_name] = dm.to_dict()
        
        return result
    
    def run_attacked(self) -> Dict:
        """Run attacked pipeline."""
        print("Running attacked pipeline...")
        
        detections = self.loader.detections.copy()
        
        # Apply camera attacks
        if self.config.camera_attack_type:
            cam_config = CameraAttackConfig(
                self.config.camera_attack_type,
                epsilon=self.config.camera_epsilon
            )
            cam_attacker = CameraAdversarialAttacker(cam_config)
            detections = self._apply_camera_attack(detections, cam_attacker)
        
        # Apply point cloud attacks
        if self.config.pointcloud_attack_type:
            pc_config = PointCloudAttackConfig(
                self.config.pointcloud_attack_type,
                epsilon=self.config.pointcloud_epsilon
            )
            pc_attacker = PointCloudAttacker(pc_config)
            detections = self._apply_pointcloud_attack(detections, pc_attacker)
        
        # Apply fusion attacks
        if self.config.fusion_attack_type:
            fusion_config = FusionAttackConfig(
                self.config.fusion_attack_type
            )
            fusion_attacker = FusionAttacker(fusion_config)
            detections = fusion_attacker.attack(detections)
        
        # Apply defense if enabled
        if self.defense:
            detections = self.defense.defend(detections)
        
        # Run tracker
        tracks = self.attacked_tracker.track(detections)
        
        # Evaluate per-timestep
        metrics = self._evaluate_per_timestep(detections)
        
        self.results['attacked'] = {
            'tracks': {tid: {'state': t['state'].tolist()} for tid, t in tracks.items()},
            'metrics': metrics,
            'num_detections': len(detections)
        }
        
        return self.results['attacked']
    
    def _apply_camera_attack(self, detections: List, attacker) -> List:
        """Apply camera attack to passive sensor detections."""
        result = []
        for det in detections:
            if det.sensor_id in [3, 4]:  # IR, EO
                attacked = attacker.attack(det)
                result.append(attacked)
            else:
                result.append(det)
        return result
    
    def _apply_pointcloud_attack(self, detections: List, attacker) -> List:
        """Apply point cloud attack to active sensor detections."""
        result = []
        for det in detections:
            if det.sensor_id in [1, 2]:  # Lidar, Radar
                attacked = attacker.attack(det)
                result.append(attacked)
            else:
                result.append(det)
        return result
    
    def compare(self) -> Dict:
        """Compare benign vs attacked results."""
        if 'benign' not in self.results or 'attacked' not in self.results:
            raise ValueError("Run both benign and attacked pipelines first")
        
        benign = self.results['benign']
        attacked = self.results['attacked']
        
        comparison = {
            'num_tracks_benign': len(benign['tracks']),
            'num_tracks_attacked': len(attacked['tracks']),
            'track_loss': len(benign['tracks']) - len(attacked['tracks']),
            'sensor_metrics': {}
        }
        
        for sensor_name in benign['metrics']:
            b_metrics = benign['metrics'][sensor_name]
            a_metrics = attacked['metrics'][sensor_name]
            
            comparison['sensor_metrics'][sensor_name] = {
                'rmse_benign': b_metrics.get('rmse', 0),
                'rmse_attacked': a_metrics.get('rmse', 0),
                'det_prob_benign': b_metrics.get('detection_probability', 0),
                'det_prob_attacked': a_metrics.get('detection_probability', 0),
                'far_benign': b_metrics.get('false_alarm_rate', 0),
                'far_attacked': a_metrics.get('false_alarm_rate', 0)
            }
        
        self.results['comparison'] = comparison
        return comparison
    
    def save_results(self):
        """Save results to JSON."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable[key] = self._make_serializable(value)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.scenario_name}_{timestamp}_results.json"
        
        with open(output_path / filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Results saved to {output_path / filename}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def run_full_evaluation(scenario_name: str, data_dir: str, output_dir: str):
    """Run full evaluation for a scenario with multiple attack types."""
    
    print("=" * 70)
    print(f"FULL EVALUATION: {scenario_name}")
    print("=" * 70)
    
    # Define attack combinations to test
    attack_combinations = [
        ("No Attack", None, None, None),
        ("Camera FGSM", CameraAttackType.FGSM, None, None),
        ("Camera PGD", CameraAttackType.PGD, None, None),
        ("Point Cloud Ghost", None, PointCloudAttackType.GHOST_INJECTION, None),
        ("Point Cloud Split", None, PointCloudAttackType.CLUSTER_SPLIT, None),
        ("Fusion Existence", None, None, FusionAttackType.EXISTENCE_SUPPRESSION),
        ("Fusion False Track", None, None, FusionAttackType.FALSE_TRACK_INJECTION),
        ("Combined Camera+PC", CameraAttackType.FGSM, PointCloudAttackType.GHOST_INJECTION, None),
    ]
    
    all_results = {}
    
    for attack_name, cam_type, pc_type, fusion_type in attack_combinations:
        print(f"\n--- Testing: {attack_name} ---")
        
        config = PipelineConfig(
            scenario_name=scenario_name,
            data_dir=data_dir,
            output_dir=output_dir,
            camera_attack_type=cam_type,
            pointcloud_attack_type=pc_type,
            fusion_attack_type=fusion_type
        )
        
        pipeline = AdversarialPipeline(config)
        
        # Run benign
        benign_results = pipeline.run_benign()
        
        # Run attacked
        attacked_results = pipeline.run_attacked()
        
        # Compare
        comparison = pipeline.compare()
        
        all_results[attack_name] = {
            'benign_tracks': comparison['num_tracks_benign'],
            'attacked_tracks': comparison['num_tracks_attacked'],
            'track_loss': comparison['track_loss']
        }
        
        print(f"  Benign tracks: {comparison['num_tracks_benign']}")
        print(f"  Attacked tracks: {comparison['num_tracks_attacked']}")
        print(f"  Track loss: {comparison['track_loss']}")
    
    # Save summary
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / f"{scenario_name}_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSummary saved to {output_path / f'{scenario_name}_summary.json'}")
    
    return all_results


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    DATA_DIR = "/Volumes/Data/maritime-adversarial-ai/data/sensor_fusion_dataset"
    OUTPUT_DIR = "/Volumes/Data/maritime-adversarial-ai/results"
    
    # Run full evaluation on scenario2
    results = run_full_evaluation("scenario2", DATA_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    for attack_name, metrics in results.items():
        print(f"\n{attack_name}:")
        print(f"  Track Loss: {metrics['track_loss']}")
