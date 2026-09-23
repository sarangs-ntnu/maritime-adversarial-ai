"""
Run Adversarial AI Pipeline on All 4 Scenarios
==============================================

Comprehensive evaluation across all maritime scenarios:
    - scenario1: Crossing from starboard
    - scenario2: Head-on crossing
    - scenario3: Overtaking
    - scenario4: Multi-target complex

Generates aggregated results and comparison tables.
"""

import sys
from pathlib import Path

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import json
from collections import defaultdict
from datetime import datetime

from data_loader import ScenarioLoader, SENSOR_NAMES
from attacks.camera_attacks import CameraAdversarialAttacker, AttackConfig as CamConfig, AttackType as CamType
from attacks.radar_lidar_attacks import PointCloudAttacker, PointCloudAttackConfig as PCConfig, PointCloudAttackType as PCType
from attacks.fusion_attacks import FusionAttacker, FusionAttackConfig as FusionConfig, FusionAttackType as FusionType, JIPDASimulator
from evaluation.metrics import DetectionEvaluator, AttackEvaluator
from defenses.defense_mechanisms import DefensePipeline, DefenseConfig, DefenseType
from pipeline import AdversarialPipeline, PipelineConfig


SCENARIOS = ['scenario1', 'scenario2', 'scenario3', 'scenario4']


def evaluate_scenario(scenario_name: str, data_dir: str = 'data/sensor_fusion_dataset'):
    """Run full evaluation on a single scenario."""
    print(f"\n{'='*70}")
    print(f"Evaluating {scenario_name}")
    print(f"{'='*70}")
    
    try:
        loader = ScenarioLoader(scenario_name, data_dir)
    except Exception as e:
        print(f"  ERROR loading {scenario_name}: {e}")
        return None
    
    dets = loader.detections.copy()
    
    # --- Benign Metrics ---
    benign_metrics = compute_metrics(dets, loader)
    
    # --- Camera Attack (FGSM) ---
    cam_config = CamConfig(CamType.FGSM, epsilon=0.1)
    cam_attacker = CameraAdversarialAttacker(cam_config)
    
    cam_attacked = []
    for d in dets:
        if d.sensor_id in [3, 4]:
            cam_attacked.append(cam_attacker.attack(d))
        else:
            cam_attacked.append(d)
    
    cam_metrics = compute_metrics(cam_attacked, loader)
    
    # --- Point Cloud Attack (Ghost Injection) ---
    pc_config = PCConfig(PCType.GHOST_INJECTION, epsilon=5.0)
    pc_attacker = PointCloudAttacker(pc_config)
    
    pc_attacked = []
    for d in dets:
        if d.sensor_id in [1, 2]:
            pc_attacked.append(pc_attacker.attack(d))
        else:
            pc_attacked.append(d)
    
    pc_metrics = compute_metrics(pc_attacked, loader)
    
    # --- Combined Attack ---
    combined_attacked = []
    for d in dets:
        if d.sensor_id in [3, 4]:
            combined_attacked.append(cam_attacker.attack(d))
        elif d.sensor_id in [1, 2]:
            combined_attacked.append(pc_attacker.attack(d))
        else:
            combined_attacked.append(d)
    
    combined_metrics = compute_metrics(combined_attacked, loader)
    
    # --- Defense (Temporal Consistency) ---
    defense_config = DefenseConfig(DefenseType.ALL)
    defense = DefensePipeline(defense_config)
    defense.temporal_checker.train(dets)
    defended_dets = defense.defend(combined_attacked)
    
    defense_metrics = compute_metrics(defended_dets, loader)
    
    # --- Tracking ---
    tracker_benign = JIPDASimulator()
    benign_tracks = tracker_benign.track(dets)
    
    tracker_attacked = JIPDASimulator()
    attacked_tracks = tracker_attacked.track(combined_attacked)
    
    tracker_defended = JIPDASimulator()
    defended_tracks = tracker_defended.track(defended_dets)
    
    results = {
        'scenario': scenario_name,
        'num_targets': len(loader.target_ids),
        'num_detections': len(dets),
        'benign': {
            'tracks': len(benign_tracks),
            'metrics': benign_metrics
        },
        'camera_attack': {
            'tracks': len(tracker_attacked.track(cam_attacked)),
            'metrics': cam_metrics
        },
        'pc_attack': {
            'tracks': len(tracker_attacked.track(pc_attacked)),
            'metrics': pc_metrics
        },
        'combined_attack': {
            'tracks': len(attacked_tracks),
            'metrics': combined_metrics
        },
        'defended': {
            'tracks': len(defended_tracks),
            'metrics': defense_metrics
        }
    }
    
    print(f"  Benign tracks: {len(benign_tracks)}")
    print(f"  Attacked tracks: {len(attacked_tracks)}")
    print(f"  Defended tracks: {len(defended_tracks)}")
    
    return results


def compute_metrics(detections, loader):
    """Compute per-sensor metrics."""
    evaluator = DetectionEvaluator(distance_threshold=50.0)
    
    dets_by_time = defaultdict(list)
    for d in detections:
        dets_by_time[d.time].append(d)
    
    gt_by_time = {}
    for i, timestep in enumerate(loader.ground_truth):
        if timestep and i < len(loader.detections):
            gt_by_time[loader.detections[i].time] = timestep
    
    metrics = {}
    for sid in [1, 2, 3, 4]:
        total_matched = 0
        total_gt = 0
        total_fa = 0
        total_dets = 0
        errors = []
        
        for t, dets_at_t in dets_by_time.items():
            gt_at_t = gt_by_time.get(t, [])
            if not gt_at_t:
                continue
            s_dets = [d for d in dets_at_t if d.sensor_id == sid]
            if not s_dets:
                continue
            m = evaluator.evaluate(s_dets, gt_at_t, sid)
            if m.rmse_position > 0:
                errors.append(m.rmse_position)
            total_matched += int(m.detection_probability * len(gt_at_t))
            total_gt += len(gt_at_t)
            total_fa += int(m.false_alarm_rate * len(s_dets))
            total_dets += len(s_dets)
        
        metrics[SENSOR_NAMES[sid]] = {
            'detection_probability': total_matched / max(total_gt, 1),
            'false_alarm_rate': total_fa / max(total_dets, 1),
            'rmse': np.mean(errors) if errors else 0
        }
    
    return metrics


def print_comparison_table(all_results):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("CROSS-SCENARIO COMPARISON")
    print("="*100)
    
    # Table 1: Track counts
    print("\n--- Track Counts ---")
    print(f"{'Scenario':<15} {'Targets':<10} {'Benign':<10} {'Camera':<10} {'PC':<10} {'Combined':<10} {'Defended':<10}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['scenario']:<15} {r['num_targets']:<10} "
              f"{r['benign']['tracks']:<10} {r['camera_attack']['tracks']:<10} "
              f"{r['pc_attack']['tracks']:<10} {r['combined_attack']['tracks']:<10} "
              f"{r['defended']['tracks']:<10}")
    
    # Table 2: Detection Probability (IR Camera)
    print("\n--- IR Camera Detection Probability ---")
    print(f"{'Scenario':<15} {'Benign':<12} {'Camera Att':<12} {'Combined':<12} {'Defended':<12} {'Recovery':<12}")
    print("-" * 77)
    for r in all_results:
        ir_key = 'IR Camera' if 'IR Camera' in r['benign']['metrics'] else 'IR'
        benign = r['benign']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        attacked = r['combined_attack']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        defended = r['defended']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        recovery = (defended / benign * 100) if benign > 0 else 0
        print(f"{r['scenario']:<15} {benign:<12.4f} {attacked:<12.4f} "
              f"{attacked:<12.4f} {defended:<12.4f} {recovery:<11.1f}%")
    
    # Table 3: Detection Probability (Lidar)
    print("\n--- Lidar Detection Probability ---")
    print(f"{'Scenario':<15} {'Benign':<12} {'PC Att':<12} {'Combined':<12} {'Defended':<12}")
    print("-" * 63)
    for r in all_results:
        lidar_key = 'Lidar' if 'Lidar' in r['benign']['metrics'] else 'LIDAR'
        benign = r['benign']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        attacked = r['combined_attack']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        defended = r['defended']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        print(f"{r['scenario']:<15} {benign:<12.4f} {attacked:<12.4f} "
              f"{attacked:<12.4f} {defended:<12.4f}")


def main():
    """Run evaluation on all scenarios."""
    print("="*100)
    print("MARITIME ADVERSARIAL AI - CROSS-SCENARIO EVALUATION")
    print("="*100)
    
    all_results = []
    
    for scenario in SCENARIOS:
        result = evaluate_scenario(scenario)
        if result:
            all_results.append(result)
    
    # Print comparison
    print_comparison_table(all_results)
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"all_scenarios_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*100}")
    print(f"Results saved to {output_file}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
