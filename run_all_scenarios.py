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
    """Run full evaluation on a single scenario with all attack types."""
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
    
    # --- Camera Attacks ---
    cam_config = CamConfig(CamType.FGSM, epsilon=0.1)
    cam_attacker = CameraAdversarialAttacker(cam_config)
    
    cam_attacked = []
    for d in dets:
        if d.sensor_id in [3, 4]:
            cam_attacked.append(cam_attacker.attack(d))
        else:
            cam_attacked.append(d)
    
    cam_metrics = compute_metrics(cam_attacked, loader)
    
    # --- Point Cloud Attacks ---
    pc_config = PCConfig(PCType.GHOST_INJECTION, epsilon=5.0)
    pc_attacker = PointCloudAttacker(pc_config)
    
    pc_attacked = []
    for d in dets:
        if d.sensor_id in [1, 2]:
            pc_attacked.append(pc_attacker.attack(d))
        else:
            pc_attacked.append(d)
    
    pc_metrics = compute_metrics(pc_attacked, loader)
    
    # --- Combined Sensor Attack ---
    combined_attacked = []
    for d in dets:
        if d.sensor_id in [3, 4]:
            combined_attacked.append(cam_attacker.attack(d))
        elif d.sensor_id in [1, 2]:
            combined_attacked.append(pc_attacker.attack(d))
        else:
            combined_attacked.append(d)
    
    combined_metrics = compute_metrics(combined_attacked, loader)
    
    # --- Fusion Attacks ---
    fusion_results = {}
    fusion_attack_types = [
        FusionType.EXISTENCE_SUPPRESSION,
        FusionType.ASSOCIATION_CONFUSION,
        FusionType.CROSS_SENSOR_CONSISTENCY,
        FusionType.FALSE_TRACK_INJECTION,
        FusionType.SENSOR_DOS,
        FusionType.TRACK_MERGE_MANIPULATION,
        FusionType.TRACK_DELETION,
        FusionType.TRACK_SWAP,
        FusionType.STEALTHY_DEGRADATION,
    ]
    
    for fat in fusion_attack_types:
        try:
            f_config = FusionConfig(fat)
            f_attacker = FusionAttacker(f_config)
            f_attacked = f_attacker.attack_scenario(loader)
            
            f_attacked_list = []
            for t in sorted(f_attacked.keys()):
                f_attacked_list.extend(f_attacked[t])
            
            f_metrics = compute_metrics(f_attacked_list, loader)
            
            tracker = JIPDASimulator()
            f_tracks = tracker.track(f_attacked_list)
            
            fusion_results[fat.value] = {
                'tracks': len(f_tracks),
                'metrics': f_metrics
            }
        except Exception as e:
            print(f"    Warning: Fusion attack {fat.value} failed: {e}")
            fusion_results[fat.value] = {'tracks': 0, 'metrics': {}}
    
    # --- Defense: ALL ---
    defense_config = DefenseConfig(DefenseType.ALL)
    defense = DefensePipeline(defense_config)
    defense.temporal_checker.train(dets)
    defended_dets = defense.defend(combined_attacked)
    defense_metrics = compute_metrics(defended_dets, loader)
    
    # --- Defense: Adversarial Training ---
    adv_train_config = DefenseConfig(DefenseType.ADVERSARIAL_TRAINING)
    adv_defense = DefensePipeline(adv_train_config)
    adv_defense.train_adversarial(dets, combined_attacked)
    adv_defended_dets = adv_defense.defend(combined_attacked)
    adv_defense_metrics = compute_metrics(adv_defended_dets, loader)
    
    # --- Tracking ---
    tracker_benign = JIPDASimulator()
    benign_tracks = tracker_benign.track(dets)
    
    tracker_attacked = JIPDASimulator()
    attacked_tracks = tracker_attacked.track(combined_attacked)
    
    tracker_defended = JIPDASimulator()
    defended_tracks = tracker_defended.track(defended_dets)
    
    tracker_adv = JIPDASimulator()
    adv_defended_tracks = tracker_adv.track(adv_defended_dets)
    
    # --- Statistical Significance ---
    from defenses.defense_mechanisms import StatisticalSignificance
    tester = StatisticalSignificance(confidence_level=0.95)
    
    sig_results = {}
    for sid in [3, 4]:  # Camera sensors
        benign_series = collect_dp_series(dets, sid, loader)
        attacked_series = collect_dp_series(combined_attacked, sid, loader)
        defended_series = collect_dp_series(defended_dets, sid, loader)
        
        if len(benign_series) > 1:
            sig_results[SENSOR_NAMES[sid]] = tester.compare_three_conditions(
                benign_series, attacked_series, defended_series
            )
    
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
        'fusion_attacks': fusion_results,
        'defended_all': {
            'tracks': len(defended_tracks),
            'metrics': defense_metrics
        },
        'defended_adv_training': {
            'tracks': len(adv_defended_tracks),
            'metrics': adv_defense_metrics
        },
        'statistical_significance': sig_results
    }
    
    print(f"  Benign tracks: {len(benign_tracks)}")
    print(f"  Attacked tracks: {len(attacked_tracks)}")
    print(f"  Defended (ALL) tracks: {len(defended_tracks)}")
    print(f"  Defended (AdvTrain) tracks: {len(adv_defended_tracks)}")
    
    return results


def collect_dp_series(detections, sensor_id, loader):
    """Collect per-timestep detection probabilities for statistical testing."""
    evaluator = DetectionEvaluator(distance_threshold=50.0)
    
    dets_by_time = defaultdict(list)
    for d in detections:
        dets_by_time[d.time].append(d)
    
    gt_by_time = {}
    for i, timestep in enumerate(loader.ground_truth):
        if timestep and i < len(loader.detections):
            gt_by_time[loader.detections[i].time] = timestep
    
    dps = []
    for t in sorted(dets_by_time.keys()):
        gt_at_t = gt_by_time.get(t, [])
        if not gt_at_t:
            continue
        s_dets = [d for d in dets_by_time[t] if d.sensor_id == sensor_id]
        if not s_dets:
            continue
        m = evaluator.evaluate(s_dets, gt_at_t, sensor_id)
        dps.append(m.detection_probability)
    
    return np.array(dps)


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
    print(f"{'Scenario':<15} {'Targets':<10} {'Benign':<10} {'Camera':<10} {'PC':<10} {'Combined':<10} {'Def(ALL)':<10} {'Def(AdvT)':<10}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['scenario']:<15} {r['num_targets']:<10} "
              f"{r['benign']['tracks']:<10} {r['camera_attack']['tracks']:<10} "
              f"{r['pc_attack']['tracks']:<10} {r['combined_attack']['tracks']:<10} "
              f"{r['defended_all']['tracks']:<10} {r['defended_adv_training']['tracks']:<10}")
    
    # Table 2: Detection Probability (IR Camera)
    print("\n--- IR Camera Detection Probability ---")
    print(f"{'Scenario':<15} {'Benign':<10} {'Attacked':<10} {'Def(ALL)':<10} {'Def(AdvT)':<10} {'Rec(ALL)':<10} {'Rec(AdvT)':<10}")
    print("-" * 85)
    for r in all_results:
        ir_key = 'IR Camera' if 'IR Camera' in r['benign']['metrics'] else 'IR'
        benign = r['benign']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        attacked = r['combined_attack']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        def_all = r['defended_all']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        def_adv = r['defended_adv_training']['metrics'].get(ir_key, {}).get('detection_probability', 0)
        rec_all = (def_all / benign * 100) if benign > 0 else 0
        rec_adv = (def_adv / benign * 100) if benign > 0 else 0
        print(f"{r['scenario']:<15} {benign:<10.4f} {attacked:<10.4f} "
              f"{def_all:<10.4f} {def_adv:<10.4f} {rec_all:<9.1f}% {rec_adv:<9.1f}%")
    
    # Table 3: Detection Probability (Lidar)
    print("\n--- Lidar Detection Probability ---")
    print(f"{'Scenario':<15} {'Benign':<10} {'Attacked':<10} {'Def(ALL)':<10} {'Def(AdvT)':<10}")
    print("-" * 65)
    for r in all_results:
        lidar_key = 'Lidar' if 'Lidar' in r['benign']['metrics'] else 'LIDAR'
        benign = r['benign']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        attacked = r['combined_attack']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        def_all = r['defended_all']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        def_adv = r['defended_adv_training']['metrics'].get(lidar_key, {}).get('detection_probability', 0)
        print(f"{r['scenario']:<15} {benign:<10.4f} {attacked:<10.4f} "
              f"{def_all:<10.4f} {def_adv:<10.4f}")
    
    # Table 4: Fusion Attack Comparison
    print("\n--- Fusion Attack Track Counts ---")
    fusion_names = [
        'existence_suppression', 'association_confusion', 'cross_sensor_consistency',
        'false_track_injection', 'sensor_dos', 'track_merge_manipulation',
        'track_deletion', 'track_swap', 'stealthy_degradation'
    ]
    header = f"{'Scenario':<15}"
    for fn in fusion_names:
        header += f" {fn[:6]:<7}"
    print(header)
    print("-" * (15 + 8 * len(fusion_names)))
    for r in all_results:
        line = f"{r['scenario']:<15}"
        for fn in fusion_names:
            tracks = r.get('fusion_attacks', {}).get(fn, {}).get('tracks', 0)
            line += f" {tracks:<7}"
        print(line)
    
    # Table 5: Statistical Significance Summary
    print("\n--- Statistical Significance (IR Camera) ---")
    print(f"{'Scenario':<15} {'BvsA p':<10} {'AvsD p':<10} {'BvsD p':<10} {'Effect Size':<12}")
    print("-" * 60)
    for r in all_results:
        sig = r.get('statistical_significance', {})
        ir_sig = sig.get('IR Camera') or sig.get('IR', {})
        if ir_sig:
            bva = ir_sig.get('benign_vs_attacked', {})
            avd = ir_sig.get('attacked_vs_defended', {})
            bvd = ir_sig.get('benign_vs_defended', {})
            print(f"{r['scenario']:<15} {bva.get('p_value', 0):<10.4f} "
                  f"{avd.get('p_value', 0):<10.4f} {bvd.get('p_value', 0):<10.4f} "
                  f"{avd.get('cohens_d', 0):<12.3f}")
        else:
            print(f"{r['scenario']:<15} N/A")


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
