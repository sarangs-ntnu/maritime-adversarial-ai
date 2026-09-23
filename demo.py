"""
Maritime Adversarial AI Framework - Comprehensive Demo
======================================================

Demonstrates all 5 phases of the adversarial AI pipeline:
    1. Data Loading & Visualization
    2. Adversarial Attacks (Camera, Radar/Lidar, Fusion)
    3. Physical Realizability (EOT)
    4. Evaluation Metrics
    5. Defense Mechanisms

Usage:
    python demo.py
"""

import sys
from pathlib import Path

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# Add src to path for imports
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data_loader import ScenarioLoader, SENSOR_NAMES
from attacks.camera_attacks import CameraAdversarialAttacker, AttackConfig as CamConfig, AttackType as CamType
from attacks.radar_lidar_attacks import PointCloudAttacker, PointCloudAttackConfig as PCConfig, PointCloudAttackType as PCType
from attacks.fusion_attacks import FusionAttacker, FusionAttackConfig as FusionConfig, FusionAttackType as FusionType, JIPDASimulator
from attacks.physical_eot import MaritimeEOT, MaritimeEnvironmentParams
from evaluation.metrics import DetectionEvaluator, AttackEvaluator
from defenses.defense_mechanisms import DefensePipeline, DefenseConfig, DefenseType, CertifiedDefense
from pipeline import AdversarialPipeline, PipelineConfig
from visualization.plot_utils import (
    plot_scenario_overview, plot_all_sensors, plot_detection_timeline,
    plot_attack_impact_bearings, plot_defense_recovery, plot_metrics_comparison,
    plot_certified_radius
)


def phase1_data_loading():
    """Phase 1: Load and inspect scenario data."""
    print("\n" + "=" * 70)
    print("PHASE 1: DATA LOADING & RECONNAISSANCE")
    print("=" * 70)
    
    loader = ScenarioLoader('scenario2', 'data/sensor_fusion_dataset')
    
    print(f"\nScenario: scenario2")
    print(f"Total detections: {len(loader.detections)}")
    print(f"Ground truth timesteps: {len(loader.ground_truth)}")
    
    # Count per sensor
    for sid in [1, 2, 3, 4]:
        dets = [d for d in loader.detections if d.sensor_id == sid]
        non_empty = [d for d in dets if len(d.measurement) > 0]
        print(f"  {SENSOR_NAMES[sid]}: {len(dets)} detections ({len(non_empty)} non-empty)")
    
    # Count unique targets
    all_targets = set()
    for timestep in loader.ground_truth:
        for gt in timestep:
            all_targets.add(gt.target_id)
    print(f"\nUnique targets: {len(all_targets)}")
    
    # Show first target trajectory
    target1_positions = []
    for timestep in loader.ground_truth[:20]:
        for gt in timestep:
            if gt.target_id == 1:
                target1_positions.append(gt.position[:2])
                break
    print(f"Target 1 first position: {target1_positions[0] if target1_positions else 'N/A'}")
    
    return loader


def phase2a_camera_attacks(loader):
    """Phase 2a: Camera adversarial attacks."""
    print("\n" + "=" * 70)
    print("PHASE 2a: CAMERA ADVERSARIAL ATTACKS")
    print("=" * 70)
    
    # Get IR detections
    ir_dets = [d for d in loader.detections if d.sensor_id == 3 and len(d.measurement) > 0]
    if not ir_dets:
        print("No IR detections found!")
        return
    
    det = ir_dets[0]
    print(f"\nOriginal IR detection:")
    print(f"  Time: {det.time:.2f}")
    print(f"  Bearing: {np.rad2deg(det.measurement[0]):.2f}°")
    
    # Test different attack types
    attack_types = [
        (CamType.FGSM, "FGSM", {"epsilon": 0.1}),
        (CamType.PGD, "PGD", {"epsilon": 0.1, "alpha": 0.02, "num_steps": 20}),
        (CamType.BIM, "BIM", {"epsilon": 0.1, "alpha": 0.02, "num_steps": 20}),
        (CamType.CW, "C&W", {"epsilon": 0.1, "num_steps": 40}),
        (CamType.UNIVERSAL, "Universal", {"epsilon": 0.1}),
        (CamType.BACKDOOR, "Backdoor", {"epsilon": 0.1}),
    ]
    
    evaluator = AttackEvaluator(loader)
    
    for attack_type, name, kwargs in attack_types:
        config = CamConfig(attack_type, **kwargs)
        attacker = CameraAdversarialAttacker(config)
        attacked = attacker.attack(det)
        
        shift = np.rad2deg(attacked.measurement[0] - det.measurement[0])
        print(f"\n  {name} Attack:")
        print(f"    Attacked bearing: {np.rad2deg(attacked.measurement[0]):.2f}°")
        print(f"    Shift: {shift:.4f}°")
        
        # Evaluate on all IR detections
        attacked_dets = [attacker.attack(d) for d in ir_dets]
        metrics = evaluator.evaluate_camera_attack(ir_dets, attacked_dets, 3)
        print(f"    Mean perturbation: {np.rad2deg(metrics.mean_perturbation):.4f}°")
        print(f"    Attack success rate: {metrics.attack_success_rate:.2%}")


def phase2b_radar_lidar_attacks(loader):
    """Phase 2b: Radar/Lidar point cloud attacks."""
    print("\n" + "=" * 70)
    print("PHASE 2b: RADAR/LIDAR POINT CLOUD ATTACKS")
    print("=" * 70)
    
    # Get radar detections
    radar_dets = [d for d in loader.detections if d.sensor_id == 2 and len(d.measurement) > 0]
    if not radar_dets:
        print("No radar detections found!")
        return
    
    det = radar_dets[0]
    print(f"\nOriginal Radar detection:")
    print(f"  Time: {det.time:.2f}")
    print(f"  Points: {len(det.measurement)}")
    print(f"  Point cloud shape: {det.measurement.shape}")
    
    # Test different attack types
    attack_types = [
        (PCType.GHOST_INJECTION, "Ghost Injection"),
        (PCType.CLUSTER_SPLIT, "Cluster Split"),
        (PCType.POINT_SUPPRESSION, "Point Suppression"),
        (PCType.NOISE_FLOOR, "Noise Floor"),
    ]
    
    for attack_type, name in attack_types:
        config = PCConfig(attack_type, epsilon=5.0)
        attacker = PointCloudAttacker(config)
        attacked = attacker.attack(det)
        
        orig_points = len(det.measurement)
        attacked_points = len(attacked.measurement)
        change = ((attacked_points - orig_points) / orig_points) * 100
        
        print(f"\n  {name} Attack:")
        print(f"    Points: {orig_points} -> {attacked_points} ({change:+.1f}%)")


def phase2c_fusion_attacks(loader):
    """Phase 2c: Fusion layer attacks on JIPDA tracker."""
    print("\n" + "=" * 70)
    print("PHASE 2c: FUSION LAYER ATTACKS")
    print("=" * 70)
    
    # Run benign tracker
    dets = [d for d in loader.detections if len(d.measurement) > 0]
    
    tracker = JIPDASimulator()
    benign_tracks = tracker.track(dets)
    print(f"\nBenign tracking: {len(benign_tracks)} confirmed tracks")
    
    # Test fusion attacks
    attack_types = [
        (FusionType.EXISTENCE_SUPPRESSION, "Existence Suppression"),
        (FusionType.FALSE_TRACK_INJECTION, "False Track Injection"),
        (FusionType.SENSOR_DOS, "Sensor DoS"),
    ]
    
    for attack_type, name in attack_types:
        config = FusionConfig(attack_type)
        attacker = FusionAttacker(config)
        attacked_dets = attacker.attack(dets)
        
        tracker_attacked = JIPDASimulator()
        attacked_tracks = tracker_attacked.track(attacked_dets)
        
        track_diff = len(attacked_tracks) - len(benign_tracks)
        print(f"\n  {name} Attack:")
        print(f"    Tracks: {len(benign_tracks)} -> {len(attacked_tracks)} ({track_diff:+d})")


def phase3_physical_eot(loader):
    """Phase 3: Physical realizability evaluation."""
    print("\n" + "=" * 70)
    print("PHASE 3: PHYSICAL REALIZABILITY (EOT)")
    print("=" * 70)
    
    # Create harsh maritime environment
    env = MaritimeEnvironmentParams(
        wave_height=2.5,
        rain_rate=15.0,
        fog_visibility=500,
        sea_state=5,
        cloud_cover=0.7
    )
    eot = MaritimeEOT(env)
    
    print(f"\nEnvironment: Wave H={env.wave_height}m, Rain={env.rain_rate}mm/h, Fog vis={env.fog_visibility}m")
    
    # Test camera attack realizability
    ir_dets = [d for d in loader.detections if d.sensor_id == 3 and len(d.measurement) > 0]
    if ir_dets:
        det = ir_dets[0]
        config = CamConfig(CamType.FGSM, epsilon=0.1)
        attacker = CameraAdversarialAttacker(config)
        attacked = attacker.attack(det)
        
        realizability = eot.evaluate_physical_realizability(det, attacked)
        
        print(f"\n  Camera FGSM Attack Realizability:")
        print(f"    Original perturbation: {realizability['original_perturbation']:.4f} rad")
        print(f"    After EOT: {realizability['transformed_perturbation']:.4f} rad")
        print(f"    Preservation ratio: {realizability['preservation_ratio']:.2%}")
        print(f"    Realizability score: {realizability['realizability_score']:.2%}")
    
    # Test lidar attack realizability
    lidar_dets = [d for d in loader.detections if d.sensor_id == 1 and len(d.measurement) > 0]
    if lidar_dets:
        det = lidar_dets[0]
        config = PCConfig(PCType.GHOST_INJECTION, epsilon=5.0)
        attacker = PointCloudAttacker(config)
        attacked = attacker.attack(det)
        
        realizability = eot.evaluate_physical_realizability(det, attacked)
        
        print(f"\n  Lidar Ghost Injection Realizability:")
        print(f"    Original perturbation: {realizability['original_perturbation']:.2f} m")
        print(f"    After EOT: {realizability['transformed_perturbation']:.2f} m")
        print(f"    Preservation ratio: {realizability['preservation_ratio']:.2%}")


def phase4_evaluation(loader):
    """Phase 4: Evaluation metrics."""
    print("\n" + "=" * 70)
    print("PHASE 4: EVALUATION METRICS")
    print("=" * 70)
    
    evaluator = DetectionEvaluator(distance_threshold=50.0)
    
    # Evaluate per sensor using per-timestep matching
    print("\nBenign Detection Performance (Per-Timestep):")
    from collections import defaultdict
    
    # Group detections by time
    dets_by_time = defaultdict(list)
    for d in loader.detections:
        dets_by_time[d.time].append(d)
    
    # Group ground truth by time
    gt_by_time = {}
    for i, timestep in enumerate(loader.ground_truth):
        if timestep and i < len(loader.detections):
            gt_by_time[loader.detections[i].time] = timestep
    
    for sid in [1, 2, 3, 4]:
        sensor_dets = [d for d in loader.detections if d.sensor_id == sid and len(d.measurement) > 0]
        
        # Per-timestep evaluation
        all_errors = []
        total_matched = 0
        total_gt = 0
        total_fa = 0
        total_dets = 0
        
        for t, dets_at_t in dets_by_time.items():
            gt_at_t = gt_by_time.get(t, [])
            if not gt_at_t:
                continue
            
            s_dets = [d for d in dets_at_t if d.sensor_id == sid]
            if not s_dets:
                continue
            
            m = evaluator.evaluate(s_dets, gt_at_t, sid)
            if m.rmse_position > 0:
                all_errors.append(m.rmse_position)
            total_matched += int(m.detection_probability * len(gt_at_t))
            total_gt += len(gt_at_t)
            total_fa += int(m.false_alarm_rate * len(s_dets))
            total_dets += len(s_dets)
        
        avg_rmse = np.mean(all_errors) if all_errors else 0
        det_prob = total_matched / max(total_gt, 1)
        far = total_fa / max(total_dets, 1)
        
        print(f"\n  {SENSOR_NAMES[sid]}:")
        print(f"    RMSE: {avg_rmse:.4f} m")
        print(f"    Detection Probability: {det_prob:.2%}")
        print(f"    False Alarm Rate: {far:.2%}")
    
    # Evaluate attack impact
    print("\n\nAttack Impact Evaluation:")
    
    # Camera attack impact
    ir_dets = [d for d in loader.detections if d.sensor_id == 3 and len(d.measurement) > 0]
    if ir_dets:
        config = CamConfig(CamType.FGSM, epsilon=0.1)
        attacker = CameraAdversarialAttacker(config)
        attacked_dets = [attacker.attack(d) for d in ir_dets]
        
        attack_eval = AttackEvaluator(loader)
        metrics = attack_eval.evaluate_camera_attack(ir_dets, attacked_dets, 3)
        
        print(f"\n  Camera FGSM Attack:")
        print(f"    Mean perturbation: {np.rad2deg(metrics.mean_perturbation):.4f}°")
        print(f"    Max perturbation: {np.rad2deg(metrics.max_perturbation):.4f}°")
        print(f"    Attack success rate: {metrics.attack_success_rate:.2%}")


def phase5_defenses(loader):
    """Phase 5: Defense mechanisms."""
    print("\n" + "=" * 70)
    print("PHASE 5: DEFENSE MECHANISMS")
    print("=" * 70)
    
    # Create attacked dataset
    dets = loader.detections.copy()
    
    # Apply attacks
    cam_config = CamConfig(CamType.FGSM, epsilon=0.1)
    cam_attacker = CameraAdversarialAttacker(cam_config)
    
    pc_config = PCConfig(PCType.GHOST_INJECTION, epsilon=5.0)
    pc_attacker = PointCloudAttacker(pc_config)
    
    attacked_dets = []
    for d in dets:
        if d.sensor_id in [3, 4]:
            attacked_dets.append(cam_attacker.attack(d))
        elif d.sensor_id in [1, 2]:
            attacked_dets.append(pc_attacker.attack(d))
        else:
            attacked_dets.append(d)
    
    print(f"\nAttacked detections: {len(attacked_dets)}")
    
    # Apply defense - train on benign data first, then apply to attacked
    defense_config = DefenseConfig(DefenseType.ALL)
    defense = DefensePipeline(defense_config)
    
    # Train temporal checker on benign data
    defense.temporal_checker.train(dets)
    
    defended_dets = defense.defend(attacked_dets)
    print(f"Defended detections: {len(defended_dets)}")
    
    # Compare metrics using per-timestep evaluation
    evaluator = DetectionEvaluator(distance_threshold=50.0)
    from collections import defaultdict
    
    def eval_per_timestep(detections_list, sensor_id):
        dets_by_time = defaultdict(list)
        for d in detections_list:
            dets_by_time[d.time].append(d)
        
        gt_by_time = {}
        for i, timestep in enumerate(loader.ground_truth):
            if timestep and i < len(loader.detections):
                gt_by_time[loader.detections[i].time] = timestep
        
        total_matched = 0
        total_gt = 0
        for t, dets_at_t in dets_by_time.items():
            gt_at_t = gt_by_time.get(t, [])
            if not gt_at_t:
                continue
            s_dets = [d for d in dets_at_t if d.sensor_id == sensor_id]
            if not s_dets:
                continue
            m = evaluator.evaluate(s_dets, gt_at_t, sensor_id)
            total_matched += int(m.detection_probability * len(gt_at_t))
            total_gt += len(gt_at_t)
        
        return total_matched / max(total_gt, 1)
    
    print("\n  Detection Probability (Benign -> Attacked -> Defended):")
    for sid in [1, 2, 3, 4]:
        benign_dp = eval_per_timestep(dets, sid)
        attacked_dp = eval_per_timestep(attacked_dets, sid)
        defended_dp = eval_per_timestep(defended_dets, sid)
        
        print(f"    {SENSOR_NAMES[sid]}: {benign_dp:.3f} -> {attacked_dp:.3f} -> {defended_dp:.3f}")


def full_pipeline_demo():
    """Run the complete end-to-end pipeline."""
    print("\n" + "=" * 70)
    print("FULL PIPELINE DEMO")
    print("=" * 70)
    
    # Config with attacks enabled
    config = PipelineConfig(
        scenario_name='scenario2',
        data_dir='data/sensor_fusion_dataset',
        output_dir='results',
        camera_attack_type=CamType.FGSM,
        camera_epsilon=0.1,
        pointcloud_attack_type=PCType.GHOST_INJECTION,
        pointcloud_epsilon=5.0,
        fusion_attack_type=FusionType.EXISTENCE_SUPPRESSION,
        use_defense=False
    )
    
    pipeline = AdversarialPipeline(config)
    
    print("\nRunning benign pipeline...")
    benign = pipeline.run_benign()
    print(f"  Tracks: {len(benign['tracks'])}")
    print(f"  Detections: {benign['num_detections']}")
    
    print("\nRunning attacked pipeline (FGSM + Ghost + Existence Suppression)...")
    attacked = pipeline.run_attacked()
    print(f"  Tracks: {len(attacked['tracks'])}")
    print(f"  Detections: {attacked['num_detections']}")
    
    print("\nComparing results...")
    comp = pipeline.compare()
    print(f"  Track loss: {comp['track_loss']}")
    
    print("\nSensor Metrics:")
    for sensor, m in comp['sensor_metrics'].items():
        print(f"  {sensor}:")
        print(f"    DetProb: {m['det_prob_benign']:.4f} -> {m['det_prob_attacked']:.4f}")
        print(f"    FAR: {m['far_benign']:.4f} -> {m['far_attacked']:.4f}")
    
    # Save results
    pipeline.save_results()
    print("\nResults saved!")
    
    # Now run with defense
    print("\n" + "-" * 50)
    print("Running with Defense (ALL)...")
    config_defense = PipelineConfig(
        scenario_name='scenario2',
        data_dir='data/sensor_fusion_dataset',
        output_dir='results',
        camera_attack_type=CamType.FGSM,
        camera_epsilon=0.1,
        pointcloud_attack_type=PCType.GHOST_INJECTION,
        pointcloud_epsilon=5.0,
        fusion_attack_type=FusionType.EXISTENCE_SUPPRESSION,
        use_defense=True,
        defense_type=DefenseType.ALL
    )
    
    pipeline_def = AdversarialPipeline(config_defense)
    benign_def = pipeline_def.run_benign()
    attacked_def = pipeline_def.run_attacked()
    comp_def = pipeline_def.compare()
    
    print(f"\nWith Defense:")
    print(f"  Track loss: {comp_def['track_loss']}")
    print(f"  Benign tracks: {len(benign_def['tracks'])}")
    print(f"  Attacked tracks: {len(attacked_def['tracks'])}")
    
    print("\nSensor Metrics (with defense):")
    for sensor, m in comp_def['sensor_metrics'].items():
        print(f"  {sensor}:")
        print(f"    DetProb: {m['det_prob_benign']:.4f} -> {m['det_prob_attacked']:.4f}")
    
    pipeline_def.save_results()
    print("\nDefense results saved!")


def phase6_visualization(loader):
    """Phase 6: Generate visualization plots."""
    print("\n" + "=" * 70)
    print("PHASE 6: VISUALIZATION")
    print("=" * 70)
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    print("\nGenerating scenario overview...")
    plot_scenario_overview(loader, save_path=results_dir / "scenario_overview.png")
    print("  Saved: scenario_overview.png")
    
    print("\nGenerating all sensors plot...")
    plot_all_sensors(loader, save_path=results_dir / "all_sensors.png")
    print("  Saved: all_sensors.png")
    
    print("\nGenerating detection timeline...")
    plot_detection_timeline(loader, save_path=results_dir / "detection_timeline.png")
    print("  Saved: detection_timeline.png")
    
    # Generate attack impact visualization
    print("\nGenerating attack impact plots...")
    
    # Camera attack
    ir_dets = [d for d in loader.detections if d.sensor_id == 3 and len(d.measurement) > 0]
    if ir_dets:
        config = CamConfig(CamType.FGSM, epsilon=0.1)
        attacker = CameraAdversarialAttacker(config)
        attacked_ir = [attacker.attack(d) for d in ir_dets]
        
        plot_attack_impact_bearings(
            ir_dets, attacked_ir, 3,
            save_path=results_dir / "attack_impact_ir.png"
        )
        print("  Saved: attack_impact_ir.png")
    
    # Defense recovery visualization
    print("\nGenerating defense recovery plots...")
    
    dets = loader.detections.copy()
    cam_config = CamConfig(CamType.FGSM, epsilon=0.1)
    cam_attacker = CameraAdversarialAttacker(cam_config)
    
    attacked_dets = []
    for d in dets:
        if d.sensor_id in [3, 4]:
            attacked_dets.append(cam_attacker.attack(d))
        else:
            attacked_dets.append(d)
    
    # Apply defense
    defense_config = DefenseConfig(DefenseType.ALL)
    defense = DefensePipeline(defense_config)
    defense.temporal_checker.train(dets)
    defended_dets = defense.defend(attacked_dets)
    
    plot_defense_recovery(
        dets, attacked_dets, defended_dets, 3,
        save_path=results_dir / "defense_recovery_ir.png"
    )
    print("  Saved: defense_recovery_ir.png")
    
    # Metrics comparison
    print("\nGenerating metrics comparison...")
    
    evaluator = DetectionEvaluator(distance_threshold=50.0)
    from collections import defaultdict
    
    def compute_metrics(detections_list):
        dets_by_time = defaultdict(list)
        for d in detections_list:
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
    
    benign_metrics = compute_metrics(dets)
    attacked_metrics = compute_metrics(attacked_dets)
    defended_metrics = compute_metrics(defended_dets)
    
    metrics_dict = {
        'Benign': benign_metrics,
        'Attacked': attacked_metrics,
        'Defended': defended_metrics
    }
    
    plot_metrics_comparison(
        metrics_dict,
        save_path=results_dir / "metrics_comparison.png"
    )
    print("  Saved: metrics_comparison.png")
    
    # Certified radius plot
    print("\nGenerating certified robustness plot...")
    certifier = CertifiedDefense(num_samples=50, noise_std=0.05)
    
    certified_results = []
    for det in ir_dets[:50]:  # Sample 50 detections
        _, radius = certifier.smooth_bearing(det)
        certified_results.append((radius, 0.1))  # Compare against epsilon=0.1
    
    plot_certified_radius(
        certified_results,
        save_path=results_dir / "certified_radius.png"
    )
    print("  Saved: certified_radius.png")
    
    print(f"\nAll plots saved to {results_dir}/")


def phase7_advanced_fusion_attacks(loader):
    """Phase 7: Advanced track-oriented fusion attacks and defenses."""
    print("\n" + "=" * 70)
    print("PHASE 7: ADVANCED FUSION ATTACKS & DEFENSES")
    print("=" * 70)
    
    from attacks.fusion_attacks import FusionAttacker, FusionAttackConfig, FusionAttackType
    from defenses.defense_mechanisms import DefensePipeline, DefenseConfig, DefenseType
    from evaluation.metrics import DetectionEvaluator
    from collections import defaultdict
    
    evaluator = DetectionEvaluator(distance_threshold=50.0)
    
    def eval_per_timestep(detections_list, sensor_id):
        dets_by_time = defaultdict(list)
        for d in detections_list:
            dets_by_time[d.time].append(d)
        
        gt_by_time = {}
        for i, timestep in enumerate(loader.ground_truth):
            if timestep and i < len(loader.detections):
                gt_by_time[loader.detections[i].time] = timestep
        
        total_matched = 0
        total_gt = 0
        for t, dets_at_t in dets_by_time.items():
            gt_at_t = gt_by_time.get(t, [])
            if not gt_at_t:
                continue
            s_dets = [d for d in dets_at_t if d.sensor_id == sensor_id]
            if not s_dets:
                continue
            m = evaluator.evaluate(s_dets, gt_at_t, sensor_id)
            total_matched += int(m.detection_probability * len(gt_at_t))
            total_gt += len(gt_at_t)
        
        return total_matched / max(total_gt, 1)
    
    # Test track deletion attack
    print("\n--- Track Deletion Attack ---")
    config = FusionAttackConfig(FusionAttackType.TRACK_DELETION)
    attacker = FusionAttacker(config)
    attacked = attacker.attack_scenario(loader)
    
    attacked_list = []
    for t in sorted(attacked.keys()):
        attacked_list.extend(attacked[t])
    
    print(f"  Attacked detections: {len(attacked_list)}")
    for sid in [1, 2, 3, 4]:
        dp = eval_per_timestep(attacked_list, sid)
        print(f"    {SENSOR_NAMES[sid]} DetProb: {dp:.3f}")
    
    # Test track swap attack
    print("\n--- Track Swap Attack ---")
    config = FusionAttackConfig(FusionAttackType.TRACK_SWAP)
    attacker = FusionAttacker(config)
    attacked = attacker.attack_scenario(loader)
    
    attacked_list = []
    for t in sorted(attacked.keys()):
        attacked_list.extend(attacked[t])
    
    print(f"  Attacked detections: {len(attacked_list)}")
    for sid in [1, 2, 3, 4]:
        dp = eval_per_timestep(attacked_list, sid)
        print(f"    {SENSOR_NAMES[sid]} DetProb: {dp:.3f}")
    
    # Test stealthy degradation attack
    print("\n--- Stealthy Degradation Attack ---")
    config = FusionAttackConfig(FusionAttackType.STEALTHY_DEGRADATION)
    attacker = FusionAttacker(config)
    attacked = attacker.attack_scenario(loader)
    
    attacked_list = []
    for t in sorted(attacked.keys()):
        attacked_list.extend(attacked[t])
    
    print(f"  Attacked detections: {len(attacked_list)}")
    for sid in [1, 2, 3, 4]:
        dp = eval_per_timestep(attacked_list, sid)
        print(f"    {SENSOR_NAMES[sid]} DetProb: {dp:.3f}")
    
    # Test track merge manipulation attack
    print("\n--- Track Merge Manipulation Attack ---")
    config = FusionAttackConfig(FusionAttackType.TRACK_MERGE_MANIPULATION)
    attacker = FusionAttacker(config)
    attacked = attacker.attack_scenario(loader)
    
    attacked_list = []
    for t in sorted(attacked.keys()):
        attacked_list.extend(attacked[t])
    
    print(f"  Attacked detections: {len(attacked_list)}")
    for sid in [1, 2, 3, 4]:
        dp = eval_per_timestep(attacked_list, sid)
        print(f"    {SENSOR_NAMES[sid]} DetProb: {dp:.3f}")
    
    # Adversarial Training Defense
    print("\n--- Adversarial Training Defense ---")
    
    # Create benign and attacked datasets
    benign_dets = loader.detections.copy()
    
    # Generate attacked data for training
    pc_config = PCConfig(PCType.GHOST_INJECTION, epsilon=5.0)
    pc_attacker = PointCloudAttacker(pc_config)
    cam_config = CamConfig(CamType.FGSM, epsilon=0.1)
    cam_attacker = CameraAdversarialAttacker(cam_config)
    
    train_attacked = []
    for d in benign_dets:
        if d.sensor_id in [1, 2]:
            train_attacked.append(pc_attacker.attack(d))
        elif d.sensor_id in [3, 4]:
            train_attacked.append(cam_attacker.attack(d))
        else:
            train_attacked.append(d)
    
    # Train adversarial defense
    defense_config = DefenseConfig(DefenseType.ADVERSARIAL_TRAINING)
    defense = DefensePipeline(defense_config)
    defense.train_adversarial(benign_dets, train_attacked)
    
    # Apply to new attacked data
    test_attacked = []
    for d in benign_dets:
        if d.sensor_id in [1, 2]:
            test_attacked.append(pc_attacker.attack(d))
        elif d.sensor_id in [3, 4]:
            test_attacked.append(cam_attacker.attack(d))
        else:
            test_attacked.append(d)
    
    defended = defense.defend(test_attacked)
    
    print("\n  Detection Probability (Benign -> Attacked -> AdvTrained):")
    for sid in [1, 2, 3, 4]:
        benign_dp = eval_per_timestep(benign_dets, sid)
        attacked_dp = eval_per_timestep(test_attacked, sid)
        defended_dp = eval_per_timestep(defended, sid)
        print(f"    {SENSOR_NAMES[sid]}: {benign_dp:.3f} -> {attacked_dp:.3f} -> {defended_dp:.3f}")
    
    # Statistical Significance Testing
    print("\n--- Statistical Significance Testing ---")
    from defenses.defense_mechanisms import StatisticalSignificance
    
    tester = StatisticalSignificance(confidence_level=0.95)
    
    # Collect per-timestep detection probabilities for each condition
    def collect_dp_series(detections_list, sensor_id):
        dets_by_time = defaultdict(list)
        for d in detections_list:
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
    
    for sid in [3, 4]:  # Camera sensors
        benign_series = collect_dp_series(benign_dets, sid)
        attacked_series = collect_dp_series(test_attacked, sid)
        defended_series = collect_dp_series(defended, sid)
        
        if len(benign_series) > 1:
            results = tester.compare_three_conditions(
                benign_series, attacked_series, defended_series
            )
            tester.print_summary(results, f"{SENSOR_NAMES[sid]} DetProb")


def phase8_all_scenarios():
    """Phase 8: Run evaluation on all available scenarios."""
    print("\n" + "=" * 70)
    print("PHASE 8: ALL SCENARIOS EVALUATION")
    print("=" * 70)
    
    from pathlib import Path
    data_dir = Path("data/sensor_fusion_dataset")
    
    # Find available scenarios
    available_scenarios = []
    for scenario_dir in data_dir.glob("scenario*"):
        if scenario_dir.is_dir():
            available_scenarios.append(scenario_dir.name)
    
    available_scenarios.sort()
    print(f"\nAvailable scenarios: {available_scenarios}")
    
    for scenario_name in available_scenarios:
        print(f"\n--- {scenario_name} ---")
        try:
            loader = ScenarioLoader(scenario_name, str(data_dir))
            print(f"  Detections: {len(loader.detections)}")
            print(f"  Targets: {len(loader.target_ids)}")
            
            # Quick attack test
            from attacks.fusion_attacks import FusionAttacker, FusionAttackConfig, FusionAttackType
            config = FusionAttackConfig(FusionAttackType.TRACK_MERGE_MANIPULATION)
            attacker = FusionAttacker(config)
            attacked = attacker.attack_scenario(loader)
            
            attacked_list = []
            for t in sorted(attacked.keys()):
                attacked_list.extend(attacked[t])
            
            # Count tracks
            from attacks.fusion_attacks import JIPDASimulator
            tracker = JIPDASimulator()
            tracks = tracker.track(attacked_list)
            print(f"  Benign tracks: {len(loader.target_ids)}")
            print(f"  Track merge tracks: {len(tracks)}")
            
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    """Run all demo phases."""
    print("\n" + "=" * 70)
    print("MARITIME ADVERSARIAL AI FRAMEWORK - COMPREHENSIVE DEMO")
    print("=" * 70)
    
    # Phase 1: Data Loading
    loader = phase1_data_loading()
    
    # Phase 2: Attacks
    phase2a_camera_attacks(loader)
    phase2b_radar_lidar_attacks(loader)
    phase2c_fusion_attacks(loader)
    
    # Phase 3: Physical Realizability
    phase3_physical_eot(loader)
    
    # Phase 4: Evaluation
    phase4_evaluation(loader)
    
    # Phase 5: Defenses
    phase5_defenses(loader)
    
    # Phase 6: Visualization
    phase6_visualization(loader)
    
    # Phase 7: Advanced Fusion Attacks & Defenses
    phase7_advanced_fusion_attacks(loader)
    
    # Phase 8: All Scenarios
    phase8_all_scenarios()
    
    # Full Pipeline
    full_pipeline_demo()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
