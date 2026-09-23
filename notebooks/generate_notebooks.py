#!/usr/bin/env python3
"""Generate all 8 Jupyter notebooks with correct API usage."""

import json
import os


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}


# ---------------------------------------------------------------------------
# Notebook 1: Data Exploration
# ---------------------------------------------------------------------------
nb1_cells = [
    md_cell("""# Notebook 1: Data Exploration

Explore the NTNU Autoferry Sensor Fusion Dataset.

**Sensors:**
- Lidar (ID=1): Active, 2D position (N, E)
- Radar (ID=2): Active, 2D position (N, E)
- IR Camera (ID=3): Passive, bearing only
- EO Camera (ID=4): Passive, bearing only"""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import load_scenario, get_all_detections, get_ground_truth, get_ownship

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 1.1 Load Scenario Data"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)
ground_truth = get_ground_truth(loader)
ownship = get_ownship(loader)

print('Loaded', len(detections), 'sensors')
print('Ground truth targets:', list(ground_truth.keys()))
print('Ownship trajectory shape:', len(ownship))"""),
    md_cell("## 1.2 Sensor Detection Overview"),
    code_cell("""for sensor_id, df in detections.items():
    print('\n=== Sensor', sensor_id, '===')
    print('Shape:', df.shape)
    print('Columns:', list(df.columns))
    print('Time range:', df['time'].min(), '-', df['time'].max(), 's')
    print(df.head(3))"""),
    md_cell("## 1.3 Plot Scenario Overview"),
    code_cell("""fig, ax = plt.subplots(figsize=(12, 8))

for sensor_id, df in detections.items():
    ax.scatter(df['x_piren'], df['y_piren'], s=2, alpha=0.4, label='Sensor ' + str(sensor_id))

for target_id, gt_df in ground_truth.items():
    ax.plot(gt_df['x_piren'], gt_df['y_piren'], 'k-', linewidth=2, label='Target ' + str(target_id))

ax.plot(ownship['x_piren'], ownship['y_piren'], 'r--', linewidth=1, label='Ownship')
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title(SCENARIO + ' Overview')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()"""),
    md_cell("## 1.4 Detection Timeline"),
    code_cell("""fig, ax = plt.subplots(figsize=(14, 6))

for sensor_id, df in detections.items():
    counts = df.groupby(df['time'].round()).size()
    ax.plot(counts.index, counts.values, label='Sensor ' + str(sensor_id), linewidth=2)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Detection Count')
ax.set_title(SCENARIO + ' Detection Timeline')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()"""),
    md_cell("## 1.5 Individual Sensor Plots"),
    code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (sensor_id, df) in enumerate(detections.items()):
    ax = axes[idx]
    ax.scatter(df['x_piren'], df['y_piren'], s=3, alpha=0.5)
    
    for tid, gt in ground_truth.items():
        ax.plot(gt['x_piren'], gt['y_piren'], 'k-', linewidth=1.5)
    
    ax.set_title('Sensor ' + str(sensor_id))
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    ax.set_aspect('equal')

plt.suptitle(SCENARIO + ' All Sensors', fontsize=14)
plt.tight_layout()
plt.show()"""),
    md_cell("## 1.6 Ground Truth Trajectories"),
    code_cell("""fig, ax = plt.subplots(figsize=(12, 8))

for target_id, gt_df in ground_truth.items():
    ax.plot(gt_df['x_piren'], gt_df['y_piren'], linewidth=2, label='Target ' + str(target_id))

ax.plot(ownship['x_piren'], ownship['y_piren'], 'r--', linewidth=1, label='Ownship')
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title(SCENARIO + ' Ground Truth Trajectories')
ax.legend()
ax.grid(True)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()"""),
    md_cell("## 1.7 Sensor Statistics"),
    code_cell("""from helpers import compute_sensor_metrics

for sensor_id, df in detections.items():
    metrics = compute_sensor_metrics(df, ground_truth, sensor_id)
    print('\n=== Sensor', sensor_id, '===')
    for k, v in metrics.items():
        print(k + ':', round(v, 4) if isinstance(v, float) else v)"""),
]

# ---------------------------------------------------------------------------
# Notebook 2: Camera Attacks
# ---------------------------------------------------------------------------
nb2_cells = [
    md_cell("""# Notebook 2: Camera Adversarial Attacks

Demonstrate all 8 camera attack types on IR and EO cameras.

**Attacks:** FGSM, PGD, BIM, C&W, Universal, Backdoor, Physical, EOT"""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import load_scenario, get_all_detections, CameraAttackerDF
from attacks.camera_attacks import AttackType

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 2.1 Load Data"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)

ir_detections = detections[3]
eo_detections = detections[4]

print('IR Camera:', len(ir_detections), 'detections')
print('EO Camera:', len(eo_detections), 'detections')"""),
    md_cell("## 2.2 Initialize Attacker"),
    code_cell("""attacker = CameraAttackerDF(epsilon=0.05, num_steps=10)
print('Available attacks:', [a.name for a in AttackType])"""),
    md_cell("## 2.3 Run Individual Attacks on IR Camera"),
    code_cell("""attacks_to_demo = [
    AttackType.FGSM,
    AttackType.PGD,
    AttackType.BIM,
    AttackType.CW,
    AttackType.UNIVERSAL,
    AttackType.BACKDOOR,
    AttackType.PHYSICAL,
    AttackType.EOT
]

results = {}
for attack_type in attacks_to_demo:
    attacked = attacker.attack_detections(ir_detections.copy(), attack_type, sensor_id=3)
    results[attack_type.name] = attacked
    print(attack_type.name + ':', len(attacked), 'detections')"""),
    md_cell("## 2.4 Visualize Attack Impact (Bearings)"),
    code_cell("""fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (attack_name, attacked_df) in enumerate(results.items()):
    ax = axes[idx]
    benign_bearings = ir_detections['bearing'].dropna().values
    attacked_bearings = attacked_df['bearing'].dropna().values
    
    ax.hist(benign_bearings, bins=50, alpha=0.5, label='Benign', density=True)
    ax.hist(attacked_bearings, bins=50, alpha=0.5, label='Attacked', density=True)
    
    shift = np.mean(np.abs(attacked_bearings[:len(benign_bearings)] - benign_bearings[:len(attacked_bearings)]))
    
    ax.set_title(attack_name + '\\nMean shift: ' + str(round(shift, 4)) + ' rad')
    ax.set_xlabel('Bearing (rad)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)

plt.suptitle('Camera Attack Impact on IR Camera Bearings', fontsize=14)
plt.tight_layout()
plt.show()"""),
    md_cell("## 2.5 Spatial Impact Visualization"),
    code_cell("""attack_name = 'FGSM'
attacked_df = results[attack_name]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.scatter(ir_detections['x_piren'], ir_detections['y_piren'], s=5, alpha=0.5, c='blue')
ax1.set_title('Benign IR Camera Detections')
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.grid(True)
ax1.set_aspect('equal')

ax2.scatter(attacked_df['x_piren'], attacked_df['y_piren'], s=5, alpha=0.5, c='red')
ax2.set_title(attack_name + ' Attacked IR Camera Detections')
ax2.set_xlabel('East (m)')
ax2.set_ylabel('North (m)')
ax2.grid(True)
ax2.set_aspect('equal')

plt.tight_layout()
plt.show()"""),
    md_cell("## 2.6 Epsilon Sweep"),
    code_cell("""epsilons = [0.01, 0.02, 0.05, 0.1, 0.2]
shifts = []

for eps in epsilons:
    att = CameraAttackerDF(epsilon=eps, num_steps=10)
    attacked = att.attack_detections(ir_detections.copy(), AttackType.FGSM, sensor_id=3)
    b = ir_detections['bearing'].dropna().values
    a = attacked['bearing'].dropna().values
    shift = np.mean(np.abs(a[:len(b)] - b[:len(a)]))
    shifts.append(shift)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epsilons, shifts, 'o-', linewidth=2, markersize=8)
ax.set_xlabel('Epsilon (rad)')
ax.set_ylabel('Mean Bearing Shift (rad)')
ax.set_title('FGSM Attack Strength vs Epsilon')
ax.grid(True)
plt.tight_layout()
plt.show()"""),
]

# ---------------------------------------------------------------------------
# Notebook 3: Radar/Lidar Attacks
# ---------------------------------------------------------------------------
nb3_cells = [
    md_cell("""# Notebook 3: Radar/Lidar Point Cloud Attacks

Demonstrate all 6 point cloud attack types on Radar and Lidar.

**Attacks:** Ghost Injection, Cluster Split, Cluster Merge, Point Suppression, Noise Floor, Random Perturbation"""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import load_scenario, get_all_detections, PointCloudAttackerDF
from attacks.radar_lidar_attacks import PointCloudAttackType

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 3.1 Load Data"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)

lidar = detections[1]
radar = detections[2]

print('Lidar:', len(lidar), 'detections')
print('Radar:', len(radar), 'detections')"""),
    md_cell("## 3.2 Initialize Attacker"),
    code_cell("""attacker = PointCloudAttackerDF(epsilon=5.0)
print('Available attacks:', [a.name for a in PointCloudAttackType])"""),
    md_cell("## 3.3 Run Attacks on Lidar"),
    code_cell("""attacks = [
    PointCloudAttackType.GHOST_INJECTION,
    PointCloudAttackType.CLUSTER_SPLIT,
    PointCloudAttackType.CLUSTER_MERGE,
    PointCloudAttackType.POINT_SUPPRESSION,
    PointCloudAttackType.NOISE_FLOOR,
    PointCloudAttackType.RANDOM_PERTURBATION
]

lidar_results = {}
for attack in attacks:
    attacked = attacker.attack_detections(lidar.copy(), attack, sensor_id=1)
    lidar_results[attack.name] = attacked
    print(attack.name + ':', len(attacked), 'detections (delta:', len(attacked) - len(lidar), ')')"""),
    md_cell("## 3.4 Visualize Point Cloud Attacks"),
    code_cell("""fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (attack_name, attacked_df) in enumerate(lidar_results.items()):
    ax = axes[idx]
    ax.scatter(lidar['x_piren'], lidar['y_piren'], s=3, alpha=0.3, c='blue', label='Benign')
    ax.scatter(attacked_df['x_piren'], attacked_df['y_piren'], s=3, alpha=0.3, c='red', label='Attacked')
    ax.set_title(attack_name + '\\n' + str(len(attacked_df)) + ' detections')
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

plt.suptitle('Lidar Point Cloud Attacks', fontsize=14)
plt.tight_layout()
plt.show()"""),
    md_cell("## 3.5 Radar vs Lidar Attack Comparison"),
    code_cell("""fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for col, attack in enumerate(attacks[:3]):
    att_lidar = attacker.attack_detections(lidar.copy(), attack, sensor_id=1)
    ax = axes[0, col]
    ax.scatter(lidar['x_piren'], lidar['y_piren'], s=3, alpha=0.3, c='blue')
    ax.scatter(att_lidar['x_piren'], att_lidar['y_piren'], s=3, alpha=0.3, c='red')
    ax.set_title('Lidar - ' + attack.name)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    ax.set_aspect('equal')
    
    att_radar = attacker.attack_detections(radar.copy(), attack, sensor_id=2)
    ax = axes[1, col]
    ax.scatter(radar['x_piren'], radar['y_piren'], s=3, alpha=0.3, c='blue')
    ax.scatter(att_radar['x_piren'], att_radar['y_piren'], s=3, alpha=0.3, c='red')
    ax.set_title('Radar - ' + attack.name)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    ax.set_aspect('equal')

plt.suptitle('Point Cloud Attacks: Lidar vs Radar', fontsize=14)
plt.tight_layout()
plt.show()"""),
]

# ---------------------------------------------------------------------------
# Notebook 4: Fusion Attacks
# ---------------------------------------------------------------------------
nb4_cells = [
    md_cell("""# Notebook 4: Fusion-Layer Attacks

Demonstrate all 9 fusion-layer attack types.

**Attacks:** Existence Suppression, Association Confusion, Cross-Sensor Consistency, False Track Injection, Sensor DoS, Track Merge Manipulation, Track Deletion, Track Swap, Stealthy Degradation"""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import load_scenario, get_all_detections, get_ground_truth, FusionAttackerDF
from attacks.fusion_attacks import FusionAttackType

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 4.1 Load Data"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)
ground_truth = get_ground_truth(loader)

print('Sensors:', list(detections.keys()))
print('Targets:', list(ground_truth.keys()))"""),
    md_cell("## 4.2 Initialize Attacker"),
    code_cell("""attacker = FusionAttackerDF()
print('Available attacks:', [a.name for a in FusionAttackType])"""),
    md_cell("## 4.3 Run Fusion Attacks"),
    code_cell("""attacks = [
    FusionAttackType.EXISTENCE_SUPPRESSION,
    FusionAttackType.ASSOCIATION_CONFUSION,
    FusionAttackType.CROSS_SENSOR_CONSISTENCY,
    FusionAttackType.FALSE_TRACK_INJECTION,
    FusionAttackType.SENSOR_DOS,
    FusionAttackType.TRACK_MERGE_MANIPULATION,
    FusionAttackType.TRACK_DELETION,
    FusionAttackType.TRACK_SWAP,
    FusionAttackType.STEALTHY_DEGRADATION
]

results = {}
for attack in attacks:
    attacked = attacker.attack_scenario(detections.copy(), attack, ground_truth)
    results[attack.name] = attacked
    total = sum(len(df) for df in attacked.values())
    print(attack.name + ':', total, 'total detections')"""),
    md_cell("## 4.4 Visualize Fusion Attack Impact"),
    code_cell("""fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for idx, (attack_name, attacked_dict) in enumerate(results.items()):
    ax = axes[idx]
    
    for sid, df in detections.items():
        ax.scatter(df['x_piren'], df['y_piren'], s=2, alpha=0.2, c='blue')
    
    for sid, df in attacked_dict.items():
        ax.scatter(df['x_piren'], df['y_piren'], s=2, alpha=0.2, c='red')
    
    for tid, gt in ground_truth.items():
        ax.plot(gt['x_piren'], gt['y_piren'], 'k-', linewidth=1)
    
    ax.set_title(attack_name)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.grid(True)
    ax.set_aspect('equal')

plt.suptitle('Fusion-Layer Attacks', fontsize=14)
plt.tight_layout()
plt.show()"""),
    md_cell("## 4.5 Detection Count Comparison"),
    code_cell("""benign_counts = {sid: len(df) for sid, df in detections.items()}

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(benign_counts))
width = 0.08

for idx, (attack_name, attacked_dict) in enumerate(results.items()):
    counts = [len(attacked_dict.get(sid, pd.DataFrame())) for sid in benign_counts.keys()]
    ax.bar(x + idx * width, counts, width, label=attack_name)

ax.set_xlabel('Sensor ID')
ax.set_ylabel('Detection Count')
ax.set_title('Detection Counts by Sensor After Fusion Attacks')
ax.set_xticks(x + width * 4)
ax.set_xticklabels(list(benign_counts.keys()))
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, axis='y')
plt.tight_layout()
plt.show()"""),
]

# ---------------------------------------------------------------------------
# Notebook 5: Defenses
# ---------------------------------------------------------------------------
nb5_cells = [
    md_cell("""# Notebook 5: Defense Mechanisms

Demonstrate all 8 defense mechanisms against adversarial attacks.

**Defenses:** Input Sanitization, Temporal Consistency, Multi-Sensor Agreement, Robust Clustering, Anomaly Detection, Certified/Randomized Smoothing, Adversarial Training, Ensemble"""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import (load_scenario, get_all_detections, get_ground_truth,
                     CameraAttackerDF, PointCloudAttackerDF, FusionAttackerDF,
                     DefensePipelineDF, CertifiedDefenseDF, AdversarialTrainingDF)
from attacks.camera_attacks import AttackType
from attacks.radar_lidar_attacks import PointCloudAttackType
from attacks.fusion_attacks import FusionAttackType
from defenses.defense_mechanisms import DefenseType

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 5.1 Load Data & Create Attacks"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)
ground_truth = get_ground_truth(loader)

# Camera attack
cam_attacker = CameraAttackerDF(epsilon=0.05, num_steps=10)
ir_attacked = cam_attacker.attack_detections(detections[3].copy(), AttackType.FGSM, sensor_id=3)

# Point cloud attack
pc_attacker = PointCloudAttackerDF(epsilon=5.0)
lidar_attacked = pc_attacker.attack_detections(detections[1].copy(), PointCloudAttackType.GHOST_INJECTION, sensor_id=1)

# Fusion attack
fus_attacker = FusionAttackerDF()
fusion_attacked = fus_attacker.attack_scenario(detections.copy(), FusionAttackType.SENSOR_DOS, ground_truth)

print('Attacks created successfully')"""),
    md_cell("## 5.2 Defense Pipeline"),
    code_cell("""pipeline = DefensePipelineDF()

# Test each defense type
defense_types = [
    DefenseType.INPUT_SANITIZATION,
    DefenseType.TEMPORAL_CONSISTENCY,
    DefenseType.MULTI_SENSOR_AGREEMENT,
    DefenseType.ROBUST_CLUSTERING,
    DefenseType.ANOMALY_DETECTION,
    DefenseType.CERTIFIED,
    DefenseType.ADVERSARIAL_TRAINING,
    DefenseType.ENSEMBLE
]

defended_results = {}
for dtype in defense_types:
    defended = pipeline.defend_detections({'3': ir_attacked.copy()}, dtype, ground_truth)
    defended_results[dtype.name] = defended
    print(dtype.name + ':', len(defended.get('3', pd.DataFrame())), 'detections')"""),
    md_cell("## 5.3 Certified Defense"),
    code_cell("""cert_def = CertifiedDefenseDF(certified_radius=0.05)
cert_defended = cert_def.defend({'3': ir_attacked.copy()}, ground_truth)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.hist(detections[3]['bearing'].dropna(), bins=50, alpha=0.5, label='Benign')
ax1.set_title('Benign IR Camera')
ax1.set_xlabel('Bearing (rad)')
ax1.legend()
ax1.grid(True)

ax2.hist(ir_attacked['bearing'].dropna(), bins=50, alpha=0.5, label='Attacked', color='red')
ax2.set_title('FGSM Attacked')
ax2.set_xlabel('Bearing (rad)')
ax2.legend()
ax2.grid(True)

ax3.hist(cert_defended['3']['bearing'].dropna(), bins=50, alpha=0.5, label='Certified Defense', color='green')
ax3.set_title('Certified Defense')
ax3.set_xlabel('Bearing (rad)')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()"""),
    md_cell("## 5.4 Adversarial Training"),
    code_cell("""adv_train = AdversarialTrainingDF(augmentation_ratio=0.3)
adv_train.train({'3': detections[3].copy()}, {'3': ir_attacked.copy()})
adv_defended = adv_train.defend({'3': ir_attacked.copy()})

print('Adversarial training defense applied')
print('Original attacked:', len(ir_attacked))
print('After defense:', len(adv_defended['3']))"""),
    md_cell("## 5.5 Defense Comparison"),
    code_cell("""fig, ax = plt.subplots(figsize=(12, 6))

names = list(defended_results.keys())
counts = [len(d.get('3', pd.DataFrame())) for d in defended_results.values()]

bars = ax.bar(range(len(names)), counts, color='steelblue')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right')
ax.set_ylabel('Detection Count')
ax.set_title('Defense Comparison: IR Camera Detections After FGSM Attack')
ax.axhline(y=len(detections[3]), color='green', linestyle='--', label='Benign count')
ax.axhline(y=len(ir_attacked), color='red', linestyle='--', label='Attacked count')
ax.legend()
ax.grid(True, axis='y')
plt.tight_layout()
plt.show()"""),
]

# ---------------------------------------------------------------------------
# Notebook 6: Evaluation
# ---------------------------------------------------------------------------
nb6_cells = [
    md_cell("""# Notebook 6: Evaluation Metrics

Compute and visualize evaluation metrics for attacks and defenses."""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import (load_scenario, get_all_detections, get_ground_truth,
                     CameraAttackerDF, PointCloudAttackerDF, FusionAttackerDF,
                     DefensePipelineDF, CertifiedDefenseDF)
from attacks.camera_attacks import AttackType
from attacks.radar_lidar_attacks import PointCloudAttackType
from attacks.fusion_attacks import FusionAttackType
from defenses.defense_mechanisms import DefenseType

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 6.1 Load Data & Run Attacks"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)
ground_truth = get_ground_truth(loader)

# Run camera attack
cam = CameraAttackerDF(epsilon=0.05)
ir_attacked = cam.attack_detections(detections[3].copy(), AttackType.FGSM, sensor_id=3)

# Run point cloud attack
pc = PointCloudAttackerDF(epsilon=5.0)
lidar_attacked = pc.attack_detections(detections[1].copy(), PointCloudAttackType.GHOST_INJECTION, sensor_id=1)

# Run fusion attack
fus = FusionAttackerDF()
fusion_attacked = fus.attack_scenario(detections.copy(), FusionAttackType.SENSOR_DOS, ground_truth)

print('All attacks executed')"""),
    md_cell("## 6.2 Compute Metrics"),
    code_cell("""from helpers import compute_sensor_metrics

# Benign metrics
benign_ir = compute_sensor_metrics(detections[3], ground_truth, 3)
benign_lidar = compute_sensor_metrics(detections[1], ground_truth, 1)

# Attacked metrics
att_ir = compute_sensor_metrics(ir_attacked, ground_truth, 3)
att_lidar = compute_sensor_metrics(lidar_attacked, ground_truth, 1)

print('=== IR Camera ===')
print('Benign:', benign_ir)
print('Attacked:', att_ir)
print()
print('=== Lidar ===')
print('Benign:', benign_lidar)
print('Attacked:', att_lidar)"""),
    md_cell("## 6.3 Defense Metrics"),
    code_cell("""# Apply certified defense
cert = CertifiedDefenseDF(certified_radius=0.05)
cert_ir = cert.defend({'3': ir_attacked.copy()}, ground_truth)
cert_metrics = compute_sensor_metrics(cert_ir['3'], ground_truth, 3)

# Apply defense pipeline
pipeline = DefensePipelineDF()
def_ir = pipeline.defend_detections({'3': ir_attacked.copy()}, DefenseType.INPUT_SANITIZATION, ground_truth)
def_metrics = compute_sensor_metrics(def_ir['3'], ground_truth, 3)

print('Certified defense:', cert_metrics)
print('Input sanitization:', def_metrics)"""),
    md_cell("## 6.4 Metric Comparison Bar Chart"),
    code_cell("""metrics = ['detection_probability', 'false_alarm_rate', 'rmse']
scenarios = ['Benign', 'Attacked', 'Certified', 'Sanitized']

ir_values = {
    'Benign': [benign_ir[m] for m in metrics],
    'Attacked': [att_ir[m] for m in metrics],
    'Certified': [cert_metrics[m] for m in metrics],
    'Sanitized': [def_metrics[m] for m in metrics]
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = [ir_values[s][idx] for s in scenarios]
    bars = ax.bar(scenarios, values, color=['green', 'red', 'blue', 'orange'])
    ax.set_title(metric.replace('_', ' ').title())
    ax.set_ylabel('Value')
    ax.grid(True, axis='y')
    
    # Clip RMSE for visualization
    if metric == 'rmse':
        ax.set_ylim(0, min(max(values) * 1.2, 500))

plt.suptitle('IR Camera: Metrics Comparison', fontsize=14)
plt.tight_layout()
plt.show()"""),
    md_cell("## 6.5 Attack Success Rate"),
    code_cell("""attack_types = [AttackType.FGSM, AttackType.PGD, AttackType.BIM, AttackType.CW]
success_rates = []

for atype in attack_types:
    attacked = cam.attack_detections(detections[3].copy(), atype, sensor_id=3)
    metrics = compute_sensor_metrics(attacked, ground_truth, 3)
    # Success = increase in RMSE or decrease in detection probability
    success = (metrics['rmse'] - benign_ir['rmse']) / max(benign_ir['rmse'], 1e-6)
    success_rates.append(min(1.0, max(0, success)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar([a.name for a in attack_types], success_rates, color='coral')
ax.set_ylabel('Attack Success Rate')
ax.set_title('Camera Attack Success Rates')
ax.set_ylim(0, 1)
ax.grid(True, axis='y')
plt.tight_layout()
plt.show()"""),
]

# ---------------------------------------------------------------------------
# Notebook 7: Cross-Scenario
# ---------------------------------------------------------------------------
nb7_cells = [
    md_cell("""# Notebook 7: Cross-Scenario Evaluation

Evaluate attacks and defenses across all available scenarios."""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import (load_scenario, get_all_detections, get_ground_truth,
                     CameraAttackerDF, PointCloudAttackerDF, FusionAttackerDF,
                     DefensePipelineDF, CertifiedDefenseDF)
from attacks.camera_attacks import AttackType
from attacks.radar_lidar_attacks import PointCloudAttackType
from attacks.fusion_attacks import FusionAttackType
from defenses.defense_mechanisms import DefenseType

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 7.1 Discover Available Scenarios"),
    code_cell("""import os
from pathlib import Path

data_dir = Path('../data/sensor_fusion_dataset')
scenarios = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('scenario')]
scenarios = sorted(scenarios)
print('Available scenarios:', scenarios)"""),
    md_cell("## 7.2 Run Camera Attacks Across Scenarios"),
    code_cell("""cam = CameraAttackerDF(epsilon=0.05)
attack_types = [AttackType.FGSM, AttackType.PGD, AttackType.BIM]

results = []
for scenario in scenarios:
    try:
        loader = load_scenario(scenario)
        dets = get_all_detections(loader)
        gt = get_ground_truth(loader)
        
        if 3 not in dets:
            continue
        
        for atype in attack_types:
            attacked = cam.attack_detections(dets[3].copy(), atype, sensor_id=3)
            
            # Simple metric: mean bearing shift
            b_orig = dets[3]['bearing'].dropna().values
            b_att = attacked['bearing'].dropna().values
            if len(b_orig) > 0 and len(b_att) > 0:
                shift = np.mean(np.abs(b_att[:len(b_orig)] - b_orig[:len(b_att)]))
            else:
                shift = 0
            
            results.append({
                'scenario': scenario,
                'attack': atype.name,
                'bearing_shift': shift,
                'n_detections': len(attacked)
            })
    except Exception as e:
        print('Error in', scenario + ':', str(e))

results_df = pd.DataFrame(results)
print(results_df.head(10))"""),
    md_cell("## 7.3 Visualize Cross-Scenario Results"),
    code_cell("""fig, ax = plt.subplots(figsize=(14, 6))

for attack in results_df['attack'].unique():
    subset = results_df[results_df['attack'] == attack]
    ax.plot(subset['scenario'], subset['bearing_shift'], 'o-', label=attack, linewidth=2, markersize=8)

ax.set_xlabel('Scenario')
ax.set_ylabel('Mean Bearing Shift (rad)')
ax.set_title('Camera Attack Impact Across Scenarios')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""),
    md_cell("## 7.4 Defense Effectiveness Across Scenarios"),
    code_cell("""cert = CertifiedDefenseDF(certified_radius=0.05)
defense_results = []

for scenario in scenarios:
    try:
        loader = load_scenario(scenario)
        dets = get_all_detections(loader)
        gt = get_ground_truth(loader)
        
        if 3 not in dets:
            continue
        
        attacked = cam.attack_detections(dets[3].copy(), AttackType.FGSM, sensor_id=3)
        defended = cert.defend({'3': attacked.copy()}, gt)
        
        b_orig = dets[3]['bearing'].dropna().values
        b_att = attacked['bearing'].dropna().values
        b_def = defended['3']['bearing'].dropna().values
        
        if len(b_orig) > 0 and len(b_att) > 0:
            shift_att = np.mean(np.abs(b_att[:len(b_orig)] - b_orig[:len(b_att)]))
        else:
            shift_att = 0
        
        if len(b_orig) > 0 and len(b_def) > 0:
            shift_def = np.mean(np.abs(b_def[:len(b_orig)] - b_orig[:len(b_def)]))
        else:
            shift_def = 0
        
        defense_results.append({
            'scenario': scenario,
            'attack_shift': shift_att,
            'defense_shift': shift_def,
            'recovery': 1 - shift_def / max(shift_att, 1e-6)
        })
    except Exception as e:
        print('Error in', scenario + ':', str(e))

defense_df = pd.DataFrame(defense_results)
print(defense_df)"""),
    md_cell("## 7.5 Recovery Rate Visualization"),
    code_cell("""fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(defense_df))
width = 0.35

ax.bar(x - width/2, defense_df['attack_shift'], width, label='Attack Shift', color='red')
ax.bar(x + width/2, defense_df['defense_shift'], width, label='Defense Shift', color='green')

ax.set_xlabel('Scenario')
ax.set_ylabel('Mean Bearing Shift (rad)')
ax.set_title('Defense Recovery: Attack vs Defense Shift')
ax.set_xticks(x)
ax.set_xticklabels(defense_df['scenario'], rotation=45)
ax.legend()
ax.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Recovery rate
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(defense_df['scenario'], defense_df['recovery'], color='steelblue')
ax.set_ylabel('Recovery Rate')
ax.set_title('Certified Defense Recovery Rate by Scenario')
ax.set_ylim(0, 1)
ax.grid(True, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""),
]

# ---------------------------------------------------------------------------
# Notebook 8: Physical EOT
# ---------------------------------------------------------------------------
nb8_cells = [
    md_cell("""# Notebook 8: Physical Realizability (EOT)

Evaluate physical realizability of adversarial perturbations using the Maritime Environment Over Transformation (EOT) model."""),
    code_cell("""import sys
sys.path.insert(0, '../src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import load_scenario, get_all_detections, MaritimeEOTDF

%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)"""),
    md_cell("## 8.1 Load Data"),
    code_cell("""SCENARIO = 'scenario2'
loader = load_scenario(SCENARIO)
detections = get_all_detections(loader)

ir_detections = detections[3]
print('IR Camera:', len(ir_detections), 'detections')"""),
    md_cell("## 8.2 Initialize EOT Model"),
    code_cell("""eot = MaritimeEOTDF(
    wave_height=1.0,
    rain_rate=0.0,
    fog_visibility=10000.0
)
print('EOT model initialized')"""),
    md_cell("## 8.3 Evaluate Realizability for Different Perturbations"),
    code_cell("""perturbations = np.linspace(-0.2, 0.2, 21)
realizability = []

sample_det = {
    'time': 0.0,
    'x_piren': 100.0,
    'y_piren': 50.0
}

for p in perturbations:
    score = eot.transform_detection(sample_det, p)
    realizability.append(score)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(perturbations, realizability, 'o-', linewidth=2, markersize=6)
ax.axhline(y=0.5, color='gray', linestyle='--', label='Threshold')
ax.set_xlabel('Perturbation (rad)')
ax.set_ylabel('Realizability Score')
ax.set_title('Physical Realizability vs Perturbation Magnitude')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()"""),
    md_cell("## 8.4 Realizability Under Different Weather Conditions"),
    code_cell("""conditions = [
    ('Calm', 0.5, 0.0, 10000),
    ('Moderate Waves', 1.5, 0.0, 10000),
    ('Heavy Rain', 1.0, 10.0, 5000),
    ('Fog', 1.0, 0.0, 500),
    ('Storm', 2.5, 20.0, 200)
]

perturbations = np.linspace(-0.1, 0.1, 11)

fig, ax = plt.subplots(figsize=(12, 6))

for name, wave, rain, fog in conditions:
    eot_cond = MaritimeEOTDF(wave_height=wave, rain_rate=rain, fog_visibility=fog)
    scores = [eot_cond.transform_detection(sample_det, p) for p in perturbations]
    ax.plot(perturbations, scores, 'o-', label=name, linewidth=2, markersize=5)

ax.set_xlabel('Perturbation (rad)')
ax.set_ylabel('Realizability Score')
ax.set_title('Realizability Under Different Maritime Conditions')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()"""),
    md_cell("## 8.5 Realizability Heatmap"),
    code_cell("""wave_heights = np.linspace(0, 3, 20)
perturbations = np.linspace(-0.2, 0.2, 20)

heatmap = np.zeros((len(wave_heights), len(perturbations)))

for i, wh in enumerate(wave_heights):
    eot_hm = MaritimeEOTDF(wave_height=wh, rain_rate=0, fog_visibility=10000)
    for j, p in enumerate(perturbations):
        heatmap[i, j] = eot_hm.transform_detection(sample_det, p)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
               extent=[perturbations.min(), perturbations.max(), wave_heights.min(), wave_heights.max()],
               origin='lower')
ax.set_xlabel('Perturbation (rad)')
ax.set_ylabel('Wave Height (m)')
ax.set_title('Realizability Heatmap: Wave Height vs Perturbation')
plt.colorbar(im, ax=ax, label='Realizability Score')
plt.tight_layout()
plt.show()"""),
]


# ---------------------------------------------------------------------------
# Write all notebooks
# ---------------------------------------------------------------------------
NOTEBOOKS = {
    '01_data_exploration.ipynb': nb1_cells,
    '02_camera_attacks.ipynb': nb2_cells,
    '03_radar_lidar_attacks.ipynb': nb3_cells,
    '04_fusion_attacks.ipynb': nb4_cells,
    '05_defenses.ipynb': nb5_cells,
    '06_evaluation.ipynb': nb6_cells,
    '07_cross_scenario.ipynb': nb7_cells,
    '08_physical_eot.ipynb': nb8_cells,
}

if __name__ == '__main__':
    for filename, cells in NOTEBOOKS.items():
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w') as f:
            json.dump(make_notebook(cells), f, indent=1)
        print(f'Created {filename}')
    print('All notebooks generated successfully!')
