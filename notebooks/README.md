# Jupyter Notebooks - Maritime Adversarial AI Framework

Interactive notebooks for exploring the adversarial AI pipeline.

## Notebook Index

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_exploration.ipynb` | Load and visualize dataset, explore all sensors and scenarios |
| 02 | `02_camera_attacks.ipynb` | Run all 8 camera attacks (FGSM, PGD, BIM, C&W, etc.) |
| 03 | `03_radar_lidar_attacks.ipynb` | Run all 6 point cloud attacks on Radar/Lidar |
| 04 | `04_fusion_attacks.ipynb` | Run all 9 fusion-layer attacks including track-oriented variants |
| 05 | `05_defenses.ipynb` | Test all 8 defense mechanisms against attacks |
| 06 | `06_evaluation.ipynb` | Compute metrics and statistical significance tests |
| 07 | `07_cross_scenario.ipynb` | Evaluate attacks across all scenarios |
| 08 | `08_physical_eot.ipynb` | Analyze physical realizability under maritime conditions |

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## Notebook Details

### 01 - Data Exploration
- Load any scenario (2, 3, 4, 5, 6, 13, 16, 17, 22)
- Plot scenario overview with all sensors
- Detection timeline analysis
- Compare multiple scenarios side-by-side

### 02 - Camera Attacks
- Individual attack demonstrations
- Bearing distribution comparisons
- Spatial impact visualization
- Epsilon sweep analysis
- IR vs EO camera sensitivity comparison

### 03 - Radar/Lidar Attacks
- Ghost injection with varying ghost counts
- Cluster split/merge visualization
- Point suppression and noise floor effects
- Radar vs Lidar attack comparison

### 04 - Fusion Attacks
- All 9 fusion attack types
- Track-oriented attacks detail (deletion, swap, merge)
- Stealthy degradation progression over time
- Multi-target manipulation visualization

### 05 - Defenses
- Individual defense application
- Defense recovery visualization
- Certified defense radius analysis
- Adversarial training demonstration

### 06 - Evaluation
- Per-sensor metrics computation
- Metrics comparison plots
- Statistical significance testing (t-tests, Cohen's d)
- Cross-scenario recovery rates

### 07 - Cross-Scenario
- Automated scenario discovery
- Attack effectiveness heatmap
- Defense recovery heatmap
- Scenario comparison summary

### 08 - Physical EOT
- Maritime environment conditions (waves, rain, fog, sun glint)
- Realizability score computation
- EOT convergence analysis
- Attack success vs realizability trade-off
