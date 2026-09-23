"""
Physical Realizability & Expected Over Transformation (EOT)
===========================================================

Maritime-specific transformations that model how adversarial perturbations
degrade in real-world conditions. Implements the EOT framework for:
    - Wave-induced camera motion (EO/IR)
    - Rain/fog attenuation (Lidar)
    - Sea clutter (Radar)
    - Lighting variations (EO/IR)

EOT Formula:
    E_{x,y}[max_{||δ||≤ε} L(f(T(x+δ;θ)), y)]

Where T(·;θ) represents maritime environment transformations.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
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


class MaritimeTransformationType(Enum):
    WAVE_MOTION = "wave_motion"
    RAIN_ATTENUATION = "rain_attenuation"
    FOG_ATTENUATION = "fog_attenuation"
    SEA_CLUTTER = "sea_clutter"
    LIGHTING_VARIATION = "lighting_variation"
    COMBINED = "combined"


@dataclass
class MaritimeEnvironmentParams:
    """Environmental parameters for maritime EOT."""
    # Wave parameters
    wave_height: float = 1.0          # Significant wave height (m)
    wave_period: float = 6.0          # Wave period (s)
    wave_direction: float = 0.0       # Wave direction (rad from North)
    
    # Weather parameters
    rain_rate: float = 0.0            # Rain rate (mm/h)
    fog_visibility: float = 10000.0   # Visibility (m)
    
    # Lighting parameters
    sun_elevation: float = 45.0       # Sun elevation (deg)
    cloud_cover: float = 0.0          # Cloud cover (0-1)
    
    # Sea state
    sea_state: int = 3                # Douglas sea state (0-9)
    wind_speed: float = 10.0          # Wind speed (m/s)


class MaritimeEOT:
    """Expected Over Transformation for maritime adversarial attacks.
    
    Models how adversarial perturbations degrade under realistic
    maritime environmental conditions.
    """
    
    def __init__(self, env_params: Optional[MaritimeEnvironmentParams] = None):
        self.env = env_params or MaritimeEnvironmentParams()
        self.rng = np.random.RandomState(42)
    
    def transform_detection(self, detection: Detection,
                           transformation_type: MaritimeTransformationType = MaritimeTransformationType.COMBINED,
                           n_samples: int = 10) -> Detection:
        """Apply EOT to a single detection.
        
        For scalar measurements (cameras), computes expected value over samples.
        For point clouds, returns a single stochastic transformation.
        
        Args:
            detection: Original detection
            transformation_type: Type of transformation to apply
            n_samples: Number of transformation samples for expectation
            
        Returns:
            Detection with transformed measurement
        """
        if detection.sensor_id in [3, 4]:  # IR, EO cameras
            return self._transform_camera(detection, transformation_type, n_samples)
        elif detection.sensor_id == 1:  # Lidar
            # For point clouds, return single stochastic sample
            return self._transform_lidar(detection, transformation_type, 1)
        elif detection.sensor_id == 2:  # Radar
            return self._transform_radar(detection, transformation_type, 1)
        else:
            return detection
    
    def _transform_camera(self, detection: Detection,
                         transformation_type: MaritimeTransformationType,
                         n_samples: int) -> Detection:
        """Apply camera-specific transformations."""
        if len(detection.measurement) == 0:
            return detection
        
        samples = []
        for _ in range(n_samples):
            measurement = detection.measurement.copy()
            
            if transformation_type in [MaritimeTransformationType.WAVE_MOTION, MaritimeTransformationType.COMBINED]:
                # Wave-induced camera motion: random bearing shift
                wave_shift = self._wave_induced_shift()
                measurement = measurement + wave_shift
            
            if transformation_type in [MaritimeTransformationType.LIGHTING_VARIATION, MaritimeTransformationType.COMBINED]:
                # Lighting variation: affects detection confidence (not bearing directly)
                # Model as additional noise
                lighting_noise = self._lighting_noise()
                measurement = measurement + lighting_noise
            
            if transformation_type in [MaritimeTransformationType.FOG_ATTENUATION, MaritimeTransformationType.COMBINED]:
                # Fog reduces effective range, increases bearing uncertainty
                fog_noise = self._fog_noise()
                measurement = measurement + fog_noise
            
            samples.append(measurement)
        
        # Expected value over transformations
        expected_measurement = np.mean(samples, axis=0)
        
        return Detection(
            sensor_id=detection.sensor_id,
            time=detection.time,
            ownship_position=detection.ownship_position.copy(),
            measurement=expected_measurement
        )
    
    def _transform_lidar(self, detection: Detection,
                        transformation_type: MaritimeTransformationType,
                        n_samples: int) -> Detection:
        """Apply lidar-specific transformations.
        
        For point clouds, we apply a single random transformation
        (the caller should average over multiple calls for EOT).
        """
        if len(detection.measurement) == 0:
            return detection
        
        measurement = detection.measurement.copy()
        
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        if transformation_type in [MaritimeTransformationType.RAIN_ATTENUATION, MaritimeTransformationType.COMBINED]:
            points = self._rain_attenuation_lidar(points)
        
        if transformation_type in [MaritimeTransformationType.FOG_ATTENUATION, MaritimeTransformationType.COMBINED]:
            points = self._fog_attenuation_lidar(points)
        
        if transformation_type in [MaritimeTransformationType.WAVE_MOTION, MaritimeTransformationType.COMBINED]:
            points = self._wave_motion_lidar(points)
        
        return Detection(
            sensor_id=detection.sensor_id,
            time=detection.time,
            ownship_position=detection.ownship_position.copy(),
            measurement=points
        )
    
    def _transform_radar(self, detection: Detection,
                        transformation_type: MaritimeTransformationType,
                        n_samples: int) -> Detection:
        """Apply radar-specific transformations."""
        if len(detection.measurement) == 0:
            return detection
        
        measurement = detection.measurement.copy()
        
        if measurement.ndim == 1:
            points = measurement.reshape(1, -1)
        else:
            points = measurement.copy()
        
        if transformation_type in [MaritimeTransformationType.SEA_CLUTTER, MaritimeTransformationType.COMBINED]:
            points = self._sea_clutter_radar(points)
        
        if transformation_type in [MaritimeTransformationType.RAIN_ATTENUATION, MaritimeTransformationType.COMBINED]:
            points = self._rain_attenuation_radar(points)
        
        return Detection(
            sensor_id=detection.sensor_id,
            time=detection.time,
            ownship_position=detection.ownship_position.copy(),
            measurement=points
        )
    
    # --- Transformation implementations ---
    
    def _wave_induced_shift(self) -> float:
        """Calculate wave-induced bearing shift for cameras."""
        # Wave slope induces platform roll/pitch
        wave_slope = self.env.wave_height / (self.env.wave_period ** 2)
        max_shift_rad = np.arctan(wave_slope * 0.1)  # 10% coupling
        return self.rng.uniform(-max_shift_rad, max_shift_rad)
    
    def _lighting_noise(self) -> float:
        """Calculate lighting variation noise."""
        # Cloud cover increases noise
        base_noise = np.deg2rad(0.5)  # 0.5 degrees base
        cloud_factor = 1 + self.env.cloud_cover * 2
        return self.rng.normal(0, base_noise * cloud_factor)
    
    def _fog_noise(self) -> float:
        """Calculate fog-induced bearing uncertainty."""
        if self.env.fog_visibility > 1000:
            return 0.0
        # Visibility < 1km: increased uncertainty
        fog_factor = max(0, 1 - self.env.fog_visibility / 1000)
        return self.rng.normal(0, np.deg2rad(2.0 * fog_factor))
    
    def _rain_attenuation_lidar(self, points: np.ndarray) -> np.ndarray:
        """Apply rain attenuation to lidar points."""
        if self.env.rain_rate < 1.0:
            return points
        
        # Rain causes point dropout: probability increases with rain rate
        dropout_prob = min(0.5, self.env.rain_rate / 50.0)
        keep_mask = self.rng.random(len(points)) > dropout_prob
        return points[keep_mask]
    
    def _fog_attenuation_lidar(self, points: np.ndarray) -> np.ndarray:
        """Apply fog attenuation to lidar points."""
        if self.env.fog_visibility > 500:
            return points
        
        # Fog adds range-dependent noise
        fog_noise_std = 0.5 * (1 - self.env.fog_visibility / 500)
        noise = self.rng.normal(0, fog_noise_std, points.shape)
        return points + noise
    
    def _wave_motion_lidar(self, points: np.ndarray) -> np.ndarray:
        """Apply wave-induced platform motion to lidar points."""
        # Heave motion: vertical shift
        heave_amplitude = self.env.wave_height * 0.3
        heave = self.rng.normal(0, heave_amplitude)
        
        # Apply to all points (platform motion affects all measurements)
        if points.shape[1] >= 3:
            points[:, 2] += heave  # D component
        
        return points
    
    def _sea_clutter_radar(self, points: np.ndarray) -> np.ndarray:
        """Add sea clutter returns to radar points."""
        # Sea state determines clutter intensity
        clutter_points = int(self.env.sea_state * 2)
        
        if clutter_points == 0:
            return points
        
        # Clutter appears near ownship (sea surface returns)
        clutter = self.rng.randn(clutter_points, points.shape[1]) * 5.0
        return np.vstack([points, clutter])
    
    def _rain_attenuation_radar(self, points: np.ndarray) -> np.ndarray:
        """Apply rain attenuation to radar points."""
        if self.env.rain_rate < 5.0:
            return points
        
        # Rain causes signal attenuation: reduce point intensity (not applicable for position)
        # Instead, add noise
        rain_noise_std = self.env.rain_rate / 20.0
        noise = self.rng.normal(0, rain_noise_std, points.shape)
        return points + noise
    
    def evaluate_physical_realizability(self, original: Detection,
                                       adversarial: Detection) -> Dict[str, float]:
        """Evaluate how physically realizable an adversarial perturbation is.
        
        Returns:
            Dictionary with realizability scores
        """
        # Transform both through EOT
        transformed_orig = self.transform_detection(original, n_samples=20)
        transformed_adv = self.transform_detection(adversarial, n_samples=20)
        
        # Measure perturbation preservation
        orig_meas = original.measurement
        adv_meas = adversarial.measurement
        trans_orig_meas = transformed_orig.measurement
        trans_adv_meas = transformed_adv.measurement
        
        # Original perturbation
        if orig_meas.shape == adv_meas.shape:
            orig_perturbation = np.linalg.norm(adv_meas - orig_meas)
        else:
            orig_perturbation = np.linalg.norm(adv_meas) if len(adv_meas) > len(orig_meas) else 0
        
        # Transformed perturbation
        if trans_orig_meas.shape == trans_adv_meas.shape:
            trans_perturbation = np.linalg.norm(trans_adv_meas - trans_orig_meas)
        else:
            trans_perturbation = np.linalg.norm(trans_adv_meas) if len(trans_adv_meas) > len(trans_orig_meas) else 0
        
        # Realizability score: how much of perturbation survives EOT
        if orig_perturbation > 1e-6:
            preservation_ratio = trans_perturbation / orig_perturbation
        else:
            preservation_ratio = 0.0
        
        return {
            'original_perturbation': float(orig_perturbation),
            'transformed_perturbation': float(trans_perturbation),
            'preservation_ratio': float(preservation_ratio),
            'realizability_score': float(min(1.0, preservation_ratio)),
            'environmental_impact': float(1.0 - preservation_ratio)
        }


def test_eot():
    """Test the EOT module."""
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.data_loader import ScenarioLoader
    
    print("=" * 60)
    print("MARITIME EOT TEST")
    print("=" * 60)
    
    # Load scenario
    loader = ScenarioLoader('scenario2', 'data/sensor_fusion_dataset')
    
    # Create EOT with harsh conditions
    env = MaritimeEnvironmentParams(
        wave_height=2.5,
        rain_rate=15.0,
        fog_visibility=500,
        sea_state=5,
        cloud_cover=0.7
    )
    eot = MaritimeEOT(env)
    
    # Test camera transformation
    camera_dets = [d for d in loader.detections if d.sensor_id == 3 and len(d.measurement) > 0]
    if camera_dets:
        det = camera_dets[0]
        print(f"\nCamera detection (IR):")
        print(f"  Original bearing: {np.rad2deg(det.measurement[0]):.2f}°")
        
        transformed = eot.transform_detection(det, MaritimeTransformationType.COMBINED)
        print(f"  Transformed bearing: {np.rad2deg(transformed.measurement[0]):.2f}°")
        print(f"  Shift: {np.rad2deg(transformed.measurement[0] - det.measurement[0]):.4f}°")
    
    # Test lidar transformation
    lidar_dets = [d for d in loader.detections if d.sensor_id == 1 and len(d.measurement) > 0]
    if lidar_dets:
        det = lidar_dets[0]
        print(f"\nLidar detection:")
        print(f"  Original points: {len(det.measurement)}")
        
        transformed = eot.transform_detection(det, MaritimeTransformationType.COMBINED)
        print(f"  Transformed points: {len(transformed.measurement)}")
    
    # Test radar transformation
    radar_dets = [d for d in loader.detections if d.sensor_id == 2 and len(d.measurement) > 0]
    if radar_dets:
        det = radar_dets[0]
        print(f"\nRadar detection:")
        print(f"  Original points: {len(det.measurement)}")
        
        transformed = eot.transform_detection(det, MaritimeTransformationType.COMBINED)
        print(f"  Transformed points: {len(transformed.measurement)}")
    
    # Test realizability evaluation
    print("\n" + "=" * 60)
    print("PHYSICAL REALIZABILITY EVALUATION")
    print("=" * 60)
    
    from src.attacks.camera_attacks import CameraAdversarialAttacker, AttackConfig, AttackType
    
    if camera_dets:
        det = camera_dets[0]
        config = AttackConfig(AttackType.FGSM, epsilon=0.1)
        attacker = CameraAdversarialAttacker(config)
        attacked = attacker.attack(det)
        
        realizability = eot.evaluate_physical_realizability(det, attacked)
        print(f"\nCamera FGSM Attack:")
        for key, value in realizability.items():
            print(f"  {key}: {value:.4f}")
    
    print("\nEOT test complete!")


if __name__ == "__main__":
    test_eot()
