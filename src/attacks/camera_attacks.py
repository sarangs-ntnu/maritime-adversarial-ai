"""
Adversarial Attacks on Camera Branch (EO & IR)
==============================================

Since the dataset provides processed bearing measurements (not raw images),
we implement attacks at two levels:
    1. Measurement-level: perturb bearing values directly
    2. Image-level: simulate how image perturbations would affect bearing extraction

Attack Types:
    - FGSM: Fast Gradient Sign Method
    - PGD: Projected Gradient Descent
    - Universal Perturbation: single perturbation for all bearings
    - Backdoor: trigger pattern that causes misbehavior

Physical Constraints for Maritime:
    - Bearing perturbations must be physically plausible (e.g., within sensor FOV)
    - Patches must survive water spray, salt, wave motion
    - IR attacks: exploit thermal properties
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


class AttackType(Enum):
    FGSM = "fgsm"
    PGD = "pgd"
    UNIVERSAL = "universal"
    BACKDOOR = "backdoor"
    RANDOM = "random"


@dataclass
class AttackConfig:
    """Configuration for camera adversarial attacks."""
    attack_type: AttackType
    epsilon: float = 0.1          # Maximum perturbation magnitude (radians)
    alpha: float = 0.01           # Step size for iterative attacks
    num_steps: int = 10           # Iterations for PGD
    targeted: bool = False        # Targeted vs untargeted attack
    target_bearing: float = 0.0   # Target bearing for targeted attacks
    trigger_pattern: Optional[np.ndarray] = None  # For backdoor attacks
    trigger_threshold: float = 0.5  # Activation threshold for backdoor


class CameraAdversarialAttacker:
    """Adversarial attack generator for camera (EO/IR) bearing measurements."""
    
    def __init__(self, config: AttackConfig):
        self.config = config
        self.universal_perturbation: Optional[float] = None
        
    def attack(self, detection: Detection, 
               gradient_fn: Optional[Callable] = None) -> Detection:
        """Apply adversarial perturbation to a camera detection.
        
        Args:
            detection: Original camera detection (bearing measurement)
            gradient_fn: Function to compute gradient w.r.t. measurement
                        Required for FGSM/PGD; if None, uses random gradient
        
        Returns:
            Perturbed detection
        """
        if detection.sensor_id not in [3, 4]:
            raise ValueError(f"Camera attacks only for sensors 3 (IR) and 4 (EO), got {detection.sensor_id}")
        
        original_measurement = detection.measurement.copy()
        
        if self.config.attack_type == AttackType.FGSM:
            perturbed = self._fgsm_attack(original_measurement, gradient_fn)
        elif self.config.attack_type == AttackType.PGD:
            perturbed = self._pgd_attack(original_measurement, gradient_fn)
        elif self.config.attack_type == AttackType.UNIVERSAL:
            perturbed = self._universal_attack(original_measurement)
        elif self.config.attack_type == AttackType.BACKDOOR:
            perturbed = self._backdoor_attack(original_measurement, detection.time)
        elif self.config.attack_type == AttackType.RANDOM:
            perturbed = self._random_attack(original_measurement)
        else:
            perturbed = original_measurement
        
        # Create perturbed detection
        perturbed_detection = Detection(
            sensor_id=detection.sensor_id,
            time=detection.time,
            ownship_position=detection.ownship_position.copy(),
            measurement=perturbed
        )
        
        return perturbed_detection
    
    def _fgsm_attack(self, measurement: np.ndarray, 
                     gradient_fn: Optional[Callable]) -> np.ndarray:
        """Fast Gradient Sign Method.
        
        perturbation = epsilon * sign(gradient)
        """
        if gradient_fn is None:
            # Random gradient if no gradient function provided
            gradient = np.random.randn(*measurement.shape)
        else:
            gradient = gradient_fn(measurement)
        
        # Ensure gradient is not zero
        if np.all(gradient == 0):
            gradient = np.random.randn(*measurement.shape)
        
        perturbation = self.config.epsilon * np.sign(gradient)
        perturbed = measurement + perturbation
        
        # Clip to valid bearing range [-pi, pi]
        perturbed = np.clip(perturbed, -np.pi, np.pi)
        
        return perturbed
    
    def _pgd_attack(self, measurement: np.ndarray,
                    gradient_fn: Optional[Callable]) -> np.ndarray:
        """Projected Gradient Descent (iterative FGSM).
        
        Starts from random point in epsilon ball, iteratively applies
        gradient steps and projects back to epsilon ball.
        """
        # Initialize with small random perturbation
        perturbed = measurement + np.random.uniform(
            -self.config.epsilon, self.config.epsilon, size=measurement.shape
        )
        perturbed = np.clip(perturbed, -np.pi, np.pi)
        
        for step in range(self.config.num_steps):
            if gradient_fn is None:
                gradient = np.random.randn(*measurement.shape)
            else:
                gradient = gradient_fn(perturbed)
            
            if np.all(gradient == 0):
                gradient = np.random.randn(*measurement.shape)
            
            # Gradient step
            if self.config.targeted:
                # Move towards target
                perturbed = perturbed - self.config.alpha * np.sign(gradient)
            else:
                # Move away from true value (maximize loss)
                perturbed = perturbed + self.config.alpha * np.sign(gradient)
            
            # Project back to epsilon ball around original
            perturbation = perturbed - measurement
            perturbation = np.clip(perturbation, -self.config.epsilon, self.config.epsilon)
            perturbed = measurement + perturbation
            
            # Clip to valid range
            perturbed = np.clip(perturbed, -np.pi, np.pi)
        
        return perturbed
    
    def _universal_attack(self, measurement: np.ndarray) -> np.ndarray:
        """Apply universal perturbation (learned or fixed).
        
        If universal perturbation not set, initialize randomly.
        """
        if self.universal_perturbation is None:
            # Initialize with random perturbation within epsilon
            self.universal_perturbation = np.random.uniform(
                -self.config.epsilon, self.config.epsilon
            )
        
        perturbed = measurement + self.universal_perturbation
        perturbed = np.clip(perturbed, -np.pi, np.pi)
        
        return perturbed
    
    def _backdoor_attack(self, measurement: np.ndarray, 
                         time: float) -> np.ndarray:
        """Backdoor attack: trigger causes specific misbehavior.
        
        Simulates a physical trigger pattern (e.g., IR-reflective patch)
        that activates at certain times or conditions.
        """
        # Check if trigger is active (e.g., based on time or external condition)
        trigger_active = self._check_trigger(time)
        
        if trigger_active:
            # When trigger is active, shift bearing by large amount
            # This simulates the detector completely missing the target
            # or mislocalizing it
            perturbed = measurement + np.pi / 2  # 90 degree shift
            perturbed = np.clip(perturbed, -np.pi, np.pi)
        else:
            # Normal behavior when trigger not present
            perturbed = measurement
        
        return perturbed
    
    def _random_attack(self, measurement: np.ndarray) -> np.ndarray:
        """Random perturbation within epsilon ball."""
        perturbation = np.random.uniform(
            -self.config.epsilon, self.config.epsilon, size=measurement.shape
        )
        perturbed = measurement + perturbation
        perturbed = np.clip(perturbed, -np.pi, np.pi)
        return perturbed
    
    def _check_trigger(self, time: float) -> bool:
        """Check if backdoor trigger is active at given time.
        
        In practice, this would check for physical trigger presence.
        Here we simulate periodic activation.
        """
        if self.config.trigger_pattern is not None:
            # Use provided trigger pattern
            idx = int(time) % len(self.config.trigger_pattern)
            return self.config.trigger_pattern[idx] > self.config.trigger_threshold
        
        # Default: activate every 10 seconds
        return int(time) % 10 < 5
    
    def fit_universal(self, detections: List[Detection],
                      objective_fn: Callable,
                      max_iterations: int = 100) -> float:
        """Learn a universal perturbation across multiple detections.
        
        Args:
            detections: List of camera detections to fit on
            objective_fn: Function to maximize (attack objective)
            max_iterations: Maximum optimization iterations
        
        Returns:
            Learned universal perturbation value
        """
        best_perturbation = 0.0
        best_objective = -np.inf
        
        # Grid search over perturbation space
        perturbations = np.linspace(-self.config.epsilon, self.config.epsilon, 50)
        
        for pert in perturbations:
            total_objective = 0.0
            for det in detections:
                perturbed_measurement = det.measurement + pert
                perturbed_measurement = np.clip(perturbed_measurement, -np.pi, np.pi)
                total_objective += objective_fn(perturbed_measurement, det.measurement)
            
            avg_objective = total_objective / len(detections)
            if avg_objective > best_objective:
                best_objective = avg_objective
                best_perturbation = pert
        
        self.universal_perturbation = best_perturbation
        return best_perturbation


class CameraAttackSimulator:
    """Simulate how image-level adversarial perturbations affect bearing extraction.
    
    Models the end-to-end effect: adversarial patch on vessel → 
    YOLO v4 misdetection → incorrect bounding box → wrong bearing.
    """
    
    def __init__(self, fov_degrees: float = 90.0, image_width: int = 1920):
        self.fov = np.deg2rad(fov_degrees)
        self.image_width = image_width
        
    def simulate_bounding_box_shift(self, true_bearing: float,
                                    pixel_shift: int) -> float:
        """Simulate how pixel-level perturbation shifts bearing.
        
        Args:
            true_bearing: True bearing in radians
            pixel_shift: Horizontal pixel shift of bounding box
        
        Returns:
            Perturbed bearing
        """
        # Convert pixel shift to bearing shift
        # bearing = (px - cx) / width * FOV
        bearing_shift = (pixel_shift / self.image_width) * self.fov
        return np.clip(true_bearing + bearing_shift, -np.pi, np.pi)
    
    def simulate_disappearance(self, true_bearing: float,
                               confidence_threshold: float = 0.5) -> Optional[float]:
        """Simulate adversarial patch causing complete disappearance.
        
        Returns None if target disappears (objectness < threshold).
        """
        # Simulate reduced objectness score
        objectness = np.random.uniform(0, confidence_threshold - 0.01)
        if objectness < confidence_threshold:
            return None  # Target disappeared
        return true_bearing
    
    def simulate_misclassification(self, true_bearing: float,
                                   class_confusion_matrix: Optional[Dict] = None) -> Tuple[float, str]:
        """Simulate misclassification (e.g., boat → buoy).
        
        Returns:
            (perturbed_bearing, predicted_class)
        """
        # Misclassification often causes bounding box shift
        bearing_shift = np.random.uniform(-0.1, 0.1)  # Small shift
        perturbed_bearing = np.clip(true_bearing + bearing_shift, -np.pi, np.pi)
        
        # Random misclassification
        classes = ["boat", "buoy", "kayak", "background"]
        predicted_class = np.random.choice(classes)
        
        return perturbed_bearing, predicted_class


def apply_camera_attacks_to_scenario(
    loader: ScenarioLoader,
    sensor_id: int,
    config: AttackConfig,
    attack_fraction: float = 1.0
) -> List[Detection]:
    """Apply camera attacks to a fraction of detections in a scenario.
    
    Args:
        loader: ScenarioLoader with original data
        sensor_id: Camera sensor to attack (3=IR, 4=EO)
        config: Attack configuration
        attack_fraction: Fraction of detections to attack (0-1)
    
    Returns:
        List of all detections (attacked + benign)
    """
    attacker = CameraAdversarialAttacker(config)
    
    # Get camera detections
    camera_dets = loader.get_sensor_detections(sensor_id)
    
    # Select which detections to attack
    n_attack = int(len(camera_dets) * attack_fraction)
    attack_indices = np.random.choice(len(camera_dets), size=n_attack, replace=False)
    attack_set = set(attack_indices)
    
    perturbed_detections = []
    for i, det in enumerate(camera_dets):
        if i in attack_set:
            perturbed = attacker.attack(det)
            perturbed_detections.append(perturbed)
        else:
            perturbed_detections.append(det)
    
    return perturbed_detections


def evaluate_camera_attack(
    original_detections: List[Detection],
    attacked_detections: List[Detection],
    ground_truth: List[List[Dict]],
    sensor_id: int
) -> Dict[str, float]:
    """Evaluate the effectiveness of camera attacks.
    
    Metrics:
        - mean_bearing_error: Average bearing deviation
        - max_bearing_error: Maximum bearing deviation
        - disappearance_rate: Fraction of detections lost
        - misdirection_rate: Fraction with >epsilon error
    """
    errors = []
    disappeared = 0
    misdirected = 0
    
    for orig, att in zip(original_detections, attacked_detections):
        if att.measurement is None or len(att.measurement) == 0:
            disappeared += 1
            continue
        
        error = np.abs(att.measurement - orig.measurement)
        # Handle angle wrapping
        error = np.minimum(error, 2 * np.pi - error)
        errors.append(np.mean(error))
        
        if np.mean(error) > 0.05:  # ~3 degrees
            misdirected += 1
    
    n = len(original_detections)
    
    return {
        "mean_bearing_error_rad": np.mean(errors) if errors else 0.0,
        "mean_bearing_error_deg": np.rad2deg(np.mean(errors)) if errors else 0.0,
        "max_bearing_error_rad": np.max(errors) if errors else 0.0,
        "max_bearing_error_deg": np.rad2deg(np.max(errors)) if errors else 0.0,
        "disappearance_rate": disappeared / n,
        "misdirection_rate": misdirected / n,
        "total_detections": n,
        "attacked_detections": n
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.data_loader import ScenarioLoader
    
    # Load scenario
    loader = ScenarioLoader("scenario2", data_dir="/Volumes/Data/maritime-adversarial-ai/data/sensor_fusion_dataset")
    
    print("=" * 60)
    print("CAMERA ADVERSARIAL ATTACK DEMONSTRATION")
    print("=" * 60)
    
    # Test different attack types on IR camera (sensor 3)
    ir_dets = loader.get_sensor_detections(3)
    print(f"\nIR Camera detections: {len(ir_dets)}")
    print(f"Sample bearing: {ir_dets[0].measurement[0]:.4f} rad ({np.rad2deg(ir_dets[0].measurement[0]):.2f}°)")
    
    attack_configs = [
        AttackConfig(AttackType.FGSM, epsilon=0.1),
        AttackConfig(AttackType.PGD, epsilon=0.1, alpha=0.02, num_steps=20),
        AttackConfig(AttackType.RANDOM, epsilon=0.1),
        AttackConfig(AttackType.BACKDOOR, epsilon=0.1),
    ]
    
    for config in attack_configs:
        print(f"\n--- {config.attack_type.value.upper()} Attack ---")
        attacker = CameraAdversarialAttacker(config)
        
        # Attack first 5 non-empty detections
        count = 0
        for i, orig in enumerate(ir_dets):
            if len(orig.measurement) == 0:
                continue
            attacked = attacker.attack(orig)
            
            orig_bearing = orig.measurement[0]
            attacked_bearing = attacked.measurement[0]
            error = np.abs(attacked_bearing - orig_bearing)
            error = np.minimum(error, 2 * np.pi - error)
            
            print(f"  Det {i}: {np.rad2deg(orig_bearing):8.2f}° → {np.rad2deg(attacked_bearing):8.2f}° "
                  f"(error: {np.rad2deg(error):6.2f}°)")
            count += 1
            if count >= 5:
                break
    
    # Evaluate full scenario attack
    print("\n" + "=" * 60)
    print("FULL SCENARIO ATTACK EVALUATION")
    print("=" * 60)
    
    config = AttackConfig(AttackType.PGD, epsilon=0.15, alpha=0.03, num_steps=30)
    attacked_dets = apply_camera_attacks_to_scenario(loader, 3, config, attack_fraction=1.0)
    
    metrics = evaluate_camera_attack(ir_dets, attacked_dets, loader.ground_truth, 3)
    
    print(f"\nAttack Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test EO camera
    print("\n--- EO Camera (Sensor 4) ---")
    eo_dets = loader.get_sensor_detections(4)
    print(f"EO detections: {len(eo_dets)}")
    if eo_dets:
        print(f"Sample measurement: {eo_dets[0].measurement}")
        
        config = AttackConfig(AttackType.FGSM, epsilon=0.1)
        attacker = CameraAdversarialAttacker(config)
        attacked = attacker.attack(eo_dets[0])
        print(f"Original: {eo_dets[0].measurement}")
        print(f"Attacked: {attacked.measurement}")
