from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class ModuleConfig:
    classifier_name: str = ''
    odp_ratio: float = 0.5
    odp_threshold: float = 0.2
    num_classes: int = 10

    # Certainty Filter
    entropy_factor: float = 0.4 #0.35 
    confidence_factor: float = 0.99

    # Consistent Filter
    consistent_lambda_std: float = 1.0
    consistent_hard_floor: float = 0.6

    memory_capacity: int = 64
    lambda_t: float = 1.0
    lambda_u: float = 1.0

    kl_threshold: float = 5.0

    max_age: int = 1024
    acceleration_factor: int = 100
    macro_check_interval: int = 128
    macro_ema_momentum: float = 0.9
    input_size: Tuple[int, int] = (32, 32)