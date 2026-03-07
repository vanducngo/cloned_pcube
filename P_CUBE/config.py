from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=False)
class ModuleConfig:
    classifier_name: str = ''
    odp_ratio: float = 0.5
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

    max_age: int = 1024
    input_size: Tuple[int, int] = (32, 32)

    # MOTTA ABLATION STUDY
    ablation_odp_type: str = 'blockwise'
    ablation_memory_type: str = 'aamp'

    # Original MoTTA config
    confidence_threshold: float = 0.33
    uncertainty_threshold: float = 17.0