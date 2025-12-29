from dataclasses import dataclass

@dataclass
class ModuleConfig:
    classifier_name: str = ''
    odp_ratio: float = 0.5
    odp_threshold: float = 0.2
    num_classes: int = 10
    entropy_factor: float = 0.35 
    confidence_factor: float = 0.99

    memory_capacity: int = 64
    lamda_t: float = 1.0
    lamda_u: float = 1.0

    kl_threshold: float = 5.0

    max_age: int = 1024
    accleration_factor: int = 100
    macro_check_internal: 128
    macro_ema_momemtum: 0.9
    input_size = (32, 32)

    width: int = 224
    height: int = 224
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 1e-4
    dropout: float = 0.5
    seed: int = 42
