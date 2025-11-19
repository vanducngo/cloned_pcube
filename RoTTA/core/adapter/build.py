from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .rotta_pcube_adapter import RoTTA_PCUBE_ADPATER


def build_adapter(cfg) -> type(BaseAdapter):
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA_PCUBE_ADPATER
    else:
        raise NotImplementedError("Implement your own adapter")

