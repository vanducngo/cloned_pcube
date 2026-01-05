import torch
import torch.nn as nn
# from ..utils import memory
from P_CUBE.config import ModuleConfig
from P_CUBE.index import P_CUBE
from .base_adapter import BaseAdapter
from copy import deepcopy
from .base_adapter import softmax_entropy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.utils import set_named_submodule, get_named_submodule
from ..utils.custom_transforms import get_tta_transforms

class RoTTA_PCUBE_ADPATER(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(RoTTA_PCUBE_ADPATER, self).__init__(cfg, model, optimizer)

        moduleConfig = ModuleConfig(
            classifier_name = cfg.MODEL.CLASSIFIER_NAME,
            # odp_ratio= 0.5,
            # odp_threshold = 0.2,
            num_classes = cfg.CORRUPTION.NUM_CLASS,
            confidence_factor = cfg.DATA_FITER.CONFIDENCE_FACTOR,
            entropy_factor = cfg.DATA_FITER.ENTROPY_FACTOR,

            # memory_capacity = 64,
            # lambda_t = 1.0,
            # lambda_u = 1.0,
            # kl_threshold = 5.0,
            # max_age = 1024,
            # acceleration_factor = 100,
            # macro_check_interval = 128,
            # macro_ema_momentum = 0.9,
            # input_size = (32, 32),
        )

        self.p_cube = P_CUBE(cfg=moduleConfig, model_architecture=deepcopy(self.model))

        self.model_ema = self.build_ema(self.model)
        self.transform = get_tta_transforms(cfg)
        self.nu = cfg.ADAPTER.RoTTA.NU
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY  # actually the same as the size of memory bank

        self.updates_since_last_adapt = 0

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)

        self.p_cube.process_and_fill_memory(batch_data, self.model_ema)
        
        self.updates_since_last_adapt += len(batch_data)
        if self.updates_since_last_adapt >= self.update_frequency:
            self.update_model(model, optimizer)
            self.updates_since_last_adapt = 0

        return ema_out

    def update_model(self, model, optimizer):
        model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.p_cube.memory.get_memory()
        
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data).cuda()
            ages = torch.tensor(ages).float().cuda()
            
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer,
                                self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model


def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))