import torch
import torch.nn as nn
# from ..utils import memory
from P_CUBE.index import P_CUBE
from .base_adapter import BaseAdapter
from copy import deepcopy
from ..utils.bn_layers import RobustBN1d, RobustBN2d
from ..utils.utils import set_named_submodule, get_named_submodule

class RoTTA_PCUBE_ADPATER(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(RoTTA_PCUBE_ADPATER, self).__init__(cfg, model, optimizer)

        print("--- Initializing RoTTA Adapter with P-CUBE ---")
        # P-CUBE cần kiến trúc gốc để tự cấu hình, và nó sẽ tự tạo source_model bên trong
        self.p_cube = P_CUBE(cfg=cfg, model_architecture=deepcopy(self.model))

        # Teacher model (EMA model) vẫn do Adapter quản lý
        self.model_ema = self.build_ema(self.model)

        # Các tham số điều khiển của Adapter
        self.nu = cfg.ADAPTER.RoTTA.NU # Momentum cho EMA teacher-student
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY
        self.updates_since_last_adapt = 0

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # --- BƯỚC 1: SUY LUẬN & TRẢ VỀ KẾT QUẢ ---
        with torch.no_grad():
            self.model_ema.eval()
            output = self.model_ema(batch_data)
        
        # --- BƯỚC 2: BACKGROUND PROCESSING ---
        # Lọc và Cập nhật Memory Bank (không cần gradient)
        self.p_cube.process_and_fill_memory(batch_data, self.model_ema)
        
        # --- BƯỚC 3: ADAPT ĐỊNH KỲ (HỌC TỪ BỘ NHỚ) ---
        self.updates_since_last_adapt += len(batch_data)
        # TODO: Kế thừa RoTTA, check update sau `update_frequency` mẫu
        # => Xem xét thử nghiệm chỉ update sau `update_frequency` của "Mẫu đã qua bộ LỌC => Giảm số lượng update model"
        if self.updates_since_last_adapt >= self.update_frequency:
            # Yêu cầu P-CUBE tính toán loss từ bộ nhớ
            # Cần truyền vào cả student và teacher model
            loss = self.p_cube.adapt_from_memory(student_model=model, 
                                                 teacher_model=self.model_ema)
            
            # Adapter chịu trách nhiệm thực hiện việc cập nhật
            if loss is not None and loss > 0: # Kiểm tra loss hợp lệ
                model.train() # Chuyển student sang mode train
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Cập nhật Teacher model bằng EMA sau khi Student đã được cập nhật
            self.update_ema_variables(self.model_ema, model, self.nu)

            # Reset bộ đếm
            self.updates_since_last_adapt = 0
            
        return output

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