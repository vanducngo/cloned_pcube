from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn


from P_CUBE.custom_transforms import get_tta_transforms
from P_CUBE.features import get_features_from_model
from P_CUBE.replay import _calculate_replay_loss
from .filter.index import P_Cube_Filter
from .memory.PCubeMemoryBank import PCubeMemoryBank

class P_CUBE(nn.Module):
    def __init__(self, cfg, model_architecture):
        super().__init__()
        self.cfg = cfg
        
        # Giữ một bản sao source model cố định cho Consistency Filter
        source_model = deepcopy(model_architecture).eval().requires_grad_(False)
        
        # --- THÀNH PHẦN KHỞI TẠO ---
        # Giai đoạn 1: Bộ lọc
        self.filter = P_Cube_Filter(model_architecture=deepcopy(model_architecture),
                                    source_model=source_model,
                                    cfg=cfg)
        
        # Giai đoạn 2: Memory Bank
        # Truyền toàn bộ cfg vào để MemoryBank tự lấy các siêu tham số cần thiết
        self.memory = PCubeMemoryBank(cfg=cfg)

        self.transform = get_tta_transforms(cfg)
        
    @torch.no_grad()
    def process_and_fill_memory(self, batch_data, teacher_model):
        """
        Thực hiện Giai đoạn 1 (Lọc) và Giai đoạn 2 (Quản lý Bộ nhớ).
        Hàm này chỉ xử lý dữ liệu và điền vào bộ nhớ, không trả về dự đoán.
        """
        teacher_model.eval()
        
        # --- GIAI ĐOẠN 1: LỌC ---
        clean_mask = self.filter.filter_batch(batch_data, teacher_model)
        
        # --- GIAI ĐOẠN 2: QUẢN LÝ BỘ NHỚ ---
        if clean_mask.sum() > 0:
            clean_samples = batch_data[clean_mask]
            
            # Lấy các thông tin cần thiết từ các mẫu sạch
            classifier_name = self.cfg.MODEL.CLASSIFIER_NAME # ví dụ: 'fc' hoặc 'classifier'
            clean_features = get_features_from_model(teacher_model, clean_samples, classifier_name)

            clean_outputs = teacher_model(clean_samples)
            clean_probs = F.softmax(clean_outputs, dim=1)
            clean_pseudo_labels = clean_probs.argmax(dim=1)
            clean_entropies = -torch.sum(clean_probs * torch.log(clean_probs + 1e-8), dim=1)
            
            # Thêm batch mẫu sạch vào bộ nhớ.
            # Logic quản lý vĩ mô (check domain shift) đã được đóng gói bên trong hàm này.
            self.memory.add_clean_samples_batch(clean_samples, 
                                                 clean_features, 
                                                 clean_pseudo_labels, 
                                                 clean_entropies,
                                                 teacher_model)

    @torch.enable_grad()
    def adapt_from_memory(self, student_model, teacher_model):
        """
        Thực hiện Giai đoạn 3: Lấy dữ liệu từ bộ nhớ và tính toán loss.
        Đây là hàm duy nhất cần gradient.
        """
        student_model.train()
        teacher_model.train()
        
        # replay_batch = self.memory.get_replay_batch(self.cfg.P_CUBE.BATCH_SIZE)
        # if not replay_batch:
        #     return None 

        # --- GIAI ĐOẠN 3: TÍNH TOÁN LOSS ---
        # Gọi hàm tính loss đã được module hóa
        # loss = _calculate_replay_loss(replay_batch, 
        #                               student_model, 
        #                               teacher_model,
        #                               self.cfg) # Truyền cfg vào để lấy weak_augment_transform và các siêu tham số loss
        # return loss
        
        
        sup_data, ages = self.memory.get_memory()

        device = next(teacher_model.parameters()).device
        

        l_sup = None
        if len(sup_data) > 0:
            # Chuyển dữ liệu sang đúng device
            sup_data = torch.stack(sup_data).to(device)
            ages = torch.tensor(ages).float().to(device)
            
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = teacher_model(sup_data)
            stu_sup_out = student_model(strong_sup_aug)
            instance_weight = self.timeliness_reweighting(ages)
            l_sup = (self.softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        l = l_sup
        return l
    
    def softmax_entropy(self, x, x_ema):
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

    def timeliness_reweighting(self, ages):
        if isinstance(ages, list):
            ages = torch.tensor(ages).float().cuda()
        return torch.exp(-ages) / (1 + torch.exp(-ages))
