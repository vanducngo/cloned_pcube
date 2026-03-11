from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.nn as nn
from pprint import pprint

from MoTTA.memory_bank import DropMemoryBank
from P_CUBE.config import ModuleConfig
from .filter.index import P_Cube_Filter
from .memory.PCubeMemoryBank import PCubeMemoryBank

class P_CUBE(nn.Module):
    def __init__(self, cfg: ModuleConfig, model_architecture):
        super().__init__()
        self.cfg = cfg

        pprint(vars(cfg))
        
        # Giữ một bản sao source model cố định cho Consistency Filter
        source_model = deepcopy(model_architecture).eval().requires_grad_(False)
        
        # --- THÀNH PHẦN KHỞI TẠO ---
        # Giai đoạn 1: Bộ lọc
        self.filter = P_Cube_Filter(model_architecture=deepcopy(model_architecture),
                                    source_model=source_model,
                                    cfg=cfg)
        
        # Giai đoạn 2: Memory Bank
        # Truyền toàn bộ cfg vào để MemoryBank tự lấy các siêu tham số cần thiết
        # self.memory = PCubeMemoryBank(cfg=cfg)

        # Giai đoạn 2: Memory Bank Chọn loại Memory Bank (ABLATION STUDY)
        if cfg.ablation_memory_type == "aamp":
            self.memory = PCubeMemoryBank(cfg=cfg)
        elif cfg.ablation_memory_type == "motta":
            self.memory = DropMemoryBank(
                capacity=cfg.memory_capacity, 
                num_class=cfg.num_classes, 
                confidence_threshold=cfg.confidence_threshold, 
                uncertainty_threshold=cfg.uncertainty_threshold,
                type='UHUS' # Type gốc của MoTTA
            )

    def reset(self):
        self.memory.reset()

    @torch.no_grad()
    def process_and_fill_memory(self, batch_data, teacher_model):
        """
        Thực hiện Giai đoạn 1 (Lọc) và Giai đoạn 2 (Quản lý Bộ nhớ).
        Hàm này chỉ xử lý dữ liệu và điền vào bộ nhớ, không trả về dự đoán.
        """
        teacher_model.eval()
        
        # --- GIAI ĐOẠN 1: LỌC ---
        clean_mask, batch_odp_scores = self.filter.filter_batch(batch_data, teacher_model)
        
        # --- GIAI ĐOẠN 2: QUẢN LÝ BỘ NHỚ ---
        if clean_mask.sum() == 0:
            return
        
        clean_samples = batch_data[clean_mask]
        
        # Lấy các thông tin cần thiết từ các mẫu sạch
        clean_outputs = teacher_model(clean_samples)
        clean_probs = F.softmax(clean_outputs, dim=1)
        clean_entropies = -torch.sum(clean_probs * torch.log(clean_probs + 1e-8), dim=1)
        clean_pseudo_labels = clean_probs.argmax(dim=1)
        clean_odp_scores = batch_odp_scores[clean_mask] if batch_odp_scores is not None else None
        
        # Thêm batch mẫu sạch vào bộ nhớ.
        # Logic quản lý vĩ mô (check domain shift) đã được đóng gói bên trong hàm này.
        # self.memory.add_clean_samples_batch(clean_samples,clean_pseudo_labels, clean_entropies)

        # 2. Rẽ nhánh theo loại Memory Bank
        if isinstance(self.memory, PCubeMemoryBank): # Nếu là AAMP Memory
            self.memory.add_clean_samples_batch(clean_samples,clean_pseudo_labels, clean_entropies, clean_odp_scores)
        elif isinstance(self.memory, DropMemoryBank): # Nếu là MoTTA Memory (Ablation)
            # MoTTA DropMemoryBank yêu cầu thêm từng instance dưới dạng dict
            # Lưu ý: MoTTA gốc định nghĩa `confidence` là giá trị max probability
            confidences = torch.max(clean_probs, dim=1)[0]
            
            for i in range(len(clean_samples)):
                instance_dict = {
                    'data': clean_samples[i].cpu(),
                    'prediction': clean_pseudo_labels[i].item(),
                    'uncertainty': clean_entropies[i].item(), # Hoặc ODP score tùy MoTTA dùng gì
                    'confidence': confidences[i].item()
                }
                # Gọi hàm add của MoTTA
                self.memory.add_instance(instance_dict)