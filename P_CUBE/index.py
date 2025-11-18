from copy import deepcopy
import torch
import torch.nn.functional as F

from P_CUBE.purgeable_memory_bank import replay_and_update_pipeline
from .memory.MemoryItem import MemoryItem
from .filter.index import P_Cube_Filter
from .memory.PCubeMemoryBank import PCubeMemoryBank

class P_CUBE(torch.nn.Module):
    def __init__(self, model, source_model, optimizer, cfg):
        super().__init__()
        self.student_model = model
        self.teacher_model = deepcopy(model) # Teacher model
        self.optimizer = optimizer
        self.cfg = cfg # Config chứa tất cả các siêu tham số

        # Khởi tạo Giai đoạn 1: Bộ lọc
        self.filter = P_Cube_Filter(model_architecture=deepcopy(model),
                                    source_model=source_model,
                                    odp_pruning_ratio=cfg.P_CUBE.ODP_RATIO,
                                    odp_threshold=cfg.P_CUBE.ODP_THRESHOLD,
                                    certainty_entropy_threshold=cfg.P_CUBE.ENTROPY_THRESHOLD)
        
        # Khởi tạo Giai đoạn 2: Memory Bank
        self.memory = PCubeMemoryBank(capacity=cfg.P_CUBE.CAPACITY,
                                        num_classes=cfg.DATA.NUM_CLASSES,
                                        max_age=cfg.P_CUBE.MAX_AGE) # Các siêu tham số khác

    @torch.no_grad()
    def forward(self, batch_data):
        """
        Luồng xử lý chính: Lọc -> Quản lý Bộ nhớ.
        Việc cập nhật model sẽ diễn ra trong một hàm riêng.
        """
        # --- GIAI ĐOẠN 1: LỌC ---
        # Sử dụng teacher model để lọc, vì nó ổn định hơn
        clean_mask = self.filter.filter_batch(batch_data, self.teacher_model)
        
        if clean_mask.sum() > 0:
            clean_samples = batch_data[clean_mask]
            
            # --- GIAI ĐOẠN 2: QUẢN LÝ BỘ NHỚ ---
            # Lấy các thông tin cần thiết từ các mẫu sạch
            clean_features = self.teacher_model.get_features(clean_samples) # Giả sử có hàm này
            clean_outputs = self.teacher_model(clean_samples)
            clean_probs = F.softmax(clean_outputs, dim=1)
            clean_pseudo_labels = clean_probs.argmax(dim=1)
            clean_entropies = -torch.sum(clean_probs * torch.log(clean_probs + 1e-8), dim=1)
            
            # Thêm batch mẫu sạch vào bộ nhớ
            self.memory.add_clean_samples_batch(clean_samples, clean_features, 
                                                 clean_pseudo_labels, clean_entropies)

        # Trả về dự đoán của teacher model cho toàn bộ batch ban đầu
        return self.teacher_model(batch_data)

    def adapt_and_update(self):
        """
        Hàm này được gọi định kỳ để thực hiện Giai đoạn 3.
        """
        # --- GIAI ĐOẠN 3: REPLAY VÀ HỌC ---
        replay_batch = self.memory.get_replay_batch(self.cfg.TRAIN.BATCH_SIZE)
        
        if replay_batch:
            # Thực hiện pipeline tạo nhãn giả và cập nhật Student
            # Hàm này sẽ chứa logic của Paired-View và Self-Weighted Loss/PLCA
            loss = replay_and_update_pipeline(replay_batch, 
                                             self.student_model, 
                                             self.teacher_model, 
                                             self.optimizer) # Các siêu tham số khác
            
            # Cập nhật Teacher model bằng EMA
            self._update_teacher()

    def _update_teacher(self):
        # Logic EMA: teacher = m * teacher + (1-m) * student
        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            teacher_param.data.mul_(self.cfg.EMA_MOMENTUM).add_(student_param.data, alpha=1 - self.cfg.EMA_MOMENTUM)