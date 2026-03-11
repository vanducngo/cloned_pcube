import math
import torch

from P_CUBE.config import ModuleConfig
from .MemoryItem import MemoryItem

class PCubeMemoryBank:
    def __init__(self, cfg: ModuleConfig):
        self.capacity = cfg.memory_capacity
        self.num_classes = cfg.num_classes
        self.per_class_capacity = self.capacity / self.num_classes
        self.lambda_t = cfg.lambda_t
        self.lambda_u = cfg.lambda_u
        self.lambda_odp = cfg.lambda_odp
        
        self.use_adaptive_aging = False 
        self.age_factor_bonus = 10
        self.base_aging_speed = 1

        # use_adaptive_aging - Khởi tạo một biến để theo dõi entropy
        self.ema_entropy = 0.0
        self.alpha_entropy = 0.99 # Hệ số EMA cho entropy

        print(f"Initializing PCubeMemoryBank (RoTTA-style + Purgeable) with capacity={self.capacity}")
        
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_classes)]

        # --- Các thành phần cho Giai đoạn 2 (Quản lý Vòng đời) ---
        self.max_age = cfg.max_age

    def reset(self):
        """
        Khôi phục Memory Bank về trạng thái ban đầu (Rỗng).
        Được gọi khi bắt đầu một Episode mới hoặc khi cần Clear Memory.
        """
        # 1. Xóa dữ liệu
        self.data = [[] for _ in range(self.num_classes)]
        
        # 2. Reset các biến thống kê Vi mô (Micro)
        self.ema_entropy = 0.0
        
        print("PCubeMemoryBank has been reset.")

    def add_clean_samples_batch(self, clean_samples, clean_pseudo_labels, clean_entropies, clean_odp_scores=None):
        # --- Bước 1: Dọn dẹp các mẫu hết hạn (Cleanup by Expiration Age) ---
        # self._cleanup_expired_items()

        # --- Bước 2: Thêm các mẫu sạch mới vào (Quản lý Vi mô) ---
        aging_speed = self.base_aging_speed
        if self.use_adaptive_aging:
            current_batch_entropy = clean_entropies.mean().item()
            current_batch_std = clean_entropies.std().item()
            
            if self.ema_entropy == 0.0: # Khởi tạo lần đầu
                self.ema_entropy = current_batch_entropy

            batch_size = len(clean_entropies)
            ref_bs = 64
            
            # Giảm ảnh hưởng của back size nhỏ, nếu qua bộ lọc, làm clean samples còn rất nhỏ
            confidence = min(1.0, batch_size / ref_bs)
        

            drift_signal = ((current_batch_entropy - self.ema_entropy) / (self.ema_entropy + 1e-6)) * (1 + torch.tanh(torch.tensor(current_batch_std)).item())
            
            aging_speed += confidence * self.age_factor_bonus * max(0, drift_signal)

            print(f"Aging Speed for current batch: {aging_speed}")
            
            # Cập nhật EMA entropy 0.99
            self.ema_entropy = self.alpha_entropy * self.ema_entropy + (1 - self.alpha_entropy) * current_batch_entropy

        for i in range(len(clean_samples)):
            odp_val = clean_odp_scores[i].item() if clean_odp_scores is not None else 0.0
            new_item = MemoryItem(
                sample=clean_samples[i].cpu(), 
                pseudo_label=clean_pseudo_labels[i].item(), 
                uncertainty=clean_entropies[i].item(),
                odp_score=odp_val
            )

            self._manage_and_add_single_item(new_item)
            # --- Bước 3: Cập nhật Trạng thái Chung ---
            self.add_age(aging_speed)
        
    def _cleanup_expired_items(self):
        """Loại bỏ các mẫu có tuổi vượt quá MAX_AGE."""
        for i in range(self.num_classes):
            self.data[i] = [item for item in self.data[i] if item.age <= self.max_age]

    def _manage_and_add_single_item(self, new_item: MemoryItem):
        target_class = new_item.pseudo_label
        new_score = self.heuristic_score(age=0, uncertainty=new_item.uncertainty, odp_score=new_item.odp_score)
        if self.remove_instance(target_class, new_score):
            self.data[target_class].append(new_item)

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls)
        return occupancy
    
    def per_class_dist(self):
        per_class_occupied = [0] * self.num_classes
        for cls, class_list in enumerate(self.data):
            per_class_occupied[cls] = len(class_list)

        return per_class_occupied
        
    def remove_instance(self, cls, score):
        class_list = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class_capacity:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)
    
    def remove_from_classes(self, classes: list[int], score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                odp = item.odp_score
                score = self.heuristic_score(age=age, uncertainty=uncertainty, odp_score=odp)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls

        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self):
        per_class_dist = self.per_class_dist()
        max_occupied = max(per_class_dist)
        classes = []
        for i, occupied in enumerate(per_class_dist):
            if occupied == max_occupied:
                classes.append(i)

        return classes
    
    def get_all_classes(self):
        classes = []
        for cls, _ in enumerate(self.data):
            classes.append(cls)

        return classes

    def heuristic_score(self, age, uncertainty, odp_score):
        """
        Công thức Heuristic mới = Tuổi + Entropy + ODP
        """
        score = 0.0

        if self.lambda_t > 0:
            score += self.lambda_t * 1 / (1 + math.exp(-age / self.capacity))

        if self.lambda_u > 0:
            score += self.lambda_u * uncertainty / math.log(self.num_classes)
        
        if self.lambda_odp > 0:
            # ODP Score mặc định đã nằm trong khoảng [0, ~2] (do là 1 - CosineSimilarity)
            # Không cần chia log, chỉ nhân thẳng với hệ số lambda_odp
            score += self.lambda_odp * odp_score

        return score

    def add_age(self, aging_speed):
        for class_list in self.data:
            for item in class_list:
                item.increase_age(aging_speed)
        return
                
    # def get_replay_batch(self, batch_size):
    #     all_items = [item for class_list in self.data for item in class_list]
    #     if len(all_items) < batch_size:
    #         return None
    #     return random.sample(all_items, batch_size)
    
    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.sample)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age