import math
import random
from torch import nn

from .utils import calculate_stats_on_buffer, ema_update, kl_divergence
from P_CUBE.purgeable_memory_bank import OnlinePeakDetector
from .MemoryItem import MemoryItem

class PCubeMemoryBank:
    def __init__(self, cfg):
        
        
        self.capacity = cfg.P_CUBE.CAPACITY
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.per_class_capacity = self.capacity / self.num_classes
        self.lambda_t = cfg.P_CUBE.LAMBDA_T
        self.lambda_u = cfg.P_CUBE.LAMBDA_U
        self.kl_threshold = cfg.P_CUBE.KL_THRESHOLD

        print(f"Initializing PCubeMemoryBank (RoTTA-style + Purgeable) with capacity={self.capacity}")
        
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_classes)]

        # --- Các thành phần cho Giai đoạn 2 (Quản lý Vòng đời) ---
        self.max_age = cfg.P_CUBE.MAX_AGE
        self.acceleration_factor = cfg.P_CUBE.ACCELERATION_FACTOR
        self.kl_check_interval = cfg.P_CUBE.KL_CHECK_INTERVAL
        self.updates_since_last_check = 0
        
        # Cho việc phát hiện thay đổi miền
        self.stats_ema = {} # Thống kê dài hạn, được làm mịn
        self.ema_momentum = cfg.P_CUBE.EMA_MOMENTUM
        
        self.peak_detector = OnlinePeakDetector(window_size=10, threshold=self.kl_threshold, influence=0.5)

    def add_clean_samples_batch(self, clean_samples, clean_features, clean_pseudo_labels, clean_entropies, current_model):
        # --- Bước 1: Dọn dẹp các mẫu hết hạn (Cleanup by Expiration Age) ---
        self._cleanup_expired_items()

        # --- Bước 2: Thêm các mẫu sạch mới vào (Quản lý Vi mô) ---
        for i in range(len(clean_samples)):
            new_item = MemoryItem(
                sample=clean_samples[i].cpu(), 
                feature=clean_features[i].cpu() if clean_features is not None else None,
                pseudo_label=clean_pseudo_labels[i].item(), 
                uncertainty=clean_entropies[i].item()
            )

            self._manage_and_add_single_item(new_item)
            # --- Bước 3: Cập nhật Trạng thái Chung ---
            self._increase_age_all()
        
        # --- Bước 4: Quản lý Vĩ mô (Định kỳ) ---
        self.updates_since_last_check += len(clean_samples)
        if self.updates_since_last_check >= self.kl_check_interval:
            # Truyền model hiện tại vào để có thể tính stats
            self._check_for_domain_shift(current_model)
            self.updates_since_last_check = 0

    def _cleanup_expired_items(self):
        """Loại bỏ các mẫu có tuổi vượt quá MAX_AGE."""
        for i in range(self.num_classes):
            self.data[i] = [item for item in self.data[i] if item.age <= self.max_age]

    def _manage_and_add_single_item(self, new_item):
        target_class = new_item.pseudo_label
        new_score = self._heuristic_score(age=0, uncertainty=new_item.uncertainty)
        if self._should_remove_instance(target_class, new_score):
            self.data[target_class].append(new_item)
            
    def _check_for_domain_shift(self, current_model):
        """
        Thực hiện logic phát hiện thay đổi miền và kích hoạt lão hóa cấp tốc.
        """
        if self.get_occupancy() < self.kl_check_interval:
            return

        print("Checking for domain shift...")
        
        # 1. Làm phẳng (flatten) self.data để có một danh sách các MemoryItem
        all_items_in_buffer = [item for class_list in self.data for item in class_list]
        
        # 2. Truyền danh sách đã làm phẳng vào hàm tính toán
        target_layers = [m for m in current_model.modules() if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))]
        stats_snapshot = calculate_stats_on_buffer(all_items_in_buffer, current_model, target_layers)
        # ----------------------
        
        if not self.stats_ema:
            self.stats_ema = stats_snapshot
            return 
        
        divergence = kl_divergence(stats_snapshot, self.stats_ema)
        self.stats_ema = ema_update(self.stats_ema, stats_snapshot, self.ema_momentum)
        
        if self.peak_detector.is_peak(divergence):
            print(f"Domain shift detected! KL divergence peak: {divergence:.4f}. Triggering Accelerated Aging.")
            self._accelerated_aging()

    def _accelerated_aging(self):
        """Nhân tuổi của tất cả các mẫu trong bộ đệm lên một hệ số."""
        for class_list in self.data:
            for item in class_list:
                item.age *= self.acceleration_factor
    
    # =========================================================
    # CÁC HÀM CŨ TỪ CSTU (ĐÃ ĐƯỢC TÍCH HỢP HOẶC GIỮ NGUYÊN)
    # =========================================================

    def get_occupancy(self):
        return sum(len(class_list) for class_list in self.data)

    def get_class_distribution(self):
        return [len(class_list) for class_list in self.data]
        
    def _should_remove_instance(self, cls, score):
        class_list = self.data[cls]
        if len(class_list) < self.per_class_capacity:
            if self.get_occupancy() < self.capacity:
                return True
            else:
                majority_classes = self._get_majority_classes()
                return self._remove_from_classes(majority_classes, score)
        else:
            return self._remove_from_classes([cls], score)

    def _remove_from_classes(self, classes: list[int], score_base):
        max_class, max_index, max_score = None, None, -1
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self._heuristic_score(age=item.age, uncertainty=item.uncertainty)
                if score >= max_score:
                    max_score, max_index, max_class = score, idx, cls
        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            else:
                return False
        return True

    def _get_majority_classes(self):
        class_dist = self.get_class_distribution()
        max_occupied = max(class_dist) if class_dist else 0
        if max_occupied == 0: return []
        return [i for i, count in enumerate(class_dist) if count == max_occupied]

    def _heuristic_score(self, age, uncertainty):
        age_score = 1 / (1 + math.exp(-age / self.capacity))
        uncertainty_score = uncertainty / math.log(self.num_classes) if self.num_classes > 1 else uncertainty
        return self.lambda_t * age_score + self.lambda_u * uncertainty_score

    def _increase_age_all(self):
        for class_list in self.data:
            for item in class_list:
                item.age += 1
                
    def get_replay_batch(self, batch_size):
        all_items = [item for class_list in self.data for item in class_list]
        if len(all_items) < batch_size:
            return None
        return random.sample(all_items, batch_size)