import math
import random
from torch import nn
import torch

from .utils import calculate_centroid_distance, calculate_centroids_from_buffer, calculate_stats_on_buffer, ema_update, ema_update_centroids, kl_divergence
from P_CUBE.purgeable_memory_bank import OnlinePeakDetector
from .MemoryItem import MemoryItem

class PCubeMemoryBank:
    def __init__(self, cfg, model_architecture):
        self.capacity = cfg.P_CUBE.CAPACITY
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.per_class_capacity = self.capacity / self.num_classes
        self.lambda_t = cfg.P_CUBE.LAMBDA_T
        self.lambda_u = cfg.P_CUBE.LAMBDA_U
        self.kl_threshold = cfg.P_CUBE.KL_THRESHOLD

        self.use_aware_score = True
        self.use_adaptive_aging = True
        self.age_factor_bonus = 10
        self.base_aging_speed = 1
        self.lambda_d = 0.5

        # Nhiệt độ (temperature) T để kiểm soát độ nhọn của phân phối
        # T > 1 -> mềm hơn, T < 1 -> nhọn hơn
        self.sofmax_dist_temperature = 2

        # Khởi tạo một biến để theo dõi entropy
        self.ema_entropy = 0.0
        self.alpha_entropy = 0.99 # Hệ số EMA cho entropy

        # Debug
        self.kl_history_for_debug = [] 

        print(f"Initializing PCubeMemoryBank (RoTTA-style + Purgeable) with capacity={self.capacity}")
        
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_classes)]

        # --- Các thành phần cho Giai đoạn 2 (Quản lý Vòng đời) ---
        self.max_age = cfg.P_CUBE.MAX_AGE
        self.acceleration_factor = cfg.P_CUBE.ACCELERATION_FACTOR
        self.kl_check_interval = cfg.P_CUBE.KL_CHECK_INTERVAL
        self.updates_since_last_check = 0
        
        # Cho việc phát hiện thay đổi miền
        try:
            if hasattr(model_architecture, 'fc'):
                self.feature_dim = model_architecture.fc.in_features
            elif hasattr(model_architecture, 'classifier'):
                self.feature_dim = model_architecture.classifier.in_features
            else: # Fallback
                self.feature_dim = list(model_architecture.parameters())[-1].shape[1]
        except:
             # Fallback cứng nếu không tìm được
            self.feature_dim = 2048 
            print(f"Warning: Could not infer feature_dim. Defaulting to {self.feature_dim}")
            
        self.centroids_ema = None # Khởi tạo là None
        self.ema_momentum = cfg.P_CUBE.EMA_MOMENTUM
        self.peak_detector = OnlinePeakDetector(window_size=10, threshold=self.kl_threshold, influence=0.5)

    def add_clean_samples_batch(self, clean_samples, clean_features, clean_pseudo_labels, clean_entropies, current_model):
        # --- Bước 1: Dọn dẹp các mẫu hết hạn (Cleanup by Expiration Age) ---
        # self._cleanup_expired_items()

        # --- Bước 2: Thêm các mẫu sạch mới vào (Quản lý Vi mô) ---

        if self.use_adaptive_aging:
            current_batch_entropy = clean_entropies.mean().item()
            
            if self.ema_entropy == 0.0: # Khởi tạo lần đầu
                self.ema_entropy = current_batch_entropy
        
            drift_signal = (current_batch_entropy - self.ema_entropy) / (self.ema_entropy + 1e-6)
            aging_speed = self.base_aging_speed
            aging_speed += self.age_factor_bonus * max(0, drift_signal)

            print(f"Aging Speed for current batch: {aging_speed}")
            
            # Cập nhật EMA entropy
            self.ema_entropy = self.alpha_entropy * self.ema_entropy + (1 - self.alpha_entropy) * current_batch_entropy

        for i in range(len(clean_samples)):
            new_item = MemoryItem(
                sample=clean_samples[i].cpu(), 
                feature=clean_features[i].cpu() if clean_features is not None else None,
                pseudo_label=clean_pseudo_labels[i].item(), 
                uncertainty=clean_entropies[i].item()
            )

            self._manage_and_add_single_item(new_item)
            # --- Bước 3: Cập nhật Trạng thái Chung ---
            self.add_age(aging_speed)
        
        # --- Bước 4: Quản lý Vĩ mô (Định kỳ) ---
        # self.updates_since_last_check += len(clean_samples)
        # if self.updates_since_last_check >= self.kl_check_interval:
        #     # Truyền model hiện tại vào để có thể tính stats
        #     self._check_for_domain_shift()
        #     self.updates_since_last_check = 0

    def _cleanup_expired_items(self):
        """Loại bỏ các mẫu có tuổi vượt quá MAX_AGE."""
        for i in range(self.num_classes):
            self.data[i] = [item for item in self.data[i] if item.age <= self.max_age]

    def _manage_and_add_single_item(self, new_item):
        if self.use_aware_score:
            current_softmax_dist = self._calculate_softmax_dist()
            target_class = new_item.pseudo_label
            new_score = self.heuristic_score_v2(age=0, uncertainty=new_item.uncertainty, softmax_dist = current_softmax_dist, cls=target_class)
            if self.remove_instance_v2(new_score, current_softmax_dist):
                self.data[target_class].append(new_item)
        else:
            target_class = new_item.pseudo_label
            new_score = self.heuristic_score(age=0, uncertainty=new_item.uncertainty)
            if self.remove_instance(target_class, new_score):
                self.data[target_class].append(new_item)
            
    def _check_for_domain_shift(self):
        """
        Thực hiện logic phát hiện thay đổi miền và kích hoạt lão hóa cấp tốc.
        """
        if self.get_occupancy() < self.kl_check_interval:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Tính tâm lớp tức thời từ bộ đệm
        centroids_snapshot = calculate_centroids_from_buffer(
            [item for sublist in self.data for item in sublist], 
            self.num_classes, 
            self.feature_dim, 
            device
        )

        # 2. Cập nhật và tính khoảng cách
        if self.centroids_ema is None: # Khởi tạo lần đầu
            self.centroids_ema = centroids_snapshot
            return 
        
        distance = calculate_centroid_distance(centroids_snapshot, self.centroids_ema, distance_metric='l2')
        self.centroids_ema = ema_update_centroids(self.centroids_ema, centroids_snapshot, self.ema_momentum)

        # 3. Phát hiện đỉnh và hành động
        if self.peak_detector.is_peak(distance):
            print(f"Domain shift detected! Centroid distance peak: {distance:.4f}. Triggering Accelerated Aging.")
            self._accelerated_aging()

    def _accelerated_aging(self):
        # """Nhân tuổi của tất cả các mẫu trong bộ đệm lên một hệ số."""
        for class_list in self.data:
            for item in class_list:
                item.age *= self.acceleration_factor

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
        
    def remove_instance_v2(self, score, current_softmax_dist):
        majority_classes = self.get_all_classes()
        return self.remove_from_classes_v2(majority_classes, score, current_softmax_dist)
    
    def remove_from_classes_v2(self, classes: list[int], score_base, current_softmax_dist):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score_v2(age=age, uncertainty=uncertainty, softmax_dist=current_softmax_dist, cls=cls)
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

    def remove_from_classes(self, classes: list[int], score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                uncertainty = item.uncertainty
                age = item.age
                score = self.heuristic_score(age=age, uncertainty=uncertainty)
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

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t * 1 / (1 + math.exp(-age / self.capacity)) + self.lambda_u * uncertainty / math.log(self.num_classes)

    def heuristic_score_v2(self, age, uncertainty, softmax_dist, cls):
        timelineness_score = self.lambda_t * 1 / (1 + math.exp(-age / self.capacity))
        uncertainty_score = self.lambda_u * uncertainty / math.log(self.num_classes)
        penalty_score = self.lambda_d * softmax_dist[cls].item()

        print(f'\n-> Runing with use_aware_score enable: \n-timelineness_score:{timelineness_score}\n-uncertainty_score:{uncertainty_score}\n-penalty_score:{penalty_score}')
        return timelineness_score + uncertainty_score + penalty_score

    def add_age(self, aging_speed):
        for class_list in self.data:
            for item in class_list:
                item.increase_age(aging_speed)
        return
                
    def get_replay_batch(self, batch_size):
        all_items = [item for class_list in self.data for item in class_list]
        if len(all_items) < batch_size:
            return None
        return random.sample(all_items, batch_size)
    
    def get_memory(self):
        tmp_data = []
        tmp_age = []

        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.sample)
                tmp_age.append(item.age)

        tmp_age = [x / self.capacity for x in tmp_age]

        return tmp_data, tmp_age
    
    def _calculate_softmax_dist(self):
        dist = torch.tensor([len(c) for c in self.data], dtype=torch.float32)
        return torch.softmax(dist / self.sofmax_dist_temperature, dim=0)