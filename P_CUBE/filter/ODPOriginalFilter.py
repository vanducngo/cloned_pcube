import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from torch.nn.utils import prune

# Hàm hỗ trợ từ code gốc của MoTTA
def apply_pruning(model, strategy="l1_unstructured", amount=0.2, target_layers="conv", n=1, dim=0):
    pruning_methods = {
        "l1_unstructured": prune.l1_unstructured,
        "ln_structured": prune.ln_structured,
        "random_unstructured": prune.random_unstructured,
        "random_structured": prune.random_structured
    }
    
    if strategy not in pruning_methods:
        raise ValueError(f"Pruning strategy {strategy} not recognized.")
    
    pruning_method = pruning_methods[strategy]

    for name, module in model.named_modules():
        if (target_layers == "conv" and isinstance(module, nn.Conv2d)) or \
           (target_layers == "linear" and isinstance(module, nn.Linear)) or \
           (target_layers == "conv&linear" and isinstance(module, (nn.Conv2d, nn.Linear))):
            if "unstructured" in strategy:
                pruning_method(module, name='weight', amount=amount)
            else:
                pruning_method(module, name='weight', amount=amount, n=n, dim=dim)

            if module.bias is not None:
                if "structured" in strategy:
                    pruning_method(module, name='bias', amount=amount, n=n, dim=dim)
                else:
                    pruning_method(module, name='bias', amount=amount)

# Hàm hỗ trợ từ code gốc của MoTTA
def split_up_model(model, arch_name: str, dataset_name: str):
    # (Đoạn này mình rút gọn các if/else không cần thiết cho mục đích chung, 
    # nhưng thực tế bạn có thể copy toàn bộ hàm split_up_model từ code gốc vào đây)
    
    if "resnet" in arch_name or "resnext" in arch_name or arch_name in {"Standard_R50"}:
        # Giả định kiến trúc phổ biến nhất theo code gốc (ví dụ ResNet50)
        # Bỏ đi lớp fully connected cuối cùng để tạo encoder
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
        return encoder, classifier
    else:
        # Bạn cần chèn lại hàm split_up_model đầy đủ từ file motta.py gốc nếu dùng model khác
        raise NotImplementedError("Please copy the full split_up_model function from motta.py")

# ==============================================================================
# CLASS CHÍNH: ODPOriginalFilter
# ==============================================================================
class ODPOriginalFilter:
    """
    Bộ lọc ODP Toàn cục (Global Pruning) - Tái hiện chính xác logic của MoTTA gốc.
    - Cắt tỉa một lần duy nhất trên toàn bộ Encoder.
    - So sánh sự thay đổi góc giữa vector đặc trưng và vector trọng số của Classifier.
    - Sử dụng ngưỡng tĩnh (Static Threshold).
    """
    def __init__(self, model_architecture, arch_name='Standard_R50', dataset_name='imagenet', prune_ratio=0.1, threshold=0.2):
        self.prune_ratio = prune_ratio
        
        # Ngưỡng tĩnh theo độ (ví dụ 0.2 độ, tuỳ thuộc vào cách định nghĩa threshold trong MoTTA)
        # Trong code gốc MoTTA không truyền trực tiếp threshold vào eval_pruning, 
        # mà nó được check ở phần "update memory": metric[i].item() <= self.uncertainty_threshold
        self.threshold = threshold 
        
        # Bước 1: Tách model thành 2 phần: Feature Extractor (Encoder) và Classifier
        # Lưu ý: cần truyền đúng arch_name và dataset_name tương tự MoTTA
        self.feature_extractor, self.classifier = split_up_model(deepcopy(model_architecture), arch_name, dataset_name)
        
        # Bước 2: Tạo một bản sao của Feature Extractor để thực hiện cắt tỉa
        self.feature_extractor_prune = deepcopy(self.feature_extractor)
        
        # Bước 3: Áp dụng cắt tỉa toàn cục lên bản sao
        # Trong MoTTA gốc, chiến lược thường dùng là "l1_unstructured"
        apply_pruning(
            self.feature_extractor_prune, 
            strategy='l1_unstructured', 
            amount=self.prune_ratio,
            target_layers='conv'
        )
        
        # Đưa các module về chế độ eval (để tránh dropout, cập nhật BatchNorm)
        self.feature_extractor.eval()
        self.feature_extractor_prune.eval()
        
        # Thiết bị
        self.device = next(self.classifier.parameters()).device

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model=None):
        """
        Kiểm tra một batch mẫu.
        Lưu ý: tham số current_model được truyền vào để đồng nhất interface với ODPBlockwise,
        nhưng ODPOriginal không sử dụng current_model (nó không đồng bộ trọng số liên tục).
        """
        # Nếu không có mẫu, trả về rỗng
        if batch_samples.numel() == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device), torch.tensor([], device=self.device)

        batch_samples = batch_samples.to(self.device)
        self.feature_extractor.to(self.device)
        self.feature_extractor_prune.to(self.device)
        self.classifier.to(self.device)

        # 1. Forward Pass Kép
        # Chạy qua encoder gốc
        feature = self.feature_extractor(batch_samples)
        # Chạy qua encoder đã bị cắt tỉa
        feature_prune = self.feature_extractor_prune(batch_samples)

        # 2. Tính toán độ lệch ODP (logic short_version=True của MoTTA)
        # Lấy trọng số của bộ phân loại
        fc_weight = self.classifier.weight
        
        # Tính độ tương đồng Cosine giữa đặc trưng và tất cả các vector của classifier
        # feature.unsqueeze(1): shape (batch_size, 1, feature_dim)
        # fc_weight.unsqueeze(0): shape (1, num_classes, feature_dim)
        # Kết quả cos_sim có shape (batch_size, num_classes)
        cos_sim = F.cosine_similarity(feature.unsqueeze(1), fc_weight.unsqueeze(0), dim=2)
        cos_sim_prune = F.cosine_similarity(feature_prune.unsqueeze(1), fc_weight.unsqueeze(0), dim=2)
        
        # Tính sự thay đổi góc lớn nhất giữa mô hình gốc và mô hình tỉa
        # torch.acos trả về Radian. 
        angle_diff = torch.abs(torch.acos(cos_sim) - torch.acos(cos_sim_prune))
        max_angle_change_feature_fc = torch.max(angle_diff, dim=1)[0]
        
        # Đổi từ Radian sang Độ (Degrees) - Đây chính là metric (ODP score)
        odp_scores = max_angle_change_feature_fc / math.pi * 180.0
        
        # 3. Tạo Mask dựa trên Ngưỡng Tĩnh (Static Threshold)
        # Trong MoTTA: metric[i].item() <= self.uncertainty_threshold
        # Điểm càng thấp (ít thay đổi góc) -> càng ổn định -> True
        is_stable_mask = (odp_scores <= self.threshold)

        # Trả về mask boolean và danh sách điểm số
        return is_stable_mask, odp_scores