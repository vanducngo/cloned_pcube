import torch
import torch.nn as nn
import numpy as np

@torch.no_grad()
def calculate_centroids_from_buffer(buffer, num_classes, feature_dim, device):
    """
    Tính toán tâm lớp (centroids) từ các mẫu hiện có trong memory bank.

    Args:
        buffer (list[MemoryItem]): Danh sách các đối tượng MemoryItem.
        num_classes (int): Tổng số lớp.
        feature_dim (int): Số chiều của vector đặc trưng.
        device: Thiết bị (GPU/CPU) để thực hiện tính toán.

    Returns:
        Tensor: Một tensor kích thước (num_classes, feature_dim) chứa các tâm lớp.
    """
    if not buffer:
        return torch.zeros(num_classes, feature_dim, device=device)

    # Khởi tạo tensor để lưu tổng và đếm
    centroids_sum = torch.zeros(num_classes, feature_dim, device=device)
    class_counts = torch.zeros(num_classes, device=device)

    # Gom nhóm các đặc trưng theo lớp
    for item in buffer:
        label = item.pseudo_label
        # Đảm bảo feature là tensor và trên đúng device
        feature = item.feature.to(device) if isinstance(item.feature, torch.Tensor) else torch.tensor(item.feature, device=device)
        centroids_sum[label] += feature
        class_counts[label] += 1

    # Tính trung bình để có tâm lớp
    # Thêm 1e-8 để tránh chia cho 0 với các lớp không có mẫu
    centroids_snapshot = centroids_sum / (class_counts.unsqueeze(1) + 1e-8)
    
    return centroids_snapshot.detach()

def ema_update_centroids(old_centroids, new_centroids, momentum):
    """
    Cập nhật các tâm lớp dài hạn bằng EMA.

    Args:
        old_centroids (Tensor): Tâm lớp dài hạn từ bước trước (stats_ema).
        new_centroids (Tensor): Tâm lớp tức thời mới (stats_snapshot).
        momentum (float): Hệ số EMA.

    Returns:
        Tensor: Tâm lớp dài hạn đã được cập nhật.
    """
    if old_centroids is None or old_centroids.numel() == 0:
        return new_centroids.clone()

    # Cập nhật bằng EMA
    updated_centroids = momentum * old_centroids + (1 - momentum) * new_centroids
    return updated_centroids.detach()

def calculate_centroid_distance(centroids_p, centroids_q, distance_metric='l2'):
    """
    Tính khoảng cách trung bình giữa hai bộ tâm lớp.

    Args:
        centroids_p (Tensor): Bộ tâm lớp P.
        centroids_q (Tensor): Bộ tâm lớp Q.
        distance_metric (str): 'l2' hoặc 'cosine'.

    Returns:
        float: Giá trị khoảng cách trung bình.
    """
    if centroids_p is None or centroids_q is None or centroids_p.numel() == 0 or centroids_q.numel() == 0:
        return 0.0

    if distance_metric == 'l2':
        # Tính khoảng cách Euclidean L2 cho từng cặp tâm lớp
        distances = torch.norm(centroids_p - centroids_q, p=2, dim=1)
    elif distance_metric == 'cosine':
        # Tính khoảng cách cosine
        distances = 1 - F.cosine_similarity(centroids_p, centroids_q, dim=1)
    else:
        raise ValueError("Invalid distance metric")

    # Chỉ tính trung bình trên các lớp có ít nhất một mẫu (tâm lớp khác 0)
    # để tránh sai lệch do các lớp "trống"
    valid_classes_mask = torch.norm(centroids_p, dim=1) > 0
    if valid_classes_mask.sum() > 0:
        return torch.mean(distances[valid_classes_mask]).item()
    else:
        return 0.0
















# ==============================================================================
# HÀM 1: calculate_stats_on_buffer
# ==============================================================================
@torch.no_grad()
def calculate_stats_on_buffer(buffer, model, target_layer_names):
    """
    Tính toán mean và variance của các lớp BN/LN trên toàn bộ buffer.
    """
    if not buffer:
        return {}

    model.eval()
    all_samples = torch.stack([item.sample for item in buffer]).to(next(model.parameters()).device)

    activations = {}
    hooks = []

    def get_activation_hook(name):
        def hook(module, input, output):
            # Lấy tên class của module hiện tại
            class_name = type(module).__name__
            
            # --- LOGIC MỚI ĐỂ XÁC ĐỊNH LOẠI LỚP ---
            # Ưu tiên các lớp có chứa "BatchNorm" trong tên
            if 'BatchNorm' in class_name or 'BN' in class_name:
                # Đối với tất cả các loại BatchNorm (BN2d, RobustBN2d, ...),
                # chúng ta luôn muốn lấy tensor đầu vào.
                activations[name] = input[0]
            # Nếu không, kiểm tra xem có phải LayerNorm không
            elif isinstance(module, nn.LayerNorm):
                # Đối với LayerNorm, chúng ta muốn lấy tensor đầu ra.
                activations[name] = output
            else:
                # Một trường hợp dự phòng: nếu là một lớp lạ,
                # tạm thời lấy đầu ra. Cần kiểm tra kỹ nếu có kiến trúc mới.
                activations[name] = output
            # ----------------------------------------
                
        return hook

    for name, layer in model.named_modules():
        # So sánh TÊN thay vì đối tượng
        if name in target_layer_names:
            # Đăng ký hook vào lớp `layer` của `model` hiện tại
            hooks.append(layer.register_forward_hook(get_activation_hook(name)))

    if not hooks:
        print("Warning: No hooks were registered in calculate_stats_on_buffer. Check target_layer_names.")
        return {}
    
    model(all_samples)

    for hook in hooks:
        hook.remove()

    buffer_stats = {}
    for name, activation_tensor in activations.items():
        if activation_tensor.dim() == 4: # N, C, H, W for BatchNorm2d
            mean = torch.mean(activation_tensor, dim=[0, 2, 3])
            var = torch.var(activation_tensor, dim=[0, 2, 3], unbiased=False)
        else: # Xử lý cho các trường hợp khác như LayerNorm
            dims_to_reduce = list(range(1, activation_tensor.dim()))
            mean = torch.mean(activation_tensor, dim=dims_to_reduce)
            var = torch.var(activation_tensor, dim=dims_to_reduce, unbiased=False)
        buffer_stats[name] = (mean.detach(), var.detach())

    return buffer_stats

# ==============================================================================
# HÀM 2: kl_divergence
# ==============================================================================
def kl_divergence(stats_p, stats_q):
    """
    Tính toán KL-Divergence trung bình giữa hai bộ thống kê (p || q).
    """
    total_kl_div = 0.0
    num_elements = 0

    common_layers = set(stats_p.keys()) & set(stats_q.keys())
    if not common_layers:
        return 0.0

    for layer_name in common_layers:
        mean_p, var_p = stats_p[layer_name]
        mean_q, var_q = stats_q[layer_name]
        
        # Thêm hằng số nhỏ để ổn định
        var_p = var_p.clamp(min=1e-8)
        var_q = var_q.clamp(min=1e-8)
        
        log_ratio = torch.log(torch.sqrt(var_q) / torch.sqrt(var_p))
        term1 = (var_p + (mean_p - mean_q).pow(2)) / (2 * var_q)
        kl_div_tensor = log_ratio + term1 - 0.5
        
        total_kl_div += torch.sum(kl_div_tensor)
        num_elements += kl_div_tensor.numel()

    if num_elements == 0:
        return 0.0
        
    return (total_kl_div / num_elements).item()

# ==============================================================================
# HÀM 3: ema_update
# ==============================================================================
def ema_update(old_stats, new_stats, momentum):
    """
    Cập nhật thống kê dài hạn (old_stats) bằng thống kê mới (new_stats).
    """
    if not old_stats:
        return new_stats.copy()

    updated_stats = {}
    for layer_name in old_stats:
        if layer_name in new_stats:
            old_mean, old_var = old_stats[layer_name]
            new_mean, new_var = new_stats[layer_name]

            updated_mean = momentum * old_mean + (1 - momentum) * new_mean
            updated_var = momentum * old_var + (1 - momentum) * new_var
            updated_stats[layer_name] = (updated_mean, updated_var)
    return updated_stats

# ==============================================================================
# LỚP 4: OnlinePeakDetector
# ==============================================================================
class OnlinePeakDetector:
    def __init__(self, window_size=10, threshold=5.0, influence=0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.influence = influence
        self.history = []
        self.mean = 0.0
        self.std = 1.0 # Khởi tạo là 1 để tránh chia cho 0

    def is_peak(self, new_value):
        if len(self.history) < self.window_size:
            self.history.append(new_value)
            if len(self.history) == self.window_size:
                self.mean = np.mean(self.history)
                self.std = np.std(self.history)
            return False

        z_score = abs(new_value - self.mean) / self.std if self.std > 0 else 0
        
        # Cập nhật cửa sổ trượt
        self.history.pop(0)
        self.history.append(new_value)

        is_peak_detected = (z_score > self.threshold)
        
        if not is_peak_detected:
            # Cập nhật mean và std một cách từ từ nếu không có đỉnh
            # Sử dụng tính toán lại trên cửa sổ trượt để chính xác hơn
            self.mean = np.mean(self.history)
            self.std = np.std(self.history)

        return is_peak_detected