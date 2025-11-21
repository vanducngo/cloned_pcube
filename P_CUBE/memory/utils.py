import torch
import torch.nn as nn
import numpy as np

# ==============================================================================
# HÀM 1: calculate_stats_on_buffer
# ==============================================================================
@torch.no_grad()
def calculate_stats_on_buffer(buffer, model, target_layers):
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
            # Với BN, input[0] là tensor activation. Với LN, output là tensor.
            activations[name] = input[0] if isinstance(module, nn.BatchNorm2d) else output
        return hook

    for name, layer in model.named_modules():
        if layer in target_layers:
            hooks.append(layer.register_forward_hook(get_activation_hook(name)))

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