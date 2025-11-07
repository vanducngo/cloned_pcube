import torch

class PurgeableMemoryBank:
    def __init__(self, capacity, num_classes, kl_threshold, acceleration_factor, ema_momentum):
        self.buffer = []
        self.capacity = capacity
        # ... c
        
        # Cho việc phát hiện thay đổi miền
        self.global_stats = self.initial_source_stats()
        self.kl_divergence_history = []
        self.updates_since_last_check = 0
        self.KL_CHECK_INTERVAL = 64


    def initial_source_stats():
        # Init ema
        pass

    def manage_and_add(self, clean_sample, model):
        # 1. Quản lý vi mô (thêm/thay thế)
        self._add_or_replace(clean_sample, model)
        for item in self.buffer:
            item.age += 1
            
        # 3. Quản lý vĩ mô (định kỳ)
        self.updates_since_last_check += 1
        if self.updates_since_last_check >= self.KL_CHECK_INTERVAL:
            self._check_for_domain_shift()
            self.updates_since_last_check = 0
            
    def _check_for_domain_shift(self):
        # 1. Tính stats_snapshot từ bộ đệm hiện tại
        stats_snapshot = self.calculate_stats_on_buffer(self.buffer)

        # 2. Tính phân kỳ KL
        divergence = self.kl_divergence(stats_snapshot, self.stats_ema)
        self.kl_divergence_history.append(divergence)
        
        # Cập nhật global_stats bằng EMA
        self.stats_ema = self.ema_update(self.stats_ema, stats_snapshot, self.ema_momentum)
        
        if self.is_peak(self.kl_divergence_history, self.kl_threshold):
            print("Domain shift detected! Triggering Accelerated Aging.")
            self._accelerated_aging()
            
    def _accelerated_aging(self):
        for item in self.buffer:
            item.age *= self.acceleration_factor

    def _add_or_replace(self, sample, model):
        # ... Logic thay thế dựa trên Heuristic H của RoTTA ...
        pass

    def calculate_stats_on_buffer(buffer, model, target_layers):
        """
        Tính toán mean và variance của các lớp BN/LN trên toàn bộ buffer.

        Args:
            buffer (list): Danh sách các MemoryItem, mỗi item chứa một mẫu dữ liệu.
            model (torch.nn.Module): Mô hình hiện tại.
            target_layers (list): Danh sách các module lớp chuẩn hóa cần tính thống kê.

        Returns:
            dict: Một từ điển chứa (mean, variance) cho mỗi lớp.
                Ví dụ: {'layer1.bn': (mean_tensor, var_tensor), ...}
        """
        if not buffer:
            return {}

        # Chuyển model sang chế độ đánh giá để không cập nhật các thống kê nội tại
        model.eval()
        
        # Gom tất cả các mẫu dữ liệu từ buffer thành một batch lớn
        all_samples = torch.stack([item.sample for item in buffer]).cuda()

        # Sử dụng hooks để "bắt" các activations tại các lớp mục tiêu
        activations = {}
        hooks = []

        def get_activation_hook(name):
            def hook(model, input, output):
                # Với BN, input[0] là tensor activation
                # Với LN, output là tensor activation
                activations[name] = input[0] if isinstance(model, torch.nn.BatchNorm2d) else output
            return hook

        for name, layer in model.named_modules():
            if layer in target_layers:
                hooks.append(layer.register_forward_hook(get_activation_hook(name)))

        # Thực hiện một lượt truyền thẳng duy nhất trên toàn bộ batch
        with torch.no_grad():
            model(all_samples)

        # Gỡ bỏ các hooks sau khi đã có activations
        for hook in hooks:
            hook.remove()

        # Tính toán thống kê từ các activations đã thu thập
        buffer_stats = {}
        for name, activation_tensor in activations.items():
            # Đối với BN (Conv2d), tính mean/var trên các chiều (N, H, W)
            if activation_tensor.dim() == 4: # N, C, H, W
                # Giữ lại chiều Channel (C)
                mean = torch.mean(activation_tensor, dim=[0, 2, 3])
                var = torch.var(activation_tensor, dim=[0, 2, 3], unbiased=False)
            # Đối với LN (Transformer), tính mean/var trên chiều đặc trưng cuối cùng
            elif activation_tensor.dim() == 3: # N, SeqLen, Dim
                # Giữ lại chiều Sequence Length và Batch
                mean = torch.mean(activation_tensor, dim=-1)
                var = torch.var(activation_tensor, dim=-1, unbiased=False)
            else: # Xử lý các trường hợp khác nếu cần
                # Mặc định tính trên tất cả các chiều trừ chiều batch
                dims_to_reduce = list(range(1, activation_tensor.dim()))
                mean = torch.mean(activation_tensor, dim=dims_to_reduce)
                var = torch.var(activation_tensor, dim=dims_to_reduce, unbiased=False)

            buffer_stats[name] = (mean.detach(), var.detach())

        return buffer_stats
    
    def kl_divergence(stats_p, stats_q):
        """
        Tính toán KL-Divergence trung bình giữa hai bộ thống kê (p || q).
        Giả định mỗi kênh là một phân phối Gaussian độc lập.

        Args:
            stats_p (dict): Từ điển thống kê P, ví dụ: stats_snapshot.
            stats_q (dict): Từ điển thống kê Q, ví dụ: stats_ema.

        Returns:
            float: Giá trị KL-Divergence trung bình trên tất cả các kênh, các lớp.
        """
        total_kl_div = 0.0
        num_elements = 0

        # Đảm bảo chỉ tính trên các lớp có trong cả hai bộ thống kê
        common_layers = set(stats_p.keys()) & set(stats_q.keys())

        for layer_name in common_layers:
            mean_p, var_p = stats_p[layer_name]
            mean_q, var_q = stats_q[layer_name]
            
            # Thêm một hằng số nhỏ để tránh log(0) hoặc chia cho 0
            var_p = var_p + 1e-8
            var_q = var_q + 1e-8
            
            # Công thức KL-Divergence cho 2 phân phối Gaussian 1-D
            # KL(P || Q) = log(sigma_q / sigma_p) + (sigma_p^2 + (mu_p - mu_q)^2) / (2 * sigma_q^2) - 0.5
            
            log_ratio = torch.log(torch.sqrt(var_q) / torch.sqrt(var_p))
            term1 = (var_p + (mean_p - mean_q)**2) / (2 * var_q)
            kl_div_tensor = log_ratio + term1 - 0.5
            
            # Lấy tổng trên tất cả các kênh của lớp hiện tại
            total_kl_div += torch.sum(kl_div_tensor)
            num_elements += kl_div_tensor.numel()

        if num_elements == 0:
            return 0.0

        average_kl_div = total_kl_div / num_elements
        return average_kl_div.item()
    
    def ema_update(old_stats, new_stats, momentum):
        """
        Cập nhật bộ thống kê dài hạn (old_stats) bằng bộ thống kê tức thời (new_stats).

        Args:
            old_stats (dict): Thống kê dài hạn cần cập nhật (ví dụ: stats_ema).
            new_stats (dict): Thống kê tức thời mới (ví dụ: stats_snapshot).
            momentum (float): Hệ số momentum cho EMA (ví dụ: 0.9).

        Returns:
            dict: Bộ thống kê dài hạn đã được cập nhật.
        """
        if not old_stats: # Nếu old_stats chưa được khởi tạo
            return new_stats.copy()

        updated_stats = {}
        common_layers = set(old_stats.keys()) & set(new_stats.keys())

        for layer_name in common_layers:
            old_mean, old_var = old_stats[layer_name]
            new_mean, new_var = new_stats[layer_name]

            # Cập nhật bằng EMA
            updated_mean = momentum * old_mean + (1 - momentum) * new_mean
            updated_var = momentum * old_var + (1 - momentum) * new_var
            
            updated_stats[layer_name] = (updated_mean, updated_var)

        return updated_stats
    
import numpy as np

class OnlinePeakDetector:
    def __init__(self, window_size, threshold, influence):
        """
        Khởi tạo thuật toán phát hiện đỉnh online dựa trên z-score.
        
        Args:
            window_size (int): Kích thước cửa sổ trượt để tính mean và std.
            threshold (float): Ngưỡng z-score để coi là một đỉnh (ví dụ: 5.0).
            influence (float): Hệ số ảnh hưởng của đỉnh mới đến mean/std (0 đến 1).
        """
        self.window_size = window_size
        self.threshold = threshold
        self.influence = influence
        self.history = [] # Lưu trữ window_size giá trị gần nhất
        self.mean = 0.0
        self.std = 0.0

    def is_peak(self, new_value):
        """
        Kiểm tra xem một giá trị mới có phải là đỉnh hay không.

        Args:
            new_value (float): Giá trị mới cần kiểm tra (ví dụ: KL divergence).

        Returns:
            bool: True nếu là đỉnh, False nếu không.
        """
        if len(self.history) < self.window_size:
            # Giai đoạn khởi tạo: chỉ thu thập dữ liệu
            self.history.append(new_value)
            if len(self.history) == self.window_size:
                self.mean = np.mean(self.history)
                self.std = np.std(self.history)
            return False

        # Tính z-score của giá trị mới so với lịch sử
        if self.std == 0: # Tránh chia cho 0
            z_score = 0.0
        else:
            z_score = abs(new_value - self.mean) / self.std

        # Cập nhật lịch sử (thêm mới, bỏ cũ)
        self.history.pop(0)
        self.history.append(new_value)

        is_peak_detected = (z_score > self.threshold)
        
        if is_peak_detected:
            # Nếu là đỉnh, không cập nhật mean/std bằng giá trị này
            # để tránh đỉnh làm sai lệch thống kê
            pass
        else:
            # Nếu không phải đỉnh, cập nhật mean và std bằng EMA
            # Hoặc tính lại trên cửa sổ trượt mới
            # Cách EMA đơn giản hơn:
            self.mean = (1 - self.influence) * self.mean + self.influence * new_value
            self.std = np.sqrt( (1 - self.influence) * self.std**2 + self.influence * (new_value - self.mean)**2 )

        return is_peak_detected
    
    # w_ent, w_dis là các siêu tham số, ví dụ: 1.0, 0.5
# weak_augment_transform là một transform của torchvision
# sce_loss là hàm tính Symmetric Cross-Entropy

def replay_and_update_pipeline(replay_batch, student_model, teacher_model, optimizer, 
                                weak_augment_transform, num_aug_checks=2, w_ent=1.0, w_dis=0.5):
    """
    Thực hiện pipeline Giai đoạn 3: Replay và Tạo Nhãn giả Chất lượng cao.
    """
    samples = torch.stack([item.sample for item in replay_batch]).cuda()

    # --- Bước 1: Tạo Nhãn giả Dự thảo bằng Paired-View ---
    with torch.no_grad():
        flipped_samples = torch.flip(samples, dims=[-1])
        probs_original = torch.softmax(teacher_model(samples), dim=-1)
        probs_flipped = torch.softmax(teacher_model(flipped_samples), dim=-1)
        y_draft = (probs_original + probs_flipped) / 2.0

    # --- Bước 2: Đánh giá Chất lượng Nhãn giả ---
    # 2a: Tính Độ Chắc chắn (Certainty)
    entropies = -torch.sum(y_draft * torch.log(y_draft + 1e-8), dim=-1)
    
    # 2b: Tính Độ Ổn định (Stability)
    disagreement_scores = torch.zeros_like(entropies)
    with torch.no_grad():
        for _ in range(num_aug_checks):
            augmented_samples = weak_augment_transform(samples)
            probs_aug = torch.softmax(teacher_model(augmented_samples), dim=-1)
            
            kl_div = torch.sum(y_draft * (torch.log(y_draft + 1e-8) - torch.log(probs_aug + 1e-8)), dim=-1)
            disagreement_scores += kl_div
    
    # --- Bước 3: Tính Loss có Trọng số Thích ứng ---
    # Tính điểm chất lượng tổng hợp (càng thấp càng tốt)
    quality_penalty = (w_ent * entropies) + (w_dis * disagreement_scores)
    
    # Chuyển thành trọng số (càng cao càng tốt)
    weights = torch.exp(-quality_penalty)
    
    # Chuẩn hóa trọng số
    normalized_weights = weights / (torch.sum(weights) + 1e-8)

    # --- Cập nhật Student Model ---
    student_outputs = student_model(samples)
    
    # Tính SCE loss cho từng mẫu, sau đó áp dụng trọng số
    individual_losses = sce_loss(student_outputs, y_draft.detach(), reduction='none')
    weighted_loss = torch.sum(normalized_weights * individual_losses)
    
    # Cập nhật
    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()

    return weighted_loss.item()

# --- Cách sử dụng trong mã giả trước ---
# Khởi tạo bên ngoài vòng lặp
# peak_detector = OnlinePeakDetector(window_size=10, threshold=5.0, influence=0.5)

# Bên trong hàm _check_for_domain_shift
# divergence = kl_divergence(...)
# if peak_detector.is_peak(divergence):
#     self._accelerated_aging()