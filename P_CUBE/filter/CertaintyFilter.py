import torch


class CertaintyFilter:
    def __init__(self, entropy_threshold):
        # Ngưỡng entropy. Mẫu có entropy cao hơn ngưỡng này sẽ bị loại.
        self.threshold = entropy_threshold

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        """
        Kiểm tra độ chắc chắn (entropy) của dự đoán trên một batch.

        Args:
            batch_samples (Tensor): Batch dữ liệu đã vượt qua các bộ lọc trước.
            current_model (torch.nn.Module): Mô hình hiện tại.

        Returns:
            Tensor: Một mask boolean cho biết mẫu nào đủ chắc chắn.
            Tensor: Một tensor chứa giá trị entropy của từng mẫu.
        """
        if batch_samples.numel() == 0:
            device = next(current_model.parameters()).device
            return torch.tensor([], dtype=torch.bool, device=device), torch.tensor([], device=device)

        current_model.eval()

        # Lấy vector xác suất đầu ra
        logits = current_model(batch_samples)
        probs = F.softmax(logits, dim=1)
        
        # Tính toán entropy cho mỗi mẫu trong batch
        # Thêm 1e-8 để tránh log(0)
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # So sánh với ngưỡng để tạo mask
        is_certain_mask = (entropies < self.threshold)
        
        return is_certain_mask, entropies