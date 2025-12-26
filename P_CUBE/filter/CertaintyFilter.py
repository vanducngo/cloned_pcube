import torch
import torch.nn.functional as F
import math
class CertaintyFilter:
    '''
    Bộ lọc Chắc chắn dựa trên entropy có điều chỉnh theo số lớp.
    '''
    def __init__(self, num_classes, threshold_factor=0.5, confidence_factor = 0.99):
        """
        Args:
            num_classes (int): Tổng số lớp của bộ dữ liệu.
            threshold_factor (float): Hệ số để tính ngưỡng (ví dụ: 0.5). 
                                     Ngưỡng cuối cùng sẽ là factor * log(num_classes).
        """

        self.confidence_factor = confidence_factor
        if num_classes <= 1:
            # Nếu chỉ có 1 lớp, entropy luôn bằng 0, đặt ngưỡng rất nhỏ
            self.threshold = 1e-6
        else:
            # Tính toán ngưỡng entropy dựa trên số lớp
            # log(C) là entropy tối đa (khi dự đoán là phân phối đều)
            self.threshold = threshold_factor * math.log(num_classes)
        
        print(f"CertaintyFilter initialized with threshold = {self.threshold:.4f} (factor={threshold_factor}, num_classes={num_classes})")

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        """
        Kiểm tra độ chắc chắn (entropy) của dự đoán trên một batch.
        """
        if batch_samples.numel() == 0:
            device = next(current_model.parameters()).device
            return torch.tensor([], dtype=torch.bool, device=device), torch.tensor([], device=device)

        current_model.eval()

        logits = current_model(batch_samples)
        probs = F.softmax(logits, dim=1)
        
        # 1. Tín hiệu Chắc chắn Thống kê (Entropy)
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        # is_entropy_certain = (entropies < self.threshold)
        is_entropy_certain = True #TODO

        # 2. Tín hiệu Độ tin cậy (Max Probability)
        max_probs, _ = torch.max(probs, dim=1)
        is_conf_high = (max_probs >= self.confidence_factor)

        # Kết hợp cả hai để tạo Certainty Mask
        is_certain_mask = is_entropy_certain & is_conf_high
        
        return is_certain_mask, entropies