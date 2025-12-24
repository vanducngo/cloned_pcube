import torch
import torch.nn.functional as F

class ConsistencyFilter:
    def __init__(self, source_model, quantile=0.5, safety_floor=0.4):
        """
        Cổng 2: Lọc Nhất quán (Consistency Filter) cải tiến - Phiên bản Adaptive.
        
        Args:
            source_model: Bản sao cố định của mô hình gốc.
            quantile: Tỷ lệ phần trăm mẫu muốn giữ lại trong mỗi batch (ví dụ 0.5 = 50%).
            safety_floor: Ngưỡng sàn tối thiểu. Nếu mẫu có cosine < safety_floor thì chắc chắn là rác, kể cả nó nằm trong top 50%.
        """
        self.source_model = source_model
        self.device = next(source_model.parameters()).device
        self.source_model.eval()
        
        self.quantile = quantile
        self.safety_floor = safety_floor

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        if batch_samples.numel() == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)

        current_model.eval()
        self.source_model.eval()
        
        # 1. Trích xuất Logits
        logits_cur = current_model(batch_samples)
        logits_src = self.source_model(batch_samples)
        
        # 2. Tính Cosine Similarity
        cos_sim = F.cosine_similarity(logits_cur, logits_src, dim=1)
        
        # 3. Tính Ngưỡng Thích ứng (Adaptive Threshold)
        # Tìm giá trị phân vị thứ (1 - quantile). 
        # Ví dụ quantile=0.5 -> lấy giá trị ở vị trí 50% khi sort tăng dần.
        # Những mẫu > giá trị này sẽ là top 50% cao nhất.
        if cos_sim.numel() > 0:
            adaptive_threshold = torch.quantile(cos_sim, q=(1 - self.quantile))
        else:
            adaptive_threshold = 0.0

        # 4. Kết hợp với Safety Floor
        # Ngưỡng cuối cùng là max(adaptive, floor).
        # Điều này đảm bảo:
        # - Batch tốt: Adaptive cao -> Lọc chặt.
        # - Batch xấu: Adaptive thấp -> Dùng Floor chặn đáy để không lấy rác.
        final_threshold = max(adaptive_threshold.item(), self.safety_floor)
        
        # 5. Tạo mask
        is_consistent_mask = (cos_sim >= final_threshold)
        
        # [Optional] Debug log
        # print(f"Consistency: Mean={cos_sim.mean():.3f}, Adaptive Thresh={adaptive_threshold:.3f}, Final Thresh={final_threshold:.3f}, Passed={is_consistent_mask.sum()}/{len(batch_samples)}")
        
        return is_consistent_mask