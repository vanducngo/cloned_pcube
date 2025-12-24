import torch
import torch.nn.functional as F

class ConsistencyFilter:
    def __init__(self, source_model, threshold=0.9):
        """
        Cổng 2: Lọc Nhất quán (Consistency Filter) cải tiến.
        Sử dụng cơ chế Soft Consistency thay vì so sánh nhãn cứng [1, 2].
        
        Args:
            source_model: Bản sao cố định của mô hình gốc (Source Model) [1, 2].
            threshold: Ngưỡng Cosine Similarity để chấp nhận mẫu (mặc định 0.9).
        """
        self.source_model = source_model
        self.device = next(source_model.parameters()).device
        self.source_model.eval()
        self.threshold = threshold

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        """
        Kiểm tra độ nhất quán của hình dạng phân phối dự đoán [1, 2].
        """
        if batch_samples.numel() == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)

        # Đảm bảo cả hai mô hình đều ở chế độ eval để trích xuất logits ổn định
        current_model.eval()
        self.source_model.eval()
        
        # 1. Trích xuất Vector Logits (Representation) từ cả hai mô hình [1, 2]
        logits_cur = current_model(batch_samples)
        logits_src = self.source_model(batch_samples)
        
        # 2. Tính toán Cosine Similarity giữa hai vector logits của từng mẫu [1, 2]
        # Kết quả trả về một tensor 1D chứa độ tương đồng (từ -1 đến 1)
        cos_sim = F.cosine_similarity(logits_cur, logits_src, dim=1)
        
        # 3. Tạo mask dựa trên ngưỡng (Threshold)
        # Mẫu được giữ lại nếu hình dạng phân phối dự đoán ổn định so với tri thức nguồn [1, 2]
        is_consistent_mask = (cos_sim > self.threshold)
        
        return is_consistent_mask