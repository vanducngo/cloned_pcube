import torch


class ConsistencyFilter:
    def __init__(self, source_model):
        # source_model: bản sao của mô hình gốc,được giữ cố định và không bao giờ được cập nhật.
        self.source_model = source_model
        # Chuyển source_model sang GPU nếu cần và đặt ở chế độ eval
        self.device = next(source_model.parameters()).device
        self.source_model.eval()

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        """
        Kiểm tra sự nhất quán của dự đoán trên một batch.

        Args:
            batch_samples (Tensor): Batch dữ liệu đã vượt qua các bộ lọc trước.
            current_model (torch.nn.Module): Mô hình hiện tại đang được TTA.

        Returns:
            Tensor: Một mask boolean cho biết mẫu nào nhất quán.
        """
        # Nếu không có mẫu nào để kiểm tra, trả về mask rỗng
        if batch_samples.numel() == 0:
            return torch.tensor([], dtype=torch.bool, device=self.device)

        # Đảm bảo cả hai mô hình đều ở chế độ eval
        current_model.eval()
        
        # Lấy dự đoán từ cả hai mô hình
        # argmax(dim=1) sẽ trả về chỉ số của lớp có xác suất cao nhất cho mỗi mẫu
        preds_current = current_model(batch_samples).argmax(dim=1)
        preds_source = self.source_model(batch_samples).argmax(dim=1)
        
        # So sánh để tạo mask. Phép so sánh '==' sẽ trả về một tensor boolean
        is_consistent_mask = (preds_current == preds_source)
        
        return is_consistent_mask
