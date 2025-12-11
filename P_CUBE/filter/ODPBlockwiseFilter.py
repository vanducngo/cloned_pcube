import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy

"""
Bộ lọc ODP (Output Difference under Pruning) hoạt động theo từng khối (Block-wise).
Mục đích: Phát hiện các mẫu outlier hoặc nhiễu bằng cách đo lường sự ổn định của
các đặc trưng trung gian khi kiến trúc mô hình bị thay đổi nhẹ (tỉa bớt).
"""
class ODPBlockwiseFilter:
    def __init__(self, model_architecture, pruning_ratio=0.1, threshold=0.2):
        '''
        - model_architecture (nn.Module): Kiến trúc mô hình gốc để phân tích cấu trúc.
        - pruning_ratio (float): Tỷ lệ kênh/trọng số cần tỉa trong mỗi khối.
        - threshold (float, optional): Ngưỡng ODP cứng. Nếu None, sẽ sử dụng ngưỡng thích ứng. Defaults to None.
        - quantile (float, optional): Phân vị để xác định ngưỡng thích ứng. Ví dụ: 0.85 nghĩa là giữ lại 85% mẫu tốt nhất.
        '''
        self.pruning_ratio = pruning_ratio
        self.threshold = threshold
        self.quantile = 0.85
        
        # Bước 1: Tìm và lưu lại TÊN của các khối có thể phân tích (ví dụ: các khối Residual)
        self.prunable_block_names = self._find_prunable_block_names(model_architecture)
        print(f"ODPFilter: Found {len(self.prunable_block_names)} prunable blocks to monitor: {self.prunable_block_names}")
        
        # Bước 2: Tạo và lưu trữ các "bộ khung" đã bị tỉa của mỗi khối MỘT LẦN DUY NHẤT.
        # Điều này giúp tránh việc deepcopy và prune lặp đi lặp lại, tối ưu hóa hiệu năng.
        self.pruned_blocks = self._create_pruned_versions(model_architecture)

    def _find_prunable_block_names(self, model):
        """Duyệt qua mô hình để tìm tên của các khối có cấu trúc phù hợp (ví dụ: BasicBlock, Bottleneck)."""
        block_names = []
        for name, module in model.named_modules():
            # Sử dụng cách nhận diện chính xác và an toàn
            if type(module).__name__ in ["BasicBlock", "Bottleneck"]:
                block_names.append(name)
        return block_names

    def _create_pruned_versions(self, model):
        """Tạo ra các phiên bản đã bị tỉa của mỗi khối và lưu trữ chúng."""
        pruned_blocks_dict = {}
        original_blocks = dict(model.named_modules())
        for name in self.prunable_block_names:
            original_block = original_blocks[name]
            # Tạo một bản sao sâu để có thể chỉnh sửa mà không ảnh hưởng đến mô hình gốc.
            pruned_block = copy.deepcopy(original_block)
            
            # Tỉa tất cả các lớp Conv2d bên trong khối này.
            for module in pruned_block.modules():
                if isinstance(module, nn.Conv2d):
                    # Áp dụng mặt nạ tỉa dựa trên L1-norm của các trọng số.
                    # Trọng số có độ lớn nhỏ nhất sẽ bị loại bỏ.
                    prune.l1_unstructured(module, name="weight", amount=self.pruning_ratio)
            
            pruned_blocks_dict[name] = pruned_block.to(next(model.parameters()).device)
        return pruned_blocks_dict

    def _synchronize_weights(self, current_block, pruned_block):
        '''
        """
        Đồng bộ trọng số từ khối đang được adapt (current_block) sang khối đã bị tỉa (pruned_block).
        Đây là bước quan trọng để đảm bảo sự khác biệt đầu ra chỉ đến từ cấu trúc bị tỉa,
        chứ không phải từ sự khác biệt về giá trị trọng số.
        """
        '''
        current_state_dict = current_block.state_dict()
        pruned_state_dict = pruned_block.state_dict()

        for key in current_state_dict:
            if key.endswith("weight_orig"): # Bỏ qua các tham số gốc của pruning
                continue
            
            # `torch.prune` đổi tên 'weight' thành 'weight_orig' và thêm 'weight_mask'.
            # Chúng ta cần sao chép giá trị của 'weight' từ khối gốc vào 'weight_orig' của khối bị tỉa.
            pruned_key = key.replace(".weight", ".weight_orig")
            if pruned_key in pruned_state_dict:
                pruned_state_dict[pruned_key].copy_(current_state_dict[key])
            elif key in pruned_state_dict: 
                # Sao chép các tham số khác không bị tỉa (bias, trọng số BatchNorm, etc.)
                pruned_state_dict[key].copy_(current_state_dict[key])
        
        pruned_block.load_state_dict(pruned_state_dict)
    
    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        """
        Kiểm tra một batch mẫu và trả về một mask boolean cho biết mẫu nào đáng tin cậy.
        """
        current_model.eval()
        
        # --- Bước 1: Forward Pass Gốc và Thu thập Dữ liệu bằng Hooks ---
        block_inputs = {}
        block_outputs_original = {}
        hooks = []

        def get_io_hook(name):
            def hook(module, input, output):
                block_inputs[name] = input[0]
                block_outputs_original[name] = output
            return hook

        # Lấy ra các module khối từ mô hình hiện tại và đăng ký hook
        current_blocks = dict(current_model.named_modules())
        for name in self.prunable_block_names:
            hooks.append(current_blocks[name].register_forward_hook(get_io_hook(name)))
        
        # Thực hiện một lượt truyền thẳng để kích hoạt các hooks
        _ = current_model(batch_samples)

        # Gỡ bỏ hooks ngay sau khi dùng xong để tránh ảnh hưởng đến các bước sau
        for hook in hooks:
            hook.remove()
        
        # --- Bước 2 & 3: Tỉa, Đồng bộ và Tái tính toán Cục bộ ---
        per_block_scores = []
        for block_name in self.prunable_block_names:
            original_block = current_blocks[block_name]
            pruned_block = self.pruned_blocks[block_name]
            
            # Đồng bộ trọng số từ khối gốc (đã adapt) sang khối bị tỉa
            self._synchronize_weights(original_block, pruned_block)
            
            # Tái tính toán đầu ra chỉ trên khối đã bị tỉa
            input_tensor = block_inputs[block_name]
            pruned_output = pruned_block(input_tensor)
            original_output = block_outputs_original[block_name]
            
            # --- Bước 4: Tính điểm ODP theo Khối ---
            # Chuẩn hóa L2 vector đặc trưng đầu ra của mỗi khối
            original_flat = F.normalize(original_output.flatten(1), p=2, dim=1)
            pruned_flat = F.normalize(pruned_output.flatten(1), p=2, dim=1)
            
            # Tính khoảng cách cosine (1 - similarity)
            scores = 1 - torch.sum(original_flat * pruned_flat, dim=1)
            per_block_scores.append(scores)
        
        # --- Bước 5: Tổng hợp Điểm và Quyết định ---
        # Lấy trung bình điểm ODP trên tất cả các khối cho mỗi mẫu
        final_odp_scores = torch.mean(torch.stack(per_block_scores, dim=0), dim=0)

        # Xác định ngưỡng lọc
            # Ngưỡng thích ứng: tính toán dựa trên phân vị của batch hiện tại
        if final_odp_scores.numel() > 0:
            current_threshold = torch.quantile(final_odp_scores, q=self.quantile)
        else:
            current_threshold = float('inf') # Nếu không có điểm nào, cho qua tất cả
        
        is_stable_mask = (final_odp_scores < current_threshold)
        
        return is_stable_mask, final_odp_scores