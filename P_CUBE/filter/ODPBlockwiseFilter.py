import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy

class ODPBlockwiseFilter:
    def __init__(self, model_architecture, pruning_ratio=0.1, threshold=0.2):
        """
        Khởi tạo bộ lọc ODP-Blockwise.
        Tự động tìm và lưu lại các khối có thể tỉa từ kiến trúc được cung cấp.
        """
        self.pruning_ratio = pruning_ratio
        self.threshold = threshold
        
        # Tự động nhận diện các khối Residual Block
        self.block_modules = self._find_residual_blocks(model_architecture)
        print(f"ODPFilter: Found {len(self.block_modules)} residual blocks to monitor.")

    def _find_residual_blocks(self, model):
        # Duyệt qua model để tìm các module là BasicBlock hoặc Bottleneck (tên class trong ResNet của torchvision)
        blocks = []
        for module in model.modules():
            if type(module).__name__ in ["BasicBlock", "Bottleneck"]:
                blocks.append(module)
        return blocks
    
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

        # Hàm hook để "bắt" đầu vào và đầu ra của mỗi khối
        def get_io_hook(name):
            def hook(module, input, output):
                block_inputs[name] = input[0]
                block_outputs_original[name] = output
            return hook

        # Đăng ký hooks
        for i, block in enumerate(self.block_modules):
            hooks.append(block.register_forward_hook(get_io_hook(f"block_{i}")))
        
        # Chạy forward pass
        _ = current_model(batch_samples)

        # Gỡ bỏ hooks ngay sau khi dùng xong
        for hook in hooks:
            hook.remove()

        # --- Bước 2 & 3: Tỉa và Tái tính toán Cục bộ ---
        per_block_scores = []
        for i, block in enumerate(self.block_modules):
            # Tạo bản sao của khối để tỉa
            pruned_block = copy.deepcopy(block)
            
            # Tỉa tất cả các lớp Conv2d bên trong khối này
            for module in pruned_block.modules():
                if isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=self.pruning_ratio)
                    prune.remove(module, 'weight')
            
            # Tái tính toán chỉ trên khối đã bị tỉa
            original_output = block_outputs_original[f"block_{i}"]
            pruned_output = pruned_block(block_inputs[f"block_{i}"])
            
            # --- Bước 4: Tính điểm ODP theo Khối ---
            # Chuẩn hóa L2 để tính cosine similarity
            original_flat = F.normalize(original_output.flatten(1), p=2, dim=1)
            pruned_flat = F.normalize(pruned_output.flatten(1), p=2, dim=1)
            
            # Cosine distance = 1 - cosine similarity
            scores = 1 - torch.sum(original_flat * pruned_flat, dim=1)
            per_block_scores.append(scores)
        
        # --- Bước 5: Tổng hợp Điểm và Quyết định ---
        final_odp_scores = torch.mean(torch.stack(per_block_scores, dim=0), dim=0)
        is_stable_mask = (final_odp_scores < self.threshold)
        
        return is_stable_mask, final_odp_scores
    