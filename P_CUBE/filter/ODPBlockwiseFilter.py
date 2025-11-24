import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy

class ODPBlockwiseFilter:
    def __init__(self, model_architecture, pruning_ratio=0.1, threshold=0.2):
        self.pruning_ratio = pruning_ratio
        self.threshold = threshold
        
        # Tự động nhận diện khối một cách linh hoạt hơn ---
        self.prunable_blocks_info = self._find_and_prepare_prunable_blocks(model_architecture)
        print(f"ODPFilter: Found {len(self.prunable_blocks_info)} prunable blocks to monitor.")
        
        # Tạo và lưu trữ các khối bị tỉa MỘT LẦN DUY NHẤT ---
        self.pruned_blocks = self._create_pruned_versions()

    def _find_and_prepare_prunable_blocks(self, model):
        prunable_blocks = []
        # Thay vì tìm theo tên, tìm theo cấu trúc: một module có chứa ít nhất một Conv2d
        # và là một 'Sequential' hoặc một module không có con là chính nó (để tránh lặp vô hạn)
        for name, module in model.named_modules():
            # Điều kiện nhận diện một "khối"
            if isinstance(module, (nn.Sequential, nn.Module)) and \
               any(isinstance(m, nn.Conv2d) for m in module.modules()) and \
               module != model: # Loại trừ toàn bộ model
                
                # Tránh lấy các khối lồng nhau (ví dụ không lấy layer1 nếu đã lấy các block bên trong nó)
                is_submodule_of_already_found = any(name.startswith(p_name) for p_name, _ in prunable_blocks)
                if not is_submodule_of_already_found:
                    # Loại bỏ các khối lớn hơn mà khối này là con của nó
                    prunable_blocks = [(p_name, p_mod) for p_name, p_mod in prunable_blocks if not p_name.startswith(name)]
                    prunable_blocks.append((name, module))
        
        return prunable_blocks

    def _create_pruned_versions(self):
        """Tạo ra các phiên bản bị tỉa của mỗi khối và lưu lại."""
        pruned_blocks_dict = {}
        for name, block in self.prunable_blocks_info:
            pruned_block = copy.deepcopy(block)
            
            # Tỉa tất cả các lớp Conv2d bên trong khối này
            for module in pruned_block.modules():
                if isinstance(module, nn.Conv2d):
                    # Quan trọng: chỉ thêm mặt nạ, không remove vĩnh viễn
                    prune.l1_unstructured(module, name="weight", amount=self.pruning_ratio)
            
            pruned_blocks_dict[name] = pruned_block
        return pruned_blocks_dict

    @torch.no_grad()
    def check_batch(self, batch_samples, current_model):
        current_model.eval()
        
        block_inputs = {}
        block_outputs_original = {}
        hooks = []

        def get_io_hook(name):
            def hook(module, input, output):
                block_inputs[name] = input[0]
                block_outputs_original[name] = output
            return hook

        # Đăng ký hooks vào mô hình ĐANG THÍCH ỨNG (current_model)
        for name, _ in self.prunable_blocks_info:
            module = dict(current_model.named_modules())[name]
            hooks.append(module.register_forward_hook(get_io_hook(name)))
        
        _ = current_model(batch_samples)

        for hook in hooks:
            hook.remove()

        # --- CẢI TIẾN 3: Sử dụng các khối bị tỉa đã được tạo sẵn ---
        per_block_scores = []
        for name, original_block in self.prunable_blocks_info:
            pruned_block = self.pruned_blocks[name]
            
            # Đồng bộ trọng số trước khi tính toán ---
            # Sao chép trọng số từ khối gốc (đã được adapt) sang khối bị tỉa
            # Mặt nạ tỉa vẫn được giữ nguyên
            pruned_block.load_state_dict(dict(current_model.named_modules())[name].state_dict())

            # Tái tính toán chỉ trên khối đã bị tỉa
            input_tensor = block_inputs[name]
            pruned_output = pruned_block(input_tensor)
            original_output = block_outputs_original[name]
            
            # Tính điểm ODP
            original_flat = F.normalize(original_output.flatten(1), p=2, dim=1)
            pruned_flat = F.normalize(pruned_output.flatten(1), p=2, dim=1)
            scores = 1 - torch.sum(original_flat * pruned_flat, dim=1)
            per_block_scores.append(scores)
        
        final_odp_scores = torch.mean(torch.stack(per_block_scores, dim=0), dim=0)
        is_stable_mask = (final_odp_scores < self.threshold)
        
        return is_stable_mask, final_odp_scores