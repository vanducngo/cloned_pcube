import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy

class ODPBlockwiseFilter:
    def __init__(self, model_architecture, pruning_ratio=0.1, threshold=0.2):
        self.pruning_ratio = pruning_ratio
        self.threshold = threshold
        
        self.prunable_block_names = self._find_prunable_block_names(model_architecture)
        print(f"ODPFilter: Found {len(self.prunable_block_names)} prunable blocks to monitor: {self.prunable_block_names}")
        
        # Tạo các khối bị tỉa MỘT LẦN DUY NHẤT từ kiến trúc
        self.pruned_blocks = self._create_pruned_versions(model_architecture)

    def _find_prunable_block_names(self, model):
        """Trả về một list các TÊN của các khối có thể tỉa."""
        block_names = []
        for name, module in model.named_modules():
            # Sử dụng cách nhận diện chính xác và an toàn
            if type(module).__name__ in ["BasicBlock", "Bottleneck"]:
                block_names.append(name)
        return block_names

    def _create_pruned_versions(self, model):
        """Tạo ra các phiên bản bị tỉa của mỗi khối và lưu lại."""
        pruned_blocks_dict = {}
        original_blocks = dict(model.named_modules())
        for name in self.prunable_block_names:
            original_block = original_blocks[name]
            pruned_block = copy.deepcopy(original_block)
            
            for module in pruned_block.modules():
                if isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=self.pruning_ratio)
            
            pruned_blocks_dict[name] = pruned_block.to(next(model.parameters()).device)
        return pruned_blocks_dict

    def _synchronize_weights(self, current_block, pruned_block):
        """Sao chép trọng số từ khối đang adapt sang khối bị tỉa."""
        current_state_dict = current_block.state_dict()
        pruned_state_dict = pruned_block.state_dict()

        for key in current_state_dict:
            if key.endswith("weight_orig"): # Bỏ qua các tham số gốc của pruning
                continue
            
            pruned_key = key.replace(".weight", ".weight_orig")
            if pruned_key in pruned_state_dict:
                pruned_state_dict[pruned_key].copy_(current_state_dict[key])
            elif key in pruned_state_dict: # Copy các tham số khác (bias, bn weights)
                pruned_state_dict[key].copy_(current_state_dict[key])
        
        pruned_block.load_state_dict(pruned_state_dict)
    
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

        current_blocks = dict(current_model.named_modules())
        for name in self.prunable_block_names:
            hooks.append(current_blocks[name].register_forward_hook(get_io_hook(name)))
        
        _ = current_model(batch_samples)

        for hook in hooks:
            hook.remove()

        per_block_scores = []
        for block_name in self.prunable_block_names:
            original_block = current_blocks[block_name]
            pruned_block = self.pruned_blocks[block_name]
            
            self._synchronize_weights(original_block, pruned_block)
            
            input_tensor = block_inputs[block_name]
            pruned_output = pruned_block(input_tensor)
            original_output = block_outputs_original[block_name]
            
            original_flat = F.normalize(original_output.flatten(1), p=2, dim=1)
            pruned_flat = F.normalize(pruned_output.flatten(1), p=2, dim=1)
            scores = 1 - torch.sum(original_flat * pruned_flat, dim=1)
            per_block_scores.append(scores)
        
        final_odp_scores = torch.mean(torch.stack(per_block_scores, dim=0), dim=0)
        is_stable_mask = (final_odp_scores < self.threshold)

        # --- THÊM PHẦN DEBUGGING ---
        print("----- ODP Filter Debug -----")
        print(f"Threshold: {self.threshold}")
        if len(final_odp_scores) > 0:
            print(f"ODP Scores (Batch of {len(final_odp_scores)}):")
            print(f"  - Min:    {final_odp_scores.min():.6f}")
            print(f"  - Max:    {final_odp_scores.max():.6f}")
            print(f"  - Mean:   {final_odp_scores.mean():.6f}")
            print(f"  - Median: {torch.median(final_odp_scores):.6f}")

            # In ra 5 giá trị cao nhất để xem chúng có gần ngưỡng không
            top5_scores, _ = torch.topk(final_odp_scores, k=min(5, len(final_odp_scores)))
            print(f"  - Top 5 Scores: {top5_scores.numpy()}")
        print("--------------------------")
        # --------------------------
        
        return is_stable_mask, final_odp_scores