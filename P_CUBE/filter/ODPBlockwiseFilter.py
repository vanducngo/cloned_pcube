class ODPBlockwiseFilter:
    def __init__(self, pruning_ratio):
        self.pruning_ratio = pruning_ratio

    def check(self, sample, model):
        # 1. Forward gốc, lưu inputs của các blocks
        z_original, block_inputs = model.forward_with_hooks(sample)
        
        # 2. Tỉa và tái tính toán cục bộ
        pruned_outputs = []
        for i, block_input in enumerate(block_inputs):
            pruned_block = self.get_pruned_version(model.blocks[i])
            pruned_outputs.append(pruned_block(block_input))
            
        # 3. Tính toán điểm ODP
        odp_score = self.calculate_score(model.original_outputs, pruned_outputs)
        
        return odp_score < self.threshold, odp_score