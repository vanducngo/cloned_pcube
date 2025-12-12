import torch
from .ODPBlockwiseFilter import ODPBlockwiseFilter
from .ConsistencyFilter import ConsistencyFilter
from .CertaintyFilter import CertaintyFilter

class P_Cube_Filter:
    def __init__(self, cfg, model_architecture, source_model):
        print("Initializing P-CUBE Filters...")
        self.source_model = source_model
        
        # Cổng 1: Lọc Ổn định
        self.odp_filter = ODPBlockwiseFilter(
            model_architecture=model_architecture,
            pruning_ratio=cfg.P_CUBE.FILTER.ODP_RATIO,
            threshold=cfg.P_CUBE.FILTER.ODP_THRESHOLD
        )

        # Cổng 2: Lọc Nhất quán
        self.consistency_filter = ConsistencyFilter(source_model)

        # Cổng 3: Lọc Chắc chắn
        self.certainty_filter = CertaintyFilter(
            num_classes=cfg.CORRUPTION.NUM_CLASS,
            threshold_factor=cfg.P_CUBE.FILTER.ENTROPY_FACTOR
        )

    @torch.no_grad()
    def filter_batch(self, batch_samples, current_model):
        num_initial_samples = len(batch_samples)
        if num_initial_samples == 0:
            return torch.tensor([], dtype=torch.bool, device=batch_samples.device)
        
        # Tạo mask ban đầu, tất cả đều là True
        final_mask = torch.ones(len(batch_samples), dtype=torch.bool, device=batch_samples.device)
        
        num_after_prev_gate = num_initial_samples

        # --- Cổng 1: Lọc Ổn định (ODP) ---
        stable_mask, _ = self.odp_filter.check_batch(batch_samples, current_model)
        final_mask &= stable_mask # Dùng phép AND logic để cập nhật mask
        
        num_after_gate1 = final_mask.sum().item()
        print(f"Gate 1 (ODP): {num_after_gate1}/{num_after_prev_gate} samples passed.")
        num_after_prev_gate = num_after_gate1
        
        if num_after_gate1 == 0:
            # Nếu không có mẫu nào còn lại, thoát sớm để tiết kiệm tính toán
            print("P-CUBE Filter: 0 samples passed after ODP filter.")
            return final_mask
        
        # --- Cổng 2: Lọc Nhất quán ---
        samples_to_check_consistency = batch_samples[final_mask]
        consistent_mask_relative = self.consistency_filter.check_batch(samples_to_check_consistency, current_model)
        
        # Cập nhật mask tổng: đặt các vị trí không nhất quán thành False
        final_mask[final_mask.clone()] = consistent_mask_relative

        num_after_gate2 = final_mask.sum().item()
        print(f"Gate 2 (Consistency): {num_after_gate2}/{num_after_prev_gate} samples passed.")
        num_after_prev_gate = num_after_gate2
        if num_after_gate2 == 0:
            print("P-CUBE Filter: 0 samples passed after Consistency filter.")
            return final_mask

        # --- Cổng 3: Lọc Chắc chắn ---
        samples_to_check_certainty = batch_samples[final_mask]
        certain_mask_relative, _ = self.certainty_filter.check_batch(samples_to_check_certainty, current_model)
        
        # Cập nhật mask tổng lần cuối
        final_mask[final_mask.clone()] = certain_mask_relative
        
        num_after_gate3 = final_mask.sum().item()
        print(f"Gate 3 (Certainty): {num_after_gate3}/{num_after_prev_gate} samples passed.")
        
        return final_mask