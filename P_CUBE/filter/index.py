import torch
from .ODPBlockwiseFilter import ODPBlockwiseFilter
from .ConsistencyFilter import ConsistencyFilter
from .CertaintyFilter import CertaintyFilter


class P_Cube_Filter:
    def __init__(self, source_model):
        self.source_model = source_model
        # Khởi tạo các thành phần

        # Config
        pruning_ratio = 0.1
        odp_threshold = 0.2
        certainty_entropy_threshold = 0.2
        
        self.odp_filter = ODPBlockwiseFilter(self.source_model, pruning_ratio, odp_threshold)
        self.consistency_filter = ConsistencyFilter(source_model)
        self.certainty_filter = CertaintyFilter(certainty_entropy_threshold)

    @torch.no_grad()
    def filter_batch(self, batch_samples, current_model):
        # Tạo mask ban đầu, tất cả đều là True
        current_mask = torch.ones(len(batch_samples), dtype=torch.bool, device=batch_samples.device)

        # Cổng 1: ODP
        stable_mask, _ = self.odp_filter.check_batch(batch_samples, current_model)
        current_mask = current_mask & stable_mask
        if current_mask.sum() == 0: return current_mask

        # Cổng 2: Consistency
        consistent_mask = self.consistency_filter.check_batch(batch_samples[current_mask], current_model)
        # Cập nhật mask tổng
        current_mask[torch.where(current_mask)[0][~consistent_mask]] = False
        if current_mask.sum() == 0: return current_mask

        # Cổng 3: Certainty
        certain_mask, _ = self.certainty_filter.check_batch(batch_samples[current_mask], current_model)
        # Cập nhật mask tổng
        current_mask[torch.where(current_mask)[0][~certain_mask]] = False
        
        return current_mask