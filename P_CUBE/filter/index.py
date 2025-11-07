from .ODPBlockwiseFilter import ODPBlockwiseFilter
from .ConsistencyFilter import ConsistencyFilter
from .CertaintyFilter import CertaintyFilter


class P_Cube_Filter:
    
    def __init__(self, source_model):
        self.source_model = source_model
        # Khởi tạo các thành phần

        pruning_ratio = 0
        T_ent = 0
        self.odp_filter = ODPBlockwiseFilter(pruning_ratio=pruning_ratio)
        self.consistency_filter = ConsistencyFilter(self.source_model)
        self.certainty_filter = CertaintyFilter(entropy_threshold=T_ent)

    def isSampleClean(self, sample, current_model):
        # === Gate 1: ODP filter ===
        # is_stable, odp_score = self.odp_filter.check(sample, current_model)
        is_stable = True #TODO remove this when start implementation
        if not is_stable:
            print(f"Sample rejected by ODP Filter")
            return False
            
        # === Gate 2: Consistency filter ===
        # is_consistent = self.consistency_filter.check(sample, current_model)
        is_consistent = True #TODO remove this when start implementation
        if not is_consistent:
            print("Sample rejected by Consistency Filter")
            return False

        # === Gate 2: Consistency filter ===
        # is_certain, entropy = self.certainty_filter.check(sample, current_model)
        is_certain = True #TODO remove this when start implementation
        if not is_certain:
            print(f"Sample rejected by Certainty Filter")
            return False
            
        # === Clean Sample. Accepted to next Stage ===        
        return True


