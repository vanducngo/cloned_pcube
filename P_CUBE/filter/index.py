import torch

from P_CUBE.config import ModuleConfig
from .ODPOriginalFilter import ODPOriginalFilter
from .ODPBlockwiseFilter import ODPBlockwiseFilter
from .ConsistencyFilter import ConsistencyFilter
from .CertaintyFilter import CertaintyFilter

class P_Cube_Filter:
    def __init__(self, cfg: ModuleConfig, model_architecture, source_model):
        print("Initializing P-CUBE Filters...")
        self.source_model = source_model
        
        # # Cổng 1: Lọc Ổn định
        # self.odp_filter = ODPBlockwiseFilter(
        #     model_architecture=model_architecture,
        #     pruning_ratio=cfg.odp_ratio,
        # )

        # Cổng 1: Lọc Ổn định
        # 1. Khởi tạo ODP Filter dựa trên Config (ABLATION STUDY)
        if cfg.ablation_odp_type == "blockwise":
            self.odp_filter = ODPBlockwiseFilter(model_architecture=model_architecture, pruning_ratio=cfg.odp_ratio)
        elif cfg.ablation_odp_type == "original":
            # Tạo một class wrapper cho eval_pruning của MoTTA gốc để nó có cùng interface (.check_batch)
            self.odp_filter = ODPOriginalFilter(model_architecture, prune_ratio=cfg.odp_ratio, threshold=cfg.uncertainty_threshold)
        else:
            self.odp_filter = None # Không dùng ODP

        # Cổng 2: Lọc Nhất quán
        self.consistency_filter = ConsistencyFilter(source_model, cfg.consistent_lambda_std, cfg.consistent_hard_floor)

        # Cổng 3: Lọc Chắc chắn
        self.certainty_filter = CertaintyFilter(
            num_classes=cfg.num_classes,
            threshold_factor=cfg.entropy_factor,
            confidence_factor=cfg.confidence_factor
        )

    @torch.no_grad()
    def filter_batch(self, batch_samples, current_model):
        num_initial_samples = len(batch_samples)
        if num_initial_samples == 0:
            return torch.tensor([], dtype=torch.bool, device=batch_samples.device), None
        
        # Tạo mask ban đầu, tất cả đều là True - chấp nhận toàn bộ batch
        final_mask = torch.ones(num_initial_samples, dtype=torch.bool, device=batch_samples.device)

        # Khởi tạo mảng ODP scores cho toàn bộ batch (mặc định là 0.0)
        # Sẽ được cập nhật ở vòng lặp khi chạy qua ODP Filter
        final_odp_scores = torch.zeros(num_initial_samples, dtype=torch.float, device=batch_samples.device)

        filter_pipeline = [
            # ("Gate 1 (Certainty)", self.certainty_filter, True),   # Nhẹ (Chỉ tính Softmax & Entropy)
            # ("Gate 2 (Consistency)", self.consistency_filter, False), # Trung bình (Tính Cosine 2 models)
            ("Gate 3 (ODP)", self.odp_filter, True)                # Nặng nhất (Hooks + Forward blocks)
        ]

        num_after_prev_gate = num_initial_samples

        # ----------------------------
        # VÒNG LẶP CHẠY LỌC TUẦN TỰ
        # ----------------------------
        for gate_name, filter_obj, returns_tuple in filter_pipeline:
            # 1. Trích xuất những mẫu còn sống sót sau các cổng trước
            samples_to_check = batch_samples[final_mask]
            
            # Nếu tất cả đã bị loại thì thoát sớm (Early Exit)
            if len(samples_to_check) == 0:
                print(f"P-CUBE Filter: 0 samples passed before reaching {gate_name}.")
                break
                
            # 2. Chạy bộ lọc hiện tại
            if returns_tuple:
                # ODP và Certainty filter trả về (mask, scores)
                current_mask, current_scores = filter_obj.check_batch(samples_to_check, current_model)

                # Nếu là cổng ODP, lưu lại điểm số vào final_odp_scores
                if gate_name == "Gate 3 (ODP)":
                    # final_mask đang giữ index của những mẫu sống sót đến trước cổng này
                    # Chỉ gán điểm cho những vị trí đó
                    final_odp_scores[final_mask.clone()] = current_scores
            else:
                # Consistency filter chỉ trả về mask
                current_mask = filter_obj.check_batch(samples_to_check, current_model)
                
            # 3. Cập nhật Mask tổng
            # final_mask.clone() được dùng làm index để chỉ ghi đè lên những vị trí còn mang giá trị True
            final_mask[final_mask.clone()] = current_mask
            
            # 4. Tính toán và In log
            # num_survivors = final_mask.sum().item()
            # print(f"{gate_name}: {num_survivors}/{num_after_prev_gate} samples passed.")  
            # Cập nhật số lượng cho vòng lặp tiếp theo
            # num_after_prev_gate = num_survivors

        return final_mask, final_odp_scores

        # num_after_prev_gate = num_initial_samples

        # # --- Cổng 1: Lọc Ổn định (ODP) ---
        # stable_mask, _ = self.odp_filter.check_batch(batch_samples, current_model)
        # final_mask &= stable_mask # Dùng phép AND logic để cập nhật mask
        
        # num_after_gate1 = final_mask.sum().item()
        # print(f"Gate 1 (ODP): {num_after_gate1}/{num_after_prev_gate} samples passed.")
        # num_after_prev_gate = num_after_gate1
        
        # if num_after_gate1 == 0:
        #     # Nếu không có mẫu nào còn lại, thoát sớm để tiết kiệm tính toán
        #     print("P-CUBE Filter: 0 samples passed after ODP filter.")
        #     return final_mask
        
        # # --- Cổng 2: Lọc Nhất quán ---
        # # samples_to_check_consistency = batch_samples[final_mask]
        # # consistent_mask_relative = self.consistency_filter.check_batch(samples_to_check_consistency, current_model)
        
        # # # Cập nhật mask tổng: đặt các vị trí không nhất quán thành False
        # # final_mask[final_mask.clone()] = consistent_mask_relative

        # # num_after_gate2 = final_mask.sum().item()
        # # print(f"Gate 2 (Consistency): {num_after_gate2}/{num_after_prev_gate} samples passed.")
        # # num_after_prev_gate = num_after_gate2
        # # if num_after_gate2 == 0:
        # #     print("P-CUBE Filter: 0 samples passed after Consistency filter.")
        # #     return final_mask

        # # --- Cổng 3: Lọc Chắc chắn ---
        # samples_to_check_certainty = batch_samples[final_mask]
        # certain_mask_relative, _ = self.certainty_filter.check_batch(samples_to_check_certainty, current_model)
        
        # # Cập nhật mask tổng lần cuối
        # final_mask[final_mask.clone()] = certain_mask_relative
        
        # num_after_gate3 = final_mask.sum().item()
        # print(f"Gate 3 (Certainty): {num_after_gate3}/{num_after_prev_gate} samples passed.")
        
        # return final_mask