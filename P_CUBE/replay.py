import torch
import torch.nn.functional as F

@torch.enable_grad()
def _calculate_replay_loss(replay_batch, student_model, teacher_model, cfg):
    """
    Thực hiện pipeline Giai đoạn 3: Lấy batch replay, tạo nhãn giả chất lượng cao,
    và tính toán loss để cập nhật Student model.

    Args:
        replay_batch (list[MemoryItem]): Một batch các đối tượng MemoryItem từ bộ nhớ.
        student_model (torch.nn.Module): Mô hình Student cần được cập nhật.
        teacher_model (torch.nn.Module): Mô hình Teacher để tạo nhãn giả.
        cfg (config object): Đối tượng config chứa các siêu tham số.

    Returns:
        Tensor or None: Giá trị loss cuối cùng.
    """
    
    # Gom dữ liệu từ batch replay
    device = next(teacher_model.parameters()).device
    samples = torch.stack([item.sample for item in replay_batch]).to(device)
    
    # --- Bước 1: Tạo Nhãn giả Dự thảo bằng Paired-View (Insight từ DPLOT) ---
    with torch.no_grad():
        teacher_model.eval()
        
        # Tạo phiên bản lật ngang
        flipped_samples = torch.flip(samples, dims=[-1])
        
        # Lấy dự đoán từ Teacher model cho cả hai phiên bản
        outputs_original = teacher_model(samples)
        outputs_flipped = teacher_model(flipped_samples)
        
        probs_original = F.softmax(outputs_original, dim=1)
        probs_flipped = F.softmax(outputs_flipped, dim=1)
        
        # y_draft là nhãn giả mềm ban đầu (trung bình cộng)
        y_draft = (probs_original + probs_flipped) / 2.0

    # --- Bước 2: Đánh giá Chất lượng Nhãn giả ---
    with torch.no_grad():
        # 2a: Tính Độ Chắc chắn (Certainty) bằng Entropy
        entropies = -torch.sum(y_draft * torch.log(y_draft + 1e-8), dim=-1)
        
        # 2b: Tính Độ Ổn định (Stability) bằng Augmentation Sensitivity
        disagreement_scores = torch.zeros_like(entropies)
        
        # Lấy transform từ config
        weak_augment_transform = get_weak_augment_transform(cfg)
        num_aug_checks = cfg.P_CUBE.REPLAY.NUM_AUG_CHECKS # ví dụ: 2

        for _ in range(num_aug_checks):
            augmented_samples = weak_augment_transform(samples)
            probs_aug = F.softmax(teacher_model(augmented_samples), dim=-1)
            
            # Tính KL Divergence giữa nhãn dự thảo và dự đoán của bản augment
            # KL(p || q) = sum(p * (log(p) - log(q)))
            kl_div = torch.sum(y_draft * (torch.log(y_draft + 1e-8) - torch.log(probs_aug + 1e-8)), dim=-1)
            disagreement_scores += kl_div
    
    # --- Bước 3: Tính Loss có Trọng số Thích ứng cho Student ---
    # Lấy các siêu tham số từ config
    w_ent = cfg.P_CUBE.REPLAY.W_ENT # ví dụ: 1.0
    w_dis = cfg.P_CUBE.REPLAY.W_DIS # ví dụ: 0.5
    
    # Tính điểm phạt chất lượng (quality penalty score)
    quality_penalty = (w_ent * entropies) + (w_dis * disagreement_scores)
    
    # Chuyển thành trọng số (mẫu có penalty thấp sẽ có trọng số cao)
    weights = torch.exp(-quality_penalty)
    
    # Chuẩn hóa trọng số để tổng bằng 1 (giúp loss ổn định)
    normalized_weights = weights / (torch.sum(weights) + 1e-8)

    # --- Tính Loss cuối cùng và Cập nhật Student ---
    # Đảm bảo student model ở chế độ train
    student_model.train()
    
    # Đưa các mẫu GỐC qua Student model
    student_outputs = student_model(samples)
    
    # Tính Symmetric Cross-Entropy (SCE) loss cho từng mẫu
    # `reduction='none'` để có thể áp dụng trọng số
    individual_losses = symmetric_cross_entropy_loss(student_outputs, y_draft.detach(), reduction='none')
    
    # Áp dụng trọng số
    # Dùng reshape(-1) để đảm bảo an toàn nếu individual_losses có nhiều hơn 1 chiều
    weighted_loss = torch.sum(normalized_weights * individual_losses.reshape(-1))
    
    return weighted_loss

def get_weak_augment_transform(cfg):
    # Hàm này sẽ trả về một đối tượng transform của torchvision
    # ví dụ: transforms.Compose([transforms.RandomResizedCrop(...), ...])
    # Dựa trên cấu hình trong cfg
    pass

def symmetric_cross_entropy_loss(outputs, targets, reduction='mean'):
    # Hàm này triển khai SCE loss
    # loss = alpha * CE(outputs, targets) + beta * RCE(outputs, targets)
    # ...
    pass
