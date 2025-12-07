import torch
import torch.nn.functional as F
from torchvision import transforms
from P_CUBE.custom_transforms import get_tta_transforms

@torch.enable_grad()
def _calculate_replay_loss_rotta_like(sup_data, ages, transform, student_model, teacher_model, cfg):
    device = next(teacher_model.parameters()).device
    
    l_sup = None
    if len(sup_data) > 0:
        # Chuyển dữ liệu sang đúng device
        sup_data = torch.stack(sup_data).to(device)
        ages = torch.tensor(ages).float().to(device)

        strong_sup_aug = transform(sup_data)
        ema_sup_out = teacher_model(sup_data)
        stu_sup_out = student_model(strong_sup_aug)
        instance_weight = timeliness_reweighting(ages)
        l_sup = (softmax_entropy(stu_sup_out, ema_sup_out.detach()) * instance_weight).mean()

        print (f"Loss over time: {l_sup}")

    return l_sup

@torch.enable_grad()
def _calculate_replay_loss(sup_data, ages, transform, student_model, teacher_model, cfg):
    device = next(teacher_model.parameters()).device
    
    l_sup = None
    if len(sup_data) > 0:
        # Chuyển dữ liệu sang đúng device
        sup_data = torch.stack(sup_data).to(device)
        ages = torch.tensor(ages).float().to(device)

        # --- TẠO NHÃN GIẢ BẰNG PAIRED-VIEW CÓ ĐIỀU KIỆN ---
        with torch.no_grad():
            teacher_model.eval()
            flipped_samples = torch.flip(sup_data, dims=[-1])
            
            outputs_original = teacher_model(sup_data)
            outputs_flipped = teacher_model(flipped_samples)
            
            probs_original = F.softmax(outputs_original, dim=1)
            probs_flipped = F.softmax(outputs_flipped, dim=1)

            conf_original, pred_original = probs_original.max(dim=1)
            conf_flipped, pred_flipped = probs_flipped.max(dim=1)

            # Lấy ngưỡng từ config
            confidence_threshold = 0.7 #cfg.P_CUBE.REPLAY.PV_CONF_THRESHOLD
            
            consistent_mask = (pred_original == pred_flipped) & \
                            (conf_original > confidence_threshold) & \
                            (conf_flipped > confidence_threshold)

            draft_logits = torch.zeros_like(outputs_original)
            
            # Với các mẫu nhất quán: lấy trung bình logits
            if consistent_mask.sum() > 0:
                draft_logits[consistent_mask] = (outputs_original[consistent_mask] + outputs_flipped[consistent_mask]) / 2.0
            
            # Với các mẫu không nhất quán: chọn logits của dự đoán tự tin hơn
            inconsistent_mask = ~consistent_mask
            if inconsistent_mask.sum() > 0:
                draft_logits[inconsistent_mask] = torch.where(
                    conf_original[inconsistent_mask].unsqueeze(1) > conf_flipped[inconsistent_mask].unsqueeze(1),
                    outputs_original[inconsistent_mask],
                    outputs_flipped[inconsistent_mask]
                )
        # -----------------------------------------------------

        strong_sup_aug = transform(sup_data)
        # ema_sup_out = teacher_model(sup_data)
        stu_sup_out = student_model(strong_sup_aug)
        instance_weight = timeliness_reweighting(ages)
        l_sup = (softmax_entropy(stu_sup_out, draft_logits.detach()) * instance_weight).mean()

        print (f"Loss over time: {l_sup}")

    return l_sup


def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))

@torch.enable_grad()
def _calculate_replay_loss_2(replay_batch, student_model, teacher_model, cfg):
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
    original_samples = torch.stack([item.sample for item in replay_batch]).to(device)
    
    # --- Bước 1: Tạo Nhãn giả Dự thảo bằng Paired-View (Insight từ DPLOT) ---
    with torch.no_grad():
        teacher_model.eval()
        
        # Tạo phiên bản lật ngang
        flipped_samples = torch.flip(original_samples, dims=[-1])
        
        # Lấy dự đoán từ Teacher model cho cả hai phiên bản
        outputs_original = teacher_model(original_samples)
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
            augmented_samples = weak_augment_transform(original_samples)
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
    
    # Temperature (T) là một siêu tham số mới để điều chỉnh độ "sắc nét" của trọng số.
    weight_temperature = cfg.P_CUBE.REPLAY.WEIGHT_TEMP # ví dụ: 0.5
    weights = F.softmax(-quality_penalty / weight_temperature, dim=0)
    normalized_weights = weights

    # --- Bước 4: Cập nhật Student với Strong Augmentation ---
    # Đảm bảo student model ở chế độ train
    student_model.train()
    

    # TODO: Thử nghiệm 2 phiên bản 
    # 1. Sử dụng các mẫu GỐC qua Student model
    # 2. Áp dụng Strong Augmentation cho Student
    
    # student_outputs = student_model(original_samples)
    strong_augment_transform = get_tta_transforms(cfg)
    student_input = strong_augment_transform(original_samples)
    student_outputs = student_model(student_input)
    
    # Tính Symmetric Cross-Entropy (SCE) loss cho từng mẫu
    individual_losses = symmetric_cross_entropy_loss(student_outputs, y_draft.detach(), reduction='none')
    
    # Áp dụng trọng số
    # Dùng reshape(-1) để đảm bảo an toàn nếu individual_losses có nhiều hơn 1 chiều
    weighted_loss = torch.sum(normalized_weights * individual_losses.reshape(-1))
    
    return weighted_loss

def get_weak_augment_transform(cfg):
    # Lấy các giá trị từ config, có thể cung cấp giá trị mặc định
    crop_scale_min = 0.8 #cfg.P_CUBE.REPLAY.AUG.CROP_SCALE_MIN # ví dụ: 0.8
    crop_scale_max = 1.0 #cfg.P_CUBE.REPLAY.AUG.CROP_SCALE_MAX # ví dụ: 1.0
    brightness_factor = 0.2 #cfg.P_CUBE.REPLAY.AUG.BRIGHTNESS # ví dụ: 0.2
    contrast_factor = 0.2 #cfg.P_CUBE.REPLAY.AUG.CONTRAST # ví dụ: 0.2
    
    # Giả sử kích thước ảnh đầu vào là 32x32 cho CIFAR
    input_size = 32 #cfg.DATA.INPUT_SIZE # ví dụ: 32

    weak_augment = transforms.Compose([
        # Cắt một vùng ngẫu nhiên có kích thước từ 80% đến 100% ảnh gốc,
        # sau đó resize lại về kích thước ban đầu.
        transforms.RandomResizedCrop(size=input_size, scale=(crop_scale_min, crop_scale_max)),
        
        # Thay đổi ngẫu nhiên độ sáng, tương phản, bão hòa và màu sắc ở mức độ nhẹ.
        transforms.ColorJitter(
            brightness=brightness_factor,
            contrast=contrast_factor,
            saturation=0, # Giữ nguyên saturation và hue để tránh thay đổi màu quá mạnh
            hue=0
        ),
        # Có thể thêm các phép augment nhẹ khác nếu cần, ví dụ:
        # transforms.RandomAffine(degrees=5), # Xoay nhẹ
    ])

    return weak_augment

def symmetric_cross_entropy_loss(outputs, targets, reduction='mean'):
    alpha = 1.0
    beta = 1.0

    # Chuyển logits thành xác suất
    probs = F.softmax(outputs, dim=1)
    
    # 1. Tính Cross-Entropy (CE)
    # CE(p, q) = -sum(q * log(p))
    ce_loss = -torch.sum(targets * torch.log(probs + 1e-8), dim=1)

    # 2. Tính Reverse Cross-Entropy (RCE)
    # RCE(p, q) = -sum(p * log(q))
    rce_loss = -torch.sum(probs * torch.log(targets + 1e-8), dim=1)

    # 3. Kết hợp lại
    loss = alpha * ce_loss + beta * rce_loss

    # Áp dụng reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
