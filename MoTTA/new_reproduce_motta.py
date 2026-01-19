import torch
import random
import numpy as np
from motta import MoTTA, normalize_model
from optimizer import build_optimizer
from imagenet_subsets import create_imagenet_subset, ALL_WNIDS, IMAGENET_R_WNIDS
from MoTTA.new_stream_loader import MoTTAStream
from torchvision import models as pt_models
from yacs.config import CfgNode as cdict

def setup_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def reproduce():
    setup_seed(2024)
    device = torch.device("cuda")

    # 1. Load Config (Đảm bảo các thông số GAM và ODP đúng như bạn đã gửi)
    cfg = cdict(new_allowed=True)
    cfg.merge_from_file('config.yml')

    # 2. Load Model & Normalization
    # Cực kỳ quan trọng: MoTTA cần ResNet50 bản chuẩn của torchvision
    base_model = pt_models.resnet50(weights=pt_models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = normalize_model(base_model, mu=(0.485, 0.456, 0.406), sigma=(0.229, 0.224, 0.225))

    # 3. Khởi tạo MoTTA (Nó sẽ tự gọi split_up_model và init_pruning bên trong __init__)
    model = MoTTA(model=backbone, **cfg.paras_adapt_model)
    model.to(device)

    # 4. Chuẩn bị Data (Dùng đúng tỷ lệ nhiễu 0.2)
    # SH Scenario
    target_ds = create_imagenet_subset("/Users/admin/Working/Data-MoTTA/imagenet-r", "imagenet_r", split="")
    SH_WNIDS = [w for w in ALL_WNIDS if w not in IMAGENET_R_WNIDS]
    from imagenet_subsets import create_file_list
    sh_noise_samples = create_file_list("/Users/admin/Working/Data-MoTTA/imagenet", SH_WNIDS, split="val")
    
    stream_dataset = MoTTAStream(target_ds.samples, sh_noise_samples, noise_ratio=0.2)
    loader = torch.utils.data.DataLoader(stream_dataset, batch_size=64, shuffle=False)

    # 5. Vòng lặp Adaptation & Evaluation
    model.eval()
    correct = 0
    total = 0

    for i, (images, labels, is_noise) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # BƯỚC QUAN TRỌNG: Forward qua MoTTA
        # MoTTA sẽ thực hiện: 
        # a) Tính ODP metric
        # b) Nếu i % update_frequency == 0: Chạy GAM Optimizer trên Memory Bank
        with torch.no_grad():
            output_dict = model(images)
            logits = output_dict['logits']

        # CHỈ TÍNH ACC TRÊN MẪU SẠCH (is_noise == 0)
        clean_idx = (is_noise == 0)
        if clean_idx.any():
            preds = logits[clean_idx].argmax(dim=1)
            correct += (preds == labels[clean_idx]).sum().item()
            total += clean_idx.sum().item()

        if i % 10 == 0:
            print(f"Batch {i} | Error Rate: {(1 - correct/total)*100:.2f}%")

    print(f"\nFinal Error Rate on ImageNet-R (SH): {(1 - correct/total)*100:.2f}%")

if __name__ == '__main__':
    reproduce()