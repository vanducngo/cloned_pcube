import os
import torch
import csv
import gc
from torchvision import models as pt_models, transforms
from torch.utils.data import DataLoader
from yacs.config import CfgNode as cdict
from torchvision.datasets import ImageFolder
from imagenet_subsets import ALL_WNIDS
import wandb

# Import các modules của bạn
from motta import MoTTA, normalize_model
from motta_aamp import MoTTA_AAMP, normalize_model as aamp_normalize_model
from MoTTA.new_stream_loader import MoTTAStream

# --- CẤU HÌNH ĐƯỜNG DẪN ---
IMAGENETC_ROOT = "./Data/imagenet-c"
NINCO_ROOT = "./Data/NINCO"
SEVERITY = 5

# Danh sách 15 loại nhiễu của ImageNet-C
CORRUPTIONS = [
    "gaussian_noise", 
    "shot_noise", 
    "impulse_noise",
    "defocus_blur", 
    "glass_blur", 
    "motion_blur", 
    "zoom_blur",
    "snow", 
    "frost", 
    "fog",
    "brightness",
    "contrast",
    "elastic_transform", 
    "pixelate", 
    "jpeg_compression"
]

# --- DATASET HELPER ---
class ImageNet1KFolder(ImageFolder):
    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir() and d.name.startswith('n')]
        if not classes:
            classes = [d.name for d in os.scandir(directory) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {}
        for cls_name in classes:
            if cls_name in ALL_WNIDS:
                class_to_idx[cls_name] = ALL_WNIDS.index(cls_name)
            else:
                class_to_idx[cls_name] = -1
        return classes, class_to_idx

def get_dataloader(corruption_type):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    path_c = os.path.join(IMAGENETC_ROOT, corruption_type, str(SEVERITY))
    
    # Kiểm tra đường dẫn tồn tại
    if not os.path.exists(path_c):
        print(f"Warning: Không tìm thấy {path_c}. Bỏ qua.")
        return None

    target_ds = ImageNet1KFolder(root=path_c, transform=transform)
    noise_ds = ImageNet1KFolder(root=NINCO_ROOT, transform=transform)
    
    stream_dataset = MoTTAStream(target_ds.samples, noise_ds.samples, noise_ratio=0.2, transform=transform)
    loader = DataLoader(stream_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    return loader

def main_continual():
    device = torch.device("cuda")
    # MODES = ["Source_Only", "MoTTA_Original", "MoTTA_AAMP"]
    MODES = ["MoTTA_AAMP"]
    
    # Mở file CSV kết quả
    with open("continual_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Corruption"] + MODES)

    for mode in MODES:
        print(f"\n[Starting Continual Run] Mode: {mode}")
        
        # 1. KHỞI TẠO MODEL (CHỈ 1 LẦN)
        mu, sigma = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        cfg = cdict(new_allowed=True); cfg.merge_from_file('config.yml')
        
        if mode == "Source_Only":
            model = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma).to(device).eval()
        elif mode == "MoTTA_Original":
            backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma).to(device)
            model = MoTTA(model=backbone, **cfg.paras_adapt_model).to(device).eval()
        else: # MoTTA_AAMP
            backbone = aamp_normalize_model(pt_models.resnet50(pretrained=True), mu, sigma).to(device)
            model = MoTTA_AAMP(model=backbone, **cfg.paras_adapt_model).to(device).eval()

        # 2. VÒNG LẶP QUA 15 CORRUPTIONS (KHÔNG RESET MODEL)
        results_row = []
        for corruption in CORRUPTIONS:
            loader = get_dataloader(corruption)
            if loader is None: continue
            
            correct, total = 0, 0
            for i, (images, labels, is_noise) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                
                with torch.no_grad():
                    # Nếu là Source Only thì chỉ predict, không adapt
                    if mode == "Source_Only":
                        logits = model(images)
                    else:
                        output_dict = model(images)
                        logits = output_dict['logits']

                clean_idx = (is_noise == 0)
                if clean_idx.any():
                    preds = logits[clean_idx].argmax(dim=1)
                    correct += (preds == labels[clean_idx]).sum().item()
                    total += clean_idx.sum().item()
            
            error_rate = 100 - (correct / total) * 100 if total > 0 else 0
            results_row.append(f"{error_rate:.2f}")
            print(f"  {corruption}: {error_rate:.2f}%")
        
        # Ghi kết quả vào CSV
        with open("continual_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([mode] + results_row)
            
        # Giải phóng GPU trước khi qua mode mới
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main_continual()