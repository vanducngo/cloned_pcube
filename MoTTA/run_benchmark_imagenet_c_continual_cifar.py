import os
import torch
import csv
import gc
from torchvision import transforms
from torch.utils.data import DataLoader
from yacs.config import CfgNode as cdict
from MoTTA.model_cifar import build_model
import wandb

# Import các modules
from motta import MoTTA, normalize_model
from motta_aamp import MoTTA_AAMP, normalize_model as aamp_normalize_model
from MoTTA.new_stream_loader import MoTTAStream
from robustbench.data import load_cifar10c, load_cifar100c # Dùng thư viện chuẩn của robustbench

# --- CẤU HÌNH ---
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

def get_cifar_loader(dataset_name, corruption, severity, batch_size=64):
    """Load dữ liệu CIFAR-C bằng robustbench."""
    data_dir = "./Data" # Đường dẫn chứa data
    n_examples = 10000 
    
    if dataset_name == "cifar10":
        x, y = load_cifar10c(n_examples, severity, data_dir, False, [corruption])
    else:
        x, y = load_cifar100c(n_examples, severity, data_dir, False, [corruption])
        
    # Tạo stream với 20% nhiễu (Dùng ngẫu nhiên một phần x,y làm noise)
    # Ở đây chúng ta mock noise bằng cách xáo trộn 20% dữ liệu
    stream_dataset = MoTTAStream(list(zip(x, y)), noise_ratio=0.2) 
    return DataLoader(stream_dataset, batch_size=batch_size, shuffle=False)

def run_continual_bench(dataset_name, mode, device):
    print(f"\n[Benchmarking] {dataset_name} - {mode}")
    
    # 1. Init Model & Config
    cfg = cdict(new_allowed=True); cfg.merge_from_file('config.yml')
    # Lưu ý: Cần chọn model phù hợp cho Cifar (ResNet18 hoặc WideResNet)
    # Hãy đảm bảo hàm build_model của bạn chọn đúng kiến trúc theo cfg
    
    if mode == "Source_Only":
        model = build_model(dataset_name).to(device).eval()
    elif mode == "MoTTA_Original":
        backbone = build_model(dataset_name).to(device)
        model = MoTTA(model=backbone, **cfg.paras_adapt_model).to(device).eval()
    else: # MoTTA_AAMP
        backbone = build_model(dataset_name).to(device)
        numclass = 10 if dataset_name == 'cifar10' else 100
        cfg.num_classes = numclass
        model = MoTTA_AAMP(model=backbone, cfg=cfg, **cfg.paras_adapt_model).to(device).eval()

    # 2. Loop qua 15 corruptions
    results = []
    for corruption in CORRUPTIONS:
        loader = get_cifar_loader(dataset_name, corruption, 5)
        correct, total = 0, 0
        for images, labels, is_noise in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(images) if mode == "Source_Only" else model(images)['logits']
            
            clean_idx = (is_noise == 0)
            if clean_idx.any():
                preds = logits[clean_idx].argmax(dim=1)
                correct += (preds == labels[clean_idx]).sum().item()
                total += clean_idx.sum().item()
        
        err = 100 - (correct/total)*100
        results.append(err)
        print(f"  {corruption}: {err:.2f}%")
        
    return results

def main():
    device = torch.device("cuda")
    datasets = ["cifar10", "cifar100"]
    MODES = ["Source_Only", "MoTTA_Original", "MoTTA_AAMP"]
    
    for ds in datasets:
        with open(f"results_{ds}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Corruption"] + MODES)
            
            # Khởi tạo ma trận kết quả
            data_to_write = [[c] for c in CORRUPTIONS]
            
            for mode in MODES:
                errs = run_continual_bench(ds, mode, device)
                for i, e in enumerate(errs):
                    data_to_write[i].append(f"{e:.2f}")
            
            writer.writerows(data_to_write)
            
if __name__ == "__main__":
    main()