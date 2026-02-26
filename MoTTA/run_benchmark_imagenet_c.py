import os
import torch
import csv
import gc
from torchvision import models as pt_models, transforms
from torch.utils.data import DataLoader
from yacs.config import CfgNode as cdict
from torchvision.datasets import ImageFolder
from imagenet_subsets import ALL_WNIDS

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
    # "gaussian_noise", 
    # "shot_noise", 
    # "impulse_noise",
    # "defocus_blur", 
    # "glass_blur", 
    # "motion_blur", 
    # "zoom_blur",
    # "snow", 
    # "frost", 
    # "fog",
    "brightness",
    "contrast",
    "elastic_transform", 
    "pixelate", 
    "jpeg_compression"
]

# CORRUPTIONS = ["zoom_blur"]

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

# --- HÀM CHẠY EXPERIMENT CHUNG ---
def run_experiment(mode, corruption_type, device):
    print(f"\n[{mode}] Running on {corruption_type}...")
    
    # 1. Prepare Data
    loader = get_dataloader(corruption_type)
    if loader is None: return None

    # 2. Init Model
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    
    if mode == "Source_Only":
        # Load backbone thuần
        model = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
        model.eval() # Quan trọng: Source Only không update BN
        
    elif mode == "MoTTA_Original":
        cfg = cdict(new_allowed=True)
        cfg.merge_from_file('config.yml')
        backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
        backbone.to(device)
        model = MoTTA(model=backbone, **cfg.paras_adapt_model)
        model.to(device)
        model.eval() # MoTTA tự chuyển sang train() nội bộ khi cần

    elif mode == "MoTTA_AAMP":
        cfg = cdict(new_allowed=True)
        cfg.merge_from_file('config.yml')
        # MoTTA_AAMP dùng hàm normalize riêng nếu cần (như trong code mẫu của bạn)
        backbone = aamp_normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
        backbone.to(device)
        model = MoTTA_AAMP(model=backbone, **cfg.paras_adapt_model)
        model.to(device)
        model.eval()

    model.to(device)
    
    # 3. Evaluation Loop
    correct = 0
    total = 0
    

    try:
        for i, (images, labels, is_noise) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            if mode == "Source_Only":
                with torch.no_grad():
                    logits = model(images)
            else:
                # MoTTA variants tự handle forward/update
                with torch.no_grad():
                    output_dict = model(images)
                    logits = output_dict['logits']

            # Chỉ tính trên mẫu sạch
            clean_idx = (is_noise == 0)
            if clean_idx.any():
                preds = logits[clean_idx].argmax(dim=1)
                correct += (preds == labels[clean_idx]).sum().item()
                total += clean_idx.sum().item()
                
            # Optional: Print progress
            if i % 5 == 0: print(f"  Batch {i}...")
            
    except Exception as e:
        print(f"Error at {corruption_type}: {e}")
        return "Error"
    
    error_rate = 100 - (correct / total) * 100
    print(f"  -> Result: {error_rate:.2f}%")
    
    # Dọn dẹp GPU
    del model, loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return error_rate

# --- MAIN LOOP ---
def main():
    device = torch.device("cuda")
    results = []
    
    # Header cho CSV
    headers = ["Corruption", "Source_Only", "MoTTA_Original", "MoTTA_AAMP"]
    
    # Mở file CSV để ghi dần (tránh mất dữ liệu nếu crash)
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    for corruption in CORRUPTIONS:
        row = [corruption]
        
        # 1. Chạy Source Only
        err_src = run_experiment("Source_Only", corruption, device)
        row.append(f"{err_src:.2f}" if isinstance(err_src, float) else err_src)
        
        # 2. Chạy MoTTA Gốc
        err_orig = run_experiment("MoTTA_Original", corruption, device)
        row.append(f"{err_orig:.2f}" if isinstance(err_orig, float) else err_orig)
        
        # 3. Chạy MoTTA AAMP
        err_aamp = run_experiment("MoTTA_AAMP", corruption, device)
        row.append(f"{err_aamp:.2f}" if isinstance(err_aamp, float) else err_aamp)
        
        # Ghi vào CSV ngay lập tức
        with open("benchmark_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print("-" * 50)

    print("Benchmark Completed! Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()