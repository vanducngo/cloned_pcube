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
    # "shot_noise", 
    # "impulse_noise",
    "defocus_blur", 
    # "glass_blur", 
    # "motion_blur", 
    # "zoom_blur",
    "snow", 
    # "frost", 
    # "fog",
    # "brightness",
    "contrast",
    # "elastic_transform", 
    # "pixelate", 
    # "jpeg_compression"
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

def run_experiment(mode, corruption_type, device):
    print(f"\n[{mode}] Running on {corruption_type}...")

    # Đặt tên Wandb Run rõ ràng để phân biệt các Ablation
    wandb.init(
        project="MoTTA_Ablation_Study", 
        name=f"{mode}_{corruption_type}",
        reinit=True # Cho phép chạy nhiều run trong cùng 1 script
    )
    
    # 1. Prepare Data
    loader = get_dataloader(corruption_type)
    if loader is None: 
        wandb.finish()
        return None

    # 2. Init Model
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    
    # Load config chuẩn
    cfg = cdict(new_allowed=True)
    cfg.merge_from_file('config.yml')
    
    # Đảm bảo có nhánh P_CUBE trong config để ghi đè
    if not hasattr(cfg, 'P_CUBE'):
        cfg.P_CUBE = cdict(new_allowed=True)
        cfg.P_CUBE.FILTER = cdict(new_allowed=True)
        cfg.P_CUBE.MEMORY = cdict(new_allowed=True)

    if mode == "Source_Only":
        model = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
        model.eval() 
        
    elif mode == "MoTTA_Original":
        backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
        backbone.to(device)
        model = MoTTA(model=backbone, **cfg.paras_adapt_model)
        model.to(device)
        model.eval()

    else:
        # --- CÁC KỊCH BẢN ABLATION DÙNG CLASS MoTTA_AAMP ---
        # Ghi đè cấu hình dựa trên Mode
        if mode == "MoTTA_AAMP":
            cfg.P_CUBE.FILTER.ODP_TYPE = "blockwise"
            cfg.P_CUBE.MEMORY.TYPE = "aamp"
        elif mode == "Ablation_ODP_Block_Mem_MoTTA":
            cfg.P_CUBE.FILTER.ODP_TYPE = "blockwise"
            cfg.P_CUBE.MEMORY.TYPE = "motta"
        elif mode == "Ablation_ODP_Orig_Mem_AAMP":
            cfg.P_CUBE.FILTER.ODP_TYPE = "original"
            cfg.P_CUBE.MEMORY.TYPE = "aamp"
        elif mode == "Ablation_ODP_Orig_Mem_MoTTA":
            # [NEW] Sanity Check: Sử dụng wrapper kiến trúc mới, nhưng ruột là hàng cũ
            cfg.P_CUBE.FILTER.ODP_TYPE = "original"
            cfg.P_CUBE.MEMORY.TYPE = "motta"

        backbone = aamp_normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
        backbone.to(device)
        
        # LƯU Ý: Phải truyền cái cfg đã được ghi đè vào MoTTA_AAMP
        # Nếu __init__ của bạn không nhận cfg trực tiếp, hãy thêm nó vào:
        # model = MoTTA_AAMP(model=backbone, p_cube_cfg=cfg.P_CUBE, **cfg.paras_adapt_model)
        model = MoTTA_AAMP(model=backbone, cfg=cfg, **cfg.paras_adapt_model) 
        
        model.to(device)
        model.eval()

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
                with torch.no_grad():
                    output_dict = model(images)
                    logits = output_dict['logits']

            clean_idx = (is_noise == 0)
            if clean_idx.any():
                preds = logits[clean_idx].argmax(dim=1)
                correct += (preds == labels[clean_idx]).sum().item()
                total += clean_idx.sum().item()
                
            if wandb.run is not None and total > 0:
                batch_acc = (preds == labels[clean_idx]).float().mean().item() * 100 if clean_idx.any() else 0
                cum_acc = (correct / total) * 100
                wandb.log({
                    "acc/batch_accuracy": batch_acc,
                    "acc/cumulative_accuracy": cum_acc,
                    "step": i
                }, commit=True)
                
            if i % 5 == 0: print(f"  Batch {i} | Cum. Acc: {(correct/total)*100:.2f}%")
            
    except Exception as e:
        print(f"Error at {corruption_type}: {e}")
        wandb.finish()
        return "Error"
    
    error_rate = 100 - (correct / total) * 100
    print(f"  -> Result: {error_rate:.2f}%")
    
    # Kết thúc WandB run hiện tại
    wandb.finish()
    
    del model, loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return error_rate

def main():
    device = torch.device("cuda")
    
    # Danh sách các kịch bản cần test
    MODES_TO_RUN = [
        "Source_Only",
        "MoTTA_Original",                # Lấy từ file motta.py gốc
        "Ablation_ODP_Orig_Mem_MoTTA",   # Sanity Check (P_CUBE Factory)
        "Ablation_ODP_Block_Mem_MoTTA",  # Test ODP
        "Ablation_ODP_Orig_Mem_AAMP",    # Test Memory
        "MoTTA_AAMP"                     # Phiên bản Full cải tiến
    ]
    
    headers = ["Corruption"] + MODES_TO_RUN
    
    with open("ablation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    for corruption in CORRUPTIONS:
        row = [corruption]
        
        for mode in MODES_TO_RUN:
            err = run_experiment(mode, corruption, device)
            row.append(f"{err:.2f}" if isinstance(err, float) else err)
        
        with open("ablation_results.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print("-" * 50)

    print("Ablation Study Completed! Results saved to ablation_results.csv")

if __name__ == "__main__":
    main()