import os
import torch
import csv
import gc
from torchvision import models as pt_models, transforms
from torch.utils.data import DataLoader, ConcatDataset
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

def get_continual_loader(corruption_list):
    """Tạo một DataLoader duy nhất chạy qua tất cả corruptions liên tiếp."""
    all_datasets = []
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    
    for corr in corruption_list:
        path_c = os.path.join(IMAGENETC_ROOT, corr, str(SEVERITY))
        if not os.path.exists(path_c): continue
        
        target_ds = ImageNet1KFolder(root=path_c, transform=transform)
        noise_ds = ImageNet1KFolder(root=NINCO_ROOT, transform=transform)
        
        # Tạo stream cho từng corruption
        stream = MoTTAStream(target_ds.samples, noise_ds.samples, noise_ratio=0.2, transform=transform)
        all_datasets.append(stream)
        
    return DataLoader(ConcatDataset(all_datasets), batch_size=64, shuffle=False, num_workers=4)

def run_continual_experiment(mode, device):
    print(f"\n[Continual Mode: {mode}] Running full sequence...")
    wandb.init(project="MoTTA_Continual_Benchmark", name=mode, reinit=True)
    
    # 1. Khởi tạo Model 1 lần duy nhất
    mu, sigma = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    cfg = cdict(new_allowed=True); cfg.merge_from_file('config.yml')

    if mode == "Source_Only":
        model = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma).to(device).eval()
    elif mode == "MoTTA_Original":
        backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma).to(device)
        model = MoTTA(model=backbone, **cfg.paras_adapt_model).to(device)
    else: # MoTTA_AAMP
        backbone = aamp_normalize_model(pt_models.resnet50(pretrained=True), mu, sigma).to(device)
        model = MoTTA_AAMP(model=backbone, cfg=cfg, **cfg.paras_adapt_model).to(device)

    # 2. Evaluation
    loader = get_continual_loader(CORRUPTIONS)
    correct, total = 0, 0
    
    # Dictionary lưu Error rate từng loại nhiễu
    corruption_errors = {corr: {"correct": 0, "total": 0} for corr in CORRUPTIONS}
    
    # Chia dữ liệu theo corruptions (giả định DataLoader trả về batch theo đúng thứ tự list)
    # Lưu ý: Bạn cần tùy chỉnh để biết batch hiện tại thuộc corruption nào
    # Cách đơn giản: Chia loader theo corruption trước khi Concat
    
    for corr in CORRUPTIONS:
        loader_corr = get_dataloader(corr) # Helper cũ của bạn
        for images, labels, is_noise in loader_corr:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                logits = model(images)['logits'] if mode != "Source_Only" else model(images)
            
            clean_idx = (is_noise == 0)
            if clean_idx.any():
                preds = logits[clean_idx].argmax(dim=1)
                corruption_errors[corr]["correct"] += (preds == labels[clean_idx]).sum().item()
                corruption_errors[corr]["total"] += clean_idx.sum().item()
        
        # Kết quả từng corruption
        c_err = 100 - (corruption_errors[corr]["correct"] / corruption_errors[corr]["total"]) * 100
        print(f"  -> {corr}: {c_err:.2f}%")
        correct += corruption_errors[corr]["correct"]
        total += corruption_errors[corr]["total"]

    avg_error = 100 - (correct / total) * 100
    print(f"\n[FINAL] Avg Error Rate: {avg_error:.2f}%")
    
    wandb.log({"final_avg_error": avg_error})
    wandb.finish()
    return avg_error, corruption_errors


def main():
    device = torch.device("cuda")
    MODES = ["Source_Only", "MoTTA_Original", "MoTTA_AAMP"]
    results = {m: {} for m in MODES}

    for mode in MODES:
        avg_err, details = run_continual_experiment(mode, device)
        results[mode]["Avg"] = avg_err
        for corr, data in details.items():
            results[mode][corr] = 100 - (data["correct"]/data["total"])*100

    # Lưu CSV
    with open("continual_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Corruption"] + MODES)
        for corr in CORRUPTIONS + ["Avg"]:
            writer.writerow([corr] + [results[m].get(corr, 0) for m in MODES])

if __name__ == "__main__":
    main()

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

# # --- HÀM CHẠY EXPERIMENT CHUNG ---
# def run_experiment(mode, corruption_type, device):
#     print(f"\n[{mode}] Running on {corruption_type}...")


#     wandb.init(project="MoTTA_Benchmark", name="ODP_Comparison_Time")

    
#     # 1. Prepare Data
#     loader = get_dataloader(corruption_type)
#     if loader is None: return None

#     # 2. Init Model
#     mu = (0.485, 0.456, 0.406)
#     sigma = (0.229, 0.224, 0.225)
    
#     if mode == "Source_Only":
#         # Load backbone thuần
#         model = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
#         model.eval() # Quan trọng: Source Only không update BN
        
#     elif mode == "MoTTA_Original":
#         cfg = cdict(new_allowed=True)
#         cfg.merge_from_file('config.yml')
#         backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
#         backbone.to(device)
#         model = MoTTA(model=backbone, **cfg.paras_adapt_model)
#         model.to(device)
#         model.eval() # MoTTA tự chuyển sang train() nội bộ khi cần

#     elif mode == "MoTTA_AAMP":
#         cfg = cdict(new_allowed=True)
#         cfg.merge_from_file('config.yml')
#         # MoTTA_AAMP dùng hàm normalize riêng nếu cần (như trong code mẫu của bạn)
#         backbone = aamp_normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
#         backbone.to(device)
#         model = MoTTA_AAMP(model=backbone, **cfg.paras_adapt_model)
#         model.to(device)
#         model.eval()

#     model.to(device)
    
#     # 3. Evaluation Loop
#     correct = 0
#     total = 0
    

#     try:
#         for i, (images, labels, is_noise) in enumerate(loader):
#             images, labels = images.to(device), labels.to(device)
            
#             if mode == "Source_Only":
#                 with torch.no_grad():
#                     logits = model(images)
#             else:
#                 # MoTTA variants tự handle forward/update
#                 with torch.no_grad():
#                     output_dict = model(images)
#                     logits = output_dict['logits']

#             # Chỉ tính trên mẫu sạch
#             clean_idx = (is_noise == 0)
#             if clean_idx.any():
#                 preds = logits[clean_idx].argmax(dim=1)
#                 correct += (preds == labels[clean_idx]).sum().item()
#                 total += clean_idx.sum().item()
                
#             if wandb.run is not None:
#                 batch_acc = (preds == labels[clean_idx]).float().mean().item() * 100
#                 wandb.log({
#                     "acc/batch_accuracy": batch_acc,
#                     "step": i # Optional
#                 }, commit=True) # Kích hoạt đẩy dữ liệu lên
#             # Optional: Print progress
#             if i % 5 == 0: print(f"  Batch {i}...")
            
#     except Exception as e:
#         print(f"Error at {corruption_type}: {e}")
#         return "Error"
    
#     error_rate = 100 - (correct / total) * 100
#     print(f"  -> Result: {error_rate:.2f}%")
    
#     # Dọn dẹp GPU
#     del model, loader
#     torch.cuda.empty_cache()
#     gc.collect()
    
#     return error_rate

# # --- MAIN LOOP ---
# def main():
#     device = torch.device("cuda")
#     results = []
    
#     # Header cho CSV
#     headers = ["Corruption", "Source_Only", "MoTTA_Original", "MoTTA_AAMP"]
    
#     # Mở file CSV để ghi dần (tránh mất dữ liệu nếu crash)
#     with open("benchmark_results.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)

#     for corruption in CORRUPTIONS:
#         row = [corruption]
        
#         # 1. Chạy Source Only
#         # err_src = run_experiment("Source_Only", corruption, device)
#         # row.append(f"{err_src:.2f}" if isinstance(err_src, float) else err_src)
        
#         # 2. Chạy MoTTA Gốc
#         # err_orig = run_experiment("MoTTA_Original", corruption, device)
#         # row.append(f"{err_orig:.2f}" if isinstance(err_orig, float) else err_orig)
        
#         # 3. Chạy MoTTA AAMP
#         err_aamp = run_experiment("MoTTA_AAMP", corruption, device)
#         row.append(f"{err_aamp:.2f}" if isinstance(err_aamp, float) else err_aamp)
        
#         # Ghi vào CSV ngay lập tức
#         with open("benchmark_results.csv", "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(row)
            
#         print("-" * 50)

#     print("Benchmark Completed! Results saved to benchmark_results.csv")

# if __name__ == "__main__":
#     main()