import os
import torch
from torchvision import models as pt_models, transforms
from torch.utils.data import DataLoader
from motta_aamp import MoTTA_AAMP, normalize_model as aamp_normalize_model
from motta import MoTTA, normalize_model
from MoTTA.new_stream_loader import MoTTAStream
from yacs.config import CfgNode as cdict
from torchvision.datasets import ImageFolder

import os
from torchvision.datasets import ImageFolder
from imagenet_subsets import ALL_WNIDS # Đảm bảo file này chứa list 1000 WNID gốc

class ImageNet1KFolder(ImageFolder):
    """
    Dataset class thông minh giúp:
    1. Loại bỏ các thư mục hệ thống như .ipynb_checkpoints.
    2. Ánh xạ (Map) nhãn từ tên thư mục (nxxxx) về đúng ID chuẩn (0-999) của ImageNet-1K.
    """
    def find_classes(self, directory):
        # 1. Tìm tất cả thư mục con thực sự là class của ImageNet (bắt đầu bằng 'n')
        # d.is_dir() giúp loại bỏ các file lẻ
        # not d.name.startswith('.') giúp loại bỏ hoàn toàn .ipynb_checkpoints
        classes = [d.name for d in os.scandir(directory) 
                   if d.is_dir() and d.name.startswith('n')]
        
        if not classes:
            # Nếu là tập NINCO (không bắt đầu bằng 'n'), lấy các thư mục không ẩn
            classes = [d.name for d in os.scandir(directory) 
                       if d.is_dir() and not d.name.startswith('.')]
        
        classes.sort()

        # 2. Tạo bảng ánh xạ nhãn (Mapping)
        class_to_idx = {}
        for cls_name in classes:
            if cls_name in ALL_WNIDS:
                # Gán nhãn bằng đúng vị trí của nó trong 1000 lớp chuẩn
                class_to_idx[cls_name] = ALL_WNIDS.index(cls_name)
            else:
                # Nếu không tìm thấy trong ImageNet-1K (như NINCO), gán nhãn tạm là -1
                # (Vì MoTTA không tính độ chính xác trên nhãn NINCO)
                class_to_idx[cls_name] = -1
                
        return classes, class_to_idx

def reproduce_c():
    print(f"REPRODUCING AAMP MOTTA")
    device = torch.device("cuda")
    
    PATH_C = "./Data/imagenet-c/blur/zoom_blur/5" # Đường dẫn tập ImageNet-C
    PATH_NINCO = "./Data/NINCO"
    
    # 2. Khởi tạo Model (1000 lớp chuẩn)
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)

    backbone = aamp_normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
    backbone.to(device) # Đưa backbone lên GPU trước

    # 3. Load Config
    cfg = cdict(new_allowed=True)
    cfg.merge_from_file('config.yml')
    
    # Khởi tạo MoTTA
    model = MoTTA_AAMP(model=backbone, **cfg.paras_adapt_model)
    model.to(device)

    # 4. Tạo Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print(f"Đang load Target từ: {PATH_C}")
    target_ds = ImageNet1KFolder(root=PATH_C, transform=transform)
    print(f"Số lượng lớp Target tìm thấy: {len(target_ds.classes)}")
    # In kiểm tra thử 1 lớp:
    folder_name = target_ds.classes[0]
    label_id = target_ds.class_to_idx[folder_name]
    print(f"Ví dụ: Thư mục {folder_name} được gán nhãn chuẩn là: {label_id}")

    # 2. Load tập Noise (NINCO)
    print(f"Đang load Noise từ: {PATH_NINCO}")
    noise_ds = ImageNet1KFolder(root=PATH_NINCO, transform=transform)

    
    # Trộn luồng dữ liệu (20% NINCO)
    stream_dataset = MoTTAStream(target_ds.samples, noise_ds.samples, noise_ratio=0.2, transform=transform)
    loader = DataLoader(stream_dataset, batch_size=64, shuffle=False)

    # 5. Evaluation Loop
    model.eval()
    correct = 0
    total = 0

    print(">>> Đang chạy MoTTA trên ImageNet-C + NINCO...")
    for i, (images, labels, is_noise) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            output_dict = model(images)
            logits = output_dict['logits']

        # Chỉ tính trên mẫu sạch (ImageNet-C)
        clean_idx = (is_noise == 0)
        if clean_idx.any():
            preds = logits[clean_idx].argmax(dim=1)
            # Vì cả model và labels đều là 1000 lớp, so sánh trực tiếp:
            correct += (preds == labels[clean_idx]).sum().item()
            total += clean_idx.sum().item()

        if i % 5 == 0:
            acc = (correct / total) * 100 if total > 0 else 0
            print(f"Batch {i} | Accuracy trên ImageNet-C: {acc:.2f}% | Error Rate: {100-acc:.2f}%")

    print(f"\n[KẾT QUẢ CUỐI CÙNG] Error Rate: {100 - (correct/total)*100:.2f}%")

def reproduce_motta_goc():
    print(f"REPRODUCING MOTTA GOC")
    device = torch.device("cuda")
    
    PATH_C = "./Data/imagenet-c/blur/zoom_blur/5" # Đường dẫn tập ImageNet-C
    PATH_NINCO = "./Data/NINCO"
    
    # 2. Khởi tạo Model (1000 lớp chuẩn)
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)

    backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
    backbone.to(device) # Đưa backbone lên GPU trước

    # 3. Load Config
    cfg = cdict(new_allowed=True)
    cfg.merge_from_file('config.yml')
    
    # Khởi tạo MoTTA
    model = MoTTA(model=backbone, **cfg.paras_adapt_model)
    model.to(device)

    # 4. Tạo Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    print(f"Đang load Target từ: {PATH_C}")
    target_ds = ImageNet1KFolder(root=PATH_C, transform=transform)
    print(f"Số lượng lớp Target tìm thấy: {len(target_ds.classes)}")
    # In kiểm tra thử 1 lớp:
    folder_name = target_ds.classes[0]
    label_id = target_ds.class_to_idx[folder_name]
    print(f"Ví dụ: Thư mục {folder_name} được gán nhãn chuẩn là: {label_id}")

    # 2. Load tập Noise (NINCO)
    print(f"Đang load Noise từ: {PATH_NINCO}")
    noise_ds = ImageNet1KFolder(root=PATH_NINCO, transform=transform)

    
    # Trộn luồng dữ liệu (20% NINCO)
    stream_dataset = MoTTAStream(target_ds.samples, noise_ds.samples, noise_ratio=0.2, transform=transform)
    loader = DataLoader(stream_dataset, batch_size=64, shuffle=False)

    # 5. Evaluation Loop
    model.eval()
    correct = 0
    total = 0

    print(">>> Đang chạy MoTTA trên ImageNet-C + NINCO...")
    for i, (images, labels, is_noise) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            output_dict = model(images)
            logits = output_dict['logits']

        # Chỉ tính trên mẫu sạch (ImageNet-C)
        clean_idx = (is_noise == 0)
        if clean_idx.any():
            preds = logits[clean_idx].argmax(dim=1)
            # Vì cả model và labels đều là 1000 lớp, so sánh trực tiếp:
            correct += (preds == labels[clean_idx]).sum().item()
            total += clean_idx.sum().item()

        if i % 5 == 0:
            acc = (correct / total) * 100 if total > 0 else 0
            print(f"Batch {i} | Accuracy trên ImageNet-C: {acc:.2f}% | Error Rate: {100-acc:.2f}%")

    print(f"\n[KẾT QUẢ CUỐI CÙNG] Error Rate: {100 - (correct/total)*100:.2f}%")

if __name__ == '__main__':
    reproduce_c()
    # reproduce_motta_goc()