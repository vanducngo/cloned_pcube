import os
import torch
from torchvision import models as pt_models, transforms
from torch.utils.data import DataLoader
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

def reproduce_source_only():
    device = torch.device("cuda")
    
    PATH_C = "./Data/imagenet-c/blur/zoom_blur/5" 
    PATH_NINCO = "./Data/NINCO"
    
    # 1. Khởi tạo mô hình gốc (Source Model)
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    
    # Chúng ta chỉ dùng backbone, không khởi tạo lớp MoTTA
    backbone = normalize_model(pt_models.resnet50(pretrained=True), mu, sigma)
    backbone.to(device)
    backbone.eval() # Quan trọng: Giữ mô hình ở chế độ eval, không cập nhật BN

    # 2. Tạo Dataset & Loader (Giữ nguyên để đảm bảo so sánh công bằng)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    target_ds = ImageNet1KFolder(root=PATH_C, transform=transform)
    noise_ds = ImageNet1KFolder(root=PATH_NINCO, transform=transform)
    
    stream_dataset = MoTTAStream(target_ds.samples, noise_ds.samples, noise_ratio=0.2, transform=transform)
    loader = DataLoader(stream_dataset, batch_size=64, shuffle=False)

    # 3. Vòng lặp dự đoán (Inference Only)
    correct = 0
    total = 0

    print(">>> Đang chạy SOURCE ONLY (Không có MoTTA)...")
    
    # Không dùng torch.enable_grad() vì không cần update model
    with torch.no_grad():
        for i, (images, labels, is_noise) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            # DỰ ĐOÁN TRỰC TIẾP TỪ BACKBONE
            logits = backbone(images)

            # Chỉ tính trên mẫu sạch (ImageNet-C)
            clean_idx = (is_noise == 0)
            if clean_idx.any():
                preds = logits[clean_idx].argmax(dim=1)
                correct += (preds == labels[clean_idx]).sum().item()
                total += clean_idx.sum().item()

            if i % 20 == 0:
                acc = (correct / total) * 100 if total > 0 else 0
                print(f"Batch {i} | Source Acc: {acc:.2f}% | Source Error: {100-acc:.2f}%")

    final_err = 100 - (correct/total)*100
    print(f"\n[KẾT QUẢ SOURCE ONLY] Final Error Rate: {final_err:.2f}%")
    return final_err

if __name__ == '__main__':
    reproduce_source_only()