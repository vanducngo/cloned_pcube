import os
import torch
from torchvision import models as pt_models, transforms
from torch.utils.data import DataLoader
from motta import MoTTA, normalize_model
from MoTTA.new_stream_loader import MoTTAStream
from yacs.config import CfgNode as cdict
from torchvision.datasets import ImageFolder

class RobustImageFolder(ImageFolder):
    """
    Tự động loại bỏ .ipynb_checkpoints và các file ẩn không phải class của ImageNet
    """
    def find_classes(self, directory):
        # Chỉ lấy các thư mục bắt đầu bằng 'n' (đúng định dạng WNID của ImageNet)
        # và thực sự là một thư mục
        classes = [d.name for d in os.scandir(directory) 
                   if d.is_dir() and d.name.startswith('n')]
        
        if not classes:
            # Nếu tập dữ liệu không bắt đầu bằng 'n' (như NINCO), 
            # thì ta lấy tất cả trừ các thư mục ẩn
            classes = [d.name for d in os.scandir(directory) 
                       if d.is_dir() and not d.name.startswith('.')]
            
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def reproduce_c():
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

    print(f"Đang load dữ liệu từ: {PATH_C}")
    target_ds = RobustImageFolder(root=PATH_C, transform=transform)
    print(f"Số lượng lớp tìm thấy trong Target: {len(target_ds.classes)}")

    noise_ds = RobustImageFolder(root=PATH_NINCO, transform=transform)
    print(f"Số lượng lớp tìm thấy trong Noise: {len(noise_ds.classes)}")
    
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