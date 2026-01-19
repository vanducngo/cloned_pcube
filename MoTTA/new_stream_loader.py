import torch
from PIL import Image
import random
from torch.utils.data import Dataset

class MoTTAStream(Dataset):
    def __init__(self, target_samples, noise_samples, noise_ratio=0.2, transform=None):
        """
        target_samples: List [(path, label), ...] từ ImageNet-R
        noise_samples: List [(path, label), ...] từ SH hoặc NINCO
        noise_ratio: Tỷ lệ nhiễu (mặc định 0.2 = 20% nhiễu)
        """
        self.target_samples = target_samples
        self.noise_samples = noise_samples
        self.noise_ratio = noise_ratio
        self.transform = transform
        
        # Tính toán số lượng mẫu nhiễu cần thiết để đạt tỷ lệ noise_ratio
        num_target = len(target_samples)
        num_noise = int(num_target * (noise_ratio / (1 - noise_ratio)))
        
        # Lấy ngẫu nhiên ảnh nhiễu từ kho dữ liệu nhiễu (SH hoặc NINCO)
        random_noise = random.choices(noise_samples, k=num_noise)
        
        # Gộp dữ liệu: (thông tin mẫu, cờ hiệu is_noise)
        # is_noise = 0 là ảnh sạch, 1 là ảnh nhiễu
        self.combined_list = [(s, 0) for s in target_samples] + [(s, 1) for s in random_noise]
        
        # Trộn ngẫu nhiên thứ tự xuất hiện trong luồng (Stream)
        random.shuffle(self.combined_list)
        print(f"Đã tạo luồng dữ liệu: {num_target} ảnh sạch + {num_noise} ảnh nhiễu.")

    def __len__(self):
        return len(self.combined_list)

    def __getitem__(self, idx):
        # Lấy mẫu từ danh sách đã trộn
        (path, label), is_noise = self.combined_list[idx]
        
        # Mở và tiền xử lý ảnh
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Lỗi khi đọc ảnh {path}: {e}")
            # Trả về ảnh đen nếu lỗi (để không dừng chương trình)
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
            
        return img, label, is_noise