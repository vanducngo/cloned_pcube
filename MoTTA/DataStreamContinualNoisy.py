import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image

class DataStreamContinualNoisy(Dataset):
    def __init__(self, target_samples, noise_samples, noise_ratio=0.2, gamma=1.0, num_slots=100):
        self.noise_ratio = noise_ratio
        
        # 1. Tính toán số lượng nhiễu tương ứng
        num_target = len(target_samples)
        num_noise = int(num_target * (noise_ratio / (1 - noise_ratio)))
        selected_noise = random.choices(noise_samples, k=num_noise)
        
        # 2. Gán nhãn tạm thời (is_noise=0: sạch, 1: nhiễu)
        data_with_flags = [(s, 0) for s in target_samples] + [(s, 1) for s in selected_noise]
        
        # 3. Gom nhóm theo nhãn để apply Dirichlet
        # Lưu ý: Noise samples thường không có nhãn class thực, 
        # nên ta sẽ gán cho chúng một nhãn giả hoặc nhóm riêng
        self.num_classes = 1000 # ImageNet
        data_by_class = [[] for _ in range(self.num_classes)]
        noise_list = []
        
        for (path, label), is_noise in data_with_flags:
            if is_noise == 0:
                data_by_class[label].append((path, label, 0))
            else:
                noise_list.append((path, -1, 1)) # Label -1 cho nhiễu
        
        # 4. Phân bổ Dirichlet
        # Chia các mẫu sạch vào các 'slots' (cụm dữ liệu nhỏ)
        slot_indices = [[] for _ in range(num_slots)]
        label_distribution = np.random.dirichlet([gamma] * num_slots, self.num_classes)

        for c in range(self.num_classes):
            c_samples = data_by_class[c]
            # Chia samples theo phân phối Dirichlet
            splits = np.split(c_samples, (np.cumsum(label_distribution[c])[:-1] * len(c_samples)).astype(int))
            for s, samples in enumerate(splits):
                slot_indices[s].extend(samples)
        
        # 5. Trộn nhiễu vào các cụm
        self.final_list = []
        for s in slot_indices:
            random.shuffle(s)
            self.final_list.extend(s)
            # Chèn ngẫu nhiên nhiễu vào cụm
            num_noise_in_slot = len(noise_list) // num_slots
            self.final_list.extend(noise_list[:num_noise_in_slot])
            noise_list = noise_list[num_noise_in_slot:]
            
        random.shuffle(self.final_list)

    def __getitem__(self, idx):
        path, label, is_noise = self.final_list[idx]
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

# class DataStreamContinualNoisy(Dataset):
    # def __init__(self, target_samples, noise_samples, noise_ratio=0.2, transform=None):
    #     """
    #     target_samples: List [(path, label), ...] từ ImageNet-R
    #     noise_samples: List [(path, label), ...] từ SH hoặc NINCO
    #     noise_ratio: Tỷ lệ nhiễu (mặc định 0.2 = 20% nhiễu)
    #     """
    #     self.target_samples = target_samples
    #     self.noise_samples = noise_samples
    #     self.noise_ratio = noise_ratio
    #     self.transform = transform
        
    #     # Tính toán số lượng mẫu nhiễu cần thiết để đạt tỷ lệ noise_ratio
    #     num_target = len(target_samples)
    #     num_noise = int(num_target * (noise_ratio / (1 - noise_ratio)))
        
    #     # Lấy ngẫu nhiên ảnh nhiễu từ kho dữ liệu nhiễu (SH hoặc NINCO)
    #     random_noise = random.choices(noise_samples, k=num_noise)
        
    #     # Gộp dữ liệu: (thông tin mẫu, cờ hiệu is_noise)
    #     # is_noise = 0 là ảnh sạch, 1 là ảnh nhiễu
    #     self.combined_list = [(s, 0) for s in target_samples] + [(s, 1) for s in random_noise]
        
    #     # Trộn ngẫu nhiên thứ tự xuất hiện trong luồng (Stream)
    #     random.shuffle(self.combined_list)
    #     print(f"Đã tạo luồng dữ liệu: {num_target} ảnh sạch + {num_noise} ảnh nhiễu.")

    # def __len__(self):
    #     return len(self.combined_list)

    # def __getitem__(self, idx):
    #     # Lấy mẫu từ danh sách đã trộn
    #     (path, label), is_noise = self.combined_list[idx]
        
    #     # Mở và tiền xử lý ảnh
    #     try:
    #         img = Image.open(path).convert('RGB')
    #     except Exception as e:
    #         print(f"Lỗi khi đọc ảnh {path}: {e}")
    #         # Trả về ảnh đen nếu lỗi (để không dừng chương trình)
    #         img = Image.new('RGB', (224, 224))
            
    #     if self.transform:
    #         img = self.transform(img)
            
    #     return img, label, is_noise