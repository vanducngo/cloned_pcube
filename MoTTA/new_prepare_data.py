import os
from imagenet_subsets import ALL_WNIDS, IMAGENET_R_WNIDS, create_file_list

ROOT_DIR = "/Users/admin/Working/Data-MoTTA"
IMAGENET_1K_VAL = f"{ROOT_DIR}/ILSVRC2012_img_val" 

# --- A. TẠO DANH SÁCH NHIỄU SH ---

# Logic: ImageNet-1K có 1000 lớp (ALL_WNIDS)
# ImageNet-R chỉ dùng 200 lớp (IMAGENET_R_WNIDS)
# Vậy 800 lớp còn lại chính là nhiễu SH (Semantic Shift)
SH_NOISE_WNIDS = [wnid for wnid in ALL_WNIDS if wnid not in IMAGENET_R_WNIDS]

# Hàm create_file_list (trong data_utils.py) sẽ quét vào 800 thư mục này 
# và lấy ra đường dẫn của từng tấm ảnh.

sh_noise_samples = create_file_list(IMAGENET_1K_VAL, SH_NOISE_WNIDS, split="")

print(f"Đã tìm thấy {len(sh_noise_samples)} ảnh nhiễu SH từ 800 lớp dư thừa.")


# --- B. TẠO DANH SÁCH NHIỄU NINCO ---

# NINCO không nằm trong ImageNet nên không dùng data_utils.py để lọc.
# Bạn dùng thư viện chuẩn của PyTorch để quét thư mục NINCO bạn đã tải.
from torchvision.datasets import ImageFolder

# Đường dẫn đến thư mục NINCO đã giải nén
NINCO_PATH = f"{ROOT_DIR}/NINCO"
ninco_ds = ImageFolder(root=NINCO_PATH)

# ninco_samples sẽ là một list các tuple: [(đường_dẫn_ảnh, nhãn), ...]
ninco_samples = ninco_ds.samples

print(f"Đã tìm thấy {len(ninco_samples)} ảnh nhiễu NINCO.")