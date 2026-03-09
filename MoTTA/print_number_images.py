import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Thay đổi đường dẫn này trỏ tới thư mục chứa dataset của bạn
DATASET_ROOT = "./Data/imagenet-c"  
# DATASET_ROOT = "./Data/cifar10-c"
SEVERITY = "5" # Thư mục con chứa mức độ nhiễu (thường là 1 đến 5)

# Danh sách 15 loại nhiễu chuẩn
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
]

def count_images_in_dir(directory):
    """
    Đếm đệ quy tổng số file ảnh trong một thư mục (bao gồm cả các thư mục con/class).
    Bỏ qua các file ẩn (như .DS_Store).
    """
    count = 0
    # Duyệt qua tất cả các thư mục và file bên trong
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Kiểm tra đuôi file (có thể thêm/bớt tùy định dạng ảnh của bạn)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.JPEG')) and not file.startswith('.'):
                count += 1
    return count

def check_dataset_integrity():
    print(f"=== KIỂM TRA DATASET: {DATASET_ROOT} (Severity: {SEVERITY}) ===\n")
    
    total_all_corruptions = 0
    missing_corruptions = []

    print(f"{'CORRUPTION TYPE':<25} | {'SỐ LƯỢNG ẢNH':<15} | {'TRẠNG THÁI'}")
    print("-" * 65)

    for corruption in CORRUPTIONS:
        # Tạo đường dẫn đầy đủ: ví dụ ./Data/imagenet-c/gaussian_noise/5
        target_path = os.path.join(DATASET_ROOT, corruption, SEVERITY)
        
        if os.path.exists(target_path):
            img_count = count_images_in_dir(target_path)
            total_all_corruptions += img_count
            
            # Kiểm tra logic: ImageNet-C thường có 50.000 ảnh mỗi corruption, CIFAR có 10.000
            status = "OK"
            if img_count == 0:
                status = "TRỐNG (0 ảnh)!"
            elif img_count not in [10000, 50000]:
                status = f"BẤT THƯỜNG ({img_count})"
                
            print(f"{corruption:<25} | {img_count:<15,} | {status}")
        else:
            print(f"{corruption:<25} | {'---':<15} | KHÔNG TÌM THẤY THƯ MỤC")
            missing_corruptions.append(corruption)

    print("-" * 65)
    print(f"TỔNG CỘNG TẤT CẢ:         {total_all_corruptions:,} ảnh")
    
    if missing_corruptions:
        print("\n[CẢNH BÁO] Thiếu các thư mục sau:")
        for c in missing_corruptions:
            print(f" - {c}")

if __name__ == "__main__":
    check_dataset_integrity()