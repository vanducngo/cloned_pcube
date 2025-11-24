import torch

def get_features_from_model(model, batch_samples, classifier_layer_name='classifier'):
    """
    Trích xuất các vector đặc trưng từ một mô hình bằng cách sử dụng hook.
    Hàm này lấy đầu vào của lớp classifier cuối cùng.

    Args:
        model (torch.nn.Module): Mô hình (ví dụ: teacher_model) để trích xuất đặc trưng.
        batch_samples (Tensor): Batch dữ liệu đầu vào.
        classifier_layer_name (str): Tên của module classifier trong mô hình. 
                                     Mặc định là 'classifier' (cho ResNeXt) hoặc 'fc' (cho ResNet/WideResNet).

    Returns:
        Tensor: Một tensor chứa các vector đặc trưng đã được flatten.
    """
    # Biến để lưu trữ đặc trưng (sử dụng list để có thể thay đổi bên trong hook)
    features_storage = []

    def hook_fn(module, input, output):
        # Đầu vào của một lớp Linear (classifier) là một tuple, 
        # phần tử đầu tiên chính là tensor đặc trưng.
        if isinstance(input, tuple) and len(input) > 0:
            features_storage.append(input[0])
        else:
            features_storage.append(input)

    # --- Tự động tìm lớp classifier ---
    try:
        # Thử tìm lớp classifier theo tên được cung cấp
        classifier_layer = dict(model.named_modules())[classifier_layer_name]
    except KeyError:
        # Nếu không tìm thấy, thử các tên phổ biến khác
        if hasattr(model, 'fc'):
            classifier_layer = model.fc
        elif hasattr(model, 'classifier'):
            classifier_layer = model.classifier
        else:
            # Nếu vẫn không tìm thấy, báo lỗi
            raise AttributeError(f"Model does not have a layer named '{classifier_layer_name}', 'fc', or 'classifier'.")
            
    # --- Đăng ký, Chạy, và Gỡ bỏ Hook ---
    # Đăng ký hook vào lớp classifier để "bắt" đầu vào của nó
    hook_handle = classifier_layer.register_forward_hook(hook_fn)

    # Thực hiện forward pass (không cần gradient)
    with torch.no_grad():
        model.eval()
        _ = model(batch_samples)

    # Gỡ bỏ hook ngay lập tức sau khi đã có đặc trưng. Rất quan trọng!
    hook_handle.remove()

    # --- Xử lý và Trả về Kết quả ---
    if not features_storage:
        raise RuntimeError("Hook did not capture any features. Check the classifier layer name and model architecture.")

    # Lấy đặc trưng và flatten nó nếu cần
    features = features_storage[0]
    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    return features.detach()