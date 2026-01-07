import optuna
import logging
import torch
import argparse
import sys
from tqdm import tqdm
from copy import deepcopy

# Import các module từ codebase hiện tại
# Đảm bảo bạn đặt file này ở root folder của project (cùng cấp với core/)
from core.configs import cfg
from core.utils import setup_logger, set_random_seed, mkdir
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter

# Cấu hình log để giảm bớt output rác, chỉ hiện thông tin quan trọng
logging.getLogger("TTA").setLevel(logging.WARNING)

def run_tta_full_experiment(trial_cfg, trial):
    """
    Chạy Full TTA Experiment trên 15 Corruptions (Giống PTTA.py).
    Trả về Average Error Rate.
    """
    dataset_name = trial_cfg.CORRUPTION.DATASET
    # Lấy danh sách 15 corruption chuẩn từ config hoặc hardcode
    # Thông thường cfg.CORRUPTION.TYPE là một list hoặc 'all'
    # Ở đây mình giả định chạy loop qua tất cả các loại
    all_corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]
    
    total_error = 0.0
    processed_count = 0
    
    # Progress bar cho các corruptions
    pbar_corr = tqdm(all_corruptions, desc=f"Trial {trial.number}")
    
    for i, corruption in enumerate(pbar_corr):
        # 1. Build Model & Adapter mới cho mỗi corruption (Reset trạng thái)
        # Đây là quy trình chuẩn của TTA benchmark
        model = build_model(trial_cfg)
        model = model.cuda()
        optimizer = build_optimizer(trial_cfg)
        tta_adapter = build_adapter(trial_cfg)
        tta_model = tta_adapter(trial_cfg, model, optimizer)
        
        # 2. Build Loader
        # Severity mặc định là 5 (chuẩn benchmark)
        loader, processor = build_loader(trial_cfg, dataset_name, corruption, 5)
        
        # 3. Chạy TTA Loop (Full Data)
        for batch_id, data_package in enumerate(loader):
            data, label, domain = data_package["image"], data_package['label'], data_package['domain']
            if len(label) == 1: continue
            
            data, label = data.cuda(), label.cuda()
            
            output = tta_model(data)
            predict = torch.argmax(output, dim=1)
            accurate = (predict == label)
            
            processor.process(accurate, domain)
            
        processor.calculate()
        
        # Lấy Accuracy -> Error Rate
        acc = processor.cumulative_acc()
        # Chuẩn hóa về 0-100 nếu processor trả về 0-1
        if acc <= 1.0: acc *= 100.0 
        error_rate = 100.0 - acc
        
        total_error += error_rate
        processed_count += 1
        
        # Log kết quả từng corruption để theo dõi
        pbar_corr.set_postfix({'curr_err': f"{error_rate:.2f}", 'avg_err': f"{total_error/processed_count:.2f}"})
        
        # Dọn dẹp GPU
        del tta_model, model, optimizer, loader
        torch.cuda.empty_cache()
        
        # --- Optuna Pruning (Cắt tỉa sớm) ---
        # Báo cáo kết quả trung bình tạm thời sau mỗi corruption
        # Nếu kết quả quá tệ so với các trial trước đó -> Dừng luôn trial này
        intermediate_value = total_error / processed_count
        trial.report(intermediate_value, i)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return total_error / len(all_corruptions)

def objective(trial):
    # 1. Gợi ý tham số
    # Tìm kiếm entropy_factor từ 0.05 đến 1.0 (step 0.05)
    # entropy_factor = trial.suggest_float("entropy_factor", 0.05, 1.0, step=0.05)
    lambda_std = trial.suggest_float("lambda_std", 0.5, 2.0, step=0.25)
    hard_floor = trial.suggest_float("hard_floor", 0.4, 0.7, step=0.05)

    
    # 2. Clone Config gốc
    trial_cfg = deepcopy(cfg)
    trial_cfg.defrost()
    
    # Cập nhật tham số cần tune
    # Đảm bảo đúng đường dẫn biến trong config của bạn
    # Ví dụ: cfg.P_CUBE.FILTER.ENTROPY_FACTOR
    if not hasattr(trial_cfg, 'P_CUBE'):
        from yacs.config import CfgNode as CN
        trial_cfg.P_CUBE = CN()
        trial_cfg.P_CUBE.FILTER = CN()
        
    # trial_cfg.P_CUBE.FILTER.ENTROPY_FACTOR = entropy_factor # TODO: Kiểm tra lại lần chạy với entropy factor
    trial_cfg.DATA_FITER.CONSISTENT_LAMBDA_STD = lambda_std
    trial_cfg.DATA_FITER.CONSISTENT_HARD_FLOOR = hard_floor
    
    # Đảm bảo các tham số khác chuẩn
    # trial_cfg.CORRUPTION.DATASET = 'cifar10' # Hoặc lấy từ args
    
    trial_cfg.freeze()
    
    # 3. Chạy thực nghiệm
    try:
        avg_error = run_tta_full_experiment(trial_cfg, trial)
        return avg_error
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"[Trial {trial.number}] Failed: {e}")
        # Trả về giá trị rất lớn để Optuna biết đây là bộ tham số tồi
        return 100.0 

def main():
    # Setup Argument Parser giống PTTA.py để dễ dùng
    parser = argparse.ArgumentParser("Optuna Tuning for RoTTA")
    parser.add_argument('-acfg', '--adapter-config-file', type=str, default="", help="path to adapter config file")
    parser.add_argument('-dcfg', '--dataset-config-file', type=str, default="", help="path to dataset config file")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Load Configs
    if args.adapter_config_file:
        cfg.merge_from_file(args.adapter_config_file)
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Setup Random Seed
    set_random_seed(cfg.SEED)
    
    print(f"Start Tuning on Dataset: {cfg.CORRUPTION.DATASET}")
    print(f"Base Config Loaded. Default Entropy Factor: {getattr(cfg.P_CUBE.FILTER, 'ENTROPY_FACTOR', 'N/A')}")

    # Setup Optuna Study
    # Lưu vào SQLite để resume nếu bị ngắt giữa chừng
    db_url = "sqlite:///optuna_rotta.db"
    study_name = f"rotta_tuning_{cfg.CORRUPTION.DATASET}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1) # Pruning sau 1 corruption đầu tiên
    )
    
    # Chạy tối ưu (Ví dụ 20 trials)
    study.optimize(objective, n_trials=20)
    
    print("\n" + "="*50)
    print("TUNING COMPLETED")
    print(f"Best Error Rate: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    print("="*50)
    
    # Xuất kết quả ra CSV
    df = study.trials_dataframe()
    df.to_csv(f"tuning_results_{cfg.CORRUPTION.DATASET}.csv")

if __name__ == "__main__":
    main()