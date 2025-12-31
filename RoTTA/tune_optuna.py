from copy import deepcopy
import logging
import torch
import optuna
import argparse

from core.configs import cfg
from core.utils import *
from core.model import build_model
from core.data import build_loader
from core.optim import build_optimizer
from core.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle

# Tắt log hệ thống, chỉ giữ lại log quan trọng
logging.getLogger("TTA").setLevel(logging.ERROR)

def testTimeAdaptation(cfg):
    logger = logging.getLogger("TTA.test_time")
    # model, optimizer
    model = build_model(cfg)

    optimizer = build_optimizer(cfg)

    tta_adapter = build_adapter(cfg)

    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.cuda()

    loader, processor = build_loader(cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY)

    tbar = tqdm(loader)
    for batch_id, data_package in enumerate(tbar):
        data, label, domain = data_package["image"], data_package['label'], data_package['domain']
        if len(label) == 1:
            continue  # ignore the final single point
        data, label = data.cuda(), label.cuda()
        output = tta_model(data)
        predict = torch.argmax(output, dim=1)
        accurate = (predict == label)
        processor.process(accurate, domain)
        if batch_id % 10 == 0:
            if hasattr(tta_model, "mem"):
                tbar.set_postfix(acc=processor.cumulative_acc(), bank=tta_model.mem.get_occupancy())
            else:
                tbar.set_postfix(acc=processor.cumulative_acc())

    processor.calculate()

    # logger.info(f"All Results\n{processor.info()}")

    acc = processor.cumulative_acc()
    # Chuẩn hóa về 0-100
    if acc <= 1.0: acc *= 100.0
    error_rate = 100.0 - acc
    
    # Dọn dẹp
    del tta_model, model, optimizer, loader
    torch.cuda.empty_cache()
    
    return error_rate

def objective(trial):
    # 1. Gợi ý tham số
    entropy_factor = trial.suggest_float("entropy_factor", 0.05, 1.0, step=0.05)
    
    # 2. Clone & Update Config
    trial_cfg = deepcopy(cfg)
    trial_cfg.defrost()
    
    # Cấu hình tham số cần tune
    if not hasattr(trial_cfg, 'P_CUBE'):
        from yacs.config import CfgNode as CN
        trial_cfg.P_CUBE = CN()
        trial_cfg.P_CUBE.FILTER = CN()
        
    trial_cfg.P_CUBE.FILTER.ENTROPY_FACTOR = entropy_factor
    
    # Đảm bảo chạy full pipeline
    # Nếu file config của bạn để CORRUPTION.TYPE là 1 loại cụ thể, hãy sửa thành list tất cả hoặc logic tương ứng
    # Ở đây mình giả định config mặc định đã cấu hình để chạy full sequence
    
    trial_cfg.freeze()
    
    # 3. Chạy Full Pipeline
    try:
        print(f"\n[Trial {trial.number}] Running with entropy_factor={entropy_factor:.3f} ...")
        error_rate = testTimeAdaptation(trial_cfg)
        print(f"[Trial {trial.number}] Result: Error Rate = {error_rate:.2f}%")
        return error_rate
    except Exception as e:
        print(f"[Trial {trial.number}] Failed: {e}")
        return 100.0 # Penalty

def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        '-ocfg',
        '--order-config-file',
        metavar="FILE",
        default="",
        help="path to order config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)

    args = parser.parse_args()

    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    if not args.order_config_file == "":
        cfg.merge_from_file(args.order_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    ds = cfg.CORRUPTION.DATASET
    adapter = cfg.ADAPTER.NAME
    setproctitle(f"TTA:{ds:>8s}:{adapter:<10s}")

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('TTA', cfg.OUTPUT_DIR, 0, filename=cfg.LOG_DEST)
    logger.info(args)

    logger.info(f"Loaded configuration file: \n"
                f"\tadapter: {args.adapter_config_file}\n"
                f"\tdataset: {args.dataset_config_file}\n"
                f"\torder: {args.order_config_file}")
    logger.info("Running with config:\n{}".format(cfg))

    set_random_seed(cfg.SEED)

    # testTimeAdaptation(cfg)

    # Setup DB
    db_url = "sqlite:///optuna_full_pipeline.db"
    study_name = f"full_tuning_{cfg.CORRUPTION.DATASET}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=db_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.SEED)
    )
    
    print(f"Start Tuning on {cfg.CORRUPTION.DATASET}. Results saved to {db_url}")
    
    # Chạy tối ưu (Ví dụ 20 trials)
    study.optimize(objective, n_trials=20)
    
    # In kết quả tốt nhất
    print("\n" + "="*50)
    print(f"BEST RESULT: Error Rate = {study.best_value:.2f}%")
    print("Best Params:", study.best_params)
    print("="*50)
    
    # Lưu CSV
    df = study.trials_dataframe()
    df.to_csv(f"tuning_results_full_{cfg.CORRUPTION.DATASET}.csv")


if __name__ == "__main__":
    main()
