from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


def build_model(dataset):
    arch = "Standard" if dataset == "cifar10" else "Hendrycks2020AugMix_ResNeXt"
    ckpt_dir = "./ckpt"
    if dataset in ["cifar10", "cifar100"]:
        base_model = load_model(arch, ckpt_dir,
                                dataset, ThreatModel.corruptions).cuda()
        
        # base_model = load_model(arch, cfg.CKPT_DIR,
        #                         cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    else:
        raise NotImplementedError()

    return base_model
