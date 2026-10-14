import os
from models import AttenUNet
from utils import load_pretrained
from .base import BaseTrainer

class AttenUNetTrainer(BaseTrainer):
    def __init__(self, cfg, projector, loaders, use_amp=True, late_fusion=False, out_dir="./results/"):
        model_name = "AttentionUnet"
        model_path = os.path.join(cfg.WEIGHTS_DIR, cfg.WEIGHT_DICT.get("attenunet.pth", f"{model_name.lower()}.pth"))

        model = AttenUNet(cfg.INPUT_CHANNELS, cfg.OUTPUT_CHANNELS, late_fusion=late_fusion).to(cfg.DEVICE)
        model = load_pretrained(model, model_path, cfg.DEVICE)
        super().__init__(cfg, projector, model, model_name, model_path, loaders, False, use_amp, out_dir)