#!/usr/bin/env python3
"""Training configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os


@dataclass
class Config:
    """Training configuration."""

    # ---- data ----
    data_root: str = "./data/zenodo"
    dataset_type: str = "zenodo"  # zenodo, covid19_20, medseg
    spatial_size: Tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 4
    num_workers: int = 4
    val_ratio: float = 0.2

    # ---- model ----
    # use_sf_gate=True -> SF-Swin-UNETR; False -> vanilla Swin-UNETR baseline.
    in_channels: int = 1
    out_channels: int = 2  # background + infection
    feature_size: int = 48
    use_checkpoint: bool = True  # gradient checkpointing to save memory
    use_sf_gate: bool = True
    sf_gate_reduction: int = 16
    sf_gate_lr_multiplier: float = 10.0  # SF-Gate uses a higher learning rate

    # pretrained backbone
    pretrained_path: Optional[str] = None
    use_ssl_pretrained: bool = True
    ssl_pretrained_url: str = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"

    # ---- training ----
    max_epochs: int = 500
    base_lr: float = 2e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 20
    optimizer: str = "adamw"  # adamw, sgd
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    loss_type: str = "dice_ce"  # dice, ce, dice_ce, focal
    dice_weight: float = 0.5
    ce_weight: float = 0.5

    # ---- validation & saving ----
    val_interval: int = 10
    save_interval: int = 50
    early_stopping: bool = True
    early_stopping_patience: int = 100

    # ---- output & logging ----
    output_dir: str = "./outputs"
    exp_name: str = "covid_sf_swin"
    use_tensorboard: bool = True
    log_interval: int = 10

    # ---- misc ----
    seed: int = 42
    deterministic: bool = False
    amp: bool = True  # mixed precision

    def __post_init__(self):
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "predictions"), exist_ok=True)

    @classmethod
    def from_args(cls, args):
        """Build a Config from parsed CLI arguments."""
        config = cls()
        for key, value in vars(args).items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        config.__post_init__()
        return config


