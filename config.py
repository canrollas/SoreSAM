"""
SoreSAM - SAM2-based Wound and Skin Segmentation
Configuration
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    root: str = "data/data_wound_seg_3class"
    train_images: str = "train_images"
    train_masks: str = "train_masks"
    test_images: str = "test_images"
    test_masks: str = "test_masks"
    class_info: str = "class_info.txt"

    # Mask color → class index mapping (RGB tuples)
    # "Other" (background) → 0 : unmarked pixels (black)
    # "Skin"              → 1 : blue  (0, 0, 255)
    # "Wound"             → 2 : red   (255, 0, 0)
    num_classes: int = 3
    class_names: List[str] = field(default_factory=lambda: ["Other", "Skin", "Wound"])
    class_colors: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (0, 0, 0),      # 0 - Other  (background / unmarked)
            (0, 0, 255),    # 1 - Skin   (blue)
            (255, 0, 0),    # 2 - Wound  (red)
        ]
    )
    # Color tolerance for mask parsing (handles JPEG compression artifacts)
    color_threshold: int = 30

    # Image resolution fed to SAM2
    image_size: int = 1024

    # ImageNet normalisation (SAM2 default)
    pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Train / validation split (fraction of train set used for validation)
    val_split: float = 0.1
    seed: int = 42


@dataclass
class ModelConfig:
    # SAM2 largest model (sam2.1_hiera_large)
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"

    # Number of learnable prompt tokens per class
    num_class_tokens: int = 4

    # Freeze SAM2 image encoder during training
    freeze_image_encoder: bool = True

    # Also freeze SAM2 prompt encoder (only class_tokens are learned)
    freeze_prompt_encoder: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    vis_dir: str = "outputs/visualizations"

    num_epochs: int = 50
    batch_size: int = 4
    num_workers: int = 4

    # Optimiser
    lr: float = 1e-4
    weight_decay: float = 1e-4
    # Separate (lower) LR for mask decoder vs class tokens
    decoder_lr_multiplier: float = 0.1

    # LR scheduler
    lr_scheduler: str = "cosine"   # "cosine" | "step" | "none"
    warmup_epochs: int = 2

    # Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0

    # Class weights for CE loss (up-weight wound & skin)
    class_weights: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])

    # Logging / saving
    log_interval: int = 10      # batches
    val_interval: int = 1       # epochs
    save_best_metric: str = "mean_iou"
    keep_last_n_checkpoints: int = 3

    # Mixed precision
    use_amp: bool = True

    # Gradient clipping
    grad_clip: float = 1.0

    device: str = "cuda"        # "cuda" | "mps" | "cpu"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train.log_dir, exist_ok=True)
        os.makedirs(self.train.vis_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Convenience singleton for import
# ---------------------------------------------------------------------------
cfg = Config()
