"""
SoreSAM - SAM2-based Wound and Skin Segmentation
Dataset class for 3-class wound segmentation.

Expected directory layout:
    data/data_wound_seg_3class/
    ├── train_images/   (*.jpg / *.png)
    ├── train_masks/    (*.png  - RGB colour-coded)
    ├── test_images/
    ├── test_masks/
    └── class_info.txt

Mask colour convention (RGB):
    (  0,   0,   0) → 0  Other  (background / unlabelled)
    (  0,   0, 255) → 1  Skin   (blue)
    (255,   0,   0) → 2  Wound  (red)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Colour → class-index mapping
# ---------------------------------------------------------------------------
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0,   0,   0),    # Other
    1: (0,   0, 255),    # Skin   (blue)
    2: (255, 0,   0),    # Wound  (red)
}


def mask_rgb_to_label(mask_rgb: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Convert an RGB colour-coded mask image to a 2-D integer label map.

    Args:
        mask_rgb:  (H, W, 3) uint8 image loaded in RGB order.
        threshold: Per-channel tolerance for colour matching
                   (handles JPEG compression artifacts).

    Returns:
        label: (H, W) uint8 array with values in {0, 1, 2}.
    """
    H, W = mask_rgb.shape[:2]
    label = np.zeros((H, W), dtype=np.uint8)  # default: Other (0)

    for class_idx, color in CLASS_COLORS.items():
        if class_idx == 0:
            continue  # Other is the default fallback
        color_arr = np.array(color, dtype=np.int32)
        dist = np.abs(mask_rgb.astype(np.int32) - color_arr).max(axis=2)
        label[dist <= threshold] = class_idx

    return label


def label_to_mask_rgb(label: np.ndarray) -> np.ndarray:
    """Convert an integer label map back to an RGB visualisation."""
    H, W = label.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        rgb[label == class_idx] = color
    return rgb


# ---------------------------------------------------------------------------
# Albumentations transforms
# ---------------------------------------------------------------------------
def get_train_transforms(image_size: int = 1024) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15),
            A.CLAHE(clip_limit=2.0),
        ], p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 1024) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WoundDataset(Dataset):
    """
    PyTorch Dataset for 3-class wound segmentation.

    Args:
        data_root:       Path to ``data_wound_seg_3class/``.
        split:           ``"train"`` | ``"val"`` | ``"test"``.
        transform:       Albumentations Compose pipeline.
        val_indices:     Indices from the train set reserved for validation.
        color_threshold: Colour-matching tolerance (pixels).
        image_size:      Target resolution (default: 1024 for SAM2).
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        val_indices: Optional[List[int]] = None,
        color_threshold: int = 30,
        image_size: int = 1024,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.data_root = Path(data_root)
        self.split = split
        self.color_threshold = color_threshold
        self.image_size = image_size

        # Resolve directories
        if split in ("train", "val"):
            img_dir = self.data_root / "train_images"
            msk_dir = self.data_root / "train_masks"
        else:
            img_dir = self.data_root / "test_images"
            msk_dir = self.data_root / "test_masks"

        # Collect paired image / mask paths
        all_pairs = self._collect_pairs(img_dir, msk_dir)

        if split == "train":
            if val_indices is not None:
                val_set = set(val_indices)
                all_pairs = [p for i, p in enumerate(all_pairs) if i not in val_set]
        elif split == "val":
            if val_indices is not None:
                all_pairs = [all_pairs[i] for i in val_indices]
            # If no val_indices supplied, val == all train pairs (unusual but safe)

        self.pairs = all_pairs

        # Default transforms if none provided
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

    # ------------------------------------------------------------------
    def _collect_pairs(
        self, img_dir: Path, msk_dir: Path
    ) -> List[Tuple[Path, Path]]:
        pairs = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in self.VALID_EXTENSIONS:
                continue
            # Try to find a matching mask (same stem, any valid extension)
            mask_path = self._find_mask(msk_dir, img_path.stem)
            if mask_path is None:
                print(f"[WoundDataset] Warning: no mask found for {img_path.name}, skipping.")
                continue
            pairs.append((img_path, mask_path))
        return pairs

    def _find_mask(self, msk_dir: Path, stem: str) -> Optional[Path]:
        for ext in self.VALID_EXTENSIONS:
            candidate = msk_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, msk_path = self.pairs[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (BGR → RGB)
        mask_rgb = cv2.imread(str(msk_path), cv2.IMREAD_COLOR)
        if mask_rgb is None:
            raise FileNotFoundError(f"Cannot read mask: {msk_path}")
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        # Convert colour mask → integer label map
        label = mask_rgb_to_label(mask_rgb, self.color_threshold)

        # Apply transforms
        transformed = self.transform(image=image, mask=label)
        image_tensor = transformed["image"]          # (3, H, W) float32
        label_tensor = transformed["mask"].long()    # (H, W) int64

        return {
            "image": image_tensor,
            "label": label_tensor,
            "image_path": str(img_path),
        }

    # ------------------------------------------------------------------
    def class_pixel_counts(self) -> np.ndarray:
        """Count pixels per class across the entire split (for class weighting)."""
        counts = np.zeros(len(CLASS_COLORS), dtype=np.int64)
        for _, msk_path in self.pairs:
            mask_rgb = cv2.imread(str(msk_path), cv2.IMREAD_COLOR)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
            label = mask_rgb_to_label(mask_rgb, self.color_threshold)
            for c in range(len(CLASS_COLORS)):
                counts[c] += (label == c).sum()
        return counts


# ---------------------------------------------------------------------------
# Helper: build train/val split indices
# ---------------------------------------------------------------------------
def make_train_val_split(
    data_root: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Return (train_indices, val_indices) from the train_images directory.
    """
    img_dir = Path(data_root) / "train_images"
    paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in WoundDataset.VALID_EXTENSIONS
    )
    n = len(paths)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = sorted(indices[:n_val].tolist())
    train_idx = sorted(indices[n_val:].tolist())
    return train_idx, val_idx
