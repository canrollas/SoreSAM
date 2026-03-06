"""
SoreSAM - Visualisation utilities.

Converts label maps → colour images and saves prediction grids.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Colour palette (RGB 0–1 for matplotlib)
# ---------------------------------------------------------------------------
CLASS_COLORS_RGB = {
    0: (0.15, 0.15, 0.15),   # Other  — dark grey
    1: (0.0,  0.3,  1.0),    # Skin   — blue
    2: (1.0,  0.15, 0.15),   # Wound  — red
}

CLASS_NAMES_DEFAULT = ["Other", "Skin", "Wound"]

# ImageNet denormalisation
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) float tensor → (H, W, 3) uint8 numpy."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * _STD + _MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def label_to_color(label: np.ndarray) -> np.ndarray:
    """(H, W) int → (H, W, 3) float RGB image for matplotlib."""
    H, W = label.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls_idx, color in CLASS_COLORS_RGB.items():
        mask = label == cls_idx
        rgb[mask] = color
    return rgb


def overlay_mask(image: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend RGB image (0-255 uint8) with label colour overlay."""
    color_mask = (label_to_color(label) * 255).astype(np.uint8)
    return ((1 - alpha) * image + alpha * color_mask).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grid plot
# ---------------------------------------------------------------------------
def save_prediction_grid(
    images: torch.Tensor,        # (N, 3, H, W)
    preds: torch.Tensor,         # (N, H, W)  int64
    labels: torch.Tensor,        # (N, H, W)  int64
    save_path: str,
    class_names: Optional[List[str]] = None,
    alpha: float = 0.45,
    dpi: int = 120,
) -> None:
    """
    Save a grid showing: image | ground truth overlay | prediction overlay
    for the first N samples.
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    N = len(images)
    fig, axes = plt.subplots(N, 3, figsize=(12, 4 * N), dpi=dpi)
    if N == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground Truth", "Prediction"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold")

    for i in range(N):
        img_np    = denormalize(images[i])
        label_np  = labels[i].cpu().numpy()
        pred_np   = preds[i].cpu().numpy()

        gt_overlay   = overlay_mask(img_np, label_np, alpha)
        pred_overlay = overlay_mask(img_np, pred_np,  alpha)

        axes[i, 0].imshow(img_np)
        axes[i, 1].imshow(gt_overlay)
        axes[i, 2].imshow(pred_overlay)

        for j in range(3):
            axes[i, j].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS_RGB[c], label=class_names[c])
        for c in range(len(class_names))
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(class_names),
               fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualise] Saved grid → {save_path}")


def plot_training_curves(log_path: str, save_dir: str) -> None:
    """
    Read a JSONL training log and plot loss + mean IoU curves.
    """
    import json

    records = []
    with open(log_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    epochs     = [r["epoch"] for r in records]
    train_loss = [r.get("train_loss", float("nan")) for r in records]
    val_loss   = [r.get("val_loss",   float("nan")) for r in records]
    mean_iou   = [r.get("mean_iou",   float("nan")) for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="darkorange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, mean_iou, label="Mean IoU", color="seagreen")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean IoU")
    ax2.set_title("Validation Mean IoU")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualise] Saved curves → {out}")
