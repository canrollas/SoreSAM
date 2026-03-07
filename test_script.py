"""
SoreSAM - Single-image inference & visualisation script.

Usage:
    python test_script.py \
        --checkpoint /content/drive/MyDrive/SoreSAM-Outputs/checkpoints/best.pth \
        --image path/to/image.jpg \
        --sam2-config /content/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
        --sam2-checkpoint /content/SoreSAM/checkpoints/sam2.1_hiera_large.pt

Optional flags:
    --output    path/to/output.png   (default: inference_result.png)
    --alpha     overlay opacity 0-1  (default: 0.45)
    --device    cuda | cpu           (default: cuda if available)
    --no-show   skip plt.show()      (useful in headless environments / Colab)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import cfg
from dataset import mask_rgb_to_label
from model import build_model
from visualize import CLASS_COLORS_RGB, CLASS_NAMES_DEFAULT, denormalize, label_to_color, overlay_mask


# ---------------------------------------------------------------------------
# ImageNet normalisation (must match training)
# ---------------------------------------------------------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str, image_size: int = 1024) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess a single image for inference.

    Returns
    -------
    tensor : (1, 3, image_size, image_size) float32
    original_rgb : (H_orig, W_orig, 3) uint8  — for display
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize keeping aspect ratio, then pad to square
    h, w = original_rgb.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to image_size × image_size
    pad_h = image_size - new_h
    pad_w = image_size - new_w
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    # Normalise
    norm = (padded.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor, original_rgb


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Run the model and return a predicted label map.

    Returns
    -------
    pred : (H, W) uint8  — class indices {0, 1, 2}
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)          # (1, num_classes, H, W)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def compute_class_coverage(pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Return percentage of pixels per class."""
    total = pred.size
    return {
        class_names[c]: round(float((pred == c).sum()) / total * 100, 2)
        for c in range(len(class_names))
    }


def plot_result(
    original_rgb: np.ndarray,
    pred: np.ndarray,
    image_size: int,
    class_names: list[str],
    alpha: float,
    save_path: str,
    show: bool,
) -> None:
    """
    3-panel figure: Original | Prediction overlay | Prediction mask
    + pixel coverage stats.
    """
    # Resize original to model input size for overlay alignment
    h_orig, w_orig = original_rgb.shape[:2]
    scale = image_size / max(h_orig, w_orig)
    new_h = int(round(h_orig * scale))
    new_w = int(round(w_orig * scale))
    img_resized = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_padded  = np.pad(img_resized, ((0, image_size - new_h), (0, image_size - new_w), (0, 0)))

    pred_color   = label_to_color(pred)                          # (H,W,3) float
    pred_overlay = overlay_mask(img_padded, pred, alpha=alpha)   # (H,W,3) uint8

    coverage = compute_class_coverage(pred, class_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
    fig.suptitle("SoreSAM — Inference Result", fontsize=15, fontweight="bold", y=1.01)

    # --- Panel 0: original ---
    axes[0].imshow(img_padded)
    axes[0].set_title("Input Image", fontsize=12)

    # --- Panel 1: overlay ---
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Prediction Overlay", fontsize=12)

    # --- Panel 2: mask only ---
    axes[2].imshow(pred_color)
    axes[2].set_title("Prediction Mask", fontsize=12)

    for ax in axes:
        ax.axis("off")

    # Legend with coverage stats
    patches = [
        mpatches.Patch(
            color=CLASS_COLORS_RGB[c],
            label=f"{class_names[c]}  {coverage[class_names[c]]:.1f}%",
        )
        for c in range(len(class_names))
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=len(class_names),
        fontsize=11, bbox_to_anchor=(0.5, -0.04),
        framealpha=0.9,
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    print(f"[Saved] {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SoreSAM single-image inference")
    p.add_argument("--checkpoint",       type=str, required=True,  help="Path to best.pth")
    p.add_argument("--image",            type=str, required=True,  help="Path to input image")
    p.add_argument("--sam2-config",      type=str, default=None)
    p.add_argument("--sam2-checkpoint",  type=str, default=None)
    p.add_argument("--output",           type=str, default="inference_result.png")
    p.add_argument("--alpha",            type=float, default=0.45, help="Overlay opacity (0-1)")
    p.add_argument("--device",           type=str, default=None,   help="cuda | cpu")
    p.add_argument("--no-show",          action="store_true",       help="Do not call plt.show()")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = cfg

    if args.sam2_config:      config.model.sam2_config     = args.sam2_config
    if args.sam2_checkpoint:  config.model.sam2_checkpoint = args.sam2_checkpoint

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Preprocess
    print(f"[Image] {args.image}")
    image_tensor, original_rgb = preprocess(args.image, config.data.image_size)

    # Build model
    model = build_model(
        sam2_config=config.model.sam2_config,
        sam2_checkpoint=config.model.sam2_checkpoint,
        num_classes=config.data.num_classes,
        num_class_tokens=config.model.num_class_tokens,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        device=str(device),
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    metric = ckpt.get("metric", None)
    metric_str = f"  |  val mean_iou={metric:.4f}" if metric is not None else ""
    print(f"[Checkpoint] epoch {epoch}{metric_str}")

    # Inference
    pred = run_inference(model, image_tensor, device)

    # Coverage stats
    coverage = compute_class_coverage(pred, config.data.class_names)
    print("[Coverage]")
    for cls_name, pct in coverage.items():
        print(f"  {cls_name:10s}: {pct:.1f}%")

    # Plot & save
    plot_result(
        original_rgb=original_rgb,
        pred=pred,
        image_size=config.data.image_size,
        class_names=config.data.class_names,
        alpha=args.alpha,
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
