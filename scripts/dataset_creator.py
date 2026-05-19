"""
SoreSAM - Batch dataset creator.

Recursively finds all JPEG images under --dir, runs inference on each,
and saves results to three subfolders under --export-dir:

    export/images/    original image  (PNG)
    export/dataset/   predicted mask  (colour PNG: Other=dark-grey, Skin=blue, Wound=red)
    export/overlay/   image + mask blend (PNG)

Usage
-----
    python scripts/dataset_creator.py --checkpoint best.pth --dir /path/to/images/

    # Custom output folder
    python scripts/dataset_creator.py --checkpoint best.pth --dir /data/ --export-dir out/

    # Disable morphological post-processing
    python scripts/dataset_creator.py --checkpoint best.pth --dir /data/ --no-morph

Optional flags
--------------
    --export-dir   root output directory         (default: export/)
    --alpha        overlay opacity 0-1           (default: 0.45)
    --device       cuda | cpu                    (default: cuda if available)
    --no-morph     disable morphological post-processing
    --close-ksize  closing kernel size in pixels (default: 15)
    --open-ksize   opening kernel size in pixels (default: 7)
    --min-area     minimum connected component area in pixels (default: 500)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import cfg
from model import build_model
from visualize import label_to_color, overlay_mask


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


# ---------------------------------------------------------------------------
# Helpers (duplicated from test_script to keep this file self-contained)
# ---------------------------------------------------------------------------

def find_images_recursive(directory: str) -> list[Path]:
    paths = sorted(
        p for p in Path(directory).rglob("*")
        if p.suffix in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No JPEG images found under: {directory}")
    return paths


def preprocess(image_path: str, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    original_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = original_rgb.shape[:2]
    scale = image_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = image_size - new_h
    pad_w = image_size - new_w
    padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    norm = (padded.astype(np.float32) / 255.0 - _MEAN) / _STD
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_rgb


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    model.eval()
    logits_np = model(image_tensor.to(device)).squeeze(0).cpu().numpy()  # (C, H, W)

    if smooth_sigma > 0.0:
        for c in range(logits_np.shape[0]):
            logits_np[c] = gaussian_filter(logits_np[c], sigma=smooth_sigma)

    e = np.exp(logits_np - logits_np.max(axis=0, keepdims=True))
    probs = e / e.sum(axis=0, keepdims=True)
    return probs.argmax(axis=0).astype(np.uint8)


def morphological_postprocess(
    pred: np.ndarray,
    num_classes: int = 3,
    close_ksize: int = 15,
    open_ksize: int = 7,
    min_area: int = 500,
) -> np.ndarray:
    H, W = pred.shape
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize,  open_ksize))
    score = np.zeros((num_classes, H, W), dtype=np.float32)

    for c in range(num_classes):
        mask = (pred == c).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        if min_area > 0 and mask.any():
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            clean = np.zeros_like(mask)
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels == lbl] = 1
            mask = clean
        score[c] = mask.astype(np.float32)

    result = score.argmax(axis=0).astype(np.uint8)
    unclaimed = score.max(axis=0) == 0
    result[unclaimed] = pred[unclaimed]

    # Absorb isolated islands: components fully enclosed by a single foreign
    # class and smaller than 4×min_area get reassigned to that class.
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        result.astype(np.uint8), connectivity=8
    )
    for lbl in range(1, n_labels):
        component_class = result[labels == lbl][0]
        border = cv2.dilate((labels == lbl).astype(np.uint8), k_dilate) - (labels == lbl).astype(np.uint8)
        neighbour_classes = result[border == 1]
        if neighbour_classes.size == 0:
            continue
        unique, counts = np.unique(neighbour_classes, return_counts=True)
        dominant = unique[counts.argmax()]
        if dominant != component_class and stats[lbl, cv2.CC_STAT_AREA] < 4 * min_area:
            result[labels == lbl] = dominant

    return result


def save_png(array_rgb: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(array_rgb, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Main export loop
# ---------------------------------------------------------------------------

def export(args: argparse.Namespace) -> None:
    config = cfg
    if args.sam2_config:     config.model.sam2_config     = args.sam2_config
    if args.sam2_checkpoint: config.model.sam2_checkpoint = args.sam2_checkpoint

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = build_model(
        sam2_config=config.model.sam2_config,
        sam2_checkpoint=config.model.sam2_checkpoint,
        num_classes=config.data.num_classes,
        num_class_tokens=config.model.num_class_tokens,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        device=str(device),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch  = ckpt.get("epoch", "?")
    metric = ckpt.get("metric", None)
    metric_str = f"  |  val mean_iou={metric:.4f}" if metric is not None else ""
    print(f"[Checkpoint] epoch {epoch}{metric_str}")

    morph_kwargs = None if args.no_morph else {
        "close_ksize": args.close_ksize,
        "open_ksize":  args.open_ksize,
        "min_area":    args.min_area,
    }
    if morph_kwargs:
        print(f"[Morphology] close={args.close_ksize}px  open={args.open_ksize}px  min_area={args.min_area}px²")
    else:
        print("[Morphology] disabled")

    image_paths = find_images_recursive(args.dir)
    print(f"[Found] {len(image_paths)} images under {args.dir}")

    out         = Path(args.export_dir)
    dir_images  = out / "images"
    dir_dataset = out / "dataset"
    dir_overlay = out / "overlay"
    for d in (dir_images, dir_dataset, dir_overlay):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[Output] {out}/")
    print(f"           images/   dataset/   overlay/\n")

    seen_stems: dict[str, Path] = {}
    n = len(image_paths)
    pad = len(str(n))

    for i, path in enumerate(image_paths, 1):
        stem = path.stem
        if stem in seen_stems:
            print(f"  [WARN] duplicate filename '{stem}' — conflicts with {seen_stems[stem]}, skipping.")
            continue
        seen_stems[stem] = path

        print(f"  [{i:>{pad}}/{n}] {path.relative_to(args.dir)}", end="\r", flush=True)

        tensor, original_rgb = preprocess(str(path), config.data.image_size)
        pred = run_inference(model, tensor, device, smooth_sigma=args.smooth_sigma)

        if morph_kwargs is not None:
            pred = morphological_postprocess(pred, num_classes=config.data.num_classes, **morph_kwargs)

        # Crop padding back to original (scaled) dimensions
        h, w  = original_rgb.shape[:2]
        scale = config.data.image_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img_display  = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pred_display = pred[:new_h, :new_w]

        mask_color = (label_to_color(pred_display) * 255).astype(np.uint8)
        blend      = overlay_mask(img_display, pred_display, alpha=args.alpha)

        save_png(img_display,  dir_images  / f"{stem}.png")
        save_png(mask_color,   dir_dataset / f"{stem}.png")
        save_png(blend,        dir_overlay / f"{stem}.png")

    print(f"\n[Done] {len(seen_stems)} images exported to {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SoreSAM batch dataset creator — recursive inference & export"
    )
    p.add_argument("--checkpoint",       type=str, required=True, help="Path to best.pth")
    p.add_argument("--dir",              type=str, required=True, help="Root directory to search recursively")
    p.add_argument("--sam2-config",      type=str, default=None)
    p.add_argument("--sam2-checkpoint",  type=str, default=None)
    p.add_argument("--export-dir",       type=str, default="export",  help="Output root (default: export/)")
    p.add_argument("--alpha",            type=float, default=0.45,    help="Overlay opacity 0-1 (default: 0.45)")
    p.add_argument("--device",           type=str,   default=None,    help="cuda | cpu")
    p.add_argument("--smooth-sigma",      type=float, default=0.0,
                   help="Gaussian sigma on logits before softmax (0=off, try 1-3 for clinical images)")
    p.add_argument("--no-morph",         action="store_true",         help="Disable morphological post-processing")
    p.add_argument("--close-ksize",      type=int,   default=15)
    p.add_argument("--open-ksize",       type=int,   default=7)
    p.add_argument("--min-area",         type=int,   default=500)
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
