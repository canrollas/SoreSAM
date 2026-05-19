"""
SoreSAM - Single-image inference & visualisation / directory gallery script.

── Single image ──────────────────────────────────────────────────────────────
    python test_script.py --checkpoint best.pth --image foto.jpg

── Directory gallery ─────────────────────────────────────────────────────────
    python test_script.py --checkpoint best.pth --dir /path/to/images/

    Gallery controls:
        → / d / Space   next image
        ← / a           previous image
        s               save current result to --output-dir
        q / Esc         quit

Optional flags:
    --output-dir   directory for saved results  (default: gallery_results/)
    --alpha        overlay opacity 0-1          (default: 0.45)
    --device       cuda | cpu                   (default: cuda if available)
    --no-morph     disable morphological post-processing
    --close-ksize  closing kernel size in pixels (default: 15)
    --open-ksize   opening kernel size in pixels (default: 7)
    --min-area     minimum connected component area in pixels (default: 500)
    --prior-weight temporal prior weight for directory mode (default: 0.3)
                   0 = disabled, higher = stronger pull toward previous mask

    (single-image mode only)
    --output      path/to/output.png           (default: inference_result.png)
    --no-show     skip plt.show()
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
from scipy.ndimage import gaussian_filter

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
    prev_probs: np.ndarray | None = None,
    prior_weight: float = 0.0,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model and return a predicted label map plus softmax probabilities.

    When *prev_probs* and *prior_weight > 0* are supplied the final prediction
    is taken from a weighted blend of the current frame's softmax distribution
    and the previous frame's, providing temporal smoothing for sequential images.

    Parameters
    ----------
    prev_probs    : (num_classes, H, W) float32 from the previous frame, or None
    prior_weight  : weight given to prev_probs (0 = disabled, 0.3 = recommended)
    smooth_sigma  : Gaussian sigma applied per-class to logits before softmax,
                    smooths blocky patch-boundary artifacts (0 = disabled, 1-3 recommended)

    Returns
    -------
    pred  : (H, W) uint8  — class indices {0, 1, 2}
    probs : (num_classes, H, W) float32 — raw softmax of *this* frame (unblended)
            pass as prev_probs to the next call
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)                                   # (1, C, H, W)
    logits_np = logits.squeeze(0).cpu().numpy()                    # (C, H, W)

    if smooth_sigma > 0.0:
        for c in range(logits_np.shape[0]):
            logits_np[c] = gaussian_filter(logits_np[c], sigma=smooth_sigma)

    # softmax manually so we stay in numpy after the optional smoothing
    e = np.exp(logits_np - logits_np.max(axis=0, keepdims=True))
    probs = e / e.sum(axis=0, keepdims=True)                       # (C, H, W)

    if prev_probs is not None and prior_weight > 0.0:
        blended = (1.0 - prior_weight) * probs + prior_weight * prev_probs
    else:
        blended = probs

    pred = blended.argmax(axis=0).astype(np.uint8)
    return pred, probs


def morphological_postprocess(
    pred: np.ndarray,
    num_classes: int = 3,
    close_ksize: int = 15,
    open_ksize: int = 7,
    min_area: int = 500,
) -> np.ndarray:
    """
    Apply per-class morphological post-processing to clean the prediction mask.

    Per-class steps:
      1. Closing  (dilation → erosion) : fills small holes inside regions
      2. Opening  (erosion → dilation) : removes small isolated noise blobs
      3. Remove connected components smaller than min_area pixels

    Post-merge step:
      4. Absorb isolated islands — any connected component completely enclosed
         within a single foreign class gets reassigned to that class. Handles
         scattered specks (e.g. Skin predictions inside a Wound region) that
         closing alone cannot reach because they are too spread out.

    Parameters
    ----------
    pred       : (H, W) uint8 label map
    num_classes: number of classes
    close_ksize: kernel size for closing (larger → fills bigger holes)
    open_ksize : kernel size for opening (larger → removes bigger blobs)
    min_area   : connected components smaller than this are discarded
    """
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

    # ── Step 4: absorb isolated islands ──────────────────────────────────────
    # For each connected component in the merged result, check whether all its
    # border-adjacent pixels belong to a single different class. If so, the
    # component is an "island" and gets absorbed into that surrounding class.
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        result.astype(np.uint8), connectivity=8
    )
    for lbl in range(1, n_labels):
        component_mask = (labels == lbl).astype(np.uint8)
        component_class = result[labels == lbl][0]

        # Dilate the component by 1 pixel to find its border neighbours
        border = cv2.dilate(component_mask, k_dilate) - component_mask
        neighbour_classes = result[border == 1]
        if neighbour_classes.size == 0:
            continue

        unique, counts = np.unique(neighbour_classes, return_counts=True)
        dominant = unique[counts.argmax()]

        # Absorb only if the component is surrounded by a single foreign class
        # and is smaller than 4× min_area (large regions are kept as-is)
        area = stats[lbl, cv2.CC_STAT_AREA]
        if dominant != component_class and area < 4 * min_area:
            result[labels == lbl] = dominant

    return result


def compute_class_coverage(pred: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Return percentage of pixels per class."""
    total = pred.size
    return {
        class_names[c]: round(float((pred == c).sum()) / total * 100, 2)
        for c in range(len(class_names))
    }


def plot_result(
    original_rgb: np.ndarray,
    pred_raw: np.ndarray,
    pred_morph: np.ndarray | None,
    image_size: int,
    class_names: list[str],
    alpha: float,
    save_path: str,
    show: bool,
) -> None:
    """
    Figure panels (varies by whether morphology was applied):

    Without morphology  → 3 panels: Original | Overlay | Mask
    With morphology     → 4 panels: Original | Raw Overlay | Morph Overlay | Morph Mask
    """
    # Resize original image to match model output resolution, then crop padding
    h_orig, w_orig = original_rgb.shape[:2]
    scale  = image_size / max(h_orig, w_orig)
    new_h  = int(round(h_orig * scale))
    new_w  = int(round(w_orig * scale))
    img_display = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Crop predictions to the valid (non-padded) region
    raw_display   = pred_raw[:new_h, :new_w]
    morph_display = pred_morph[:new_h, :new_w] if pred_morph is not None else None

    use_morph = morph_display is not None
    n_panels  = 4 if use_morph else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), dpi=120)
    fig.suptitle("SoreSAM — Inference Result", fontsize=15, fontweight="bold", y=1.02)

    # Panel 0 — original
    axes[0].imshow(img_display)
    axes[0].set_title("Input Image", fontsize=12)

    if use_morph:
        # Panel 1 — raw prediction overlay
        axes[1].imshow(overlay_mask(img_display, raw_display, alpha=alpha))
        axes[1].set_title("Raw Prediction", fontsize=12)

        # Panel 2 — morphological prediction overlay
        axes[2].imshow(overlay_mask(img_display, morph_display, alpha=alpha))
        axes[2].set_title("After Morphology", fontsize=12)

        # Panel 3 — morphological mask only
        axes[3].imshow(label_to_color(morph_display))
        axes[3].set_title("Morph Mask", fontsize=12)

        coverage = compute_class_coverage(morph_display, class_names)
    else:
        # Panel 1 — raw overlay
        axes[1].imshow(overlay_mask(img_display, raw_display, alpha=alpha))
        axes[1].set_title("Prediction Overlay", fontsize=12)

        # Panel 2 — raw mask only
        axes[2].imshow(label_to_color(raw_display))
        axes[2].set_title("Prediction Mask", fontsize=12)

        coverage = compute_class_coverage(raw_display, class_names)

    for ax in axes:
        ax.axis("off")

    # Legend with coverage percentages
    patches = [
        mpatches.Patch(
            color=CLASS_COLORS_RGB[c],
            label=f"{class_names[c]}  {coverage[class_names[c]]:.1f}%",
        )
        for c in range(len(class_names))
    ]
    fig.legend(
        handles=patches, loc="lower center", ncol=len(class_names),
        fontsize=11, bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    print(f"[Saved] {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Directory gallery
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def find_images(directory: str) -> list[Path]:
    """Return sorted list of JPEG paths in *directory* (non-recursive)."""
    paths = sorted(
        p for p in Path(directory).iterdir()
        if p.suffix in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No JPEG images found in: {directory}")
    return paths


class Gallery:
    """
    Interactive matplotlib gallery.
    Runs inference on ALL images upfront, then opens the viewer.

    Keys:
        → / d / Space   next
        ← / a           previous
        s               save current figure to output_dir
        q / Esc         quit
    """

    def __init__(
        self,
        image_paths: list[Path],
        model: torch.nn.Module,
        device: torch.device,
        config,
        morph_kwargs: dict,
        alpha: float,
        output_dir: str,
        prior_weight: float = 0.3,
        smooth_sigma: float = 0.0,
    ):
        self.paths        = image_paths
        self.config       = config
        self.morph_kwargs = morph_kwargs
        self.alpha        = alpha
        self.output_dir   = Path(output_dir)
        self.smooth_sigma = smooth_sigma
        self.idx          = 0

        # ── Batch inference ───────────────────────────────────────────
        n = len(image_paths)
        self.results: list[tuple] = []   # (pred_raw, pred_morph, img_padded)

        if prior_weight > 0:
            print(f"[Gallery] Temporal prior enabled (prior_weight={prior_weight:.2f})")
        else:
            print("[Gallery] Temporal prior disabled (--prior-weight 0)")

        print(f"[Gallery] Processing {n} images…")
        prev_probs = None
        for i, path in enumerate(image_paths, 1):
            print(f"  [{i:>{len(str(n))}}/{n}] {path.name}", end="\r", flush=True)

            tensor, original_rgb = preprocess(str(path), config.data.image_size)
            pred_raw, prev_probs = run_inference(
                model, tensor, device,
                prev_probs=prev_probs,
                prior_weight=prior_weight,
                smooth_sigma=self.smooth_sigma,
            )

            pred_morph = None
            if morph_kwargs is not None:
                pred_morph = morphological_postprocess(
                    pred_raw,
                    num_classes=config.data.num_classes,
                    **morph_kwargs,
                )

            h, w = original_rgb.shape[:2]
            scale = config.data.image_size / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img_display = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            self.results.append((pred_raw, pred_morph, img_display, new_h, new_w))

        print(f"\n[Gallery] Done. Opening viewer…\n")

        # ── Open viewer ───────────────────────────────────────────────
        n_panels = 4 if morph_kwargs is not None else 3
        self.fig, self.axes = plt.subplots(
            1, n_panels, figsize=(6 * n_panels, 6), dpi=110
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._render()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def _render(self) -> None:
        pred_raw, pred_morph, img_display, new_h, new_w = self.results[self.idx]
        raw_display   = pred_raw[:new_h, :new_w]
        morph_display = pred_morph[:new_h, :new_w] if pred_morph is not None else None
        final  = morph_display if morph_display is not None else raw_display
        cnames = self.config.data.class_names

        for ax in self.axes:
            ax.cla()
            ax.axis("off")

        self.axes[0].imshow(img_display)
        self.axes[0].set_title("Input Image", fontsize=11)

        if morph_display is not None:
            self.axes[1].imshow(overlay_mask(img_display, raw_display,   alpha=self.alpha))
            self.axes[1].set_title("Raw Prediction", fontsize=11)
            self.axes[2].imshow(overlay_mask(img_display, morph_display, alpha=self.alpha))
            self.axes[2].set_title("After Morphology", fontsize=11)
            self.axes[3].imshow(label_to_color(morph_display))
            self.axes[3].set_title("Morph Mask", fontsize=11)
        else:
            self.axes[1].imshow(overlay_mask(img_display, raw_display, alpha=self.alpha))
            self.axes[1].set_title("Prediction Overlay", fontsize=11)
            self.axes[2].imshow(label_to_color(raw_display))
            self.axes[2].set_title("Prediction Mask", fontsize=11)

        # Coverage legend
        coverage = compute_class_coverage(final, cnames)
        patches = [
            mpatches.Patch(
                color=CLASS_COLORS_RGB[c],
                label=f"{cnames[c]}  {coverage[cnames[c]]:.1f}%",
            )
            for c in range(len(cnames))
        ]
        # Remove previous legend if any
        if self.fig.legends:
            self.fig.legends[-1].remove()
        self.fig.legend(
            handles=patches, loc="lower center", ncol=len(cnames),
            fontsize=10, bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
        )

        fname = self.paths[self.idx].name
        self.fig.suptitle(
            f"[{self.idx + 1}/{len(self.paths)}]  {fname}"
            f"   |   ← → navigate   s save   q quit",
            fontsize=11, y=1.01,
        )
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _save_current(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.paths[self.idx].stem
        out  = self.output_dir / f"{stem}_result.png"
        self.fig.savefig(out, bbox_inches="tight", dpi=120)
        print(f"[Saved] {out}")

    def _on_key(self, event) -> None:
        if event.key in ("right", "d", " "):
            self.idx = (self.idx + 1) % len(self.paths)
            self._render()
        elif event.key in ("left", "a"):
            self.idx = (self.idx - 1) % len(self.paths)
            self._render()
        elif event.key == "s":
            self._save_current()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SoreSAM inference — single image or directory gallery")
    p.add_argument("--checkpoint",      type=str, required=True, help="Path to best.pth")
    # Input: exactly one of --image or --dir
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",         type=str, help="Single image path")
    src.add_argument("--dir",           type=str, help="Directory of JPEG images (gallery mode)")
    p.add_argument("--sam2-config",     type=str, default=None)
    p.add_argument("--sam2-checkpoint", type=str, default=None)
    # Output
    p.add_argument("--output",          type=str, default="inference_result.png", help="Single-image output path")
    p.add_argument("--output-dir",      type=str, default="gallery_results",      help="Directory for gallery saves")
    p.add_argument("--alpha",           type=float, default=0.45, help="Overlay opacity (0-1)")
    p.add_argument("--device",          type=str,   default=None, help="cuda | cpu")
    p.add_argument("--no-show",         action="store_true",      help="Skip plt.show() (single-image mode)")
    # Smoothing (reduces blocky patch-boundary artifacts on OOD images)
    p.add_argument("--smooth-sigma",    type=float, default=0.0,
                   help="Gaussian sigma on logits before softmax (0=off, try 1-3 for clinical images)")
    # Morphological post-processing
    p.add_argument("--no-morph",        action="store_true",      help="Disable morphological post-processing")
    p.add_argument("--close-ksize",     type=int, default=15,     help="Closing kernel size (fills holes)")
    p.add_argument("--open-ksize",      type=int, default=7,      help="Opening kernel size (removes noise)")
    p.add_argument("--min-area",        type=int, default=500,    help="Min connected component area (pixels)")
    p.add_argument("--prior-weight",    type=float, default=0.3,
                   help="Temporal prior weight in directory mode: weight given to the "
                        "previous frame's softmax map (0=disabled, default=0.3)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = cfg

    if args.sam2_config:      config.model.sam2_config     = args.sam2_config
    if args.sam2_checkpoint:  config.model.sam2_checkpoint = args.sam2_checkpoint

    # Device
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Build model + load checkpoint (shared by both modes)
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

    # ── Directory gallery mode ────────────────────────────────────────────
    if args.dir:
        image_paths = find_images(args.dir)
        print(f"[Gallery] {len(image_paths)} images in {args.dir}")
        Gallery(
            image_paths=image_paths,
            model=model,
            device=device,
            config=config,
            morph_kwargs=morph_kwargs,
            alpha=args.alpha,
            output_dir=args.output_dir,
            prior_weight=args.prior_weight,
            smooth_sigma=args.smooth_sigma,
        )
        return

    # ── Single image mode ─────────────────────────────────────────────────
    print(f"[Image] {args.image}")
    image_tensor, original_rgb = preprocess(args.image, config.data.image_size)
    pred_raw, _ = run_inference(model, image_tensor, device, smooth_sigma=args.smooth_sigma)

    pred_morph = None
    if morph_kwargs:
        pred_morph = morphological_postprocess(
            pred_raw, num_classes=config.data.num_classes, **morph_kwargs
        )

    # Coverage stats
    header = f"\n{'Class':<12} {'Raw':>8}   {'Morph':>8}" if pred_morph is not None \
             else f"\n{'Class':<12} {'Coverage':>8}"
    print(header)
    print("-" * (34 if pred_morph is not None else 22))
    for c, cls_name in enumerate(config.data.class_names):
        raw_pct = float((pred_raw == c).sum()) / pred_raw.size * 100
        if pred_morph is not None:
            morph_pct = float((pred_morph == c).sum()) / pred_morph.size * 100
            print(f"  {cls_name:<10} {raw_pct:>7.1f}%  {morph_pct:>7.1f}%")
        else:
            print(f"  {cls_name:<10} {raw_pct:>7.1f}%")
    print()

    plot_result(
        original_rgb=original_rgb,
        pred_raw=pred_raw,
        pred_morph=pred_morph,
        image_size=config.data.image_size,
        class_names=config.data.class_names,
        alpha=args.alpha,
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
