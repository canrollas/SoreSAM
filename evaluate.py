"""
SoreSAM - Evaluation script.

Runs inference on the test set and reports per-class and mean metrics.

Usage:
    python evaluate.py --checkpoint outputs/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from config import Config, cfg
from dataset import WoundDataset
from metrics import SegmentationMetrics
from model import build_model
from visualize import save_prediction_grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SoreSAM evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--sam2-config", type=str, default=None)
    p.add_argument("--sam2-checkpoint", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--vis-n", type=int, default=8, help="Samples to visualise")
    p.add_argument("--output-dir", type=str, default="outputs/eval")
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    metrics: SegmentationMetrics,
    device: torch.device,
    use_amp: bool = True,
    vis_n: int = 8,
    vis_dir: str = "outputs/eval",
) -> dict:
    model.eval()
    metrics.reset()

    vis_images, vis_preds, vis_labels = [], [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)

        preds = logits.argmax(dim=1)
        metrics.update(preds, labels)

        # Collect samples for visualisation
        if len(vis_images) < vis_n:
            remaining = vis_n - len(vis_images)
            vis_images.append(images[:remaining].cpu())
            vis_preds.append(preds[:remaining].cpu())
            vis_labels.append(labels[:remaining].cpu())

    results = metrics.compute()
    metrics.print_table(results)

    # Save visualisation grid
    if vis_images:
        import torch as _torch
        Path(vis_dir).mkdir(parents=True, exist_ok=True)
        save_prediction_grid(
            images=_torch.cat(vis_images)[:vis_n],
            preds=_torch.cat(vis_preds)[:vis_n],
            labels=_torch.cat(vis_labels)[:vis_n],
            save_path=str(Path(vis_dir) / "test_predictions.png"),
            class_names=metrics.class_names,
        )

    return results


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config: Config = cfg
    if args.data_root:       config.data.root = args.data_root
    if args.sam2_config:     config.model.sam2_config = args.sam2_config
    if args.sam2_checkpoint: config.model.sam2_checkpoint = args.sam2_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Data
    test_ds = WoundDataset(
        config.data.root,
        split="test",
        color_threshold=config.data.color_threshold,
        image_size=config.data.image_size,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"[Data] Test set: {len(test_ds)} samples")

    # Model
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
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    print(f"[Checkpoint] Loaded from epoch {epoch}: {args.checkpoint}")

    # Evaluate
    seg_metrics = SegmentationMetrics(
        num_classes=config.data.num_classes,
        class_names=config.data.class_names,
    )
    results = evaluate(
        model, test_loader, seg_metrics, device,
        use_amp=(args.device == "cuda"),
        vis_n=args.vis_n,
        vis_dir=args.output_dir,
    )

    # Save JSON results
    out_json = Path(args.output_dir) / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {out_json}")


if __name__ == "__main__":
    main()
