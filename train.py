"""
SoreSAM - Training script.

Usage:
    python train.py [--config overrides via argparse]

Key design choices:
    - SAM2 image encoder frozen by default (only decoder + class tokens trained)
    - Combined CE + Dice loss
    - AMP (mixed precision) for memory efficiency
    - Cosine LR schedule with linear warmup
    - Saves best checkpoint by mean IoU on validation set
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import warnings
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import Config, cfg
from dataset import WoundDataset, make_train_val_split
from losses import CombinedSegLoss
from metrics import SegmentationMetrics
from model import build_model


# ---------------------------------------------------------------------------
# Argument parsing (minimal — override config fields)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SoreSAM training")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--sam2-config", type=str, default=None)
    p.add_argument("--sam2-checkpoint", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def apply_args(config: Config, args: argparse.Namespace) -> Config:
    if args.data_root:        config.data.root = args.data_root
    if args.sam2_config:      config.model.sam2_config = args.sam2_config
    if args.sam2_checkpoint:  config.model.sam2_checkpoint = args.sam2_checkpoint
    if args.output_dir:
        config.train.output_dir = args.output_dir
        config.train.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        config.train.log_dir = os.path.join(args.output_dir, "logs")
        config.train.vis_dir = os.path.join(args.output_dir, "visualizations")
    if args.epochs:           config.train.num_epochs = args.epochs
    if args.batch_size:       config.train.batch_size = args.batch_size
    if args.lr:               config.train.lr = args.lr
    if args.device:           config.train.device = args.device
    return config


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataloaders(config: Config):
    train_idx, val_idx = make_train_val_split(
        config.data.root,
        val_fraction=config.data.val_split,
        seed=config.data.seed,
    )

    train_ds = WoundDataset(
        config.data.root, split="train",
        val_indices=val_idx,
        color_threshold=config.data.color_threshold,
        image_size=config.data.image_size,
    )
    val_ds = WoundDataset(
        config.data.root, split="val",
        val_indices=val_idx,
        color_threshold=config.data.color_threshold,
        image_size=config.data.image_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, config.train.batch_size // 2),
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Scheduler with warmup
# ---------------------------------------------------------------------------
def build_scheduler(optimiser, config: Config, steps_per_epoch: int):
    T_max = (config.train.num_epochs - config.train.warmup_epochs) * steps_per_epoch

    cosine = CosineAnnealingLR(optimiser, T_max=T_max, eta_min=1e-6)

    if config.train.warmup_epochs > 0:
        warmup = LinearLR(
            optimiser,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.train.warmup_epochs * steps_per_epoch,
        )
        scheduler = SequentialLR(
            optimiser,
            schedulers=[warmup, cosine],
            milestones=[config.train.warmup_epochs * steps_per_epoch],
        )
    else:
        scheduler = cosine

    return scheduler


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedSegLoss,
    optimiser: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: Config,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    n_batches = len(loader)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimiser.zero_grad()

        with autocast("cuda", enabled=config.train.use_amp):
            logits = model(images)
            loss, loss_dict = criterion(logits, labels)

        scaler.scale(loss).backward()

        if config.train.grad_clip > 0:
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        total_loss += loss_dict["loss"].item()
        total_ce += loss_dict["loss_ce"].item()
        total_dice += loss_dict["loss_dice"].item()

        if (step + 1) % config.train.log_interval == 0:
            lr = optimiser.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch:03d} | {step+1:4d}/{n_batches}] "
                f"loss={loss_dict['loss'].item():.4f}  "
                f"ce={loss_dict['loss_ce'].item():.4f}  "
                f"dice={loss_dict['loss_dice'].item():.4f}  "
                f"lr={lr:.2e}"
            )

    return {
        "train_loss": total_loss / n_batches,
        "train_ce": total_ce / n_batches,
        "train_dice": total_dice / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedSegLoss,
    metrics: SegmentationMetrics,
    device: torch.device,
    config: Config,
) -> Dict[str, float]:
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast("cuda", enabled=config.train.use_amp):
            logits = model(images)
            loss, _ = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        metrics.update(preds, labels)

    results = metrics.compute()
    results["val_loss"] = total_loss / len(loader)
    return results


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    optimiser,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    metric: float,
    config: Config,
    filename: str,
) -> None:
    path = Path(config.train.checkpoint_dir) / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metric": metric,
        },
        path,
    )
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimiser=None,
    scheduler=None,
    scaler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimiser and "optimiser_state_dict" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}: {path}")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config = apply_args(cfg, args)
    config.__post_init__()

    # Device
    if config.train.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        config.train.device = "cpu"
    device = torch.device(config.train.device)
    print(f"[Device] Using: {device}")

    # Reproducibility
    torch.manual_seed(config.data.seed)

    # Data
    train_loader, val_loader = build_dataloaders(config)

    # Model
    model = build_model(
        sam2_config=config.model.sam2_config,
        sam2_checkpoint=config.model.sam2_checkpoint,
        num_classes=config.data.num_classes,
        num_class_tokens=config.model.num_class_tokens,
        freeze_image_encoder=config.model.freeze_image_encoder,
        freeze_prompt_encoder=config.model.freeze_prompt_encoder,
        device=str(device),
    )

    # Loss
    criterion = CombinedSegLoss(
        num_classes=config.data.num_classes,
        ce_weight=config.train.ce_weight,
        dice_weight=config.train.dice_weight,
        class_weights=config.train.class_weights,
    ).to(device)

    # Optimiser — separate LRs for class tokens vs decoder
    param_groups = model.parameter_groups(
        lr=config.train.lr,
        decoder_lr_mult=config.train.decoder_lr_multiplier,
    )
    optimiser = AdamW(param_groups, weight_decay=config.train.weight_decay)

    # Scheduler + AMP
    scheduler = build_scheduler(optimiser, config, steps_per_epoch=len(train_loader))
    scaler = GradScaler("cuda", enabled=config.train.use_amp)
    # SequentialLR calls step() internally during init which triggers a false-positive
    # "step before optimizer.step()" warning — suppress it.
    warnings.filterwarnings(
        "ignore",
        message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
        category=UserWarning,
    )

    # Metrics
    metrics = SegmentationMetrics(
        num_classes=config.data.num_classes,
        class_names=config.data.class_names,
    )

    # Resume
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimiser, scheduler, scaler) + 1

    # Log file
    log_path = Path(config.train.log_dir) / "train_log.jsonl"
    log_file = open(log_path, "a")

    best_metric = 0.0
    best_epoch = 0

    print(f"\n[Train] Starting training for {config.train.num_epochs} epochs")
    print(f"        Trainable params: {model.num_trainable_params():,}\n")

    for epoch in range(start_epoch, config.train.num_epochs + 1):
        t0 = time.time()
        print(f"{'─'*60}")
        print(f"Epoch {epoch}/{config.train.num_epochs}")

        # Train
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimiser, scheduler, scaler,
            device, epoch, config,
        )

        # Validate
        val_stats: Dict = {}
        if epoch % config.train.val_interval == 0:
            val_stats = validate(model, val_loader, criterion, metrics, device, config)
            metrics.print_table(val_stats)

            # Save best
            current_metric = val_stats.get(config.train.save_best_metric, 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                save_checkpoint(model, optimiser, scheduler, scaler, epoch,
                                 current_metric, config, "best.pth")
                print(f"  [Best] {config.train.save_best_metric}={best_metric:.4f} @ epoch {epoch}")

        # Always save latest
        save_checkpoint(model, optimiser, scheduler, scaler, epoch,
                         best_metric, config, "latest.pth")

        # Log
        log_entry = {
            "epoch": epoch,
            "time_s": round(time.time() - t0, 1),
            **train_stats,
            **val_stats,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        elapsed = time.time() - t0
        print(f"  Epoch time: {elapsed:.1f}s")

    log_file.close()
    print(f"\n[Done] Best {config.train.save_best_metric}={best_metric:.4f} at epoch {best_epoch}")
    print(f"       Checkpoints: {config.train.checkpoint_dir}")


if __name__ == "__main__":
    main()
