"""
SoreSAM - Evaluation metrics.

Per-class and mean metrics: IoU (Jaccard), Dice (F1), Precision, Recall.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class SegmentationMetrics:
    """
    Accumulates confusion matrix over batches, then computes metrics.

    Usage
    -----
    metrics = SegmentationMetrics(num_classes=3, class_names=["Other","Skin","Wound"])
    for batch in dataloader:
        preds = model(batch["image"]).argmax(dim=1)
        metrics.update(preds, batch["label"])
    results = metrics.compute()
    metrics.reset()
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        ignore_index: Optional[int] = None,
    ):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.ignore_index = ignore_index
        self._conf = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._conf[:] = 0

    def update(
        self,
        preds: torch.Tensor,   # (B, H, W)  int64
        targets: torch.Tensor, # (B, H, W)  int64
    ) -> None:
        preds_np = preds.detach().cpu().numpy().flatten()
        tgts_np = targets.detach().cpu().numpy().flatten()

        if self.ignore_index is not None:
            keep = tgts_np != self.ignore_index
            preds_np = preds_np[keep]
            tgts_np = tgts_np[keep]

        # Clamp to valid range
        valid = (tgts_np >= 0) & (tgts_np < self.num_classes) & \
                (preds_np >= 0) & (preds_np < self.num_classes)
        preds_np = preds_np[valid]
        tgts_np = tgts_np[valid]

        np.add.at(self._conf, (tgts_np, preds_np), 1)

    # ------------------------------------------------------------------
    def compute(self) -> Dict[str, float]:
        conf = self._conf.astype(np.float64)
        results: Dict[str, float] = {}

        per_class_iou: List[float] = []
        per_class_dice: List[float] = []
        per_class_prec: List[float] = []
        per_class_rec: List[float] = []

        for c in range(self.num_classes):
            tp = conf[c, c]
            fp = conf[:, c].sum() - tp
            fn = conf[c, :].sum() - tp

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

            name = self.class_names[c]
            results[f"iou_{name}"] = iou
            results[f"dice_{name}"] = dice
            results[f"precision_{name}"] = prec
            results[f"recall_{name}"] = rec

            per_class_iou.append(iou)
            per_class_dice.append(dice)
            per_class_prec.append(prec)
            per_class_rec.append(rec)

        # Macro averages (NaN-safe)
        results["mean_iou"] = float(np.nanmean(per_class_iou))
        results["mean_dice"] = float(np.nanmean(per_class_dice))
        results["mean_precision"] = float(np.nanmean(per_class_prec))
        results["mean_recall"] = float(np.nanmean(per_class_rec))

        # Overall pixel accuracy
        total_correct = np.diag(conf).sum()
        total_pixels = conf.sum()
        results["pixel_accuracy"] = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        return results

    # ------------------------------------------------------------------
    def print_table(self, results: Optional[Dict[str, float]] = None) -> None:
        if results is None:
            results = self.compute()

        print("\n" + "=" * 65)
        print(f"{'Class':<12} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}")
        print("-" * 65)
        for name in self.class_names:
            iou = results.get(f"iou_{name}", float("nan"))
            dice = results.get(f"dice_{name}", float("nan"))
            prec = results.get(f"precision_{name}", float("nan"))
            rec = results.get(f"recall_{name}", float("nan"))
            print(f"{name:<12} {iou:>8.4f} {dice:>8.4f} {prec:>8.4f} {rec:>8.4f}")
        print("-" * 65)
        print(
            f"{'Mean':<12} {results['mean_iou']:>8.4f} "
            f"{results['mean_dice']:>8.4f} "
            f"{results['mean_precision']:>8.4f} "
            f"{results['mean_recall']:>8.4f}"
        )
        print(f"\nPixel Accuracy: {results['pixel_accuracy']:.4f}")
        print("=" * 65 + "\n")
