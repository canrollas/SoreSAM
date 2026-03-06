"""
SoreSAM - Loss functions.

Combined segmentation loss: weighted Cross-Entropy + Dice.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class segmentation.

    Computes per-class Dice and averages over classes (macro Dice).
    Ignores classes whose ground-truth is absent in the batch (prevents NaN).

    Args:
        smooth:           Laplace smoothing to avoid division by zero.
        ignore_index:     Class index to exclude from the loss (optional).
        from_logits:      If True, apply softmax to predictions first.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(
        self,
        logits: torch.Tensor,   # (B, C, H, W)
        targets: torch.Tensor,  # (B, H, W)  int64
    ) -> torch.Tensor:
        num_classes = logits.shape[1]

        if self.from_logits:
            probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        else:
            probs = logits

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes)     # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Mask out ignored class if specified
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # Compute Dice per class
        dims = (0, 2, 3)  # reduce over batch, H, W
        intersection = (probs * targets_one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Only average over classes present in the batch
        present = targets_one_hot.sum(dim=dims) > 0
        if present.any():
            dice_loss = 1.0 - dice_per_class[present].mean()
        else:
            dice_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return dice_loss


class CombinedSegLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice loss.

    Loss = ce_weight * CE + dice_weight * Dice

    Args:
        num_classes:   Number of segmentation classes.
        ce_weight:     Scalar weight for the CE term.
        dice_weight:   Scalar weight for the Dice term.
        class_weights: Optional per-class weight vector for CE
                       (useful when classes are imbalanced).
        ignore_index:  Label index to ignore (e.g. -1 for unlabelled pixels).
    """

    def __init__(
        self,
        num_classes: int = 3,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

        weight_tensor: Optional[torch.Tensor] = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.ce = nn.CrossEntropyLoss(
            weight=weight_tensor,
            ignore_index=ignore_index,
        )
        self.dice = DiceLoss(
            smooth=1.0,
            ignore_index=ignore_index if ignore_index >= 0 else None,
            from_logits=True,
        )

    def forward(
        self,
        logits: torch.Tensor,   # (B, C, H, W)
        targets: torch.Tensor,  # (B, H, W)  int64
    ) -> tuple[torch.Tensor, dict]:
        # Move class_weights to same device as logits (handles device changes)
        if self.ce.weight is not None:
            self.ce.weight = self.ce.weight.to(logits.device)

        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        loss_dict = {
            "loss": total,
            "loss_ce": ce_loss,
            "loss_dice": dice_loss,
        }
        return total, loss_dict
