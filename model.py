"""
SoreSAM - SAM2-based Wound and Skin Segmentation
Model: SAM2 fine-tuning for 3-class semantic segmentation.

Architecture overview
---------------------
1. SAM2 Hiera-Large image encoder   (frozen by default)
2. SAM2 mask decoder                (fine-tuned)
3. Per-class learnable sparse prompt tokens
   - NUM_CLASS_TOKENS tokens per class replace hand-crafted prompts
   - Trained from scratch; everything else in the decoder is pre-trained

Forward pass
------------
  images (B, 3, 1024, 1024)
      │
      ▼
  image_encoder → backbone_features
      │
      ▼
  [for each class c]:
      prompt_tokens[c] + dense_no_mask_embed → mask_decoder → binary_mask_c
      │
      ▼
  stack → (B, num_classes, H, W) logits
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sam2.build_sam import build_sam2
    from sam2.modeling.sam2_base import SAM2Base
except ImportError as e:
    raise ImportError(
        "SAM2 is not installed. Please install it from: "
        "https://github.com/facebookresearch/sam2\n"
        f"Original error: {e}"
    )


class SAM2WoundSegmenter(nn.Module):
    """
    SAM2-based 3-class semantic segmentation model for wound images.

    Args:
        sam2_config:          Path to the SAM2 YAML config file.
        sam2_checkpoint:      Path to the SAM2 pretrained weights.
        num_classes:          Number of segmentation classes (default: 3).
        num_class_tokens:     Learnable prompt tokens per class (default: 4).
        freeze_image_encoder: Freeze the SAM2 image encoder (default: True).
        freeze_prompt_encoder: Freeze the SAM2 prompt encoder (default: True).
        image_size:           Input image resolution (default: 1024).
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_checkpoint: str,
        num_classes: int = 3,
        num_class_tokens: int = 4,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        image_size: int = 1024,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_class_tokens = num_class_tokens
        self.image_size = image_size

        # ------------------------------------------------------------------
        # Load SAM2 and extract sub-modules
        # ------------------------------------------------------------------
        sam2: SAM2Base = build_sam2(sam2_config, sam2_checkpoint)

        self.image_encoder = sam2.image_encoder
        self.prompt_encoder = sam2.sam_prompt_encoder
        self.mask_decoder = sam2.sam_mask_decoder

        # ------------------------------------------------------------------
        # Selective freezing
        # ------------------------------------------------------------------
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

        if freeze_prompt_encoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad_(False)

        # ------------------------------------------------------------------
        # Learnable class-specific sparse prompt tokens
        # Shape: (num_classes, num_class_tokens, embed_dim)
        # These replace hand-crafted point/box prompts at training and inference.
        # ------------------------------------------------------------------
        embed_dim: int = self.prompt_encoder.embed_dim  # 256 for SAM2
        self.class_tokens = nn.Parameter(
            torch.empty(num_classes, num_class_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.class_tokens, std=0.02)

        # Cache the spatial size of the image embedding (set in first forward)
        self._embed_h: Optional[int] = None
        self._embed_w: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run the SAM2 image encoder + FPN neck.

        Returns
        -------
        image_embed : (B, C, H_e, W_e)  — coarsest feature map (for attention)
        high_res_feats : list[(B, C_i, H_i, W_i)]  — finer feature maps
        """
        # image_encoder is a Hiera backbone; output is a dict:
        #   {"vision_features": ..., "vision_pos_enc": [...], "backbone_fpn": [...]}
        backbone_out = self.image_encoder(images)

        # The FPN neck is attached to the mask decoder in SAM2
        fpn_features: List[torch.Tensor] = self.mask_decoder.neck(backbone_out)
        # fpn_features is a list ordered coarse → fine (or fine → coarse).
        # SAM2 convention: index -1 is the coarsest (64×64 for 1024 input),
        # indices [:-1] are high-res feature maps.

        image_embed = fpn_features[-1]          # (B, 256, 64, 64)
        high_res_feats = fpn_features[:-1]      # [(B,32,256,256),(B,64,128,128)]

        self._embed_h, self._embed_w = image_embed.shape[-2:]
        return image_embed, high_res_feats

    def _decode_class(
        self,
        image_embed: torch.Tensor,
        high_res_feats: List[torch.Tensor],
        class_idx: int,
    ) -> torch.Tensor:
        """
        Decode a binary mask for one class using class-specific prompt tokens.

        Returns
        -------
        mask : (B, H_e*4, W_e*4) — upsampled by SAM2 mask decoder
        """
        B = image_embed.shape[0]
        H_e, W_e = self._embed_h, self._embed_w

        # ---- Sparse prompt: class tokens ----------------------------------
        # (num_class_tokens, D) → (B, num_class_tokens, D)
        sparse_emb = self.class_tokens[class_idx].unsqueeze(0).expand(B, -1, -1)

        # ---- Dense prompt: no-mask embedding ------------------------------
        dense_emb = self.prompt_encoder.no_mask_embed.weight  # (1, D)
        dense_emb = dense_emb.reshape(1, -1, 1, 1).expand(B, -1, H_e, W_e)

        # ---- Positional encoding for the image embedding ------------------
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, D, H_e, W_e)

        # ---- Mask decoder -------------------------------------------------
        # Returns: (low_res_masks, iou_preds, sam_tokens_out, obj_score_logits)
        low_res_masks, _, _, _ = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_feats,
        )
        # low_res_masks: (B, 1, H_m, W_m)  — H_m ≈ 256 with high-res feats
        return low_res_masks.squeeze(1)  # (B, H_m, W_m)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        images : (B, 3, 1024, 1024) — normalised with ImageNet stats

        Returns
        -------
        logits : (B, num_classes, 1024, 1024) — raw logits (before softmax)
        """
        # 1. Encode image once (shared across all classes)
        image_embed, high_res_feats = self._encode_image(images)

        # 2. Decode one binary mask per class
        class_masks: List[torch.Tensor] = [
            self._decode_class(image_embed, high_res_feats, c)
            for c in range(self.num_classes)
        ]

        # 3. Stack into multi-class logit tensor: (B, num_classes, H_m, W_m)
        logits = torch.stack(class_masks, dim=1)

        # 4. Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return logits

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_groups(self, lr: float, decoder_lr_mult: float = 0.1) -> List[dict]:
        """
        Return parameter groups for the optimiser:
          - class_tokens:   full LR  (learned from scratch)
          - mask_decoder:   reduced LR (fine-tuned from pretrained)
        """
        decoder_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and "class_tokens" not in n
        ]
        return [
            {"params": [self.class_tokens], "lr": lr, "name": "class_tokens"},
            {"params": decoder_params,      "lr": lr * decoder_lr_mult, "name": "mask_decoder"},
        ]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = self.num_trainable_params()
        return (
            f"SAM2WoundSegmenter("
            f"num_classes={self.num_classes}, "
            f"num_class_tokens={self.num_class_tokens}, "
            f"total_params={total:,}, "
            f"trainable_params={trainable:,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(
    sam2_config: str,
    sam2_checkpoint: str,
    num_classes: int = 3,
    num_class_tokens: int = 4,
    freeze_image_encoder: bool = True,
    freeze_prompt_encoder: bool = True,
    device: str = "cuda",
) -> SAM2WoundSegmenter:
    model = SAM2WoundSegmenter(
        sam2_config=sam2_config,
        sam2_checkpoint=sam2_checkpoint,
        num_classes=num_classes,
        num_class_tokens=num_class_tokens,
        freeze_image_encoder=freeze_image_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder,
    )
    model = model.to(device)
    print(model)
    return model
