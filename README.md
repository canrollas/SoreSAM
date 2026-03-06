# SoreSAM: SAM2-Based Semantic Segmentation for Wound and Skin Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Abstract

**SoreSAM** is a semantic segmentation framework that adapts the Segment Anything Model 2 (SAM2, Meta AI Research) for clinical wound imaging. The model performs 3-class pixel-wise classification — distinguishing *background/other*, *healthy skin*, and *wound tissue* — without requiring explicit human-provided prompts at inference time.

The key novelty of SoreSAM is the introduction of **per-class learnable sparse prompt tokens** that replace hand-crafted point or bounding-box prompts. These tokens are jointly trained with the SAM2 mask decoder while the large Hiera-Large image encoder remains frozen, dramatically reducing computational cost while retaining the powerful visual representations learned during SAM2's large-scale pretraining.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Dataset Preparation](#5-dataset-preparation)
6. [Configuration](#6-configuration)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Visualisation](#9-visualisation)
10. [Loss Functions](#10-loss-functions)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Design Decisions & Ablations](#12-design-decisions--ablations)
13. [Citation](#13-citation)

---

## 1. Background & Motivation

Wound assessment is a critical yet labour-intensive task in clinical practice. Automated segmentation of wound tissue from the surrounding healthy skin enables:

- Objective wound area quantification
- Longitudinal healing tracking
- Decision support for treatment planning

Foundation models such as SAM (Kirillov *et al.*, 2023) and its successor SAM2 (Ravi *et al.*, 2024) have demonstrated remarkable zero-shot segmentation capabilities, but they require interactive prompts (points, boxes, masks) at inference time. SoreSAM eliminates this requirement by learning class-conditioned dense prompt representations during supervised fine-tuning, enabling **fully automatic, prompt-free semantic segmentation**.

---

## 2. Architecture

### 2.1 Overview

```
Input: (B, 3, 1024, 1024)  — ImageNet-normalised RGB
         │
         ▼
┌─────────────────────────────────────────┐
│  SAM2 Hiera-Large Image Encoder (frozen) │
│  + FPN neck (frozen)                     │
└───────────┬─────────────────────────────┘
            │  image_embed: (B, 256, 64, 64)
            │  high_res_feats: [(B,32,256,256), (B,64,128,128)]
            │
            ▼
  ┌──────────────────────┐
  │  For each class c:   │  ← runs 3 times (Other, Skin, Wound)
  │                      │
  │  class_tokens[c]     │  learnable (K × 256), trained from scratch
  │    + no_mask_embed   │  frozen SAM2 dense embedding
  │         │            │
  │         ▼            │
  │  SAM2 Mask Decoder   │  fine-tuned
  │  (two-way attn.)     │
  │         │            │
  │  binary_mask_c       │  (B, H_m, W_m)
  └──────────┬───────────┘
             │  stack → (B, 3, H_m, W_m)
             ▼
     Bilinear upsample to 1024×1024
             │
             ▼
Output: (B, 3, 1024, 1024) logits
```

### 2.2 Component Details

| Component | Status | Parameters |
|---|---|---|
| SAM2 Hiera-Large Encoder | **Frozen** | ~307 M |
| SAM2 Prompt Encoder | **Frozen** | ~6 M |
| SAM2 Mask Decoder | **Fine-tuned** (×0.1 LR) | ~4 M |
| Class Prompt Tokens | **Trained** (full LR) | `num_classes × K × 256` |

- **K** (`num_class_tokens`): number of learnable tokens per class (default: **4**)
- The mask decoder is fine-tuned at 1/10th the learning rate of the class tokens, a standard practice for pre-trained decoder adaptation.
- All three classes share the same decoder weights but receive **different sparse prompt embeddings**, causing the decoder to produce class-specific masks.

### 2.3 Learnable Prompt Tokens

The class token tensor has shape `(C, K, D)` where:
- `C` = number of classes (3)
- `K` = tokens per class (default 4)
- `D` = decoder embedding dimension (256 for SAM2)

Tokens are initialised with `trunc_normal_(std=0.02)`, matching ViT-style initialisation. During the forward pass, the tokens for class `c` are expanded along the batch dimension and passed as sparse prompt embeddings to the SAM2 two-way attention-based mask decoder.

---

## 3. Repository Structure

```
SoreSAM/
├── model.py            # SAM2WoundSegmenter: main model class + build_model()
├── dataset.py          # WoundDataset + colour-mask parser + augmentations
├── train.py            # Full training loop with AMP, LR schedule, checkpointing
├── evaluate.py         # Test-set evaluation script
├── metrics.py          # Confusion-matrix-based IoU, Dice, Precision, Recall
├── losses.py           # DiceLoss + CombinedSegLoss (CE + Dice)
├── visualize.py        # Label colorisation + prediction grid + training curves
├── config.py           # DataConfig, ModelConfig, TrainConfig dataclasses
├── requirements.txt    # Python dependencies
└── scripts/
    └── download_weights.sh   # Downloads SAM2.1 Hiera-Large checkpoint
```

---

## 4. Installation

### 4.1 Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU (recommended ≥ 12 GB VRAM for batch size 4 at 1024²)
- CUDA ≥ 11.8

### 4.2 Step-by-step Setup

**Step 1 — Clone this repository**
```bash
git clone https://github.com/<your-org>/SoreSAM.git
cd SoreSAM
```

**Step 2 — Install SAM2** (required dependency, not on PyPI)
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

**Step 3 — Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Download SAM2.1 Hiera-Large pretrained weights**
```bash
bash scripts/download_weights.sh
```
This downloads `sam2.1_hiera_large.pt` (~900 MB) from Meta's servers into `checkpoints/`.

> **Note:** The SAM2 config YAML (`configs/sam2.1/sam2.1_hiera_l.yaml`) is shipped with the SAM2 package. Ensure the `sam2` Python package is installed and importable before running any scripts.

---

## 5. Dataset Preparation

### 5.1 Required Directory Layout

```
data/
└── data_wound_seg_3class/
    ├── train_images/       # RGB images (*.jpg / *.png)
    ├── train_masks/        # Colour-coded RGB masks (*.png)
    ├── test_images/
    ├── test_masks/
    └── class_info.txt      # Optional: human-readable class descriptions
```

### 5.2 Mask Colour Convention

Segmentation masks use the following **RGB colour coding**:

| Class Index | Class Name | RGB Colour | Description |
|---|---|---|---|
| 0 | Other | `(0, 0, 0)` | Background / unlabelled pixels |
| 1 | Skin | `(0, 0, 255)` | Healthy skin tissue (blue) |
| 2 | Wound | `(255, 0, 0)` | Wound / lesion area (red) |

> **JPEG Tolerance:** A per-channel threshold of ±30 (configurable via `color_threshold`) is applied during colour→class conversion to handle JPEG compression artefacts.

### 5.3 Naming Convention

Image and mask files must share the same **stem** (filename without extension). Extension can differ:
- `train_images/patient_001.jpg` → `train_masks/patient_001.png` ✔
- `train_images/patient_002.png` → `train_masks/patient_002.jpg` ✔

### 5.4 Train / Validation Split

90% of `train_images/` is used for training, 10% for validation (configurable via `val_split` in `config.py`). The split is deterministic, seeded by `seed=42`.

---

## 6. Configuration

All hyperparameters are centralised in `config.py` using Python dataclasses. The global singleton `cfg` can be imported directly or overridden via command-line arguments.

### 6.1 DataConfig

| Field | Default | Description |
|---|---|---|
| `root` | `data/data_wound_seg_3class` | Dataset root directory |
| `num_classes` | `3` | Number of segmentation classes |
| `class_names` | `["Other","Skin","Wound"]` | Class labels |
| `image_size` | `1024` | Image resolution fed to SAM2 |
| `val_split` | `0.1` | Fraction of train set used for validation |
| `color_threshold` | `30` | Per-channel colour tolerance for mask parsing |

### 6.2 ModelConfig

| Field | Default | Description |
|---|---|---|
| `sam2_config` | `configs/sam2.1/sam2.1_hiera_l.yaml` | SAM2 architecture config |
| `sam2_checkpoint` | `checkpoints/sam2.1_hiera_large.pt` | SAM2 pretrained weights |
| `num_class_tokens` | `4` | Learnable prompt tokens per class |
| `freeze_image_encoder` | `True` | Freeze Hiera-Large backbone |
| `freeze_prompt_encoder` | `True` | Freeze SAM2 prompt encoder |

### 6.3 TrainConfig

| Field | Default | Description |
|---|---|---|
| `num_epochs` | `50` | Total training epochs |
| `batch_size` | `4` | Training batch size |
| `lr` | `1e-4` | Base learning rate (for class tokens) |
| `decoder_lr_multiplier` | `0.1` | LR scale for mask decoder parameters |
| `lr_scheduler` | `cosine` | LR schedule type |
| `warmup_epochs` | `2` | Linear warmup epochs |
| `ce_weight` | `1.0` | CE loss coefficient |
| `dice_weight` | `1.0` | Dice loss coefficient |
| `class_weights` | `[0.5, 1.0, 2.0]` | Per-class CE weights (Other, Skin, Wound) |
| `use_amp` | `True` | Automatic Mixed Precision |
| `grad_clip` | `1.0` | Gradient clipping norm |
| `save_best_metric` | `mean_iou` | Metric used to track best checkpoint |

---

## 7. Training

### 7.1 Basic Training

```bash
python train.py
```

### 7.2 Custom Arguments

```bash
python train.py \
    --data-root /path/to/data_wound_seg_3class \
    --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --output-dir outputs/experiment_01 \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --device cuda
```

### 7.3 Resuming from Checkpoint

```bash
python train.py --resume outputs/checkpoints/latest.pth
```

### 7.4 Outputs

| Path | Description |
|---|---|
| `outputs/checkpoints/best.pth` | Best checkpoint (highest mean IoU on validation) |
| `outputs/checkpoints/latest.pth` | Checkpoint from the last completed epoch |
| `outputs/logs/train_log.jsonl` | Per-epoch JSON training log (loss + metrics) |
| `outputs/visualizations/` | Prediction grid images saved during training |

### 7.5 Training Log Format

Each line in `train_log.jsonl` is a JSON object:
```json
{
  "epoch": 10,
  "time_s": 142.3,
  "train_loss": 0.4821,
  "train_ce": 0.2314,
  "train_dice": 0.2507,
  "val_loss": 0.3917,
  "mean_iou": 0.7345,
  "iou_Other": 0.8812,
  "iou_Skin": 0.7601,
  "iou_Wound": 0.5622
}
```

### 7.6 Learning Rate Schedule

Training employs a **cosine annealing schedule with linear warmup**:

```
Epochs 1–2:   LR linearly scales from 0.01×lr to lr  (warmup)
Epochs 3–50:  LR decays from lr to 1e-6 following cosine curve
```

Two parameter groups receive different learning rates:
- `class_tokens`: full `lr` (e.g. 1e-4)
- `mask_decoder`: `lr × decoder_lr_multiplier` (e.g. 1e-5)

---

## 8. Evaluation

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/best.pth \
    --data-root /path/to/data_wound_seg_3class \
    --device cuda \
    --batch-size 4 \
    --vis-n 8 \
    --output-dir outputs/eval
```

**Outputs:**
- `outputs/eval/results.json` — Per-class and mean metrics as JSON
- `outputs/eval/test_predictions.png` — Prediction grid: original image | ground truth | prediction

**Sample `results.json`:**
```json
{
  "iou_Other": 0.91,
  "iou_Skin": 0.83,
  "iou_Wound": 0.74,
  "mean_iou": 0.83,
  "mean_dice": 0.89,
  "mean_precision": 0.87,
  "mean_recall": 0.91,
  "pixel_accuracy": 0.92
}
```

---

## 9. Visualisation

### 9.1 Training Curves

```python
from visualize import plot_training_curves

plot_training_curves(
    log_path="outputs/logs/train_log.jsonl",
    save_dir="outputs/plots"
)
```
Generates `outputs/plots/training_curves.png` with loss and mean IoU curves.

### 9.2 Prediction Grid

```python
from visualize import save_prediction_grid

save_prediction_grid(
    images=image_batch,   # (N, 3, H, W) tensor
    preds=pred_batch,     # (N, H, W) int64 tensor
    labels=label_batch,   # (N, H, W) int64 tensor
    save_path="outputs/predictions.png",
    class_names=["Other", "Skin", "Wound"],
    alpha=0.45
)
```

Each row shows: **Raw Image | Ground Truth Overlay | Prediction Overlay**

---

## 10. Loss Functions

### 10.1 Combined Loss

$$\mathcal{L}_{\text{total}} = w_{\text{CE}} \cdot \mathcal{L}_{\text{CE}} + w_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}}$$

Default weights: $w_{\text{CE}} = 1.0$, $w_{\text{Dice}} = 1.0$

### 10.2 Cross-Entropy Loss

Weighted cross-entropy is applied to handle class imbalance, with per-class weights:
- Other: 0.5 (dominant background class)
- Skin: 1.0
- Wound: 2.0 (clinically important minority class)

### 10.3 Soft Dice Loss

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot |P_c \cap G_c| + \varepsilon}{|P_c| + |G_c| + \varepsilon}$$

Where $P_c$ and $G_c$ are the predicted probability map and ground-truth binary mask for class $c$, and $\varepsilon = 1.0$ is a Laplace smoothing term. Only classes present in the batch contribute to the mean, preventing NaN gradients for absent classes.

---

## 11. Evaluation Metrics

All metrics are computed from the accumulated full-dataset confusion matrix.

| Metric | Formula |
|---|---|
| **IoU (Jaccard)** | $\text{TP} / (\text{TP} + \text{FP} + \text{FN})$ |
| **Dice (F1)** | $2\text{TP} / (2\text{TP} + \text{FP} + \text{FN})$ |
| **Precision** | $\text{TP} / (\text{TP} + \text{FP})$ |
| **Recall** | $\text{TP} / (\text{TP} + \text{FN})$ |
| **Pixel Accuracy** | $\sum_c \text{TP}_c / N_{\text{total}}$ |

**Mean IoU (mIoU)** is the primary benchmark metric and is used to select the best checkpoint. All averages are macro (NaN-safe nanmean), so absent classes do not penalise the score.

---

## 12. Design Decisions & Ablations

### Why freeze the image encoder?
The SAM2 Hiera-Large encoder contains ~307 M parameters. Fine-tuning it end-to-end requires large GPU memory and risks destroying the rich universal representations learnt during pretraining. Freezing the encoder allows training on consumer-grade hardware while still achieving strong downstream performance.

### Why learnable prompt tokens instead of actual prompts?
Manual prompting (e.g., clicking on wound centres) introduces inter-annotator variability and is not scalable for automated deployment. Per-class tokens directly optimise for segmentation loss, **acting as soft, learned class queries** analogous to query tokens in DETR-family object detectors.

### Why a combined CE + Dice loss?
Cross-Entropy treats each pixel independently and is sensitive to class imbalance. Dice loss directly optimises the overlap metric, making it robust to imbalance but prone to instability when class presence is low. Their combination offers complementary gradient signals.

### Why separate LR for tokens vs. decoder?
Class tokens are randomly initialised and must be learned aggressively. The mask decoder starts from a good pre-trained state and should change more conservatively. A 10× LR multiplier difference prevents catastrophic forgetting in the decoder while allowing fast token learning.

---

## 13. Citation

If you use SoreSAM in your research, please cite:

```bibtex
@software{soresam2024,
  title   = {SoreSAM: SAM2-Based Semantic Segmentation for Wound and Skin Analysis},
  year    = {2024},
  url     = {https://github.com/<your-org>/SoreSAM}
}
```

Please also cite the original SAM2 paper:

```bibtex
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and others},
  journal = {arXiv preprint arXiv:2408.00714},
  year    = {2024}
}
```

---

## Acknowledgements

This project builds upon [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) by Meta AI Research. We thank the SAM2 team for releasing their pretrained models under a permissive licence.
