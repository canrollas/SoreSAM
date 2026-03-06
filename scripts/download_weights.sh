#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download SAM2.1 Hiera-Large pretrained weights
# ---------------------------------------------------------------------------
set -euo pipefail

CHECKPOINT_DIR="$(dirname "$0")/../checkpoints"
mkdir -p "$CHECKPOINT_DIR"

MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
MODEL_FILE="$CHECKPOINT_DIR/sam2.1_hiera_large.pt"

if [ -f "$MODEL_FILE" ]; then
    echo "[OK] Checkpoint already exists: $MODEL_FILE"
    exit 0
fi

echo "[Download] SAM2.1 Hiera-Large → $MODEL_FILE"
curl -L --progress-bar "$MODEL_URL" -o "$MODEL_FILE"
echo "[Done] Checkpoint saved to $MODEL_FILE"
