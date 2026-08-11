#!/usr/bin/env bash
# =============================================================================
# Kaggle GPU Launcher — Arwen Policy QLoRA Training
# =============================================================================
#
# Purpose: Set up environment on a Kaggle GPU notebook, validate GPU pipeline,
#          and (when approved) launch the canonical training script.
#
# This script ONLY handles environment/setup.
# The training logic lives in: scripts/train_qlora.py
# The GPU validation lives in:  scripts/gpu_validate.py
#
# Usage (in Kaggle notebook terminal):
#     bash scripts/kaggle_train.sh validate
#     bash scripts/kaggle_train.sh train
# =============================================================================

set -euo pipefail

MODE="${1:-validate}"

echo "============================================"
echo "Arwen Policy — Kaggle GPU Launcher"
echo "Mode: ${MODE}"
echo "============================================"

# ---- 1. Verify we are in the right directory ----
if [ ! -f "pyproject.toml" ]; then
    echo "ERROR: Run from the repository root."
    exit 1
fi

# ---- 2. Ensure uv is available ----
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# ---- 3. Sync GPU dependencies ----
echo ""
echo "Installing GPU dependencies..."
uv sync --extra gpu

# ---- 4. Verify GPU is visible ----
echo ""
echo "Checking GPU..."
uv run python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {vram:.1f} GB')
else:
    print('WARNING: No GPU detected.')
"

# ---- 5. Run the requested mode ----
echo ""
echo "============================================"
if [ "$MODE" = "validate" ]; then
    echo "Running GPU validation (smoke test)..."
    echo "============================================"
    uv run python scripts/gpu_validate.py
elif [ "$MODE" = "train" ]; then
    echo "Step 1/2: Running GPU validation before training..."
    echo "============================================"
    if uv run python scripts/gpu_validate.py; then
        echo ""
        echo "============================================"
        echo "Step 2/2: GPU validation PASSED. Launching full training..."
        echo "============================================"
        uv run python scripts/train_qlora.py --epochs 20
    else
        echo ""
        echo "============================================"
        echo "ABORTED: GPU validation FAILED."
        echo "Fix the issues above before training."
        echo "============================================"
        exit 1
    fi
else
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash scripts/kaggle_train.sh [validate|train]"
    exit 1
fi
