#!/usr/bin/env bash
# =============================================================================
# Kaggle GPU Launcher — Arwen Policy QLoRA Training
# =============================================================================
#
# Purpose: Set up environment on a Kaggle GPU notebook, run preflight gates,
#          validate GPU pipeline, launch training, and evaluate results.
#
# This script ONLY handles environment/setup/diagnostics.
# The training logic lives in:   scripts/train_qlora.py
# The GPU validation lives in:   scripts/gpu_validate.py
# The preflight gates live in:   scripts/preflight.py
#
# Usage (in Kaggle notebook terminal):
#     bash scripts/kaggle_train.sh validate
#     bash scripts/kaggle_train.sh train
#     bash scripts/kaggle_train.sh evaluate
# =============================================================================

set -euo pipefail

# ===========================================================================
# SINGLE-GPU ENFORCEMENT — MUST be set before ANY Python/PyTorch import
# ===========================================================================
# The previous Kaggle run silently halved training steps because the
# HuggingFace Trainer detected 2 GPUs (n_gpu=2) even though the model was
# placed on cuda:0.  CUDA_VISIBLE_DEVICES=0 forces PyTorch to see exactly
# one GPU, ensuring Trainer.world_size = Trainer.n_gpu = 1.
export CUDA_VISIBLE_DEVICES=0

# Reduce CUDA allocator fragmentation — critical for T4 with seq_len=2048
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

# ---- 2. Print environment ----
echo ""
echo "=== Environment ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'(unset)'}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

# ---- 3. Disk check ----
echo ""
echo "=== Disk usage (pre-flight) ==="
df -h /kaggle/working 2>/dev/null || df -h . || true
echo ""
AVAIL_KB=$(df --output=avail /kaggle/working 2>/dev/null | tail -1 | tr -d ' ' || echo 0)
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "${AVAIL_KB}" -gt 0 ] 2>/dev/null && [ "${AVAIL_GB}" -lt 5 ]; then
    echo "WARNING: Less than 5 GB available on /kaggle/working (${AVAIL_GB} GB)."
    echo "Training requires approximately 3-4 GB for artifacts plus model cache."
fi

# ---- 4. Ensure uv is available ----
if ! command -v uv &> /dev/null; then
    echo ""
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.cargo/env" 2>/dev/null || true
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo ""
echo "uv version: $(uv --version)"

# ---- 5. Sync GPU dependencies ----
echo ""
echo "Installing GPU dependencies..."
uv sync --extra gpu

# ---- 6. Verify GPU visibility ----
echo ""
echo "=== GPU visibility check ==="
echo "Note: 'ModuleNotFoundError: No module named \"wrapt\"' is a Kaggle"
echo "      sitecustomize warning — not an Arwen Policy issue. Ignore it."
uv run python -c "
import os
import torch

print(f'CUDA_VISIBLE_DEVICES: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"(unset)\")}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count (must be 1): {torch.cuda.device_count()}')

if torch.cuda.device_count() != 1:
    print('')
    print('=' * 60)
    print('PREFLIGHT FAILED: Expected exactly 1 visible GPU,')
    print(f'got {torch.cuda.device_count()}.')
    print('CUDA_VISIBLE_DEVICES=0 must be set before this script runs.')
    print('Do NOT launch training with multiple visible GPUs.')
    print('=' * 60)
    raise SystemExit(1)

torch.cuda.set_device(0)
print(f'Current device: {torch.cuda.current_device()}')
print(f'GPU name: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU total memory: {vram:.1f} GB')

if vram < 14.5:
    print('')
    print('=' * 60)
    print(f'WARNING: GPU VRAM = {vram:.1f} GB (< 15 GB).')
    print('Full training may OOM on this GPU.')
    print('=' * 60)

print('Single-GPU check: PASS')
"

# ---- 7. Run the requested mode ----
echo ""
echo "============================================"
if [ "$MODE" = "validate" ]; then
    echo "Running preflight validation..."
    echo "============================================"
    echo ""
    echo "--- Step 1/3: CPU/dataset preflight ---"
    uv run python scripts/preflight.py --cpu-only
    echo ""
    echo "--- Step 2/3: GPU smoke test ---"
    uv run python scripts/gpu_validate.py
    echo ""
    echo "--- Step 3/3: Trainer configuration preflight ---"
    uv run python scripts/preflight.py --trainer-config
    echo ""
    echo "============================================"
    echo "ALL PREFLIGHT GATES PASSED"
    echo "Ready for training."
    echo "============================================"

elif [ "$MODE" = "train" ]; then
    echo "Step 1/3: Running preflight validation..."
    echo "============================================"
    if ! uv run python scripts/preflight.py --cpu-only; then
        echo "PREFLIGHT FAILED — aborting training."
        exit 1
    fi
    if ! uv run python scripts/gpu_validate.py; then
        echo "GPU VALIDATION FAILED — aborting training."
        exit 1
    fi
    if ! uv run python scripts/preflight.py --trainer-config; then
        echo "TRAINER CONFIG VALIDATION FAILED — aborting training."
        exit 1
    fi
    echo ""
    echo "============================================"
    echo "Step 2/3: All gates passed. Launching training..."
    echo "============================================"
    uv run python scripts/train_qlora.py --epochs 20
    echo ""
    echo "============================================"
    echo "Step 3/3: Running test-set evaluation..."
    echo "============================================"
    BEST_CKPT=$(ls -d artifacts/qlora_arwen_8b/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "")
    if [ -n "$BEST_CKPT" ]; then
        uv run python scripts/eval_test_set.py --adapter-path "$BEST_CKPT" --loss-only
    else
        echo "WARNING: No checkpoint found for evaluation."
    fi

elif [ "$MODE" = "evaluate" ]; then
    echo "Running test-set evaluation..."
    echo "============================================"
    ADAPTER="${2:-artifacts/qlora_arwen_8b/final_adapter}"
    if [ -d "$ADAPTER" ]; then
        uv run python scripts/eval_test_set.py --adapter-path "$ADAPTER" --full
    else
        echo "ERROR: Adapter not found at $ADAPTER"
        echo "Usage: bash scripts/kaggle_train.sh evaluate [adapter-path]"
        exit 1
    fi

else
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash scripts/kaggle_train.sh [validate|train|evaluate]"
    exit 1
fi
