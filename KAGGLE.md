# Kaggle GPU Notebook — Arwen Policy QLoRA Training

This runbook explains how to validate and train the Arwen Policy QLoRA pipeline
on a **Kaggle GPU notebook** using `uv` (not pip).

## Prerequisites

- Kaggle account with GPU quota (T4 ×2, L4, or A10G minimum)
- Kaggle notebook with **GPU accelerator enabled**
- Internet access enabled in the notebook

## Quick Start

### Step 1: Clone the repository

In a Kaggle notebook terminal (or a `!` cell):

```bash
git clone https://github.com/soyames/arwen-policy.git
cd arwen-policy
```

### Step 2: Install uv (if not already available)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Or use the one-liner from the launcher:

```bash
bash scripts/kaggle_train.sh validate
```

The launcher auto-installs `uv` if missing.

### Step 3: Run GPU validation

```bash
bash scripts/kaggle_train.sh validate
```

This runs `scripts/gpu_validate.py` which:

1. Checks CUDA, GPU name, VRAM
2. Loads Qwen3-8B with 4-bit NF4 quantization
3. Attaches LoRA adapters
4. Tokenizes the 304 training examples with **assistant-only label masking**
5. Creates a real training batch via the exact collator
6. Runs a forward pass with labels
7. Verifies loss is finite
8. Runs `backward()` and verifies non-zero LoRA gradients
9. Reports peak VRAM

**No optimizer step. No training. No weight updates.**

### Step 4: Review validation output

Expected output should show:

```
GPU PIPELINE VALIDATED — FULL TRAINING APPROVED
```

If the validation fails, **do not proceed** to training. Fix the issue first.

### Step 5: Launch full training (only after approved validation)

```bash
bash scripts/kaggle_train.sh train
```

This runs `scripts/train_qlora.py` — the canonical training script.

## Manual Commands

If you prefer to run each step individually:

```bash
# Install GPU dependencies only
uv sync --extra gpu

# Verify GPU
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"

# GPU validation (smoke test)
uv run python scripts/gpu_validate.py

# Full training (1 epoch, ~304 examples)
uv run python scripts/train_qlora.py
```

## Expected GPU Requirements

| GPU | VRAM | Sufficient? |
|---|---|---|
| Tesla T4 | 16 GB | ✅ (peak ~12 GB reserved) |
| T4 ×2 | 16 GB | ✅ |
| L4 | 24 GB | ✅ |
| A10G | 24 GB | ✅ |
| A100 | 40 GB | ✅ |
| P100 | 16 GB | ⚠️ (tight) |

Previous T4 run: ~11.01 GB allocated, ~12.09 GB reserved.

## Training Configuration

| Setting | Value |
|---|---|
| Model | Qwen/Qwen3-8B |
| Epochs | **20** |
| Micro batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | **8** |
| Steps per epoch | ~38 (ceil(304 / 8)) |
| Total optimizer steps | ~760 |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Max sequence length | 2048 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |

**Model selection**: The adapter from the epoch with the **best validation loss**
is used as the final adapter — not blindly epoch 20.

**Test set** (35 examples) is **completely held out** — never used for training,
evaluation during training, checkpoint selection, or early stopping.

## File Structure

```
scripts/
├── train_qlora.py       # Canonical training (provider-neutral, 20 epochs)
├── gpu_validate.py      # GPU smoke test / pipeline validation
├── verify_adapter.py    # Adapter inference and qualitative eval
├── validate_pipeline.py # CPU-side dataset+tokenization validation
└── kaggle_train.sh      # Kaggle launcher (env setup only)
```

## Troubleshooting

### CUDA out of memory
- Ensure `per_device_eval_batch_size=1` and `eval_accumulation_steps=1` (set in train_qlora.py)
- Reduce `MAX_SEQ_LENGTH` from 2048 to 1536 if needed
- Use a GPU with >= 16 GB VRAM

### uv not found
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

### bitsandbytes not available
```bash
uv sync --extra gpu
```
bitandbytes is included in the `gpu` extra.

### 95/95 tests should pass before training
```bash
uv run pytest -q
```
