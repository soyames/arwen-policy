#!/bin/bash
# Arwen Policy — Full QLoRA Training on Lightning T4
# 1 epoch, validation, checkpointing, best model, adapter save, eval
set -euo pipefail

echo "============================================"
echo "ARWEN POLICY — QLORA TRAINING (T4)"
echo "============================================"

# ---- Setup ----
cd arwen-policy
git pull origin main
git log --oneline -3

uv sync --extra gpu

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo ""
echo "=== Dataset integrity ==="
uv run python -c "
import json
from pathlib import Path
for s in ('train','validation','test'):
    p = Path('datasets/sft_final') / f'{s}.jsonl'
    n = len([l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]) if p.exists() else 0
    print(f'  {s}: {n}')
"

echo ""
echo "=== Running full training (1 epoch) ==="
uv run python scripts/train_qlora.py

echo ""
echo "=== Training complete ==="
echo "Check artifacts/qlora_arwen_8b/training_report.json"
ls -la artifacts/qlora_arwen_8b/
