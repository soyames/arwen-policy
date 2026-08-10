#!/bin/bash
# Arwen Policy — Lightning GPU QLoRA Smoke Test
# Copy-paste into a Lightning GPU Studio terminal.
set -euo pipefail

echo "============================================"
echo "ARWEN POLICY — LIGHTNING GPU QLORA SMOKE TEST"
echo "============================================"

# ---- 1. Verify hardware ----
echo ""
echo "=== 1. GPU VERIFICATION ==="
nvidia-smi
echo ""
echo "uv version: $(uv --version)"
echo "git version: $(git --version)"

# ---- 2. Clone repository ----
echo ""
echo "=== 2. CLONE REPOSITORY ==="
if [ ! -d arwen-policy ]; then
    git clone https://github.com/soyames/arwen-policy.git
fi
cd arwen-policy
git log --oneline -5
git status

# ---- 3. Install dependencies with uv ----
echo ""
echo "=== 3. UV SYNC (GPU) ==="
uv sync --extra gpu
echo "  Done."

# ---- 4. Verify GPU PyTorch ----
echo ""
echo "=== 4. VERIFY GPU PYTORCH ==="
uv run python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('vram:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
else:
    print('ERROR: CUDA not available — cannot run smoke test')
    exit(1)
print('torch OK')
"

# ---- 5. Verify all training deps ----
echo ""
echo "=== 5. VERIFY TRAINING DEPENDENCIES ==="
uv run python -c "
import torch, transformers, peft, bitsandbytes, accelerate, datasets, trl
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('accelerate:', accelerate.__version__)
print('datasets:', datasets.__version__)
print('trl:', trl.__version__)
print('All imports OK')
"

# ---- 6. GPU memory before ----
echo ""
echo "=== 6. GPU MEMORY BEFORE MODEL LOAD ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# ---- 7. Run the QLoRA smoke test ----
echo ""
echo "=== 7. QLORA SMOKE TEST ==="
uv run python scripts/qlora_smoke_test.py \
    --model Qwen/Qwen3-8B \
    --data-dir datasets/sft_final \
    --output-dir artifacts/qlora_smoke_test_8b \
    --max-steps 2

# For larger GPUs (A100/L40S), use instead:
# uv run python scripts/qlora_smoke_test.py --model Qwen/Qwen3.6-27B

# ---- 8. GPU memory after ----
echo ""
echo "=== 8. GPU MEMORY AFTER SMOKE TEST ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# ---- 9. Verify adapter artifacts ----
echo ""
echo "=== 9. VERIFY ADAPTER ARTIFACTS ==="
ls -la artifacts/qlora_smoke_test/adapter_* 2>/dev/null || echo "  (adapter saved by PEFT)"
ls -la artifacts/qlora_smoke_test/

# ---- 10. Verify dataset integrity ----
echo ""
echo "=== 10. DATASET INTEGRITY ==="
uv run python -c "
import json
from pathlib import Path
for split in ('train', 'validation', 'test'):
    path = Path('datasets/sft_final') / f'{split}.jsonl'
    if path.exists():
        lines = [l for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        print(f'  {split}: {len(lines)}')
    else:
        print(f'  {split}: MISSING')
"

echo ""
echo "============================================"
echo "SMOKE TEST COMPLETE"
echo "============================================"
echo "Check artifacts/qlora_smoke_test/smoke_test_metadata.json for results."
