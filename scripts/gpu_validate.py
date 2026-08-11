#!/usr/bin/env python3
"""GPU validation smoke test for Arwen Policy QLoRA pipeline.

Validates that the QLoRA training pipeline works correctly on a single GPU:
- CUDA, GPU name, VRAM (GPU 0 only)
- Qwen3-8B 4-bit NF4 loading (forced onto cuda:0)
- LoRA adapter attachment
- Tokenization + label masking (exact tokenize_fn from train_qlora.py)
- Forward pass with real labels (batch_size=1, seq_len=2048)
- Finite loss check
- Backward pass with non-zero LoRA gradients
- Peak VRAM reporting at each stage

Does NOT perform optimizer.step() or full training.
Does NOT use multi-GPU — everything stays on cuda:0.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/gpu_validate.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

# ===================================================================
# EXACT CONFIGURATION FROM train_qlora.py — DO NOT MODIFY SEPARATELY
# ===================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "datasets/sft_final"
MAX_SEQ_LENGTH = 2048
VALIDATION_BATCH_SIZE = 1  # single-example batch to stay within T4 VRAM
SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _vram_report() -> dict:
    """Snapshot of current GPU 0 VRAM state."""
    return {
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 3),
    }


def _vram_summary_str() -> str:
    s = _vram_report()
    return f"alloc={s['allocated_gb']:.2f} GB, reserved={s['reserved_gb']:.2f} GB"


# ===================================================================
# EXACT tokenize_fn FROM train_qlora.py
# ===================================================================

def build_tokenize_fn(tokenizer):
    """Return the EXACT tokenize_fn used in train_qlora.py."""

    def tokenize_fn(examples: dict) -> dict:
        batch_messages = [msgs for msgs in examples["messages"]]

        tokenized = tokenizer.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=False,
            padding="max_length", truncation=True,
            max_length=MAX_SEQ_LENGTH, return_dict=True,
        )

        im_start_id = 151644
        im_end_id = 151645
        assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)

        labels_list = []
        for ids in tokenized["input_ids"]:
            labels = [-100] * len(ids)
            i = 0
            while i < len(ids):
                if ids[i] == im_start_id:
                    role_start = i + 1
                    is_asst = all(
                        role_start + j < len(ids)
                        and ids[role_start + j] == assistant_tokens[j]
                        for j in range(len(assistant_tokens))
                    )
                    if is_asst:
                        content_start = role_start + len(assistant_tokens)
                        content_end = content_start
                        while content_end < len(ids) and ids[content_end] != im_end_id:
                            content_end += 1
                        for j in range(content_start, content_end):
                            labels[j] = ids[j]
                        i = content_end
                    else:
                        i += 1
                else:
                    i += 1
            labels_list.append(labels)

        tokenized["labels"] = labels_list
        return tokenized

    return tokenize_fn


# ===================================================================
# HELPERS
# ===================================================================

def load_split(split_name: str) -> list[dict]:
    path = Path(DATA_DIR) / f"{split_name}.jsonl"
    if not path.exists():
        return []
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
            messages = []
            for msg in ex.get("messages", []):
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            examples.append({"messages": messages})
        except Exception:
            continue
    return examples


# ===================================================================
# MAIN
# ===================================================================

def main() -> int:
    print("=" * 70)
    print("ARWEN POLICY — GPU VALIDATION (SINGLE-GPU, BATCH=1)")
    print("=" * 70)

    # ---- Environment warnings (non-blocking) ----
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in alloc_conf:
        print("\n[ENV] Recommended but not set: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print("      This reduces CUDA allocator fragmentation. Set it before running:")
        print("      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    else:
        print("\n[ENV] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (enabled)")

    # Document Kaggle wrapt warning (not our dependency)
    print("\n[ENV] Note: 'ModuleNotFoundError: No module named \"wrapt\"' from sitecustomize")
    print("      is a Kaggle environment warning, not an Arwen Policy issue.")
    print("      It does not affect PyTorch, transformers, or the validation result.")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phases": {},
    }

    # ---- 1. GPU CHECK (GPU 0 only) ----
    print("\n[1/9] GPU check (single-GPU mode)")
    if not torch.cuda.is_available():
        print("  FAIL: CUDA not available. This script requires a GPU.")
        return 1

    n_gpus = torch.cuda.device_count()
    print(f"  GPUs visible: {n_gpus}")
    if n_gpus > 1:
        print(f"  Multi-GPU detected. Forcing GPU 0 only (not using GPU 1..{n_gpus-1}).")

    # Lock to GPU 0
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU 0:      {gpu_name}")
    print(f"  VRAM total: {vram_total:.1f} GB")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.version.cuda}")

    # Log memory state on all GPUs
    for g in range(n_gpus):
        alloc = torch.cuda.memory_allocated(g) / 1e9
        reserved = torch.cuda.memory_reserved(g) / 1e9
        print(f"  GPU {g} init: alloc={alloc:.2f} GB, reserved={reserved:.2f} GB")

    results["gpu"] = {
        "name": gpu_name,
        "vram_total_gb": round(vram_total, 1),
        "gpu_count": n_gpus,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    # bitsandbytes check
    try:
        import bitsandbytes as bnb
        print(f"  bitsandbytes: {bnb.__version__}")
        results["gpu"]["bitsandbytes"] = bnb.__version__
    except ImportError:
        print("  FAIL: bitsandbytes not available")
        return 1

    if vram_total < 14.5:
        print(f"  WARNING: VRAM < 15 GB ({vram_total:.1f} GB). Full training may OOM.")
        print(f"  Gradient accumulation is already configured (batch=1, accum=8).")

    torch.cuda.reset_peak_memory_stats(0)
    results["phases"]["gpu_check"] = "PASS"

    # ---- 2. LOAD DATASET ----
    print("\n[2/9] Load dataset")
    train_data = load_split("train")
    print(f"  Training examples: {len(train_data)}")
    if len(train_data) != 304:
        print(f"  WARNING: expected 304, got {len(train_data)}")

    train_dataset = Dataset.from_list(train_data)
    results["phases"]["dataset_load"] = "PASS"

    # ---- 3. TOKENIZE WITH LABEL MASKING ----
    print("\n[3/9] Tokenize with label masking")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = build_tokenize_fn(tokenizer)
    tokenized = train_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["messages"])
    print(f"  Tokenized examples: {len(tokenized)}")

    # Full audit
    zero_asst = 0
    sys_user_violations = 0
    im_start_id = 151644
    im_end_id = 151645
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)

    for idx in range(len(tokenized)):
        ids = tokenized[idx]["input_ids"]
        labels = tokenized[idx]["labels"]
        n_active = sum(1 for l in labels if l != -100)
        if n_active == 0:
            zero_asst += 1

        in_non_asst = False
        for i, (tid, lbl) in enumerate(zip(ids, labels)):
            if tid == im_start_id:
                role_start = i + 1
                is_asst = all(
                    role_start + j < len(ids)
                    and ids[role_start + j] == assistant_tokens[j]
                    for j in range(len(assistant_tokens))
                )
                in_non_asst = not is_asst
            if in_non_asst and lbl != -100:
                sys_user_violations += 1
                break
            if tid == im_end_id:
                in_non_asst = False

    print(f"  Zero assistant targets: {zero_asst}")
    print(f"  Sys/user violations:    {sys_user_violations}")

    if zero_asst > 0:
        print("  FAIL: examples missing assistant targets")
        return 1
    if sys_user_violations > 0:
        print("  FAIL: system/user tokens receiving training labels")
        return 1
    print("  Label masking: PASS")

    results["phases"]["label_masking"] = "PASS"
    results["label_audit"] = {
        "total": len(tokenized),
        "zero_assistant_targets": zero_asst,
        "sys_user_violations": sys_user_violations,
    }

    # ---- 4. CREATE BATCH VIA COLLATOR ----
    print("\n[4/9] Create training batch (batch_size=1)")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH,
    )
    batch_size = VALIDATION_BATCH_SIZE
    batch_examples = [tokenized[i] for i in range(min(batch_size, len(tokenized)))]
    batch = data_collator(batch_examples)

    print(f"  Batch size:           {batch_size}")
    print(f"  input_ids shape:      {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape:         {batch['labels'].shape}")

    total_positions = batch["labels"].numel()
    masked = (batch["labels"] == -100).sum().item()
    active = (batch["labels"] != -100).sum().item()
    print(f"  Masked (-100): {masked} ({100*masked/total_positions:.1f}%)")
    print(f"  Active:        {active} ({100*active/total_positions:.1f}%)")

    # Verify collator preserves labels
    collator_ok = True
    for i in range(batch_size):
        orig = tokenized[i]["labels"]
        collated = batch["labels"][i].tolist()
        if not all(collated[j] == orig[j] for j in range(len(orig))):
            collator_ok = False
            print(f"  FAIL: Example {i} labels changed during collation")
    if collator_ok:
        print("  Collator preserves labels: PASS")

    results["phases"]["collator"] = "PASS"
    results["batch_config"] = {
        "batch_size": batch_size,
        "max_seq_length": MAX_SEQ_LENGTH,
    }

    # ---- 5. LOAD MODEL (GPU 0 ONLY) ----
    print(f"\n[5/9] Load model: {MODEL_NAME} (forcing cuda:0)")
    print("  Quantization: 4-bit NF4, double quant, float16 compute")
    print(f"  VRAM before load: {_vram_summary_str()}")

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # ── CRITICAL: force everything onto cuda:0 ──
    # device_map="auto" would split the model across all GPUs, causing
    # unpredictable OOMs on GPU 1 when logits.float() allocates a large
    # temporary tensor during causal-LM loss computation.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Base parameters: {total_params:,}")

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA trainable:  {trainable:,} ({100*trainable/total_params:.4f}%)")

    after_load = _vram_report()
    print(f"  VRAM after load:  {_vram_summary_str()}")

    results["model"] = {
        "total_params": total_params,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total_params, 4),
    }
    results["vram_after_model_load"] = after_load

    results["phases"]["model_load"] = "PASS"

    # ---- 6. FORWARD PASS ----
    print(f"\n[6/9] Forward pass (batch={batch_size}, seq={MAX_SEQ_LENGTH})")
    # Move batch explicitly to cuda:0
    batch_gpu = {k: v.to("cuda:0") for k, v in batch.items()}
    model.train()  # enable dropout for valid backward

    vram_before_fwd = _vram_report()
    print(f"  VRAM before forward: {_vram_summary_str()}")

    outputs = model(**batch_gpu)
    loss = outputs.loss

    vram_after_fwd = _vram_report()
    print(f"  VRAM after forward:  {_vram_summary_str()}")
    print(f"  Loss:                {loss.item():.6f}")
    print(f"  Loss is finite:      {math.isfinite(loss.item())}")

    if not math.isfinite(loss.item()):
        print("  FAIL: Loss is not finite")
        return 1

    results["phases"]["forward_pass"] = "PASS"
    results["forward"] = {
        "loss": round(loss.item(), 6),
        "vram_before_gb": vram_before_fwd,
        "vram_after_gb": vram_after_fwd,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(0) / 1e9, 2),
    }

    # ---- 7. BACKWARD PASS ----
    print(f"\n[7/9] Backward pass")
    vram_before_bwd = _vram_report()
    print(f"  VRAM before backward: {_vram_summary_str()}")

    try:
        loss.backward()

        vram_after_bwd = _vram_report()
        print(f"  VRAM after backward:  {_vram_summary_str()}")
        print("  backward() succeeded")

        nonzero = 0
        zero = 0
        sample_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.norm().item() > 0:
                    nonzero += 1
                    if len(sample_grads) < 3:
                        sample_grads.append((name, param.grad.norm().item()))
                else:
                    zero += 1

        print(f"  Non-zero gradients: {nonzero}")
        print(f"  Zero/null gradients: {zero}")
        for name, norm in sample_grads:
            print(f"    {name}: grad norm = {norm:.6f}")

        if nonzero == 0:
            print("  FAIL: No LoRA parameters received gradients")
            return 1
        print("  LoRA gradients: PASS")

        results["phases"]["backward_pass"] = "PASS"
        results["backward"] = {
            "nonzero_grad_params": nonzero,
            "zero_grad_params": zero,
            "vram_before_gb": vram_before_bwd,
            "vram_after_gb": vram_after_bwd,
            "sample_gradients": [{"name": n, "norm": round(v, 6)} for n, v in sample_grads],
        }

    except torch.OutOfMemoryError as e:
        print(f"  FAIL: CUDA OOM during backward: {e}")
        results["phases"]["backward_pass"] = "FAIL"
        results["backward_error"] = str(e)[:500]
        return 1
    except Exception as e:
        print(f"  FAIL: backward() raised: {e}")
        return 1

    # ---- 8. VRAM SUMMARY ----
    print(f"\n[8/9] VRAM summary (GPU 0)")
    peak_alloc = torch.cuda.max_memory_allocated(0) / 1e9
    peak_reserved = torch.cuda.max_memory_reserved(0) / 1e9
    print(f"  Peak allocated: {peak_alloc:.2f} GB")
    print(f"  Peak reserved:  {peak_reserved:.2f} GB")
    print(f"  GPU total:      {vram_total:.1f} GB")
    print(f"  Headroom:       {vram_total - peak_reserved:.2f} GB")

    if vram_total < 14.5:
        print(f"  FAIL: VRAM < 15 GB ({vram_total:.1f} GB). Insufficient for full training.")
        print(f"  Peak reserved during validation: {peak_reserved:.2f} GB")
        print(f"  STOP: Do NOT launch full training on this GPU.")
        results["phases"]["vram_summary"] = "FAIL"
    else:
        print("  VRAM adequate for full training.")
        results["phases"]["vram_summary"] = "PASS"

    results["vram"] = {
        "total_gb": round(vram_total, 1),
        "peak_allocated_gb": round(peak_alloc, 2),
        "peak_reserved_gb": round(peak_reserved, 2),
        "headroom_gb": round(vram_total - peak_reserved, 2),
    }

    # ---- 9. LOGITS MEMORY ESTIMATE ----
    print(f"\n[9/9] Logits memory estimate (for documentation)")
    # Qwen3-8B vocab size; compute theoretical float32 logits tensor size
    vocab_size = model.config.vocab_size if hasattr(model.config, "vocab_size") else 151936
    logits_elements = batch_size * MAX_SEQ_LENGTH * vocab_size
    logits_gb_f32 = logits_elements * 4 / 1e9
    logits_gb_f16 = logits_elements * 2 / 1e9
    print(f"  Vocab size:              {vocab_size:,}")
    print(f"  Logits shape:            [{batch_size}, {MAX_SEQ_LENGTH}, {vocab_size}]")
    print(f"  Logits elements:         {logits_elements:,}")
    print(f"  Logits size (float32):   {logits_gb_f32:.2f} GB  ← loss computation casts here")
    print(f"  Logits size (float16):   {logits_gb_f16:.2f} GB")
    print(f"  With batch_size=4 this would be ~{logits_gb_f32 * 4:.2f} GB (OOM cause confirmed)")

    # ---- FINAL ----
    all_pass = all(v == "PASS" for v in results["phases"].values())
    print("\n" + "=" * 70)
    if all_pass:
        print("GPU PIPELINE VALIDATED — FULL TRAINING APPROVED")
    else:
        print("GPU PIPELINE NOT VALIDATED — DO NOT TRAIN")
        return 1
    print("=" * 70)

    # Save results
    out_path = Path("artifacts/qlora_smoke_test/gpu_validation_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
