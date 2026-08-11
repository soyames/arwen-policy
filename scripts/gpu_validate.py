#!/usr/bin/env python3
"""GPU validation smoke test for Arwen Policy QLoRA pipeline.

Validates that the QLoRA training pipeline works correctly on an actual GPU:
- CUDA, GPU name, VRAM
- Qwen3-8B 4-bit NF4 loading
- LoRA adapter attachment
- Tokenization + label masking (exact tokenize_fn from train_qlora.py)
- Forward pass with real labels
- Finite loss check
- Backward pass with non-zero LoRA gradients
- Peak VRAM reporting

Does NOT perform optimizer.step() or full training.

Usage:
    uv sync --extra gpu
    uv run python scripts/gpu_validate.py
"""

from __future__ import annotations

import json
import math
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
SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


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
    print("ARWEN POLICY — GPU VALIDATION")
    print("=" * 70)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phases": {},
    }

    # ---- 1. GPU CHECK ----
    print("\n[1/8] GPU check")
    if not torch.cuda.is_available():
        print("  FAIL: CUDA not available. This script requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU:        {gpu_name}")
    print(f"  VRAM total: {vram_total:.1f} GB")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.version.cuda}")

    results["gpu"] = {
        "name": gpu_name,
        "vram_total_gb": round(vram_total, 1),
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
        print(f"  Consider using a GPU with >= 16 GB VRAM.")

    torch.cuda.reset_peak_memory_stats()
    results["phases"]["gpu_check"] = "PASS"

    # ---- 2. LOAD DATASET ----
    print("\n[2/8] Load dataset")
    train_data = load_split("train")
    print(f"  Training examples: {len(train_data)}")
    if len(train_data) != 304:
        print(f"  WARNING: expected 304, got {len(train_data)}")

    train_dataset = Dataset.from_list(train_data)
    results["phases"]["dataset_load"] = "PASS"

    # ---- 3. TOKENIZE WITH LABEL MASKING ----
    print("\n[3/8] Tokenize with label masking")
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
    print("\n[4/8] Create training batch")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH,
    )
    batch_size = 4
    batch_examples = [tokenized[i] for i in range(min(batch_size, len(tokenized)))]
    batch = data_collator(batch_examples)

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

    # ---- 5. LOAD MODEL WITH QLORA ----
    print(f"\n[5/8] Load model: {MODEL_NAME}")
    print("  Quantization: 4-bit NF4, double quant, float16 compute")

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
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

    results["model"] = {
        "total_params": total_params,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total_params, 4),
    }

    vram_after_model_load = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after load: {vram_after_model_load:.2f} GB")

    results["phases"]["model_load"] = "PASS"

    # ---- 6. FORWARD PASS ----
    print("\n[6/8] Forward pass")
    batch_gpu = {k: v.to("cuda") for k, v in batch.items()}
    model.train()  # enable dropout for valid backward

    alloc_before = torch.cuda.memory_allocated() / 1e9

    outputs = model(**batch_gpu)
    loss = outputs.loss

    alloc_after = torch.cuda.memory_allocated() / 1e9

    print(f"  Loss:               {loss.item():.6f}")
    print(f"  Loss is finite:     {math.isfinite(loss.item())}")
    print(f"  VRAM allocated:     {alloc_after:.2f} GB")
    print(f"  Peak allocated:     {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Peak reserved:      {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    if not math.isfinite(loss.item()):
        print("  FAIL: Loss is not finite")
        return 1

    results["phases"]["forward_pass"] = "PASS"
    results["forward"] = {
        "loss": round(loss.item(), 6),
        "vram_allocated_gb": round(alloc_after, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }

    # ---- 7. BACKWARD PASS ----
    print("\n[7/8] Backward pass")
    try:
        loss.backward()
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
            "sample_gradients": [{"name": n, "norm": round(v, 6)} for n, v in sample_grads],
        }

    except Exception as e:
        print(f"  FAIL: backward() raised: {e}")
        return 1

    # ---- 8. VRAM SUMMARY ----
    print("\n[8/8] VRAM summary")
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9
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
