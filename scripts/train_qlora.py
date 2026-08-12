#!/usr/bin/env python3
"""Full QLoRA training for Arwen Policy on Qwen3-8B.

Runs 20 epochs, evaluates on validation at every epoch, saves
epoch checkpoints, tracks best validation loss, and produces the
final adapter from the best checkpoint (not blind last epoch).

Usage:
    uv sync --extra gpu
    uv run python scripts/train_qlora.py [--epochs 20]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ===================================================================
# CONFIGURATION
# ===================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "datasets/sft_final"
OUTPUT_DIR = "artifacts/qlora_arwen_8b"
MAX_SEQ_LENGTH = 2048
SEED = 42

# Training
NUM_EPOCHS = 20
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# Effective batch size = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 8
# Steps per epoch ≈ ceil(N / 8) — calculated from actual dataset size at runtime
# Total steps ≈ 20 * steps_per_epoch — computed dynamically, not hardcoded

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Checkpointing — per-epoch evaluation and save
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2  # keep only best + latest; no optimizer accumulation needed


# ===================================================================
# MAIN
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA training for Arwen Policy")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in the output directory")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume from a specific checkpoint path")
    args = parser.parse_args()

    start_time = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # Determine resume checkpoint
    resume_from_checkpoint = None
    if args.resume_from:
        resume_from_checkpoint = args.resume_from
        if not Path(resume_from_checkpoint).exists():
            print(f"ERROR: Resume checkpoint not found: {resume_from_checkpoint}")
            return 1
        print(f"Resume requested from: {resume_from_checkpoint}")
    elif args.resume:
        # Auto-detect latest checkpoint
        existing = sorted(out.glob("checkpoint-*"))
        if existing:
            resume_from_checkpoint = str(existing[-1])
            print(f"Resume auto-detected: {resume_from_checkpoint}")
        else:
            print("WARNING: --resume specified but no checkpoint found. Starting fresh.")

    num_epochs = args.epochs
    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    # ---- 1. GPU check ----
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return 1

    torch.cuda.set_device(0)  # lock to GPU 0 — no multi-GPU on dual-T4 Kaggle

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}")

    # ---- 1b. Single-GPU assertion ----
    n_gpu_visible = torch.cuda.device_count()
    print(f"Visible GPUs: {n_gpu_visible}")
    if n_gpu_visible != 1:
        print("=" * 60)
        print("CONFIGURATION ERROR: Expected exactly 1 visible GPU,"
              f" got {n_gpu_visible}.")
        print("Set CUDA_VISIBLE_DEVICES=0 before launching this script.")
        print("The previous Kaggle run silently halved training steps")
        print("because Trainer detected 2 GPUs (n_gpu=2).")
        print("Training ABORTED.")
        print("=" * 60)
        return 1

    # ---- 1c. Disk check ----
    import shutil
    disk = shutil.disk_usage(OUTPUT_DIR if Path(OUTPUT_DIR).exists() else ".")
    free_gb = disk.free / 1e9
    print(f"Disk free: {free_gb:.1f} GB")
    if free_gb < 3.0:
        print("=" * 60)
        print(f"WARNING: Only {free_gb:.1f} GB free. Training may fail.")
        print("Expected need: ~1.7 GB checkpoints + model cache.")
        print("=" * 60)

    # ---- 2. Load dataset ----
    print("\n=== Loading dataset ===")
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

    train_data = load_split("train")
    val_data = load_split("validation")
    test_data = load_split("test")  # held out, not used in training

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) < 300:
        print(f"WARNING: Expected >=300 train examples, got {len(train_data)}")

    # Compute expected steps from actual dataset size
    n_train = len(train_data)
    steps_per_epoch = math.ceil(n_train / effective_batch)
    total_steps_expected = steps_per_epoch * num_epochs
    print(f"Effective batch size: {effective_batch}")
    print(f"Steps per epoch:      {steps_per_epoch} (ceil({n_train} / {effective_batch}))")
    print(f"Expected total steps: {total_steps_expected} ({num_epochs} epochs × {steps_per_epoch} steps)")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    # ---- 3. Tokenizer ----
    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples: dict) -> dict:
        """Tokenize chat messages and create label masks.

        apply_chat_template() produces input_ids but no labels.
        We create labels by masking system/user tokens with -100,
        keeping only assistant content tokens as training targets.
        """
        batch_messages = [msgs for msgs in examples["messages"]]

        tokenized = tokenizer.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=False,
            truncation=True, max_length=MAX_SEQ_LENGTH, return_dict=True,
        )

        # Build labels: mask everything except assistant content
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

    train_tokenized = train_dataset.map(tokenize_fn, batched=True)
    train_tokenized = train_tokenized.remove_columns(["messages"])
    print(f"Train tokenized: {len(train_tokenized)}")

    if val_dataset is not None:
        val_tokenized = val_dataset.map(tokenize_fn, batched=True)
        val_tokenized = val_tokenized.remove_columns(["messages"])
        print(f"Val tokenized: {len(val_tokenized)}")
    else:
        val_tokenized = None

    # ---- 4. Model with QLoRA ----
    print(f"\n=== Loading model: {MODEL_NAME} ===")
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
        device_map={"": "cuda:0"},  # single-GPU only — prevents OOM on dual-T4 Kaggle
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # ---- 5. LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} ({100 * trainable / total_params:.4f}%)")

    # ---- 6. Training arguments ----
    # Check if save_only_model is supported (transformers >= 4.46)
    import transformers as _tf
    _tf_version = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _save_only_supported = _tf_version >= (4, 46)
    if _save_only_supported:
        print(f"transformers {_tf.__version__}: save_only_model=True supported")
    else:
        print(f"transformers {_tf.__version__}: save_only_model NOT supported, "
              "checkpoints will include optimizer state")

    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        save_only_model=_save_only_supported,  # skip optimizer.pt per checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        seed=SEED,
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0,
    )

    from transformers import DataCollatorForSeq2Seq, TrainerCallback
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    # Per-epoch logging callback
    epoch_log: list[dict] = []
    epoch_start_time = [time.time()]
    best_val_so_far = [float("inf")]

    class EpochLoggerCallback(TrainerCallback):
        def on_epoch_end(self, trainer_args, state, control, **kwargs):
            epoch_idx = int(state.epoch) if state.epoch else 0
            elapsed = time.time() - epoch_start_time[0]
            train_loss = None
            val_loss = None
            for entry in reversed(state.log_history):
                if "eval_loss" in entry and val_loss is None:
                    val_loss = entry["eval_loss"]
                if "loss" in entry and train_loss is None:
                    train_loss = entry["loss"]
                if train_loss is not None and val_loss is not None:
                    break
            lr = None
            for entry in reversed(state.log_history):
                if "learning_rate" in entry:
                    lr = entry["learning_rate"]
                    break

            # VRAM
            vram_alloc = torch.cuda.memory_allocated(0) / 1e9
            vram_reserved = torch.cuda.memory_reserved(0) / 1e9
            vram_peak = torch.cuda.max_memory_allocated(0) / 1e9

            # Disk
            import shutil
            disk = shutil.disk_usage(str(out))
            disk_free_gb = disk.free / 1e9

            # Best so far?
            is_best = val_loss is not None and val_loss < best_val_so_far[0]
            if is_best:
                best_val_so_far[0] = val_loss

            entry = {
                "epoch": epoch_idx,
                "step": state.global_step,
                "train_loss": round(train_loss, 6) if train_loss is not None else None,
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "learning_rate": round(lr, 8) if lr is not None else None,
                "elapsed_s": round(elapsed, 1),
                "vram_alloc_gb": round(vram_alloc, 2),
                "vram_peak_gb": round(vram_peak, 2),
                "disk_free_gb": round(disk_free_gb, 1),
                "is_best": is_best,
            }
            epoch_log.append(entry)
            best_marker = " <-- BEST" if is_best else ""
            print(f"\n--- Epoch {epoch_idx}{best_marker} ---")
            print(f"  Step:      {state.global_step}")
            print(f"  Train loss: {train_loss:.6f}" if train_loss is not None else "  Train loss: N/A")
            print(f"  Val loss:   {val_loss:.6f}" if val_loss is not None else "  Val loss: N/A")
            print(f"  LR:         {lr:.2e}" if lr is not None else "  LR: N/A")
            print(f"  VRAM:       alloc={vram_alloc:.1f}GB peak={vram_peak:.1f}GB")
            print(f"  Disk free:  {disk_free_gb:.1f} GB")
            print(f"  Time:       {elapsed:.1f}s")

    # ---- 7. Construct Trainer ----
    print(f"\n=== Constructing Trainer ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        callbacks=[EpochLoggerCallback()],
    )

    # ---- 7b. VALIDATE Trainer configuration BEFORE training ----
    print(f"\n=== Trainer Configuration Validation ===")
    train_dl = trainer.get_train_dataloader()
    dl_len = len(train_dl)
    actual_steps_per_epoch = dl_len // GRADIENT_ACCUMULATION_STEPS
    actual_max_steps = trainer.state.max_steps
    if actual_max_steps is None or actual_max_steps <= 0:
        actual_max_steps = actual_steps_per_epoch * num_epochs

    print(f"  len(train_dataset):            {len(train_tokenized)}")
    print(f"  len(train_dataloader):         {dl_len}")
    print(f"  per_device_train_batch_size:   {trainer.args.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps:   {trainer.args.gradient_accumulation_steps}")
    print(f"  num_train_epochs:              {trainer.args.num_train_epochs}")
    print(f"  n_gpu:                         {trainer.args.n_gpu}")
    print(f"  world_size:                    {trainer.args.world_size}")
    print(f"  max_steps:                     {actual_max_steps}")

    # Validate each critical value
    config_errors = []
    if trainer.args.n_gpu != 1:
        config_errors.append(
            f"n_gpu={trainer.args.n_gpu} (expected 1). "
            "Trainer sees multiple GPUs. Steps will be halved. "
            "Set CUDA_VISIBLE_DEVICES=0."
        )
    if trainer.args.world_size != 1:
        config_errors.append(
            f"world_size={trainer.args.world_size} (expected 1)"
        )
    if actual_steps_per_epoch != steps_per_epoch:
        config_errors.append(
            f"Actual steps/epoch={actual_steps_per_epoch} != "
            f"expected {steps_per_epoch}"
        )
    if actual_max_steps != total_steps_expected:
        config_errors.append(
            f"Actual max_steps={actual_max_steps} != "
            f"expected {total_steps_expected}"
        )

    if config_errors:
        print("\n" + "=" * 60)
        print("TRAINER CONFIGURATION MISMATCH — ABORTING")
        print("=" * 60)
        for err in config_errors:
            print(f"  ERROR: {err}")
        print("=" * 60)
        return 1

    print(f"  Configuration validation: PASS")
    print(f"  Expected steps/epoch:     {steps_per_epoch}")
    print(f"  Actual steps/epoch:       {actual_steps_per_epoch}")
    print(f"  Expected total steps:     {total_steps_expected}")
    print(f"  Actual max_steps:         {actual_max_steps}")

    # ---- 8. Train ----
    print(f"\n=== Starting training ({num_epochs} epochs, {len(train_tokenized)} examples) ===")
    print(f"TRAINING PLAN")
    print(f"  Dataset:     train={len(train_tokenized)} val={len(val_tokenized) if val_tokenized else 0} test={len(test_data)}")
    print(f"  Epochs:      {num_epochs}")
    print(f"  Micro batch: {PER_DEVICE_BATCH_SIZE}")
    print(f"  Grad accum:  {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Eff batch:   {effective_batch}")
    print(f"  Steps/epoch: {actual_steps_per_epoch}")
    print(f"  Total steps: {actual_max_steps}")
    print(f"  GPU:         {gpu_name} ({vram:.1f} GB)")
    print(f"  GPU count:   {n_gpu_visible}")
    print(f"  World size:  {trainer.args.world_size}")
    print(f"  Distributed: False (single-GPU)")

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - start_time

    # ---- 8. Metrics ----
    best_epoch = None
    best_val = float("inf")
    for entry in epoch_log:
        if entry["val_loss"] is not None and entry["val_loss"] < best_val:
            best_val = entry["val_loss"]
            best_epoch = entry["epoch"]

    final_epoch_val = None
    for entry in reversed(epoch_log):
        if entry["val_loss"] is not None:
            final_epoch_val = entry["val_loss"]
            break

    steps_actual = trainer.state.global_step
    epochs_completed = len(epoch_log)
    training_completed_fully = (epochs_completed >= num_epochs)

    # Checkpoint count and artifact size
    checkpoint_dirs = sorted(out.glob("checkpoint-*"))
    checkpoint_count = len(checkpoint_dirs)
    total_artifact_bytes = sum(
        sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        for d in out.glob("*") if d.is_dir()
    )

    # Disk remaining
    import shutil
    disk_final = shutil.disk_usage(str(out))
    disk_remaining_gb = disk_final.free / 1e9

    metrics = {
        "training_completed": training_completed_fully,
        "epochs_configured": num_epochs,
        "epochs_completed": epochs_completed,
        "total_optimizer_steps": steps_actual,
        "steps_per_epoch_configured": steps_per_epoch,
        "steps_per_epoch_actual": round(steps_actual / max(1, epochs_completed), 1),
        "final_train_loss": float(train_result.training_loss),
        "best_validation_loss": float(trainer.state.best_metric) if trainer.state.best_metric else None,
        "best_epoch": best_epoch,
        "final_epoch_validation_loss": final_epoch_val,
        "best_checkpoint": str(trainer.state.best_model_checkpoint),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "steps_per_second": round(steps_actual / elapsed, 2) if elapsed > 0 else 0,
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "checkpoint_count": checkpoint_count,
        "total_artifact_size_gb": round(total_artifact_bytes / 1e9, 3),
        "disk_remaining_gb": round(disk_remaining_gb, 1),
    }

    if training_completed_fully:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"TRAINING INCOMPLETE")
        print(f"  Configured epochs: {num_epochs}")
        print(f"  Completed epochs:  {epochs_completed}")
        print(f"  Reason: Trainer exited before completing all epochs.")
        print(f"{'='*60}")

    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Per-epoch summary
    print(f"\n=== Per-epoch summary ===")
    print(f"  {'Epoch':<7} {'Step':<7} {'Train':<12} {'Val':<12} {'LR':<12} {'VRAM':<10} {'Disk':<10}")
    for entry in epoch_log:
        tl = f"{entry['train_loss']:.6f}" if entry["train_loss"] is not None else "N/A"
        vl = f"{entry['val_loss']:.6f}" if entry["val_loss"] is not None else "N/A"
        lr = f"{entry['learning_rate']:.2e}" if entry["learning_rate"] is not None else "N/A"
        vr = f"{entry.get('vram_peak_gb', '?'):.1f}GB"
        dk = f"{entry.get('disk_free_gb', '?'):.0f}GB"
        marker = " <-- BEST" if entry.get("is_best") else ""
        print(f"  {entry['epoch']:<7} {entry['step']:<7} {tl:<12} {vl:<12} {lr:<12} {vr:<10} {dk:<10}{marker}")

    # ---- 9. Save final adapter ----
    adapter_path = out / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")

    # ---- 10. Reload and test ----
    # Capture best checkpoint before freeing trainer
    best_checkpoint = trainer.state.best_model_checkpoint or str(adapter_path)
    print(f"\n=== Reloading best adapter: {best_checkpoint} ===")
    del model, trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    loaded = PeftModel.from_pretrained(base_model, best_checkpoint)
    print("Adapter reloaded successfully")

    # ---- 11. Qualitative evaluation ----
    print("\n=== Qualitative evaluation ===")
    eval_prompts = [
        "What is Internet governance and what institutions are involved in it?",
        "How does ICANN's multistakeholder model address DNS policy?",
        "What was the role of RFC 1591 in the domain name system?",
        "How do different stakeholders view digital sovereignty?",
        "What trade-offs exist between security and openness in Internet policy?",
    ]

    # Import canonical system prompt
    from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT as SYSTEM_PROMPT

    comparisons = []
    for prompt in eval_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Explicit tensor path — NEVER pass BatchEncoding to generate()
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = formatted["input_ids"].to("cuda")
        attention_mask = formatted.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to("cuda")
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            out_tokens = loaded.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens (after the prompt)
        new_tokens = out_tokens[0][prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        comparisons.append({"prompt": prompt, "response": response[:500]})
        print(f"  Q: {prompt[:80]}...")
        print(f"  A: {response[:200]}...")
        print()

    # ---- 12. Save complete report ----
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "gpu": {"name": gpu_name, "vram_gb": round(vram, 1)},
        "model": {"name": MODEL_NAME, "total_params": total_params,
                   "trainable_params": trainable,
                   "trainable_pct": round(100 * trainable / total_params, 4)},
        "qlora": {"bits": 4, "nf4": True, "double_quant": True,
                   "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES},
        "training_config": {
            "epochs": num_epochs,
            "micro_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": effective_batch,
            "learning_rate": LEARNING_RATE,
            "scheduler": LR_SCHEDULER,
            "warmup_ratio": WARMUP_RATIO,
            "max_seq_length": MAX_SEQ_LENGTH,
            "steps_per_epoch_expected": steps_per_epoch,
            "total_steps_expected": total_steps_expected,
        },
        "dataset": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
        "per_epoch": epoch_log,
        "metrics": metrics,
        "qualitative_eval": comparisons,
        "adapter_path": str(adapter_path),
        "best_checkpoint": best_checkpoint,
    }

    report_path = out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {report_path}")

    if training_completed_fully:
        print("TRAINING COMPLETE — all epochs finished.")
    else:
        print("TRAINING INCOMPLETE — not all epochs completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
