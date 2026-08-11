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
# Steps per epoch ≈ ceil(304 / 8) = 38
# Total steps ≈ 20 * 38 = 760

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
SAVE_TOTAL_LIMIT = 25  # enough for all epoch checkpoints


# ===================================================================
# MAIN
# ===================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="QLoRA training for Arwen Policy")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of training epochs (default: {NUM_EPOCHS})")
    args = parser.parse_args()

    start_time = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    num_epochs = args.epochs
    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    # ---- 1. GPU check ----
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"CUDA: {torch.version.cuda}")

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
            padding="max_length", truncation=True,
            max_length=MAX_SEQ_LENGTH, return_dict=True,
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
        device_map="auto",
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

    from transformers import DataCollatorWithPadding, TrainerCallback
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH)

    # Per-epoch logging callback
    epoch_log: list[dict] = []
    epoch_start_time = [time.time()]

    class EpochLoggerCallback(TrainerCallback):
        def on_epoch_end(self, trainer_args, state, control, **kwargs):
            epoch_idx = int(state.epoch) if state.epoch else 0
            elapsed = time.time() - epoch_start_time[0]
            train_loss = None
            val_loss = None
            # Extract losses from log_history
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
            entry = {
                "epoch": epoch_idx,
                "step": state.global_step,
                "train_loss": round(train_loss, 6) if train_loss is not None else None,
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "learning_rate": round(lr, 8) if lr is not None else None,
                "elapsed_s": round(elapsed, 1),
            }
            epoch_log.append(entry)
            print(f"\n--- Epoch {epoch_idx} ---")
            print(f"  Step: {state.global_step}")
            print(f"  Train loss: {train_loss:.6f}" if train_loss is not None else "  Train loss: N/A")
            print(f"  Val loss:   {val_loss:.6f}" if val_loss is not None else "  Val loss: N/A")
            print(f"  LR:         {lr:.2e}" if lr is not None else "  LR: N/A")
            print(f"  Time:       {elapsed:.1f}s")

    # ---- 7. Train ----
    print(f"\n=== Starting training ({num_epochs} epochs, {len(train_tokenized)} examples) ===")
    print(f"Expected: {steps_per_epoch} steps/epoch, ~{total_steps_expected} total steps")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        callbacks=[EpochLoggerCallback()],
    )

    train_result = trainer.train()
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
    metrics = {
        "training_completed": True,
        "epochs_configured": num_epochs,
        "epochs_completed": len(epoch_log),
        "total_optimizer_steps": steps_actual,
        "steps_per_epoch_configured": steps_per_epoch,
        "steps_per_epoch_actual": round(steps_actual / max(1, len(epoch_log)), 1),
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
    }
    print(f"\n=== Training complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Per-epoch summary
    print(f"\n=== Per-epoch summary ===")
    print(f"  {'Epoch':<7} {'Step':<7} {'Train Loss':<12} {'Val Loss':<12} {'LR':<12}")
    for entry in epoch_log:
        tl = f"{entry['train_loss']:.6f}" if entry["train_loss"] is not None else "N/A"
        vl = f"{entry['val_loss']:.6f}" if entry["val_loss"] is not None else "N/A"
        lr = f"{entry['learning_rate']:.2e}" if entry["learning_rate"] is not None else "N/A"
        marker = " <-- BEST" if entry["epoch"] == best_epoch else ""
        print(f"  {entry['epoch']:<7} {entry['step']:<7} {tl:<12} {vl:<12} {lr:<12}{marker}")

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
        device_map="auto",
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

    comparisons = []
    for prompt in eval_prompts:
        messages = [
            {"role": "system", "content": (
                "You are a policy analysis AI. Answer questions using only the supplied source "
                "evidence. Attribute claims to documented sources. Disclose uncertainty. "
                "Do not invent facts, dates, stakeholders, or positions."
            )},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        prompt_len = formatted.shape[1]
        with torch.no_grad():
            out_tokens = loaded.generate(
                formatted, max_new_tokens=200, do_sample=True,
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
    print("DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
