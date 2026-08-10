#!/usr/bin/env python3
"""Full QLoRA training for Arwen Policy on Qwen3-8B + Tesla T4.

Runs 1 epoch, evaluates on validation, checkpoints regularly,
tracks best validation loss, saves final adapter, reloads and
runs qualitative evaluation.

Usage (on Lightning T4):
    uv sync --extra gpu
    uv run python scripts/train_qlora.py
"""

from __future__ import annotations

import json
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
    DataCollatorForLanguageModeling,
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
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Checkpointing
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 5


# ===================================================================
# MAIN
# ===================================================================

def main() -> int:
    start_time = time.time()
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

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
                text_parts = []
                for msg in ex.get("messages", []):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                examples.append({"text": "\n".join(text_parts)})
            except Exception:
                continue
        return examples

    train_data = load_split("train")
    val_data = load_split("validation")
    test_data = load_split("test")  # held out, not used in training

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) != 503:
        print(f"WARNING: Expected 503 train examples, got {len(train_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    # ---- 3. Tokenizer ----
    print("\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples["text"], truncation=True, padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )

    train_tokenized = train_dataset.map(tokenize_fn, batched=True)
    train_tokenized = train_tokenized.remove_columns(["text"])
    print(f"Train tokenized: {len(train_tokenized)}")

    if val_dataset is not None:
        val_tokenized = val_dataset.map(tokenize_fn, batched=True)
        val_tokenized = val_tokenized.remove_columns(["text"])
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
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- 7. Train ----
    print(f"\n=== Starting training ({NUM_EPOCHS} epoch, {len(train_tokenized)} examples) ===")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    elapsed = time.time() - start_time

    # ---- 8. Metrics ----
    metrics = {
        "training_completed": True,
        "epochs": NUM_EPOCHS,
        "total_steps": trainer.state.global_step,
        "train_loss": float(train_result.training_loss),
        "best_validation_loss": float(trainer.state.best_metric) if trainer.state.best_metric else None,
        "best_checkpoint": str(trainer.state.best_model_checkpoint),
        "elapsed_seconds": round(elapsed, 1),
        "elapsed_minutes": round(elapsed / 60, 1),
        "steps_per_second": round(trainer.state.global_step / elapsed, 2) if elapsed > 0 else 0,
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }
    print(f"\n=== Training complete ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ---- 9. Save final adapter ----
    adapter_path = out / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path}")

    # ---- 10. Reload and test ----
    print("\n=== Reloading best adapter ===")
    del model, trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    best_checkpoint = trainer.state.best_model_checkpoint or str(adapter_path)
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
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_tokens = loaded.generate(
                **inputs, max_new_tokens=100, do_sample=True,
                temperature=0.7, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        comparisons.append({"prompt": prompt, "response": response[len(prompt):][:300]})
        print(f"  Q: {prompt[:80]}...")
        print(f"  A: {response[len(prompt):][:200]}...")
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
        "dataset": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
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
