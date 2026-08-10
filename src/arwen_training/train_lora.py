"""LoRA fine-tuning script for Arwen Policy on Qwen3.6-27B.

Usage:
    python -m arwen_training.train_lora --data_dir data/extracted --output_dir ./lora-output

Requirements:
    uv sync

This script is designed to be run when the corpus is sufficient (>= 100 documents).
For development/testing, use --dev_mode with the local 8.2B model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _check_corpus(data_dir: str) -> dict[str, Any]:
    """Check corpus readiness and return stats."""
    from arwen_training.validation import check_corpus_quality
    return check_corpus_quality(data_dir)


def _load_training_data(data_dir: str, max_examples: int = 500) -> list[dict[str, Any]]:
    """Load/built training examples from the corpus."""
    from arwen_training.builder import build_corpus_training_examples
    examples = build_corpus_training_examples(data_dir, max_examples=max_examples)
    if not examples:
        print("WARNING: build_corpus_training_examples returned empty list")
    return examples


def _prepare_dataset(examples: list[dict[str, Any]]) -> Any:
    """Convert training examples to a Hugging Face Dataset."""
    try:
        from datasets import Dataset
    except ImportError:
        raise RuntimeError("Install with: uv sync --extra dev")

    formatted: list[dict[str, str]] = []
    for ex in examples:
        messages = ex.get("messages", [])
        # Convert chat messages to a single text for causal LM training
        text_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        formatted.append({"text": "\n".join(text_parts)})

    return Dataset.from_list(formatted)


def _get_target_modules(model_name: str) -> list[str]:
    """Return LoRA target modules based on model architecture.

    Qwen3.6-27B uses qwen3_5 architecture with mixed Gated DeltaNet
    and Gated Attention layers. We target the attention projection
    matrices that are common across both layer types.
    """
    # Standard Qwen attention projections — these exist in both
    # Gated Attention and Gated DeltaNet layers.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    print(f"LoRA target modules: {target_modules}")
    print("NOTE: Verify module names against actual model with:")
    print("  from transformers import AutoModel; m = AutoModel.from_pretrained(...)")
    print("  for n, _ in m.named_modules(): print(n)")
    return target_modules


def train(
    data_dir: str = "data/extracted",
    output_dir: str = "./lora-output",
    model_name: str = "Qwen/Qwen3.6-27B",
    dev_mode: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    """Run LoRA fine-tuning.

    In dev_mode, uses a smaller model for local testing.
    """
    start_time = datetime.now(UTC)

    # ---- 1. Corpus check ----
    quality = _check_corpus(data_dir)
    if not quality["ready"] and not dev_mode:
        return {
            "status": "BLOCKED",
            "reason": quality["reason"],
            "corpus_stats": quality,
        }

    print(f"Corpus check: {quality}")

    # ---- 2. Load training data ----
    examples = _load_training_data(data_dir)
    if not examples:
        return {
            "status": "BLOCKED",
            "reason": "No training examples could be built from corpus",
            "corpus_stats": quality,
        }

    print(f"Training examples: {len(examples)}")

    # ---- 3. Import training dependencies ----
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
    except ImportError as e:
        return {
            "status": "BLOCKED",
            "reason": f"Missing dependency: {e}. Install: uv sync",
            "corpus_stats": quality,
        }

    # ---- 4. Dev mode override ----
    if dev_mode:
        # Use local Ollama-compatible model path or a smaller HF model
        model_name = os.environ.get("DEV_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        print(f"DEV MODE: using {model_name}")

    # ---- 5. Load model and tokenizer ----
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    except Exception as e:
        return {
            "status": "BLOCKED",
            "reason": f"Failed to load model {model_name}: {e}",
            "corpus_stats": quality,
        }

    # ---- 6. Configure LoRA ----
    target_modules = _get_target_modules(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 7. Prepare dataset ----
    dataset = _prepare_dataset(examples)

    def tokenize(examples_dict: dict) -> dict:
        return tokenizer(
            examples_dict["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # ---- 8. Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # ---- 9. Train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    # ---- 10. Save ----
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Write training metadata
    metadata = {
        "base_model": model_name,
        "adapter_type": "lora",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "training_examples": len(examples),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "seed": seed,
        "corpus_stats": quality,
        "started_at": start_time.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "dev_mode": dev_mode,
    }
    metadata_path = Path(output_dir) / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Training complete. Output: {output_dir}")
    return {"status": "COMPLETED", "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Arwen Policy LoRA Training")
    parser.add_argument("--data_dir", default="data/extracted")
    parser.add_argument("--output_dir", default="./lora-output")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--dev_mode", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    result = train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        dev_mode=args.dev_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print(json.dumps(result, indent=2))
    if result["status"] == "BLOCKED":
        sys.exit(1)


if __name__ == "__main__":
    main()
