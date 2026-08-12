#!/usr/bin/env python3
"""QLoRA smoke test for Arwen Policy on Qwen3.6-27B.

Validates the entire pipeline: config → quantize → LoRA → forward → backward →
checkpoint → reload → inference.  Uses 1-2 optimizer steps only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Load .env
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()


# ===================================================================
# 1. ENVIRONMENT / GPU DETECTION
# ===================================================================

def detect_environment() -> dict[str, Any]:
    """Detect hardware and dependencies without loading the model."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cuda_available": False,
        "gpu_name": "NONE",
        "gpu_vram_gb": 0.0,
        "gpu_count": 0,
        "recommended_dtype": "float32",
    }

    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / 1e9, 1)
            # Determine compute dtype
            if props.major >= 8:  # Ampere+
                info["recommended_dtype"] = "bfloat16"
            else:
                info["recommended_dtype"] = "float16"
    except ImportError:
        info["torch"] = "NOT INSTALLED"

    for lib in ["transformers", "peft", "accelerate", "bitsandbytes", "datasets"]:
        try:
            m = __import__(lib)
            info[lib] = getattr(m, "__version__", "?")
        except ImportError:
            info[lib] = "NOT INSTALLED"

    return info


# ===================================================================
# 2. TOKENIZATION AUDIT
# ===================================================================

def tokenize_audit(data_dir: str = "datasets/sft_final",
                   model_name: str = "Qwen/Qwen3.6-27B") -> dict[str, Any]:
    """Tokenize the full dataset and report statistics."""
    from transformers import AutoTokenizer

    print("Loading tokenizer for tokenization audit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_lengths = []
    total_examples = 0
    truncated = 0
    max_length = 2048

    for split in ("train", "validation", "test"):
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                # Build chat template text
                text_parts = []
                for msg in ex.get("messages", []):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                full_text = "\n".join(text_parts)

                tokens = tokenizer.encode(full_text, add_special_tokens=False)
                tok_len = len(tokens)
                all_lengths.append(tok_len)
                total_examples += 1
                if tok_len > max_length:
                    truncated += 1
            except Exception:
                continue

    if not all_lengths:
        return {"error": "No examples tokenized"}

    sorted_lens = sorted(all_lengths)
    n = len(sorted_lens)
    stats = {
        "total_examples": total_examples,
        "min": sorted_lens[0],
        "max": sorted_lens[-1],
        "mean": round(sum(sorted_lens) / n, 1),
        "median": sorted_lens[n // 2],
        "p95": sorted_lens[int(n * 0.95)],
        "p99": sorted_lens[int(n * 0.99)],
        "max_length": max_length,
        "truncated": truncated,
        "truncation_pct": round(100 * truncated / n, 1),
    }
    return stats


# ===================================================================
# 3. GPU SMOKE TEST
# ===================================================================

def run_smoke_test(
    model_name: str = "Qwen/Qwen3.6-27B",
    data_dir: str = "datasets/sft_final",
    output_dir: str = "artifacts/qlora_smoke_test",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_steps: int = 2,
    max_seq_length: int = 2048,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the QLoRA smoke test: quantize → LoRA → train 2 steps → save → reload."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    if not torch.cuda.is_available():
        return {
            "status": "BLOCKED",
            "reason": "No CUDA GPU available. Qwen3.6-27B requires GPU for QLoRA.",
            "gpu_info": detect_environment(),
        }

    start_time = time.time()
    result: dict[str, Any] = {"status": "STARTED", "steps": []}

    # ---- Determine compute dtype ----
    props = torch.cuda.get_device_properties(0)
    compute_dtype = torch.bfloat16 if props.major >= 8 else torch.float16
    result["compute_dtype"] = str(compute_dtype)

    # ---- 4-bit quantization config ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    result["quantization"] = "4-bit NF4 + double quant"

    # ---- Load tokenizer ----
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    result["tokenizer"] = "loaded"

    # ---- Load model ----
    print(f"Loading model with 4-bit quantization: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.gradient_checkpointing_enable()
    result["model_loaded"] = True

    # ---- Count parameters ----
    total_params = sum(p.numel() for p in model.parameters())
    result["total_parameters"] = total_params
    result["total_parameters_B"] = round(total_params / 1e9, 1)

    # ---- LoRA config ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["trainable_parameters"] = trainable_params
    result["trainable_parameters_M"] = round(trainable_params / 1e6, 1)
    result["trainable_pct"] = round(100 * trainable_params / total_params, 3)
    model.print_trainable_parameters()

    # ---- Load dataset ----
    def load_split(split_name):
        path = Path(data_dir) / f"{split_name}.jsonl"
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

    from datasets import Dataset
    train_data = load_split("train")[:16]  # tiny subset for smoke test
    train_dataset = Dataset.from_list(train_data)

    def tokenize_fn(examples):
        batch_msgs = [msgs for msgs in examples["messages"]]
        tokenized = tokenizer.apply_chat_template(
            batch_msgs, tokenize=True, add_generation_prompt=False,
            padding="max_length", truncation=True, max_length=max_seq_length,
            return_dict=True,
        )

        # Build labels: mask everything except assistant content
        im_start_id = 151644
        im_end_id = 151645
        asst_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)

        labels_list = []
        for ids in tokenized["input_ids"]:
            labels = [-100] * len(ids)
            i = 0
            while i < len(ids):
                if ids[i] == im_start_id:
                    rs = i + 1
                    is_asst = all(rs + j < len(ids) and ids[rs + j] == asst_tokens[j]
                                  for j in range(len(asst_tokens)))
                    if is_asst:
                        cs = rs + len(asst_tokens)
                        ce = cs
                        while ce < len(ids) and ids[ce] != im_end_id:
                            ce += 1
                        for j in range(cs, ce):
                            labels[j] = ids[j]
                        i = ce
                    else:
                        i += 1
                else:
                    i += 1
            labels_list.append(labels)
        tokenized["labels"] = labels_list
        return tokenized

    tokenized = train_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["messages"])
    result["dataset_train_samples"] = len(tokenized)

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max_steps,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        seed=seed,
        report_to="none",
        remove_unused_columns=True,
    )

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_seq_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # ---- Train (smoke test: 2 steps) ----
    print(f"Starting smoke test training ({max_steps} steps)...")
    train_result = trainer.train()
    result["train_loss"] = round(float(train_result.training_loss), 4)
    result["train_steps"] = max_steps
    result["steps_completed"] = True

    # ---- Memory ----
    torch.cuda.memory_stats()
    result["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    result["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)

    # ---- Save adapter ----
    print(f"Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    result["checkpoint_saved"] = True

    # ---- Reload and test inference ----
    print("Reloading adapter for inference test...")
    del model, trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    from peft import PeftModel
    reloaded = PeftModel.from_pretrained(base_model, output_dir)
    result["adapter_reloaded"] = True

    # Test inference
    inputs = tokenizer("What is Internet governance?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = reloaded.generate(**inputs, max_new_tokens=20)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result["inference_test"] = decoded[:100]
    result["inference_ok"] = len(decoded) > 0

    result["status"] = "COMPLETED"
    result["elapsed_s"] = round(time.time() - start_time, 1)
    return result


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Arwen Policy QLoRA Smoke Test")
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="Foundation model (default: Qwen3-8B for T4. Use Qwen/Qwen3.6-27B for A100+)")
    p.add_argument("--data-dir", default="datasets/sft_final")
    p.add_argument("--output-dir", default="artifacts/qlora_smoke_test")
    p.add_argument("--max-steps", type=int, default=2)
    p.add_argument("--skip-gpu", action="store_true", help="Skip GPU test, run audit only")
    args = p.parse_args()

    print("=" * 60)
    print("ARWEN POLICY — QLORA SMOKE TEST")
    print("=" * 60)

    # 1. Environment
    env = detect_environment()
    print("\n=== ENVIRONMENT ===")
    for k, v in env.items():
        print(f"  {k}: {v}")

    # 2. Tokenization audit
    print("\n=== TOKENIZATION AUDIT ===")
    stats = tokenize_audit(args.data_dir, args.model)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 3. GPU smoke test (if available)
    if args.skip_gpu or not env["cuda_available"]:
        print("\n=== GPU SMOKE TEST: SKIPPED (no GPU) ===")
        print(f"  Reason: {'--skip-gpu flag' if args.skip_gpu else 'CUDA not available'}")
        result = {"status": "SKIPPED", "reason": "No GPU", "environment": env, "tokenization": stats}
    else:
        print("\n=== GPU SMOKE TEST ===")
        result = run_smoke_test(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
        )
        for k, v in result.items():
            if k != "steps":
                print(f"  {k}: {v}")

    # Save metadata
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "tokenization": stats,
        "result": result,
    }
    meta_path = Path(args.output_dir) / "smoke_test_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nMetadata saved to {meta_path}")
