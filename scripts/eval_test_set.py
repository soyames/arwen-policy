#!/usr/bin/env python3
"""Evaluate a QLoRA adapter on the held-out test set.

Computes per-example and aggregate causal-LM loss using the exact same
tokenization and label masking as training. The test set was never passed
to Trainer — this is the first time the model sees these 35 examples.

Usage:
    uv run python scripts/eval_test_set.py \
        --adapter-path artifacts/qlora_arwen_8b/checkpoint-38
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_DIR = "datasets/sft_final"
MAX_SEQ_LENGTH = 2048


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


def build_tokenize_fn(tokenizer):
    """Exact tokenize_fn from train_qlora.py — assistant-only label masking."""

    def tokenize_fn(examples: dict) -> dict:
        batch_messages = [msgs for msgs in examples["messages"]]

        tokenized = tokenizer.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=False,
            truncation=True, max_length=MAX_SEQ_LENGTH, return_dict=True,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate adapter on held-out test set")
    parser.add_argument(
        "--adapter-path", required=True,
        help="Path to LoRA adapter checkpoint (e.g., artifacts/qlora_arwen_8b/checkpoint-38)",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--qualitative-only", action="store_true",
        help="Skip 35-example loss evaluation; run only the qualitative generation sample.",
    )
    parser.add_argument(
        "--loss-only", action="store_true",
        help="Run only the 35-example loss evaluation; skip qualitative generation.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run both loss evaluation and qualitative generation (default behavior).",
    )
    args = parser.parse_args()

    # Resolve mode: --full or no flag = do both; --loss-only = loss only; --qualitative-only = qual only
    run_loss = not args.qualitative_only
    run_qual = not args.loss_only

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found: {adapter_path}")
        return 1

    # Verify adapter files
    safetensors = adapter_path / "adapter_model.safetensors"
    config = adapter_path / "adapter_config.json"
    if not safetensors.exists():
        print(f"ERROR: adapter_model.safetensors not found in {adapter_path}")
        return 1
    print(f"adapter_model.safetensors: {safetensors.stat().st_size / 1e6:.0f} MB")

    # ---- GPU check ----
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for evaluation.")
        return 1

    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({vram:.1f} GB)")

    # ---- Load test set ----
    print(f"\n=== Loading test set ===")
    test_data = load_split("test")
    print(f"Test examples: {len(test_data)}")
    assert len(test_data) == 35, f"Expected 35 test examples, got {len(test_data)}"

    # ---- Tokenize ----
    print("Tokenizing with assistant-only label masking...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenize_fn = build_tokenize_fn(tokenizer)
    test_dataset = Dataset.from_list(test_data)
    test_tokenized = test_dataset.map(tokenize_fn, batched=True)
    test_tokenized = test_tokenized.remove_columns(["messages"])

    # Verify label masking
    zero_asst = sum(
        1 for i in range(len(test_tokenized))
        if all(l == -100 for l in test_tokenized[i]["labels"])
    )
    print(f"Test examples with zero assistant targets: {zero_asst}")
    if zero_asst > 0:
        print("ERROR: Some test examples have no assistant targets.")
        return 1

    # ---- Load model + adapter ----
    print(f"\n=== Loading base model: {args.model} ===")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print("Model + adapter loaded.")

    # ---- Evaluate per-example (skip if qualitative-only) ----
    losses = []
    perplexities = []
    skipped = 0
    n_evaluated = 0

    if run_loss:
        print(f"\n=== Test-set evaluation ({len(test_tokenized)} examples) ===")
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
        )

        for i in range(len(test_tokenized)):
            example = test_tokenized[i]
            n_active = sum(1 for l in example["labels"] if l != -100)
            if n_active == 0:
                skipped += 1
                continue

            # Create single-example batch
            batch = collator([example])
            batch = {k: v.to("cuda:0") for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss.item()

            if math.isfinite(loss):
                losses.append(loss)
                perplexities.append(math.exp(loss))
            else:
                skipped += 1

        # ---- Report ----
        n_evaluated = len(losses)
        print(f"\nEvaluated: {n_evaluated}/{len(test_tokenized)}")
        if skipped > 0:
            print(f"Skipped: {skipped} (no active labels or non-finite loss)")

        if n_evaluated == 0:
            print("ERROR: No examples evaluated.")
            return 1

        avg_loss = sum(losses) / n_evaluated
        avg_perplexity = sum(perplexities) / n_evaluated
        min_loss = min(losses)
        max_loss = max(losses)
        sorted_losses = sorted(losses)
        median_loss = sorted_losses[n_evaluated // 2]

        print(f"\n=== Test-set Results ===")
        print(f"  Examples evaluated:  {n_evaluated}")
        print(f"  Average loss:        {avg_loss:.6f}")
        print(f"  Average perplexity:  {avg_perplexity:.2f}")
        print(f"  Median loss:         {median_loss:.6f}")
        print(f"  Min loss:            {min_loss:.6f}")
        print(f"  Max loss:            {max_loss:.6f}")

        # Top-5 and bottom-5
        indexed = list(enumerate(losses))
        indexed.sort(key=lambda x: x[1])

        print(f"\n  Best 5 (lowest loss):")
        for idx, loss_val in indexed[:5]:
            msgs = test_data[idx]["messages"]
            q = next((m["content"][:80] for m in msgs if m["role"] == "user"), "?")
            print(f"    [{idx}] loss={loss_val:.4f}  Q: {q}...")

        print(f"\n  Worst 5 (highest loss):")
        for idx, loss_val in indexed[-5:]:
            msgs = test_data[idx]["messages"]
            q = next((m["content"][:80] for m in msgs if m["role"] == "user"), "?")
            print(f"    [{idx}] loss={loss_val:.4f}  Q: {q}...")

    # Qualitative sample (skip if --loss-only)
    if run_qual:
        print(f"\n=== Qualitative sample ===")
        from arwen_etl.engine.arwen_prompt import ARWEN_SYSTEM_PROMPT
        for idx in range(min(3, len(test_data))):
            msgs = test_data[idx]["messages"]
            q = next((m["content"] for m in msgs if m["role"] == "user"), "?")
            messages = [
                {"role": "system", "content": ARWEN_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            # Explicit tensor path — NEVER pass BatchEncoding to generate()
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            input_ids = formatted["input_ids"].to("cuda:0")
            attention_mask = formatted.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to("cuda:0")
            prompt_len = input_ids.shape[1]
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200, do_sample=True,
                    temperature=0.2, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"\n  Q{idx}: {q[:120]}...")
            print(f"  A{idx}: {response[:300]}")

    # Write report (skip if qualitative-only — no loss data)
    if run_loss and n_evaluated > 0:
        avg_loss = sum(losses) / n_evaluated
        avg_perplexity = sum(perplexities) / n_evaluated
        report = {
            "adapter_path": str(adapter_path),
            "test_examples": len(test_data),
            "evaluated": n_evaluated,
            "skipped": skipped,
            "avg_loss": round(avg_loss, 6),
            "avg_perplexity": round(avg_perplexity, 2),
            "median_loss": round(sorted(losses)[n_evaluated // 2], 6),
            "min_loss": round(min(losses), 6),
            "max_loss": round(max(losses), 6),
            "per_example_losses": [round(l, 6) for l in losses],
        }
        report_path = adapter_path / "test_set_eval.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nReport saved to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
