#!/usr/bin/env python3
"""Verify a trained QLoRA adapter — reload and qualitative eval.

Usage: uv run python scripts/verify_adapter.py [--adapter-path artifacts/qlora_arwen_8b/checkpoint-63]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-8B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify QLoRA adapter")
    parser.add_argument("--adapter-path", default="artifacts/qlora_arwen_8b/checkpoint-63")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"ERROR: Adapter path not found: {adapter_path}")
        print("Available paths:")
        for p in Path("artifacts/qlora_arwen_8b").glob("**/adapter_config.json"):
            print(f"  {p.parent}")
        return 1

    print(f"Adapter: {adapter_path}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    print(f"Loading base model: {args.model}...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16,
    )

    # Load adapter
    print(f"Loading adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Adapter loaded successfully.")

    # Qualitative eval
    prompts = [
        "What is Internet governance and what institutions are involved in it?",
        "How does ICANN's multistakeholder model address DNS policy?",
        "What was the role of RFC 1591 in the domain name system?",
        "How do different stakeholders view digital sovereignty?",
        "What trade-offs exist between security and openness in Internet policy?",
    ]

    print("\n=== Qualitative Evaluation ===")
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=True,
                                 temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response[len(prompt):][:400]})
        print(f"  Q: {prompt[:80]}...")
        print(f"  A: {response[len(prompt):][:200]}...")
        print()

    print("Verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
