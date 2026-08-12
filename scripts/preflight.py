#!/usr/bin/env python3
"""Pre-flight validation gates for Arwen Policy QLoRA training.

Validates environment, GPU, dataset, and Trainer configuration BEFORE
launching a multi-hour training run.  Every gate that can fail before
training starts should fail here, not at epoch 10.

Gates:
  G1  — Environment (CUDA_VISIBLE_DEVICES, working directory)
  G2  — Exactly one visible GPU
  G3  — GPU memory adequate
  G4  — Dataset counts match expected
  G5  — Label masking audit
  G6  — DataLoader configuration
  G7  — Trainer world_size / n_gpu
  G8  — Optimizer steps per epoch
  G9  — max_steps
  G10 — Disk space
  G11 — Checkpoint save/load smoke test
  G12 — Generation smoke test

Every gate prints EXPECTED vs ACTUAL and PASS/FAIL explicitly.
No gate reports "PASS" merely because it found a value — it must match.

Usage:
    uv run python scripts/preflight.py              # all gates
    uv run python scripts/preflight.py --cpu-only   # CPU-only gates (G1-G6, G10)
    uv run python scripts/preflight.py --trainer-config  # Trainer config gates (G6-G9)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

# =============================================================================
# Expected configuration — single source of truth
# =============================================================================

EXPECTED = {
    "train_examples": 330,
    "val_examples": 39,
    "test_examples": 37,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_epochs": 20,
    "effective_batch_size": 1 * 8,  # per_device_batch * grad_accum
    "n_gpu": 1,
    "world_size": 1,
    "max_seq_length": 2048,
}

# Derived expected values
EXPECTED["steps_per_epoch"] = math.ceil(
    EXPECTED["train_examples"] / EXPECTED["effective_batch_size"]
)  # ceil(330/8) = 42
EXPECTED["max_steps"] = EXPECTED["steps_per_epoch"] * EXPECTED["num_epochs"]  # 42*20 = 840

DATA_DIR = "datasets/sft_final"
MODEL_NAME = "Qwen/Qwen3-8B"

# =============================================================================
# Report helpers
# =============================================================================

_passed = 0
_failed = 0
_skipped = 0


def gate(name: str, expected, actual) -> bool:
    """Report a gate result.  Returns True if passed."""
    global _passed, _failed
    match = actual == expected
    if match:
        _passed += 1
        status = "PASS"
    else:
        _failed += 1
        status = "FAIL"
    print(f"  [{status}] {name}")
    print(f"         Expected: {expected!r}")
    print(f"         Actual:   {actual!r}")
    if not match:
        print(f"         *** MISMATCH — this will cause training problems ***")
    return match


def skip(name: str, reason: str = "") -> None:
    global _skipped
    _skipped += 1
    print(f"  [SKIP] {name}  {reason}")


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def report() -> str:
    total = _passed + _failed + _skipped
    print(f"\n{'='*60}")
    print(f"  PREFLIGHT RESULT: {_passed}/{total} passed"
          + (f", {_skipped} skipped" if _skipped else "")
          + (f", {_failed} FAILED" if _failed else ""))
    print(f"{'='*60}")
    if _failed > 0:
        print("  PREFLIGHT FAILED — DO NOT LAUNCH TRAINING")
        return "FAIL"
    print("  PREFLIGHT PASSED — training may proceed")
    return "PASS"


# =============================================================================
# G1: Environment
# =============================================================================

def g1_environment() -> None:
    section("G1 — Environment")
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    print(f"  CUDA_VISIBLE_DEVICES = {cuda_vis}")
    if cuda_vis != "0":
        print(f"  [WARN] CUDA_VISIBLE_DEVICES should be '0' for single-GPU training.")
        print(f"         Current value: {cuda_vis}")
    print(f"  Working directory: {os.getcwd()}")
    pyproject = Path("pyproject.toml")
    gate("pyproject.toml exists", True, pyproject.exists())
    data_dir = Path(DATA_DIR)
    gate(f"{DATA_DIR}/ exists", True, data_dir.is_dir())


# =============================================================================
# G2: Exactly one visible GPU
# =============================================================================

def g2_single_gpu() -> None:
    section("G2 — Single GPU visibility")
    try:
        import torch
    except ImportError:
        skip("G2-G3", "torch not installed (CPU-only mode)")
        return

    n_gpu = torch.cuda.device_count()
    gate("torch.cuda.device_count()", 1, n_gpu)

    if n_gpu >= 1:
        torch.cuda.set_device(0)
        gate("torch.cuda.current_device()", 0, torch.cuda.current_device())
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"         GPU 0: {name} ({vram:.1f} GB)")
        gate("CUDA available", True, torch.cuda.is_available())


# =============================================================================
# G3: GPU memory adequate
# =============================================================================

def g3_gpu_memory() -> None:
    section("G3 — GPU memory")
    try:
        import torch
    except ImportError:
        skip("G3", "torch not installed")
        return

    if torch.cuda.device_count() < 1:
        skip("G3", "no GPU")
        return

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    adequate = vram >= 14.5
    gate("VRAM >= 14.5 GB (Tesla T4)", True, adequate)
    print(f"         Total VRAM: {vram:.1f} GB")
    if not adequate:
        print(f"         WARNING: Training may OOM on this GPU.")


# =============================================================================
# G4: Dataset counts
# =============================================================================

def g4_dataset_counts() -> None:
    section("G4 — Dataset counts")

    for split, expected in [("train", EXPECTED["train_examples"]),
                             ("validation", EXPECTED["val_examples"]),
                             ("test", EXPECTED["test_examples"])]:
        path = Path(DATA_DIR) / f"{split}.jsonl"
        if not path.exists():
            gate(f"{split}.jsonl exists", True, False)
            continue
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        actual = len(lines)
        gate(f"{split} examples", expected, actual)

    # Quick schema check on first example
    path = Path(DATA_DIR) / "train.jsonl"
    if path.exists():
        ex = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        has_msgs = "messages" in ex
        has_src = "source_document_ids" in ex
        gate("Example has 'messages' field", True, has_msgs)
        gate("Example has 'source_document_ids' field", True, has_src)


# =============================================================================
# G5: Label masking audit
# =============================================================================

def g5_label_masking() -> None:
    section("G5 — Label masking audit")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        skip("G5", "transformers not installed")
        return

    import json as _json
    from datasets import Dataset

    path = Path(DATA_DIR) / "train.jsonl"
    if not path.exists():
        skip("G5", "train.jsonl not found")
        return

    data = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = _json.loads(line)
        msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
        data.append({"messages": msgs})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reuse tokenize_fn logic
    def tokenize_fn(examples):
        batch_messages = [msgs for msgs in examples["messages"]]
        tokenized = tokenizer.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=False,
            truncation=True, max_length=EXPECTED["max_seq_length"], return_dict=True,
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

    ds = Dataset.from_list(data)
    tokenized = ds.map(tokenize_fn, batched=True)

    zero_asst = sum(
        1 for i in range(len(tokenized))
        if all(l == -100 for l in tokenized[i]["labels"])
    )
    gate("Zero assistant target examples", 0, zero_asst)

    # Check for sys/user token violations (non-asst tokens receiving labels)
    im_start_id = 151644
    im_end_id = 151645
    assistant_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    violations = 0
    for idx in range(min(50, len(tokenized))):
        ids = tokenized[idx]["input_ids"]
        labels = tokenized[idx]["labels"]
        in_non_asst = False
        for tid, lbl in zip(ids, labels):
            if tid == im_start_id:
                role_start = ids.index(tid) + 1 if tid in ids else -1
                # simpler check — scan for im_start tokens
                break
        # Full audit on first 50
        i = 0
        while i < len(ids):
            if ids[i] == im_start_id:
                role_start = i + 1
                is_asst = all(
                    role_start + j < len(ids)
                    and ids[role_start + j] == assistant_tokens[j]
                    for j in range(len(assistant_tokens))
                )
                if not is_asst:
                    # scan forward until im_end or next im_start
                    j = role_start
                    while j < len(ids) and ids[j] != im_end_id:
                        if labels[j] != -100:
                            violations += 1
                            break
                        j += 1
                i = role_start
            else:
                i += 1
    gate("System/user label violations (sampled)", 0, violations)


# =============================================================================
# G6-G9: Trainer configuration validation
# =============================================================================

def _load_training_dataset():
    """Load training data in the exact format train_qlora.py uses."""
    import json as _json
    path = Path(DATA_DIR) / "train.jsonl"
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ex = _json.loads(line)
        messages = []
        for msg in ex.get("messages", []):
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        examples.append({"messages": messages})
    return examples


def g6_g9_trainer_config() -> None:
    """Validate Trainer configuration G6-G9.

    Constructs a REAL Trainer and inspects its DataLoader, world_size, n_gpu,
    max_steps, etc.  Then compares every value against EXPECTED.

    This is the gate that would have caught the previous 19-vs-38 bug.
    """
    section("G6-G9 — Trainer configuration validation")

    try:
        import torch
        from transformers import (
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
    except ImportError as e:
        skip("G6-G9", f"import failed: {e}")
        return

    # ---- Load data ----
    train_data = _load_training_dataset()
    n_train = len(train_data)
    gate("G6 — len(train_dataset)", EXPECTED["train_examples"], n_train)

    # ---- Minimal model for DataLoader inspection ----
    # We need a model for the Trainer constructor, but we can use a
    # tiny dummy to avoid loading 8B params for a config check.
    # However, the Trainer needs a real model.  We load the real tokenizer
    # and create the tokenized dataset, then construct the Trainer with
    # a real model loaded in 4-bit to match production.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize (same fn as train_qlora.py)
    def tokenize_fn(examples):
        batch_messages = [msgs for msgs in examples["messages"]]
        tokenized = tokenizer.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=False,
            truncation=True, max_length=EXPECTED["max_seq_length"], return_dict=True,
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

    ds = Dataset.from_list(train_data)
    tokenized = ds.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["messages"])

    # ---- Load real model (4-bit) to match production ----
    print("  Loading model for Trainer construction (4-bit)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

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
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)

    # ---- Construct Trainer with EXACT production TrainingArguments ----
    training_args = TrainingArguments(
        output_dir="artifacts/preflight_test",
        num_train_epochs=EXPECTED["num_epochs"],
        per_device_train_batch_size=EXPECTED["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=EXPECTED["gradient_accumulation_steps"],
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        seed=42,
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # ---- G6: DataLoader inspection ----
    train_dl = trainer.get_train_dataloader()
    dl_len = len(train_dl)
    # Expected DataLoader length: ceil(330 / 1) = 330 (with n_gpu=1, world_size=1)
    expected_dl_len = math.ceil(EXPECTED["train_examples"] / EXPECTED["per_device_batch_size"])
    gate("G6 — len(train_dataloader)", expected_dl_len, dl_len)

    # ---- G7: world_size and n_gpu ----
    gate("G7 — trainer.args.n_gpu", EXPECTED["n_gpu"], trainer.args.n_gpu)
    gate("G7 — trainer.args.world_size", EXPECTED["world_size"], trainer.args.world_size)

    # ---- G8: optimizer steps per epoch ----
    actual_steps_per_epoch = dl_len // EXPECTED["gradient_accumulation_steps"]
    gate("G8 — optimizer steps/epoch", EXPECTED["steps_per_epoch"], actual_steps_per_epoch)

    # ---- G9: max_steps ----
    # Trainer computes max_steps from num_train_epochs * steps_per_epoch
    actual_max_steps = trainer.state.max_steps
    # max_steps might be negative (meaning "use epochs") or set by Trainer
    if actual_max_steps is not None and actual_max_steps > 0:
        gate("G9 — trainer.state.max_steps", EXPECTED["max_steps"], actual_max_steps)
    else:
        # Trainer didn't set max_steps explicitly; derive from steps_per_epoch
        expected_max = EXPECTED["max_steps"]
        actual_dl_based = actual_steps_per_epoch * EXPECTED["num_epochs"]
        gate("G9 — derived max_steps (steps/epoch × epochs)", expected_max, actual_dl_based)

    # ---- Cleanup ----
    del model, trainer
    import shutil as _shutil
    _shutil.rmtree("artifacts/preflight_test", ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Trainer config validation summary:")
    print(f"    DataLoader length:   {dl_len}")
    print(f"    n_gpu:               {trainer.args.n_gpu}")
    print(f"    world_size:          {trainer.args.world_size}")
    print(f"    steps/epoch:         {actual_steps_per_epoch}")
    print(f"    max_steps:           {actual_max_steps}")
    print(f"    If any of these had differed from expected, training")
    print(f"    would have proceeded with a silently wrong config.")
    print(f"    This gate prevents that.")


# =============================================================================
# G10: Disk space
# =============================================================================

def g10_disk_space() -> None:
    section("G10 — Disk space")
    target = "/kaggle/working"
    if not Path(target).exists():
        target = "."
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1e9
    total_gb = usage.total / 1e9
    used_gb = usage.used / 1e9
    print(f"  Path:     {target}")
    print(f"  Total:    {total_gb:.1f} GB")
    print(f"  Used:     {used_gb:.1f} GB")
    print(f"  Free:     {free_gb:.1f} GB")
    # We need: ~1.7 GB checkpoints (save_only_model) + model cache + overhead ≈ 5 GB
    min_free = 5.0
    adequate = free_gb >= min_free
    gate(f"Free space >= {min_free:.0f} GB", True, adequate)
    if not adequate:
        print(f"  WARNING: Insufficient disk space.")
        print(f"  Training artifacts need approximately {min_free:.0f} GB.")


# =============================================================================
# G11: Checkpoint save/load smoke test
# =============================================================================

def g11_checkpoint_smoke() -> None:
    """Verify that a LoRA adapter can be saved and reloaded."""
    section("G11 — Checkpoint save/load smoke test")
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    except ImportError as e:
        skip("G11", f"import failed: {e}")
        return

    test_dir = Path("artifacts/preflight_checkpoint_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        compute_dtype = torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config,
            device_map={"": "cuda:0"}, trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

        # Save
        save_path = test_dir / "test_adapter"
        model.save_pretrained(str(save_path))
        gate("save_pretrained() succeeded", True, save_path.exists())
        gate("adapter_config.json present", True, (save_path / "adapter_config.json").exists())
        gate("adapter_model.safetensors present", True,
             (save_path / "adapter_model.safetensors").exists())

        del model
        torch.cuda.empty_cache()

        # Reload
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config,
            device_map={"": "cuda:0"}, trust_remote_code=True,
            torch_dtype=compute_dtype,
        )
        reloaded = PeftModel.from_pretrained(base, str(save_path))
        gate("PEFT reload succeeded", True, reloaded is not None)
        del base, reloaded
        torch.cuda.empty_cache()

    except Exception as e:
        gate("Checkpoint smoke test", "PASS", f"EXCEPTION: {e}")
    finally:
        import shutil as _shutil
        _shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# G12: Generation smoke test
# =============================================================================

def g12_generation_smoke() -> None:
    """Verify generation works with explicit tensor path (never BatchEncoding)."""
    section("G12 — Generation smoke test")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        skip("G12", f"import failed: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map={"": "cuda:0"}, trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."},
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
            max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    gate("Generation produced output", True, len(response) > 0)
    print(f"         Response: {response[:80]}")

    # Regression: verify we did NOT pass a BatchEncoding
    gate("input_ids is a Tensor (not BatchEncoding)",
         "Tensor", type(input_ids).__name__)

    del model
    torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Arwen Policy Pre-flight Gates")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run only CPU-safe gates (G1, G4, G5, G10)")
    parser.add_argument("--trainer-config", action="store_true",
                        help="Run only Trainer configuration gates (G6-G9)")
    parser.add_argument("--gpu-gates", action="store_true",
                        help="Run only GPU gates (G2, G3, G11, G12)")
    args = parser.parse_args()

    run_all = not (args.cpu_only or args.trainer_config or args.gpu_gates)

    print("=" * 60)
    print("  ARWEN POLICY — PRE-FLIGHT VALIDATION")
    if args.cpu_only:
        print("  Mode: CPU-only")
    elif args.trainer_config:
        print("  Mode: Trainer configuration")
    elif args.gpu_gates:
        print("  Mode: GPU gates")
    print("=" * 60)

    # CPU gates
    if run_all or args.cpu_only:
        g1_environment()
        g4_dataset_counts()
        g5_label_masking()
        g10_disk_space()

    # GPU gates (load real model)
    if run_all or args.gpu_gates:
        g2_single_gpu()
        g3_gpu_memory()
        g11_checkpoint_smoke()
        g12_generation_smoke()

    # Trainer config gates (load real model + construct Trainer)
    if run_all or args.trainer_config:
        g6_g9_trainer_config()

    result = report()
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
