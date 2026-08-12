"""Regression tests for QLoRA data pipeline — dataset, tokenization, collation.

Catches the exact error seen on T4:
    ValueError: too many dimensions 'str'
    Unable to create tensor ... features ('text' in this case)
"""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sft_example():
    """A single SFT example matching the exact dataset schema."""
    return {
        "schema_version": "3.0.0",
        "task_type": "policy_question",
        "messages": [
            {
                "role": "system",
                "content": "You are Arwen Policy, a multistakeholder policy-analysis AI. Combine policy reasoning with source evidence when available — evidence grounds your analysis but is not a prerequisite for policy reasoning. Distinguish between general stakeholder perspectives and documented organizational positions. For substantive policy questions, provide multistakeholder analysis. Attribute specific claims to documented sources. Disclose uncertainty. Do not fabricate facts, dates, stakeholder positions, or organizational positions.",
            },
            {
                "role": "user",
                "content": "What policy issue does this document address?",
            },
            {
                "role": "assistant",
                "content": "The document addresses Internet governance and domain name policy.",
            },
        ],
        "source_document_ids": ["d5befe3f-45cc-4070-bfc0-eb9ff7c0e5e5"],
        "source_hashes": ["60ee347bb6a355d262613430e6775cd7514fb4c8ca0a811010f3331101d69db6"],
        "source_urls": ["https://www.icann.org/en/government-engagement"],
        "evidence": [
            {
                "document_id": "d5befe3f-45cc-4070-bfc0-eb9ff7c0e5e5",
                "quote_or_excerpt": "Stay up-to-date on the latest government engagement information.",
                "relevance": "Shows ICANN's engagement with governments.",
            }
        ],
        "stakeholders_mentioned": ["ICANN", "governments"],
        "policy_topics": ["internet-governance"],
        "language": "en",
        "example_id": "test-001",
        "split": "train",
    }


class TestDatasetLoading:
    """Final SFT dataset loads correctly."""

    def test_sft_final_exists(self):
        sft = Path("datasets/sft_final")
        assert sft.is_dir()

    def test_all_splits_present(self):
        for split in ("train", "validation", "test"):
            path = Path("datasets/sft_final") / f"{split}.jsonl"
            assert path.exists(), f"Missing {split}.jsonl"

    def test_split_counts(self):
        expected = {"train": 330, "validation": 39, "test": 37}
        for split, count in expected.items():
            path = Path("datasets/sft_final") / f"{split}.jsonl"
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == count, f"{split}: expected {count}, got {len(lines)}"

    def test_example_has_required_fields(self, sft_example):
        for field in ("messages", "source_document_ids", "source_hashes", "evidence", "task_type"):
            assert field in sft_example, f"Missing field: {field}"

    def test_messages_have_roles(self, sft_example):
        roles = {m["role"] for m in sft_example["messages"]}
        assert "user" in roles or "system" in roles
        assert "assistant" in roles


class TestChatFormatting:
    """Chat template formatting produces strings, not lists."""

    def test_build_text_from_messages(self, sft_example):
        text_parts = []
        for msg in sft_example["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        text = "\n".join(text_parts)

        assert isinstance(text, str)
        assert len(text) > 50
        assert "<|im_start|>" in text
        assert "<|im_end|>" in text

    def test_no_raw_messages_passed_as_text(self, sft_example):
        """Ensure messages list is NOT accidentally passed as a text field."""
        text_parts = []
        for msg in sft_example["messages"]:
            text_parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        text = "\n".join(text_parts)

        # The text must be a plain string, not a list
        assert not isinstance(text, list)
        assert isinstance(text, str)


class TestTokenization:
    """Tokenization produces integer token IDs, not strings."""

    def test_tokenizer_produces_int_ids(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Simulate the exact data path from qlora_smoke_test.py
        text = "<|im_start|>user\nWhat is Internet governance?<|im_end|>\n<|im_start|>assistant\nInternet governance is the system of rules and practices that coordinate the global Internet.<|im_end|>"
        result = tokenizer(text, truncation=True, padding="max_length", max_length=256)

        # input_ids must be integers
        assert "input_ids" in result
        ids = result["input_ids"]
        assert isinstance(ids, list)
        assert all(isinstance(t, int) for t in ids), f"Expected ints, got types: {set(type(t) for t in ids[:5])}"

        # attention_mask must be integers
        assert "attention_mask" in result
        mask = result["attention_mask"]
        assert all(isinstance(m, int) for m in mask)

    def test_batched_tokenization(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        texts = [
            "<|im_start|>user\nWhat is ICANN?<|im_end|>\n<|im_start|>assistant\nICANN is the Internet Corporation for Assigned Names and Numbers.<|im_end|>",
            "<|im_start|>user\nWhat is the IETF?<|im_end|>\n<|im_start|>assistant\nThe IETF develops Internet standards.<|im_end|>",
        ]

        result = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
        ids = result["input_ids"]

        # Batched result must be list of lists of ints
        assert isinstance(ids, list)
        assert len(ids) == 2
        for i, id_list in enumerate(ids):
            assert isinstance(id_list, list), f"Batch {i}: expected list, got {type(id_list)}"
            assert all(isinstance(t, int) for t in id_list), f"Batch {i}: non-int token found"


class TestCollation:
    """Data collator works with tokenized data."""

    def test_datacollator_with_tokenized_batch(self):
        from transformers import AutoTokenizer, DataCollatorForLanguageModeling

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Simulate the exact pipeline
        texts = [
            "<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\nAnswer!<|im_end|>",
            "<|im_start|>user\nQ2?<|im_end|>\n<|im_start|>assistant\nA2!<|im_end|>",
        ]

        tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=128)

        # Simulate HuggingFace Dataset batch format
        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        result = collator([{k: v for k, v in zip(batch.keys(), vals, strict=False)} for vals in zip(*batch.values(), strict=False)])

        assert "input_ids" in result
        assert "labels" in result
        import torch
        assert result["input_ids"].dtype in (torch.long, torch.int, torch.int32, torch.int64)


class TestNoRawStringInBatch:
    """Regression: the exact error 'too many dimensions str' must not occur."""

    def test_text_column_removed_after_tokenization(self):
        """After tokenization, the 'text' column must be removed."""
        from datasets import Dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build a dataset exactly like the smoke test does
        examples = [
            {
                "text": "<|im_start|>user\nWhat is ICANN?<|im_end|>\n<|im_start|>assistant\nICANN coordinates Internet names and numbers.<|im_end|>"
            },
            {
                "text": "<|im_start|>user\nWhat is IETF?<|im_end|>\n<|im_start|>assistant\nIETF develops Internet standards.<|im_end|>"
            },
        ]
        ds = Dataset.from_list(examples)

        def tokenize_fn(exs):
            return tokenizer(exs["text"], truncation=True, padding="max_length", max_length=128)

        tokenized = ds.map(tokenize_fn, batched=True)

        # THIS IS THE FIX: remove the raw 'text' column
        tokenized = tokenized.remove_columns(["text"])

        # Verify text is gone
        assert "text" not in tokenized.column_names
        assert "input_ids" in tokenized.column_names
        assert "attention_mask" in tokenized.column_names

    def test_collator_rejects_text_column(self):
        """Verify that passing a 'text' string field to the collator causes the exact T4 error."""
        from transformers import AutoTokenizer, DataCollatorForLanguageModeling

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Simulate what happens when text column is NOT removed
        bad_batch = [
            {
                "text": "<|im_start|>user\nHello<|im_end|>",
                "input_ids": tokenizer.encode("<|im_start|>user\nHello<|im_end|>"),
                "attention_mask": [1] * 10,
            }
        ]

        # This MUST raise an error about too many dimensions / str
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            collator(bad_batch)


class TestGPUValidationBatchSize:
    """Regression: gpu_validate.py must use batch_size=1, not 4.

    Batch size 4 with seq_len=2048 and Qwen3-8B's ~152K vocab causes CUDA OOM
    on Tesla T4 because logits.float() inside ForCausalLMLoss allocates ~4.64 GB
    per batch dimension for the float32 conversion. Single-example batches keep
    this tensor at ~1.16 GB, leaving headroom on the 14.56 GiB T4.
    """

    def test_validation_batch_size_is_one(self):
        """gpu_validate.py VALIDATION_BATCH_SIZE must be exactly 1."""
        # Parse the constant from the source file
        import ast
        from pathlib import Path

        script = Path("scripts/gpu_validate.py").read_text(encoding="utf-8")
        tree = ast.parse(script)

        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "VALIDATION_BATCH_SIZE":
                        if isinstance(node.value, ast.Constant):
                            assert node.value.value == 1, (
                                f"VALIDATION_BATCH_SIZE must be 1, got {node.value.value}. "
                                "batch_size > 1 causes CUDA OOM on T4 (logits.float() ~4.64 GB at batch=4). "
                                "See the Kaggle validation failure for details."
                            )
                            found = True
        assert found, "VALIDATION_BATCH_SIZE constant not found in gpu_validate.py"

    def test_validation_batch_size_is_not_four(self):
        """Explicit guard: batch_size must not be 4."""
        import ast
        from pathlib import Path

        script = Path("scripts/gpu_validate.py").read_text(encoding="utf-8")
        tree = ast.parse(script)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "VALIDATION_BATCH_SIZE":
                        if isinstance(node.value, ast.Constant):
                            assert node.value.value != 4, (
                                "VALIDATION_BATCH_SIZE=4 causes CUDA OOM on T4. Must be 1."
                            )

    def test_device_map_is_not_auto(self):
        """device_map must be explicit ({"": "cuda:0"}), not "auto".

        device_map="auto" splits the model across all GPUs, which on a dual-T4
        system can cause unpredictable OOMs when GPU 1 receives the logits.float()
        allocation during loss computation.
        """
        script = Path("scripts/gpu_validate.py").read_text(encoding="utf-8")
        # Must contain the exact device_map restriction
        assert 'device_map={"": "cuda:0"}' in script or \
               "device_map={'': 'cuda:0'}" in script or \
               'device_map={"":"cuda:0"}' in script, (
            "gpu_validate.py must use device_map={\"\": \"cuda:0\"} to force single-GPU. "
            "device_map=\"auto\" splits across GPUs and causes OOM on dual-T4 Kaggle."
        )

    def test_cuda_set_device_is_called(self):
        """torch.cuda.set_device(0) must be called to lock the default device."""
        script = Path("scripts/gpu_validate.py").read_text(encoding="utf-8")
        assert "torch.cuda.set_device(0)" in script, (
            "gpu_validate.py must call torch.cuda.set_device(0) to prevent "
            "accidental allocation on other GPUs."
        )

    def test_batch_moved_to_explicit_cuda0(self):
        """Batch must be moved to 'cuda:0', not 'cuda' (which may default elsewhere)."""
        script = Path("scripts/gpu_validate.py").read_text(encoding="utf-8")
        # The batch move must target cuda:0 explicitly
        assert 'to("cuda:0")' in script, (
            "Batch must be moved to 'cuda:0' explicitly, not 'cuda'. "
            "The default device may change if another GPU is initialized first."
        )


class TestTrainingSingleGPU:
    """Regression: train_qlora.py must force single-GPU (cuda:0) for 20-epoch run.

    device_map="auto" on dual-T4 Kaggle splits the model across both GPUs.
    During training, ForCausalLMLoss.logits.float() allocates a [1, 2048, 151936]
    float32 tensor (~1.16 GB) on whichever GPU holds lm_head. With model layers
    already on that GPU, this can OOM mid-training — potentially after hours.
    """

    def test_training_device_map_is_not_auto(self):
        """train_qlora.py model loading must use device_map={"": "cuda:0"}, not "auto"."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert 'device_map={"": "cuda:0"}' in script or \
               "device_map={'': 'cuda:0'}" in script or \
               'device_map={"":"cuda:0"}' in script, (
            "train_qlora.py must use device_map={\"\": \"cuda:0\"} to force single-GPU. "
            "device_map=\"auto\" splits across GPUs and can OOM mid-training on dual-T4."
        )

    def test_training_device_map_not_auto_string(self):
        """The string 'device_map=\"auto\"' must not appear in the model loading section."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        # Only check the model-loading section (around the AutoModelForCausalLM call)
        # The word "auto" may appear elsewhere (e.g., device_map="auto" in comments)
        import re
        # Find all device_map assignments
        matches = re.findall(r'device_map\s*=\s*"auto"', script)
        assert len(matches) == 0, (
            f"Found device_map=\"auto\" in train_qlora.py ({len(matches)} occurrence(s)). "
            "Must be device_map={\"\": \"cuda:0\"} for single-GPU training."
        )

    def test_training_cuda_set_device(self):
        """train_qlora.py must call torch.cuda.set_device(0) before model loading."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "torch.cuda.set_device(0)" in script, (
            "train_qlora.py must call torch.cuda.set_device(0) to lock the default device."
        )


class TestDynamicPadding:
    """Regression: tokenization must use dynamic padding, not static max_length.

    Static padding to 2048 forces every training batch to [1, 2048], inflating
    attention memory by ~12× (2048² vs actual ~170² per example). On T4 with
    14.56 GiB VRAM, the wasted attention + hidden-state memory pushes peak VRAM
    over the limit when combined with AdamW optimizer states (~350 MB).
    Dynamic padding reduces peak VRAM from ~14.6 GiB to ~4-6 GiB.
    """

    def test_tokenize_fn_has_no_static_max_length_padding(self):
        """tokenize_fn must NOT use padding=\"max_length\"."""
        import re
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        # Find padding="max_length" inside the tokenize_fn (not in comments)
        matches = re.findall(r'padding\s*=\s*"max_length"', script)
        assert len(matches) == 0, (
            f"Found padding=\"max_length\" in train_qlora.py ({len(matches)} occurrence(s)). "
            "Static padding to 2048 causes CUDA OOM on T4. Use dynamic padding."
        )

    def test_collator_uses_dynamic_padding(self):
        """Collator must use DataCollatorForSeq2Seq with padding=True.

        DataCollatorWithPadding cannot pad the 'labels' field when sequences
        have different lengths — tokenizer.pad() only handles input_ids and
        attention_mask. DataCollatorForSeq2Seq explicitly pads labels with
        label_pad_token_id=-100.
        """
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "DataCollatorForSeq2Seq" in script, (
            "Must use DataCollatorForSeq2Seq (not DataCollatorWithPadding) for dynamic padding. "
            "DataCollatorWithPadding cannot pad labels with varying lengths."
        )
        assert "label_pad_token_id=-100" in script, (
            "DataCollatorForSeq2Seq must use label_pad_token_id=-100 to pad label positions."
        )
        assert "padding=True" in script, (
            "Collator must use padding=True for dynamic per-batch padding."
        )

    def test_gpu_validate_has_no_static_padding(self):
        """gpu_validate.py must also use dynamic padding with DataCollatorForSeq2Seq."""
        import re
        script = Path("scripts/gpu_validate.py").read_text(encoding="utf-8")
        matches = re.findall(r'padding\s*=\s*"max_length"', script)
        assert len(matches) == 0, (
            f"Found padding=\"max_length\" in gpu_validate.py ({len(matches)} occurrence(s))."
        )
        assert "DataCollatorForSeq2Seq" in script, (
            "gpu_validate.py must use DataCollatorForSeq2Seq for label-compatible dynamic padding."
        )

    def test_dynamic_padding_produces_variable_lengths(self):
        """Tokenized examples must have different lengths (not all 2048)."""
        from pathlib import Path as P
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use the EXACT tokenize_fn logic
        def tokenize_fn(examples):
            batch_messages = [msgs for msgs in examples["messages"]]
            tokenized = tokenizer.apply_chat_template(
                batch_messages, tokenize=True, add_generation_prompt=False,
                truncation=True, max_length=2048, return_dict=True,
            )
            return tokenized

        from datasets import Dataset as DS
        data = []
        for line in P("datasets/sft_final/train.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = [{"role": m["role"], "content": m["content"]} for m in ex["messages"]]
            data.append({"messages": msgs})
            if len(data) >= 10:
                break

        ds = DS.from_list(data)
        tok = ds.map(tokenize_fn, batched=True)
        lengths = [len(row["input_ids"]) for row in tok]

        # At least one example must be shorter than 2048
        assert any(l < 2048 for l in lengths), (
            f"All tokenized lengths are >= 2048: {lengths}. "
            "Dynamic padding is not working — static 2048-padding wastes GPU memory."
        )
        # Examples must have varying lengths (proof of no static padding)
        assert len(set(lengths)) > 1, (
            f"All lengths identical ({lengths[0]}). Static padding detected."
        )


class TestEvalTestSetCLI:
    """Regression: eval_test_set.py --qualitative-only flag must be accepted."""

    def test_qualitative_only_flag_accepted(self):
        """--qualitative-only must be a recognized argument."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/eval_test_set.py", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--qualitative-only" in result.stdout, (
            "eval_test_set.py must accept --qualitative-only flag."
        )


class TestEarlyStoppingAndSafety:
    """Regression: late-training early stopping + post-training memory safety."""

    def test_early_stopping_start_epoch_is_10(self):
        """Early stopping must not start before epoch 10."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "EARLY_STOPPING_START_EPOCH = 10" in script

    def test_early_stopping_patience_is_2(self):
        """Patience must be 2 completed validation evaluations."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "EARLY_STOPPING_PATIENCE = 2" in script

    def test_late_early_stopping_callback_exists(self):
        """The custom LateEarlyStoppingCallback must be defined."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "class LateEarlyStoppingCallback" in script
        assert "should_training_stop" in script

    def test_early_stopping_resets_patience_on_improvement(self):
        """Improvement must reset the patience counter."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "self.patience_counter = 0" in script

    def test_no_stale_step_calculation(self):
        """No ceil() or // step pre-calculation must remain."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "math.ceil(n_train" not in script
        assert "steps_per_epoch = math.ceil" not in script

    def test_trainer_is_authoritative_for_steps(self):
        """The Trainer's own values must be used, not independent calc."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "trainer.get_train_dataloader()" in script
        assert "Trainer steps/epoch" in script

    def test_model_released_before_qualitative_eval(self):
        """Training model must be released before qualitative generation."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "del model, trainer" in script
        assert "gc.collect()" in script
        assert "torch.cuda.empty_cache()" in script

    def test_qualitative_eval_has_oom_guard(self):
        """Qualitative eval failure must not fail training."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "torch.cuda.OutOfMemoryError" in script
        assert "qual_failed" in script

    def test_artifact_preservation_guard(self):
        """Artifact must be preserved before qualitative generation."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "arwen_policy_training_output.tar.gz" in script
        assert "artifact_preserved" in script

    def test_no_multi_gpu(self):
        """No DDP, torchrun, deepspeed, or DataParallel."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "torchrun" not in script
        assert "deepspeed" not in script
        assert "DataParallel" not in script

    def test_save_total_limit_is_two(self):
        """Checkpoint retention must be 2."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "SAVE_TOTAL_LIMIT = 2" in script

    def test_save_only_model(self):
        """save_only_model must be configured."""
        script = Path("scripts/train_qlora.py").read_text(encoding="utf-8")
        assert "save_only_model" in script
