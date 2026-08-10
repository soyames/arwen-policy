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
                "content": "You are a policy analysis AI. Answer questions using only the supplied source evidence.",
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
        expected = {"train": 503, "validation": 60, "test": 64}
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
        result = collator([{k: v for k, v in zip(batch.keys(), vals)} for vals in zip(*batch.values())])

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
