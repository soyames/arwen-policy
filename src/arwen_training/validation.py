"""Training-data quality validation for Arwen Policy.

All validators are deterministic — no external services or models required.
"""

from __future__ import annotations

import hashlib
from typing import Any


def validate_training_example(example: dict[str, Any]) -> list[str]:
    """Validate a single training example against the Arwen schema."""
    errors: list[str] = []
    if not example.get("messages"):
        errors.append("messages is required")
    else:
        messages = example["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append("messages must be a list with at least 2 entries")
        else:
            roles = {m.get("role") for m in messages if isinstance(m, dict)}
            if "user" not in roles:
                errors.append("messages must contain a 'user' role")
            if "assistant" not in roles:
                errors.append("messages must contain an 'assistant' role")
    if not example.get("evidence"):
        errors.append("evidence is required")
    if not example.get("stakeholder_perspectives"):
        errors.append("stakeholder_perspectives is required")
    return errors


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def validate_batch(
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a batch of training examples and return a quality report."""
    if not examples:
        return {
            "valid": True,
            "total": 0,
            "errors": [],
            "summary": "Empty batch — no examples to validate.",
        }

    all_errors: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    duplicates = 0

    for i, example in enumerate(examples):
        errs = validate_training_example(example)
        if errs:
            all_errors.append({"index": i, "errors": errs})

        # Duplicate detection via content hash
        content = (
            str(example.get("messages", ""))
            + str(example.get("evidence", ""))
        )
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if content_hash in seen_hashes:
            duplicates += 1
            all_errors.append({"index": i, "errors": ["DUPLICATE_EXAMPLE"]})
        seen_hashes.add(content_hash)

    return {
        "valid": len(all_errors) == 0,
        "total": len(examples),
        "error_count": len(all_errors),
        "duplicates": duplicates,
        "errors": all_errors[:20],  # cap for reporting
        "summary": (
            f"{len(examples)} examples: "
            f"{len(all_errors)} errors, {duplicates} duplicates"
        ),
    }


def check_corpus_quality(data_dir: str = "data/extracted") -> dict[str, Any]:
    """Check whether the corpus is sufficient for training."""
    import json
    from collections import Counter
    from pathlib import Path

    extracted_dir = Path(data_dir)
    if not extracted_dir.is_dir():
        return {
            "ready": False,
            "reason": "Extracted data directory does not exist",
            "documents": 0,
        }

    docs = []
    for fpath in extracted_dir.glob("*.json"):
        try:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= 100:
                docs.append(doc)
        except Exception:
            continue

    if len(docs) < 50:
        return {
            "ready": False,
            "reason": f"Only {len(docs)} valid documents — need at least 50 for training",
            "documents": len(docs),
            "recommendation": "Run ETL pipeline against more sources before training",
        }

    # Check source diversity
    sources = Counter(d.get("source_id", "?")[:12] for d in docs)
    if len(sources) < 5:
        return {
            "ready": False,
            "reason": f"Only {len(sources)} unique sources — need at least 5 for diversity",
            "documents": len(docs),
            "unique_sources": len(sources),
        }

    # Check total text volume
    total_chars = sum(len(d.get("text", "")) for d in docs)
    if total_chars < 500_000:
        return {
            "ready": False,
            "reason": f"Only {total_chars} characters — need at least 500K",
            "documents": len(docs),
            "total_characters": total_chars,
        }

    return {
        "ready": True,
        "documents": len(docs),
        "unique_sources": len(sources),
        "total_characters": total_chars,
        "source_distribution": dict(sources.most_common()),
    }
