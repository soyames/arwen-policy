"""Supervised training dataset builder for Arwen Policy.

Transforms canonical corpus documents into provenance-preserving
SFT examples using an OpenRouter teacher model.

Design:
    1. Load canonical corpus documents
    2. Split at document level (train/val/test)
    3. For each document, generate N task-type examples via teacher
    4. Validate provenance, deduplicate, write output

The canonical corpus is NEVER modified.
Training data is versioned and separated from source corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"
SOURCE_CORPUS_REVISION = "911c82f"  # verified HF revision

TASK_TYPES = [
    "policy_question",
    "stakeholder_position",
    "evidence_extraction",
    "argument_identification",
    "historical_context",
    "institutional_role",
    "document_understanding",
]

# Documents too short for meaningful training examples
MIN_DOC_CHARS = 500

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_canonical_docs(corpus_dir: str = "corpus") -> list[dict[str, Any]]:
    """Load all valid canonical documents."""
    docs = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= MIN_DOC_CHARS:
                docs.append(doc)
        except Exception:
            continue
    return docs


def extract_doc_meta(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract standardized metadata from a canonical document."""
    meta = doc.get("metadata") or {}
    return {
        "document_id": doc.get("document_id", ""),
        "artifact_sha256": doc.get("artifact_sha256", ""),
        "source_url": doc.get("source_url", "") or doc.get("final_url", ""),
        "title": meta.get("title", "Untitled"),
        "source": classify_source_from_url(
            doc.get("source_url", "") or doc.get("final_url", "")
        ),
        "published_at": str(meta.get("published_at", "")),
        "language": meta.get("language", "en"),
        "content_type": doc.get("content_type", ""),
        "extraction_method": doc.get("extraction_method", ""),
    }


def classify_source_from_url(url: str) -> str:
    u = url.lower()
    for domain, name in [
        ("icann.org", "ICANN"), ("ietf.org", "IETF"), ("rfc-editor.org", "IETF"),
        ("itu.int", "ITU"), ("intgovforum.org", "IGF"),
        ("internetsociety.org", "ISOC"), ("oecd.org", "OECD"),
        ("un.org", "UN"), ("unesco.org", "UNESCO"),
        ("arin.net", "ARIN"), ("ripe.net", "RIPE"),
        ("apnic.net", "APNIC"), ("lacnic.net", "LACNIC"),
        ("afrinic.net", "AFRINIC"), ("europa.eu", "EU"),
        ("arxiv.org", "Academic"), ("iana.org", "IANA"),
        ("wto.org", "WTO"),
    ]:
        if domain in u:
            return name
    return "other"


# ---------------------------------------------------------------------------
# Document-level split
# ---------------------------------------------------------------------------


def split_documents(
    docs: list[dict[str, Any]],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split documents deterministically at the document level.

    No document appears in more than one split, preventing leakage.
    """
    import random

    rng = random.Random(seed)
    shuffled = list(docs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


# ---------------------------------------------------------------------------
# Example ID and hashing
# ---------------------------------------------------------------------------


def compute_example_id(example: dict[str, Any]) -> str:
    """Deterministic example ID from content."""
    content = json.dumps({
        "question": (example.get("messages") or [{}])[-1].get("content", ""),
        "answer": (example.get("messages") or [{}])[-1].get("content", ""),
        "doc_ids": example.get("source_document_ids", []),
        "task": example.get("task_type", ""),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def example_content_hash(example: dict[str, Any]) -> str:
    """Hash for duplicate detection."""
    question = ""
    answer = ""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            question = msg.get("content", "")
        elif msg.get("role") == "assistant":
            answer = msg.get("content", "")
    content = question + answer
    return hashlib.sha256(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_example(example: dict[str, Any]) -> list[str]:
    """Comprehensive validation of a training example."""
    errors = []

    # Required fields
    if not example.get("messages"):
        errors.append("missing messages")
    else:
        msgs = example["messages"]
        roles = {m.get("role") for m in msgs if isinstance(m, dict)}
        if "user" not in roles:
            errors.append("missing user message")
        if "assistant" not in roles:
            errors.append("missing assistant message")

    # Provenance
    if not example.get("source_document_ids"):
        errors.append("missing source_document_ids")
    if not example.get("source_hashes"):
        errors.append("missing source_hashes")
    if not example.get("evidence"):
        errors.append("missing evidence")

    # Task type
    if example.get("task_type") not in TASK_TYPES:
        errors.append(f"invalid task_type: {example.get('task_type')}")

    # Content quality
    for msg in example.get("messages", []):
        content = msg.get("content", "")
        if len(content) < 10:
            errors.append(f"{msg.get('role')} message too short ({len(content)} chars)")

    # No empty evidence
    valid_evidence = [
        e for e in example.get("evidence", [])
        if isinstance(e, dict) and e.get("quote_or_excerpt")
    ]
    if not valid_evidence and example.get("evidence"):
        errors.append("evidence exists but no valid entries")

    return errors


def validate_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a batch and detect duplicates."""
    if not examples:
        return {"valid": True, "total": 0, "error_count": 0, "duplicates": 0, "errors": []}

    errors_list = []
    seen_hashes: set[str] = set()
    duplicates = 0

    for i, ex in enumerate(examples):
        errs = validate_example(ex)
        if errs:
            errors_list.append({"index": i, "errors": errs})

        h = example_content_hash(ex)
        if h in seen_hashes:
            duplicates += 1
            errors_list.append({"index": i, "errors": ["DUPLICATE"]})
        seen_hashes.add(h)

    return {
        "valid": len(errors_list) == 0,
        "total": len(examples),
        "error_count": len(errors_list),
        "duplicates": duplicates,
        "errors": errors_list[:30],
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_example_for_doc(
    doc: dict[str, Any],
    task_type: str,
    openrouter_available: bool = False,
) -> dict[str, Any] | None:
    """Generate one training example for a document.

    If openrouter_available is False, uses deterministic template-based
    generation that preserves provenance but doesn't use a teacher model.
    """
    meta = extract_doc_meta(doc)
    text = doc.get("text", "")
    doc_id = meta["document_id"]
    doc_hash = meta["artifact_sha256"]
    source_url = meta["source_url"]
    title = meta["title"]

    if openrouter_available:
        return _generate_with_teacher(doc, meta, task_type)

    # Deterministic template-based generation (no teacher model)
    return _generate_template(doc, meta, task_type)


def _generate_template(
    doc: dict[str, Any],
    meta: dict[str, Any],
    task_type: str,
) -> dict[str, Any] | None:
    """Template-based example generation. Always deterministic."""
    text = doc.get("text", "")
    title = meta["title"]
    source = meta["source"]
    doc_id = meta["document_id"]
    doc_hash = meta["artifact_sha256"]
    source_url = meta["source_url"]
    pub_date = meta.get("published_at", "unknown")

    # Extract a representative passage for evidence
    passage = _extract_key_passage(text, task_type)

    templates = {
        "document_understanding": {
            "question": f"What is the policy significance of '{title}' ({source})?",
            "answer": (
                f"The document '{title}' from {source}"
                + (f" (published {pub_date})" if pub_date and pub_date != "unknown" else "")
                + f" addresses Internet governance and digital policy. "
                f"Key content includes: {passage[:300]}... "
                f"[Source: {source_url}, Document ID: {doc_id}]"
            ),
        },
        "policy_question": {
            "question": f"What policy issue does '{title}' from {source} address?",
            "answer": (
                f"Based on the document '{title}' ({source}), the policy issues addressed "
                f"include Internet governance and digital policy matters. "
                f"The document content indicates: {passage[:300]}... "
                f"[Grounded in document {doc_id}, accessed via {source_url}]"
            ),
        },
        "stakeholder_position": {
            "question": f"What stakeholders are represented in '{title}' ({source})?",
            "answer": (
                f"The document '{title}' from {source} involves stakeholders relevant to "
                f"Internet governance and digital policy. "
                f"Based on the source: {passage[:300]}... "
                f"[Source document: {doc_id}]"
            ),
        },
        "evidence_extraction": {
            "question": f"What factual claims does '{title}' make about digital policy?",
            "answer": (
                f"From '{title}' ({source}): The document presents information about "
                f"Internet governance. Key content: {passage[:300]}... "
                f"[Evidence from document {doc_id}, {source_url}]"
            ),
        },
        "argument_identification": {
            "question": f"What policy arguments are presented in '{title}'?",
            "answer": (
                f"The document '{title}' from {source} presents arguments related to "
                f"digital policy and Internet governance. "
                f"Content analysis: {passage[:300]}... "
                f"[Source: {doc_id}]"
            ),
        },
        "historical_context": {
            "question": f"What is the historical context of '{title}' ({source})?",
            "answer": (
                f"The document '{title}' from {source}"
                + (f" dates to {pub_date}" if pub_date and pub_date != "unknown" else "")
                + f". It provides perspective on Internet governance during its period. "
                f"Document excerpt: {passage[:300]}... "
                f"[Historical source: {doc_id}, {source_url}]"
            ),
        },
        "institutional_role": {
            "question": f"What is the institutional role of {source} as reflected in '{title}'?",
            "answer": (
                f"{source} is a key institution in Internet governance and digital policy. "
                f"The document '{title}' reflects its role in this domain. "
                f"Content: {passage[:300]}... "
                f"[Institutional source: {doc_id}]"
            ),
        },
    }

    tmpl = templates.get(task_type, templates["document_understanding"])

    evidence = [{
        "document_id": doc_id,
        "source_hash": doc_hash,
        "source_url": source_url,
        "quote_or_excerpt": passage[:500],
        "explanation": f"Extracted from document body for {task_type} task",
    }]

    return {
        "task_type": task_type,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Arwen Policy, a multistakeholder policy-analysis AI. "
                    "Combine policy reasoning with source evidence when available. "
                    "Distinguish between general stakeholder perspectives and "
                    "documented organizational positions. Preserve stakeholder "
                    "disagreement and disclose missing perspectives. "
                    "Attribute specific claims to documented sources."
                ),
            },
            {"role": "user", "content": tmpl["question"]},
            {"role": "assistant", "content": tmpl["answer"]},
        ],
        "source_document_ids": [doc_id],
        "source_hashes": [doc_hash],
        "source_urls": [source_url],
        "evidence": evidence,
        "stakeholders_mentioned": [],
        "policy_topics": [],
        "language": meta.get("language", "en"),
    }


def _generate_with_teacher(
    doc: dict[str, Any],
    meta: dict[str, Any],
    task_type: str,
) -> dict[str, Any] | None:
    """Generate using OpenRouter teacher model."""
    from arwen_etl.engine.openrouter_provider import (
        build_teacher_prompt,
        call_openrouter,
        parse_teacher_response,
    )

    messages = build_teacher_prompt(doc.get("text", ""), meta, task_type)
    result = call_openrouter(messages)

    if not result["success"]:
        logger.warning("OpenRouter failed for %s: %s", meta["document_id"][:12], result["error"])
        return None

    example = parse_teacher_response(result["content"], task_type, meta)
    if example:
        example["generation_model"] = result.get("model", "unknown")
        example["generation_timestamp"] = datetime.now(timezone.utc).isoformat()
    return example


def _extract_key_passage(text: str, task_type: str) -> str:
    """Extract a representative passage from document text."""
    # Skip navigation-heavy beginnings, find substantive content
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    if not lines:
        return text[:500]

    # Find a substantive paragraph (skip headers, navigation)
    for line in lines:
        if len(line) > 80 and not line.startswith(("Skip", "Search", "Menu", "Home", "Log")):
            return line[:500]

    return lines[0][:500] if lines else text[:500]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_sft_dataset(
    corpus_dir: str = "corpus",
    output_dir: str = "datasets/sft",
    task_types: list[str] | None = None,
    tasks_per_doc: int = 2,
    use_teacher: bool = False,
    seed: int = 42,
    dry_run: bool = False,
    _skip_min_check: bool = False,
) -> dict[str, Any]:
    """Build the complete SFT dataset from canonical corpus.

    Returns statistics and writes dataset files.
    """
    start = time.time()
    task_types = task_types or TASK_TYPES

    # 1. Load documents
    docs = load_canonical_docs(corpus_dir)
    if len(docs) < 10 and not _skip_min_check:
        return {"status": "BLOCKED", "reason": f"Only {len(docs)} valid documents"}

    # 2. Document-level split
    splits = split_documents(docs, seed=seed)

    # 3. Generate examples
    output = Path(output_dir)
    stats = {
        "status": "OK",
        "total_documents": len(docs),
        "documents_per_split": {k: len(v) for k, v in splits.items()},
        "corpus_revision": SOURCE_CORPUS_REVISION,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "task_types": task_types,
        "tasks_per_doc": tasks_per_doc,
        "use_teacher": use_teacher,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if use_teacher:
        openrouter_ok = bool(os.environ.get("OPENROUTER_API_KEY"))
        if not openrouter_ok:
            return {"status": "BLOCKED", "reason": "OPENROUTER_API_KEY not set"}

    all_split_data: dict[str, list[dict[str, Any]]] = {}
    total_examples = 0
    failures = 0
    rejects = 0

    for split_name in ("train", "validation", "test"):
        split_docs = splits[split_name]
        examples: list[dict[str, Any]] = []
        split_failures = 0

        for doc in split_docs:
            # Select task types for this document
            import random
            rng = random.Random(hash(doc.get("document_id", "")) + seed)
            doc_tasks = rng.sample(task_types, min(tasks_per_doc, len(task_types)))

            for task in doc_tasks:
                try:
                    example = generate_example_for_doc(
                        doc, task, openrouter_available=use_teacher,
                    )
                    if example:
                        example["example_id"] = compute_example_id(example)
                        example["split"] = split_name
                        example["schema_version"] = SCHEMA_VERSION
                        examples.append(example)
                    else:
                        rejects += 1
                except Exception as e:
                    logger.warning("Failed doc %s task %s: %s",
                                   doc.get("document_id", "?")[:12], task, e)
                    split_failures += 1
                    failures += 1

            if use_teacher and not dry_run:
                time.sleep(0.3)  # Rate limiting for free tier

        all_split_data[split_name] = examples
        total_examples += len(examples)

    # 4. Validate
    all_examples = []
    for split_name in ("train", "validation", "test"):
        all_examples.extend(all_split_data[split_name])

    validation = validate_batch(all_examples)

    # 5. Compute distributions
    year_dist = Counter()
    source_dist = Counter()
    task_dist = Counter()
    lang_dist = Counter()

    for ex in all_examples:
        task_dist[ex.get("task_type", "?")] += 1
        lang_dist[ex.get("language", "und")] += 1
        for doc_id in ex.get("source_document_ids", []):
            for d in docs:
                if d.get("document_id") == doc_id:
                    src = classify_source_from_url(d.get("source_url", "") or d.get("final_url", ""))
                    source_dist[src] += 1
                    meta = d.get("metadata") or {}
                    pub = meta.get("published_at", "")
                    if pub and len(str(pub)) >= 4:
                        try:
                            year_dist[int(str(pub)[:4])] += 1
                        except ValueError:
                            pass
                    break

    # 6. Write output
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "validation", "test"):
            split_data = all_split_data[split_name]
            if not split_data:
                continue
            path = output / f"{split_name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for ex in split_data:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # Write metadata
        stats["total_examples"] = total_examples
        stats["validation"] = validation
        stats["year_distribution"] = dict(sorted(year_dist.items()))
        stats["source_distribution"] = dict(source_dist.most_common())
        stats["task_distribution"] = dict(task_dist)
        stats["language_distribution"] = dict(lang_dist)
        stats["failures"] = failures
        stats["rejects"] = rejects
        stats["elapsed_seconds"] = round(time.time() - start, 1)

        meta_path = output / "dataset_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    stats["total_examples"] = total_examples
    stats["validation_summary"] = {
        "valid": validation["valid"],
        "errors": validation["error_count"],
        "duplicates": validation["duplicates"],
    }
    stats["year_distribution"] = dict(sorted(year_dist.items()))
    stats["source_distribution"] = dict(source_dist.most_common())
    stats["task_distribution"] = dict(task_dist)
    stats["failures"] = failures
    stats["rejects"] = rejects
    stats["elapsed_seconds"] = round(time.time() - start, 1)
    stats["output_dir"] = str(output) if not dry_run else "(dry-run)"

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Arwen Policy SFT Dataset Builder")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--output-dir", default="datasets/sft")
    parser.add_argument("--tasks-per-doc", type=int, default=2)
    parser.add_argument("--use-teacher", action="store_true",
                        help="Use OpenRouter teacher model (requires OPENROUTER_API_KEY)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0,
                        help="Process only N documents (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate stats without writing files")
    args = parser.parse_args()

    if args.sample > 0:
        # Sample mode: limit documents for testing
        docs = load_canonical_docs(args.corpus_dir)
        import random
        rng = random.Random(args.seed)
        sampled = rng.sample(docs, min(args.sample, len(docs)))

        # Write sampled docs to temp corpus
        tmp_dir = Path("data/sft_sample_corpus")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for d in sampled:
            path = tmp_dir / f"{d['document_id']}.json"
            path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        args.corpus_dir = str(tmp_dir)

    stats = build_sft_dataset(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        tasks_per_doc=args.tasks_per_doc,
        use_teacher=args.use_teacher,
        seed=args.seed,
        dry_run=args.dry_run,
        _skip_min_check=args.sample > 0,
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
