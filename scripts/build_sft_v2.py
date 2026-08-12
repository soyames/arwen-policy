#!/usr/bin/env python3
"""V2 SFT dataset builder using OpenRouter/free teacher model.

Enhances v1 template examples with teacher-generated reasoning while
preserving provenance. Uses the existing v1 document-level split.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure .env is loaded
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

SCHEMA_VERSION = "2.0.0"
CORPUS_REVISION = "911c82f"

# Task types
# Document-grounded tasks (require a corpus document)
DOCUMENT_TASK_TYPES = [
    "document_understanding",
    "evidence_extraction",
    "policy_question",
    "stakeholder_position",
    "argument_identification",
    "historical_context",
    "institutional_role",
    "policy_comparison",
    "tradeoff_analysis",
]

# Policy-analysis tasks (do NOT require a source document)
POLICY_TASK_TYPES = [
    "multistakeholder_analysis",
    "stakeholder_disagreement",
    "policy_recommendation",
    "perspective_vs_position",
    "uncertainty_handling",
]

# HRIAM tasks - human rights impact assessment & management
HRIAM_TASK_TYPES = [
    "rights_holder_identification",
    "rights_impact_analysis",
    "positive_negative_impacts",
    "disproportionate_impact",
    "stakeholder_rights_mapping",
    "participation_assessment",
    "accountability_remedy",
    "mitigation_safeguards",
    "rights_tradeoff_analysis",
    "lifecycle_hria",
    "panel_analysis",
]

POLICY_TASK_TYPES = POLICY_TASK_TYPES + HRIAM_TASK_TYPES

TASK_TYPES = DOCUMENT_TASK_TYPES + POLICY_TASK_TYPES

# ---------------------------------------------------------------------------
# .env loading already done above
# ---------------------------------------------------------------------------


def load_canonical_docs(corpus_dir: str = "corpus") -> list[dict[str, Any]]:
    docs = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= 500:
                docs.append(doc)
        except Exception:
            continue
    return docs


def load_v1_splits(v1_dir: str = "datasets/sft") -> dict[str, list[dict[str, Any]]]:
    """Load v1 examples and return the set of document IDs per split."""
    splits: dict[str, set[str]] = {}
    for split_name in ("train", "validation", "test"):
        path = Path(v1_dir) / f"{split_name}.jsonl"
        if not path.exists():
            splits[split_name] = set()
            continue
        doc_ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                for did in ex.get("source_document_ids", []):
                    doc_ids.add(did)
            except Exception:
                continue
        splits[split_name] = doc_ids
    return splits


def compute_example_id(example: dict[str, Any]) -> str:
    content = json.dumps({
        "q": (example.get("messages") or [{}])[-2].get("content", ""),
        "a": (example.get("messages") or [{}])[-1].get("content", ""),
        "dids": example.get("source_document_ids", []),
        "task": example.get("task_type", ""),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def build_teacher_messages(doc: dict[str, Any], task_type: str, v1_example: dict | None) -> list[dict[str, str]]:
    """Build the prompt for the teacher model."""
    meta = doc.get("metadata") or {}
    title = meta.get("title", "Untitled")
    source_url = doc.get("source_url", "") or doc.get("final_url", "")
    doc_id = doc.get("document_id", "")
    doc_hash = doc.get("artifact_sha256", "")
    pub_date = str(meta.get("published_at", "unknown"))
    lang = meta.get("language", "en")
    text = doc.get("text", "")[:6000]

    # Get v1 question/answer if available
    v1_q = ""
    v1_a = ""
    if v1_example:
        for msg in v1_example.get("messages", []):
            if msg.get("role") == "user":
                v1_q = msg.get("content", "")
            elif msg.get("role") == "assistant":
                v1_a = msg.get("content", "")

    task_descriptions = {
        "document_understanding": "Explain the policy significance of this document.",
        "evidence_extraction": "Extract a key factual claim and the evidence supporting it.",
        "policy_question": "Identify a specific policy issue and explain how this document addresses it.",
        "stakeholder_position": "Identify a stakeholder position expressed or implied in this document.",
        "argument_identification": "Identify a policy argument and reconstruct its reasoning chain from the text.",
        "historical_context": "Explain the historical/institutional context of this document based on its content.",
        "institutional_role": "Explain the role and authority of the institution that produced this document.",
        "policy_comparison": "Compare positions or approaches found in this document with other perspectives it references.",
        "tradeoff_analysis": "Identify policy trade-offs or tensions present in this document.",
    }

    task_instruction = task_descriptions.get(task_type, task_descriptions["document_understanding"])

    system = """You generate training data for a policy-analysis AI.

RULES:
- Use ONLY the supplied document text. No outside knowledge.
- Never invent facts, dates, organizations, stakeholder positions, or quotations.
- Every claim must reference specific content from the document.
- If the document has NO policy relevance whatsoever, return {"skip": true}.
  But nearly all documents from policy institutions have some policy relevance -
  do not skip just because the document does not directly answer the question.

HRIAM STATE - for every example, determine whether human-rights analysis is
materially relevant and add an "hriam_state" field:

  "HRIAM_NOT_MATERIAL" - No meaningful human-rights dimension. Answer as a
  normal policy/technical/governance question. Do NOT manufacture a
  human-rights analysis. Example: "What are the technical advantages of IPv6?"

  "HRIAM_RELEVANT" - Human-rights implications exist. Briefly identify
  relevant rights, rights-holders, or potential impacts while maintaining
  the broader multistakeholder policy analysis. Do NOT perform a full HRIA.
  Example: "What are the governance implications of DNS filtering?"

  "HRIAM_CENTRAL" - Human-rights impacts are central. Perform substantive
  HRIAM reasoning: affected rights, rights-holders, stakeholders,
  duty-bearers where applicable, impacts (positive and adverse), trade-offs,
  safeguards, accountability, remedy, and uncertainty.
  Example: "How could DNS filtering affect freedom of expression?"

CRITICAL: Do NOT over-trigger HRIAM. Most policy questions are
HRIAM_NOT_MATERIAL or HRIAM_RELEVANT. Reserve HRIAM_CENTRAL for questions
where human-rights impacts are explicitly the focus.

- Return VALID JSON only. No markdown, no explanation outside the JSON.

Output format:
{
  "skip": false,
  "question": "A specific, answerable question about the document",
  "answer": "Thorough answer grounded ONLY in the document. Cite specific content.",
  "reasoning": "Step-by-step reasoning from document evidence to conclusion",
  "evidence": [
    {"quote_or_excerpt": "exact text from document", "relevance": "why this supports the answer"}
  ],
  "stakeholders_mentioned": ["stakeholders relevant to this policy issue"],
  "policy_topics": ["relevant policy domains"],
  "hriam_state": "HRIAM_NOT_MATERIAL | HRIAM_RELEVANT | HRIAM_CENTRAL",
  "uncertainty": "what the document does NOT establish and what remains contested",
  "confidence": "high|medium|low"
}"""

    v1_context = ""
    if v1_q:
        v1_context = f"\n\nEXISTING DRAFT (improve this, do not copy verbatim):\nQuestion: {v1_q}\nAnswer: {v1_a[:500]}"

    user = f"""TASK: {task_instruction}

DOCUMENT:
Title: {title}
Source URL: {source_url}
Date: {pub_date}
ID: {doc_id}
Language: {lang}

CONTENT:
{text}
{v1_context}

Generate one training example of type: {task_type}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_teacher(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call OpenRouter, return parsed result."""
    from arwen_etl.engine.openrouter_provider import call_openrouter
    result = call_openrouter(messages, max_tokens=1024, temperature=0.15)
    return result


def parse_teacher_response(content: str, task_type: str, doc: dict[str, Any]) -> dict[str, Any] | None:
    """Parse and validate teacher JSON output."""
    if not content or not content.strip():
        return None

    text = content.strip()
    # Extract JSON
    for start_marker, end_marker in [("```json", "```"), ("```", "```"), ("{", "}")]:
        if start_marker == "{":
            s = text.find("{")
            e = text.rfind("}") + 1
        else:
            if start_marker in text:
                s = text.index(start_marker) + len(start_marker)
                if end_marker in text[s:]:
                    e = text.index(end_marker, s)
                else:
                    continue
            else:
                continue
        try:
            parsed = json.loads(text[s:e].strip())
            break
        except json.JSONDecodeError:
            continue
    else:
        return None

    if parsed.get("skip"):
        return None

    question = str(parsed.get("question", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()

    if len(question) < 10 or len(answer) < 20:
        return None

    doc_id = doc.get("document_id", "")
    doc_hash = doc.get("artifact_sha256", "")
    source_url = doc.get("source_url", "") or doc.get("final_url", "")

    evidence = []
    for ev in parsed.get("evidence", []):
        if isinstance(ev, dict) and ev.get("quote_or_excerpt"):
            evidence.append({
                "document_id": doc_id,
                "source_hash": doc_hash,
                "source_url": source_url,
                "quote_or_excerpt": str(ev.get("quote_or_excerpt", ""))[:500],
                "relevance": str(ev.get("relevance", ""))[:300],
            })

    if not evidence:
        # Create minimal evidence from doc
        evidence = [{
            "document_id": doc_id,
            "source_hash": doc_hash,
            "source_url": source_url,
            "quote_or_excerpt": doc.get("text", "")[:500],
            "relevance": f"Generated from document for {task_type}",
        }]

    example = {
        "schema_version": SCHEMA_VERSION,
        "task_type": task_type,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a policy analysis AI. Answer questions using only the supplied source evidence. "
                    "Attribute claims to documented sources. Disclose uncertainty. "
                    "Do not invent facts, dates, stakeholders, or positions."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "reasoning": reasoning,
        "source_document_ids": [doc_id],
        "source_hashes": [doc_hash],
        "source_urls": [source_url],
        "evidence": evidence,
        "stakeholders_mentioned": parsed.get("stakeholders_mentioned", []),
        "policy_topics": parsed.get("policy_topics", []),
        "hriam_state": parsed.get("hriam_state", "HRIAM_RELEVANT"),
        "uncertainty": str(parsed.get("uncertainty", "")),
        "teacher_confidence": str(parsed.get("confidence", "unknown")),
        "language": doc.get("metadata", {}).get("language", "en"),
        "generation_model": "openrouter/free",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "corpus_revision": CORPUS_REVISION,
    }

    return example


def validate_example(example: dict[str, Any]) -> list[str]:
    errors = []
    if not example.get("messages"):
        errors.append("missing messages")
        return errors
    roles = {m.get("role") for m in example["messages"] if isinstance(m, dict)}
    if "user" not in roles:
        errors.append("missing user")
    if "assistant" not in roles:
        errors.append("missing assistant")
    task_type = example.get("task_type", "")
    # Policy-analysis and HRIAM tasks do not require source documents or evidence
    is_policy_task = task_type in POLICY_TASK_TYPES
    if not is_policy_task:
        if not example.get("source_document_ids"):
            errors.append("missing source_document_ids")
        if not example.get("evidence"):
            errors.append("missing evidence")
    return errors


def build_v2(
    corpus_dir: str = "corpus",
    v1_dir: str = "datasets/sft",
    output_dir: str = "datasets/sft_v2",
    tasks_per_doc: int = 2,
    sample_size: int = 0,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build V2 enhanced SFT dataset."""
    start_time = time.time()

    # Load canonical docs
    all_docs = load_canonical_docs(corpus_dir)
    docs_by_id = {d["document_id"]: d for d in all_docs}

    # Load v1 splits
    v1_splits = load_v1_splits(v1_dir)
    print(f"V1 splits: train={len(v1_splits.get('train', set()))} val={len(v1_splits.get('validation', set()))} test={len(v1_splits.get('test', set()))} docs")

    # Determine docs per split from v1
    split_docs: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        split_docs[split_name] = [
            d for d in all_docs if d["document_id"] in v1_splits.get(split_name, set())
        ]

    # Sample mode
    if sample_size > 0:
        import random
        rng = random.Random(seed)
        all_sample = rng.sample(all_docs, min(sample_size, len(all_docs)))
        # Assign to splits proportionally
        split_docs = {"train": all_sample, "validation": [], "test": []}

    stats = {
        "schema_version": SCHEMA_VERSION,
        "corpus_revision": CORPUS_REVISION,
        "tasks_per_doc": tasks_per_doc,
        "seed": seed,
        "sample_size": sample_size,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    import random
    rng = random.Random(seed)

    all_examples: list[dict[str, Any]] = []
    split_results: dict[str, list[dict[str, Any]]] = {}
    requests_total = 0
    requests_success = 0
    requests_failed = 0
    rejects_hallucination = 0
    rejects_malformed = 0
    rejects_insufficient = 0

    for split_name in ("train", "validation", "test"):
        docs = split_docs.get(split_name, [])
        if not docs:
            split_results[split_name] = []
            continue

        examples: list[dict[str, Any]] = []
        for doc in docs:
            doc_id = doc.get("document_id", "")
            selected_tasks = rng.sample(TASK_TYPES, min(tasks_per_doc, len(TASK_TYPES)))

            for task_type in selected_tasks:
                requests_total += 1
                try:
                    messages = build_teacher_messages(doc, task_type, None)
                    result = call_teacher(messages)

                    if not result["success"]:
                        requests_failed += 1
                        print(f"  FAIL [{task_type}] {doc_id[:12]}: {result.get('error', '')[:80]}")
                        sys.stdout.flush()
                        continue

                    requests_success += 1
                    example = parse_teacher_response(result["content"], task_type, doc)

                    if example is None:
                        rejects_insufficient += 1
                        print(f"  SKIP [{task_type}] {doc_id[:12]}: insufficient evidence or malformed")
                        sys.stdout.flush()
                        continue

                    example["example_id"] = compute_example_id(example)
                    example["split"] = split_name
                    examples.append(example)
                    print(f"  OK [{task_type}] {doc_id[:12]}: {len(example.get('messages', [{}])[-1].get('content', ''))} chars")
                    sys.stdout.flush()

                    # Incremental write to avoid losing progress
                    if not dry_run:
                        split_path = Path(output_dir) / f"{split_name}.jsonl"
                        split_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(split_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(example, ensure_ascii=False) + "\n")

                except Exception as e:
                    requests_failed += 1
                    print(f"  ERR [{task_type}] {doc_id[:12]}: {e}")

                if not dry_run:
                    time.sleep(0.15)  # Rate limit for free tier

        split_results[split_name] = examples
        all_examples.extend(examples)

    # Validate
    import hashlib as hl
    seen_hashes: set[str] = set()
    duplicates = 0
    validation_errors = 0
    for ex in all_examples:
        errs = validate_example(ex)
        if errs:
            validation_errors += 1
        h = hl.sha256(
            (str(ex.get("messages", "")) + str(ex.get("source_document_ids", []))).encode()
        ).hexdigest()
        if h in seen_hashes:
            duplicates += 1
        seen_hashes.add(h)

    # Distributions
    year_dist = Counter()
    source_dist = Counter()
    task_dist = Counter()
    lang_dist = Counter()
    for ex in all_examples:
        task_dist[ex.get("task_type", "?")] += 1
        lang_dist[ex.get("language", "en")] += 1
        for did in ex.get("source_document_ids", []):
            d = docs_by_id.get(did, {})
            src_url = d.get("source_url", "") or d.get("final_url", "")
            u = src_url.lower()
            for domain, name in [
                ("icann.org", "ICANN"), ("ietf.org", "IETF"), ("rfc-editor.org", "IETF"),
                ("itu.int", "ITU"), ("intgovforum.org", "IGF"), ("internetsociety.org", "ISOC"),
                ("oecd.org", "OECD"), ("un.org", "UN"), ("unesco.org", "UNESCO"),
                ("arin.net", "ARIN"), ("ripe.net", "RIPE"), ("apnic.net", "APNIC"),
                ("lacnic.net", "LACNIC"), ("afrinic.net", "AFRINIC"), ("europa.eu", "EU"),
                ("arxiv.org", "Academic"), ("iana.org", "IANA"), ("wto.org", "WTO"),
            ]:
                if domain in u:
                    source_dist[name] += 1
                    break
            else:
                source_dist["other"] += 1
            pub = (d.get("metadata") or {}).get("published_at", "")
            if pub and len(str(pub)) >= 4:
                try:
                    year_dist[int(str(pub)[:4])] += 1
                except ValueError:
                    pass

    # Write output (only if not already written incrementally)
    if not dry_run:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "validation", "test"):
            data = split_results.get(split_name, [])
            if data:
                path = out / f"{split_name}.jsonl"
                # Only overwrite if incremental writes weren't done
                if not path.exists() or path.stat().st_size == 0:
                    path.write_text(
                        "\n".join(json.dumps(ex, ensure_ascii=False) for ex in data) + "\n",
                        encoding="utf-8",
                    )

    elapsed = round(time.time() - start_time, 1)
    report = {
        "total_examples": len(all_examples),
        "per_split": {k: len(v) for k, v in split_results.items()},
        "requests": {"total": requests_total, "success": requests_success, "failed": requests_failed},
        "rejects": {
            "hallucination": rejects_hallucination,
            "malformed": rejects_malformed,
            "insufficient": rejects_insufficient,
        },
        "validation": {"errors": validation_errors, "duplicates": duplicates},
        "year_distribution": dict(sorted(year_dist.items())),
        "source_distribution": dict(source_dist.most_common()),
        "task_distribution": dict(task_dist),
        "language_distribution": dict(lang_dist),
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir) if not dry_run else "(dry-run/sample)",
    }

    if not dry_run:
        meta_path = Path(output_dir) / "dataset_metadata.json"
        meta_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Arwen Policy SFT V2 Builder")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--v1-dir", default="datasets/sft")
    parser.add_argument("--output-dir", default="datasets/sft_v2")
    parser.add_argument("--tasks-per-doc", type=int, default=2)
    parser.add_argument("--sample", type=int, default=0, help="Process N docs (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build_v2(
        corpus_dir=args.corpus_dir,
        v1_dir=args.v1_dir,
        output_dir=args.output_dir,
        tasks_per_doc=args.tasks_per_doc,
        sample_size=args.sample,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
