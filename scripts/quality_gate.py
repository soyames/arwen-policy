#!/usr/bin/env python3
"""SFT quality gate: inspect, compare V1/V2, audit coverage, build final dataset."""

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Load .env
_env = Path(".env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() not in os.environ:
                os.environ[_k.strip()] = _v.strip()

SCHEMA_VERSION = "3.0.0"
CORPUS_REVISION = "911c82f"


# ===================================================================
# 1. LOAD DATA
# ===================================================================

def load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_canonical_docs(corpus_dir: str = "corpus") -> dict[str, dict]:
    docs = {}
    for f in Path(corpus_dir).glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            docs[d.get("document_id", "")] = d
        except Exception:
            continue
    return docs


def classify_source(url: str) -> str:
    u = url.lower()
    for domain, name in [
        ("icann.org", "ICANN"), ("ietf.org", "IETF"), ("rfc-editor.org", "IETF"),
        ("itu.int", "ITU"), ("intgovforum.org", "IGF"), ("internetsociety.org", "ISOC"),
        ("oecd.org", "OECD"), ("un.org", "UN"), ("unesco.org", "UNESCO"),
        ("arin.net", "ARIN"), ("ripe.net", "RIPE"), ("apnic.net", "APNIC"),
        ("lacnic.net", "LACNIC"), ("afrinic.net", "AFRINIC"), ("europa.eu", "EU"),
        ("arxiv.org", "Academic"), ("iana.org", "IANA"), ("wto.org", "WTO"),
    ]:
        if domain in u:
            return name
    return "other"


# ===================================================================
# 2. INSPECT V2 EXAMPLES BY TASK TYPE
# ===================================================================

def inspect_v2(v2_dir: str = "datasets/sft_v2") -> dict:
    """Inspect v2 examples across all task types."""
    all_examples = []
    for split in ("train", "validation", "test"):
        all_examples.extend(load_jsonl(f"{v2_dir}/{split}.jsonl"))

    by_task = defaultdict(list)
    for ex in all_examples:
        by_task[ex.get("task_type", "?")].append(ex)

    print("=" * 60)
    print(f"V2 EXAMPLES: {len(all_examples)} total across {len(by_task)} task types")
    print("=" * 60)

    has_reasoning = 0
    has_uncertainty = 0
    short_answers = 0

    for task_type, examples in sorted(by_task.items()):
        print(f"\n--- {task_type} ({len(examples)} examples) ---")
        for i, ex in enumerate(examples[:2]):  # Show first 2 per task
            msgs = ex.get("messages", [])
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            reasoning = ex.get("reasoning", "")
            uncertainty = ex.get("uncertainty", "")
            evidence = ex.get("evidence", [])

            print(f"\n  Example {i+1}:")
            print(f"  Q: {q[:200]}")
            print(f"  A: {a[:400]}")
            print(f"  Reasoning: {reasoning[:300]}")
            print(f"  Uncertainty: {uncertainty[:200]}")
            print(f"  Evidence entries: {len(evidence)}")
            print(f"  Source: {ex.get('source_document_ids', ['?'])[0][:20]}")

            if reasoning:
                has_reasoning += 1
            if uncertainty:
                has_uncertainty += 1
            if len(a) < 50:
                short_answers += 1

    print("\n=== V2 SUMMARY ===")
    print(f"With reasoning: {has_reasoning}/{len(all_examples)}")
    print(f"With uncertainty: {has_uncertainty}/{len(all_examples)}")
    print(f"Short answers (<50 chars): {short_answers}")

    return {"by_task": dict(by_task), "total": len(all_examples),
            "has_reasoning": has_reasoning, "has_uncertainty": has_uncertainty}


# ===================================================================
# 3. REMOVE PRIVATE CHAIN-OF-THOUGHT
# ===================================================================

def transform_reasoning(reasoning: str) -> dict:
    """Convert private reasoning into structured evidence-grounded analysis."""
    if not reasoning:
        return {}

    # Check if it reads like private/internal reasoning
    private_markers = [
        "I think", "let me", "first, I", "I'll", "I will",
        "step by step", "let's", "we need to", "we should",
        "okay", "now I", "I note", "I see",
    ]
    is_private = any(m in reasoning.lower()[:100] for m in private_markers)

    result = {}

    if is_private:
        # Extract useful analysis, drop internal monologue
        # Keep: evidence statements, factual analysis, conclusions
        lines = reasoning.split("\n")
        evidence_lines = []
        analysis_lines = []
        conclusion_lines = []
        in_analysis = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            # Skip self-referential/internal monologue
            if any(m in lower for m in ["i think", "let me", "i'll", "i will", "okay", "now i"]):
                continue
            # Classify
            if any(w in lower for w in ["evidence:", "source:", "the document", "states", "according"]):
                evidence_lines.append(stripped)
                in_analysis = True
            elif any(w in lower for w in ["therefore", "thus", "conclusion", "in summary", "overall"]):
                conclusion_lines.append(stripped)
            elif in_analysis or len(stripped) > 40:
                analysis_lines.append(stripped)

        if evidence_lines:
            result["evidence_summary"] = " ".join(evidence_lines)[:500]
        if analysis_lines:
            result["policy_analysis"] = " ".join(analysis_lines[:3])[:500]
        if conclusion_lines:
            result["justification"] = " ".join(conclusion_lines)[:300]
    else:
        # Already evidence-grounded, keep as-is but structure it
        result["policy_analysis"] = reasoning[:500]

    return result


def remove_private_cot(v2_dir: str = "datasets/sft_v2") -> list[dict]:
    """Strip private chain-of-thought from v2 examples, convert to structured analysis."""
    all_examples = []
    for split in ("train", "validation", "test"):
        all_examples.extend(load_jsonl(f"{v2_dir}/{split}.jsonl"))

    cleaned = []
    cot_removed = 0
    for ex in all_examples:
        reasoning = ex.pop("reasoning", "")
        uncertainty = ex.pop("uncertainty", "")
        ex.pop("teacher_confidence", "")

        if reasoning:
            transformed = transform_reasoning(reasoning)
            ex["analysis"] = transformed
            if "policy_analysis" not in transformed:
                cot_removed += 1

        if uncertainty:
            ex["limitations"] = uncertainty[:300]

        ex["schema_version"] = SCHEMA_VERSION
        cleaned.append(ex)

    print("\n=== COT CLEANUP ===")
    print(f"Total: {len(cleaned)}, private reasoning removed: {cot_removed}")
    return cleaned


# ===================================================================
# 4. PAIRED V1/V2 COMPARISON
# ===================================================================

def compare_v1_v2(v1_dir: str = "datasets/sft", v2_dir: str = "datasets/sft_v2",
                  corpus_dir: str = "corpus") -> dict:
    """Compare V1 and V2 examples for the same document+task pairs."""
    v1_all = load_jsonl(f"{v1_dir}/train.jsonl") + load_jsonl(f"{v1_dir}/validation.jsonl") + load_jsonl(f"{v1_dir}/test.jsonl")
    v2_all = load_jsonl(f"{v2_dir}/train.jsonl") + load_jsonl(f"{v2_dir}/validation.jsonl") + load_jsonl(f"{v2_dir}/test.jsonl")

    # Index by (doc_id, task_type)
    v1_index: dict[tuple, dict] = {}
    for ex in v1_all:
        for did in ex.get("source_document_ids", []):
            key = (did, ex.get("task_type", ""))
            v1_index[key] = ex

    v2_index: dict[tuple, dict] = {}
    for ex in v2_all:
        for did in ex.get("source_document_ids", []):
            key = (did, ex.get("task_type", ""))
            v2_index[key] = ex

    # Find paired examples
    common_keys = set(v1_index.keys()) & set(v2_index.keys())

    scores = {"v2_better": 0, "v1_better": 0, "equivalent": 0, "unclear": 0}
    comparisons = []

    for key in sorted(common_keys)[:30]:
        v1 = v1_index[key]
        v2 = v2_index[key]
        did, task = key

        v1_a = next((m["content"] for m in v1.get("messages", []) if m["role"] == "assistant"), "")
        v2_a = next((m["content"] for m in v2.get("messages", []) if m["role"] == "assistant"), "")
        v2_reasoning = v2.get("reasoning", "")

        # Score
        grounding = 0
        specificity = 0
        policy = 0

        # Grounding: check for source references
        v1_has_ref = any(w in v1_a.lower() for w in ["source:", "[source", "document", "evidence"])
        v2_has_ref = any(w in v2_a.lower() for w in ["source:", "[source", "document", "evidence"])
        if v2_has_ref and not v1_has_ref:
            grounding = 2  # v2 better
        elif v1_has_ref and not v2_has_ref:
            grounding = -2
        elif v2_has_ref and v1_has_ref:
            grounding = 1 if len(v2_a) > len(v1_a) * 0.5 else -1

        # Specificity: longer, more detailed answers are better if grounded
        if len(v2_a) > len(v1_a) * 1.5 and v2_has_ref:
            specificity = 2
        elif len(v1_a) > len(v2_a) * 2:
            specificity = -1

        # Policy relevance
        v2_policy = any(w in v2_a.lower() for w in ["policy", "governance", "regulation", "stakeholder", "standard"])
        v1_policy = any(w in v1_a.lower() for w in ["policy", "governance", "regulation", "stakeholder", "standard"])
        if v2_policy and not v1_policy:
            policy = 2
        elif v1_policy and not v2_policy:
            policy = -1

        # Additional: reasoning present
        reasoning_bonus = 1 if v2_reasoning and len(v2_reasoning) > 100 else 0
        uncertainty_bonus = 1 if v2.get("uncertainty") and len(str(v2.get("uncertainty", ""))) > 20 else 0

        total = grounding + specificity + policy + reasoning_bonus + uncertainty_bonus

        if total >= 3:
            scores["v2_better"] += 1
        elif total <= -2:
            scores["v1_better"] += 1
        elif total >= 1:
            scores["v2_better"] += 1  # margin
        else:
            scores["equivalent"] += 1

        comparisons.append({
            "doc_id": did[:20], "task": task,
            "v1_len": len(v1_a), "v2_len": len(v2_a),
            "grounding": grounding, "specificity": specificity,
            "policy": policy, "reasoning_bonus": reasoning_bonus,
            "uncertainty_bonus": uncertainty_bonus, "total": total,
        })

    print("\n=== V1/V2 COMPARISON ===")
    print(f"Paired examples: {len(comparisons)}")
    print(f"V2 better: {scores['v2_better']}")
    print(f"V1 better: {scores['v1_better']}")
    print(f"Equivalent: {scores['equivalent']}")
    print(f"Unclear: {scores['unclear']}")

    if comparisons:
        v2_avg_score = sum(c["total"] for c in comparisons) / len(comparisons)
        print(f"V2 average score: {v2_avg_score:.1f}")

    return {"scores": scores, "compared": len(comparisons),
            "sample": comparisons[:3], "v2_avg_score": v2_avg_score if comparisons else 0}


# ===================================================================
# 5. SOURCE COVERAGE AUDIT
# ===================================================================

def audit_source_coverage(corpus_dir: str = "corpus", v2_dir: str = "datasets/sft_v2") -> dict:
    """Compare corpus source families with v2 coverage."""
    corpus_docs = load_canonical_docs(corpus_dir)
    v2_all = []
    for split in ("train", "validation", "test"):
        v2_all.extend(load_jsonl(f"{v2_dir}/{split}.jsonl"))

    # Corpus sources
    corpus_sources = Counter()
    for d in corpus_docs.values():
        url = d.get("source_url", "") or d.get("final_url", "")
        corpus_sources[classify_source(url)] += 1

    # V2 sources
    v2_sources = Counter()
    v2_doc_ids = set()
    for ex in v2_all:
        for did in ex.get("source_document_ids", []):
            v2_doc_ids.add(did)
            d = corpus_docs.get(did, {})
            url = d.get("source_url", "") or d.get("final_url", "")
            v2_sources[classify_source(url)] += 1

    print("\n=== SOURCE COVERAGE ===")
    print(f"{'Source':15} {'Corpus':>8} {'V2':>8} {'Status'}")
    print("-" * 45)
    missing = []
    for src in sorted(corpus_sources.keys()):
        c = corpus_sources[src]
        v = v2_sources.get(src, 0)
        status = "OK" if v > 0 else "MISSING"
        if status == "MISSING":
            missing.append(src)
        print(f"{src:15} {c:>8} {v:>8} {status}")

    # Missing doc IDs
    if missing:
        print("\n=== MISSING SOURCE FAMILIES ===")
        for src in missing:
            print(f"\n{src}:")
            missing_docs = []
            for did, d in corpus_docs.items():
                url = d.get("source_url", "") or d.get("final_url", "")
                if classify_source(url) == src and did not in v2_doc_ids:
                    meta = d.get("metadata") or {}
                    text_len = len(d.get("text", ""))
                    missing_docs.append((did, meta.get("title", "?"), text_len))
            for did, title, tl in missing_docs[:5]:
                print(f"  {did[:20]} | {str(title)[:60]} | {tl} chars")

    return {"corpus_sources": dict(corpus_sources), "v2_sources": dict(v2_sources),
            "missing": missing, "v2_doc_count": len(v2_doc_ids)}


# ===================================================================
# 6. BUILD FINAL SFT DATASET
# ===================================================================

def build_final(v1_dir: str = "datasets/sft", v2_dir: str = "datasets/sft_v2",
                output_dir: str = "datasets/sft_final", corpus_dir: str = "corpus") -> dict:
    """Build final SFT dataset: filtered V1 + cleaned V2."""
    corpus_docs = load_canonical_docs(corpus_dir)

    # Load V1 splits
    v1_splits = {}
    v1_doc_per_split = {}
    for split in ("train", "validation", "test"):
        exs = load_jsonl(f"{v1_dir}/{split}.jsonl")
        v1_splits[split] = exs
        dids = set()
        for ex in exs:
            for did in ex.get("source_document_ids", []):
                dids.add(did)
        v1_doc_per_split[split] = dids

    # Load cleaned V2
    v2_cleaned = remove_private_cot(v2_dir)

    # Assign V2 to splits based on document IDs matching V1 splits
    v2_splits = {"train": [], "validation": [], "test": []}
    for ex in v2_cleaned:
        for did in ex.get("source_document_ids", []):
            for split in ("train", "validation", "test"):
                if did in v1_doc_per_split.get(split, set()):
                    ex["split"] = split
                    v2_splits[split].append(ex)
                    break

    # Filter V1: keep only examples with substantive answers
    def is_substantive(ex):
        a = next((m["content"] for m in ex.get("messages", []) if m["role"] == "assistant"), "")
        next((m["content"] for m in ex.get("messages", []) if m["role"] == "user"), "")
        # Must have specific question and non-generic answer
        has_source = any(w in a.lower() for w in ["source:", "document", "evidence", "[", "http"])
        has_detail = len(a) > 100
        return has_source or has_detail

    final_splits = {}
    final_counts = {}
    for split in ("train", "validation", "test"):
        v1_filtered = [ex for ex in v1_splits.get(split, []) if is_substantive(ex)]
        v2_in_split = v2_splits.get(split, [])

        # Combine, preferring V2 over V1 for same (doc, task)
        seen_keys = set()
        combined = []
        for ex in v2_in_split + v1_filtered:
            dids = tuple(sorted(ex.get("source_document_ids", [])))
            task = ex.get("task_type", "")
            key = (dids, task)
            if key not in seen_keys:
                seen_keys.add(key)
                ex["split"] = split
                combined.append(ex)

        final_splits[split] = combined
        final_counts[split] = len(combined)

    # Write output
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        data = final_splits.get(split, [])
        if data:
            (out / f"{split}.jsonl").write_text(
                "\n".join(json.dumps(ex, ensure_ascii=False) for ex in data) + "\n",
                encoding="utf-8",
            )

    # Validate
    validation = validate_final(final_splits, corpus_docs, v1_doc_per_split)

    # Quality scoring
    quality = score_final(final_splits)

    # Metadata
    all_examples = []
    for split in ("train", "validation", "test"):
        all_examples.extend(final_splits.get(split, []))

    # Distributions
    year_dist = Counter()
    source_dist = Counter()
    task_dist = Counter()
    for ex in all_examples:
        task_dist[ex.get("task_type", "?")] += 1
        for did in ex.get("source_document_ids", []):
            d = corpus_docs.get(did, {})
            url = d.get("source_url", "") or d.get("final_url", "")
            source_dist[classify_source(url)] += 1
            pub = (d.get("metadata") or {}).get("published_at", "")
            if pub and len(str(pub)) >= 4:
                try:
                    year_dist[int(str(pub)[:4])] += 1
                except ValueError:
                    pass

    decade_dist = Counter()
    for y, c in year_dist.items():
        if y >= 2020:
            decade_dist["2020-2026"] += c
        elif y >= 2010:
            decade_dist["2010-2019"] += c
        elif y >= 2000:
            decade_dist["2000-2009"] += c
        elif y >= 1990:
            decade_dist["1990-1999"] += c

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "corpus_revision": CORPUS_REVISION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "splits": final_counts,
        "total": sum(final_counts.values()),
        "v1_original": sum(len(v) for v in v1_splits.values()),
        "v1_filtered_kept": sum(len(final_splits[s]) for s in ("train", "validation", "test")
                                if any(ex.get("schema_version") != SCHEMA_VERSION for ex in final_splits.get(s, []))),
        "v2_cleaned_kept": sum(1 for s in ("train", "validation", "test")
                               for ex in final_splits.get(s, []) if ex.get("schema_version") == SCHEMA_VERSION),
        "validation": validation,
        "quality": quality,
        "year_distribution": dict(sorted(year_dist.items())),
        "decade_distribution": dict(decade_dist),
        "source_distribution": dict(source_dist.most_common()),
        "task_distribution": dict(task_dist),
    }

    (out / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "quality_report.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FINAL DATASET ===")
    print(f"Train: {final_counts.get('train', 0)}")
    print(f"Validation: {final_counts.get('validation', 0)}")
    print(f"Test: {final_counts.get('test', 0)}")
    print(f"Total: {sum(final_counts.values())}")
    print(f"Validation: {validation}")
    print(f"Sources: {len(source_dist)}")
    print(f"1990-1999: {decade_dist.get('1990-1999', 0)}")
    print(f"Output: {output_dir}/")

    return metadata


def validate_final(splits: dict, corpus_docs: dict, v1_doc_per_split: dict) -> dict:
    """Validate final dataset."""
    errors = []
    all_docs_per_split = {}

    # Check document-level splits
    for split_name, examples in splits.items():
        split_docs = set()
        for ex in examples:
            for did in ex.get("source_document_ids", []):
                split_docs.add(did)

                # Validate source hash
                d = corpus_docs.get(did, {})
                expected_hash = d.get("artifact_sha256", "")
                for sh in ex.get("source_hashes", []):
                    if sh and expected_hash and sh != expected_hash:
                        errors.append(f"Hash mismatch: {did[:20]} expected {expected_hash[:12]} got {sh[:12]}")

        all_docs_per_split[split_name] = split_docs

    # Check split leakage
    leakage = {}
    for s1, s2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = all_docs_per_split.get(s1, set()) & all_docs_per_split.get(s2, set())
        leakage[f"{s1}/{s2}"] = len(overlap)
        if overlap:
            errors.append(f"LEAKAGE {s1}/{s2}: {len(overlap)} docs")

    # Check provenance
    missing_provenance = 0
    for examples in splits.values():
        for ex in examples:
            if not ex.get("source_document_ids"):
                missing_provenance += 1
            if not ex.get("source_hashes"):
                missing_provenance += 1

    # Check duplicate content
    seen_content = set()
    duplicates = 0
    for examples in splits.values():
        for ex in examples:
            content = hashlib.sha256(
                (str(ex.get("messages", ""))).encode()
            ).hexdigest()
            if content in seen_content:
                duplicates += 1
            seen_content.add(content)

    has_cot = 0
    for examples in splits.values():
        for ex in examples:
            if "chain_of_thought" in ex or "private_reasoning" in ex:
                has_cot += 1
            if ex.get("reasoning") and len(str(ex.get("reasoning", ""))) > 500:
                has_cot += 1

    result = {
        "leakage": leakage,
        "missing_provenance": missing_provenance,
        "duplicates": duplicates,
        "private_cot_fields": has_cot,
        "errors": errors[:20],
        "valid": len(errors) == 0,
    }
    return result


def score_final(splits: dict) -> dict:
    """Score final dataset quality."""
    all_examples = []
    for s in ("train", "validation", "test"):
        all_examples.extend(splits.get(s, []))

    if not all_examples:
        return {}

    grounding_scores = []
    evidence_scores = []
    policy_scores = []
    specificity_scores = []
    reasoning_scores = []
    provenance_scores = []

    for ex in all_examples:
        a = next((m["content"] for m in ex.get("messages", []) if m["role"] == "assistant"), "")
        q = next((m["content"] for m in ex.get("messages", []) if m["role"] == "user"), "")

        # Grounding: source references in answer
        has_ref = any(w in a.lower() for w in ["source", "document", "[", "http", "evidence", "states", "according"])
        has_ev = len(ex.get("evidence", [])) > 0
        grounding_scores.append(min(5, (3 if has_ref else 0) + (2 if has_ev else 0)))

        # Evidence quality
        evidence_scores.append(min(5, len(ex.get("evidence", [])) * 2 if has_ev else 1))

        # Policy relevance
        policy_terms = sum(1 for w in ["policy", "governance", "regulation", "stakeholder", "standard",
                                        "internet", "digital", "rights", "public", "multistakeholder"]
                          if w in (q + a).lower())
        policy_scores.append(min(5, policy_terms))

        # Specificity
        specificity_scores.append(min(5, len(a) // 100))

        # Reasoning
        has_analysis = bool(ex.get("analysis", {}))
        reasoning_scores.append(3 if has_analysis else 1)

        # Provenance
        has_id = len(ex.get("source_document_ids", [])) > 0
        has_hash = len(ex.get("source_hashes", [])) > 0
        provenance_scores.append(min(5, (3 if has_id else 0) + (2 if has_hash else 0)))

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    quality = {
        "avg_grounding": avg(grounding_scores),
        "avg_evidence": avg(evidence_scores),
        "avg_policy_relevance": avg(policy_scores),
        "avg_specificity": avg(specificity_scores),
        "avg_reasoning": avg(reasoning_scores),
        "avg_provenance": avg(provenance_scores),
    }
    return quality


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SFT Quality Gate")
    p.add_argument("--inspect", action="store_true", help="Inspect v2 examples")
    p.add_argument("--compare", action="store_true", help="Compare V1 vs V2")
    p.add_argument("--audit-sources", action="store_true", help="Audit source coverage")
    p.add_argument("--build-final", action="store_true", help="Build final SFT dataset")
    p.add_argument("--all", action="store_true", help="Run all quality gate steps")
    args = p.parse_args()

    if args.all or args.inspect:
        inspect_v2()

    if args.all or args.compare:
        compare_v1_v2()

    if args.all or args.audit_sources:
        audit_source_coverage()

    if args.all or args.build_final:
        build_final()
