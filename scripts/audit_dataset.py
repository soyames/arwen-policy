#!/usr/bin/env python3
"""Audit the Arwen Policy SFT dataset composition.

Reports:
  - Split counts
  - Task type distribution
  - System prompt distribution (restrictive vs. V1 template vs. V3 multistakeholder)
  - Stakeholder coverage
  - Missing behavior types (general policy, multistakeholder, disagreement, etc.)

Does NOT modify the dataset.
Does NOT regenerate anything.

Usage:
    uv run python scripts/audit_dataset.py
    uv run python scripts/audit_dataset.py --json  # machine-readable output
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path("datasets/sft_final")

# Known system prompts to classify
RESTRICTIVE_MARKER = "only the supplied source evidence"
V1_TEMPLATE_MARKER = "using supplied evidence"
V3_MULTISTAKEHOLDER_MARKER = "not a prerequisite for policy reasoning"


def classify_prompt(content: str) -> str:
    """Classify a system prompt into its generation/variant."""
    if RESTRICTIVE_MARKER in content:
        return "restrictive_v3"
    if V3_MULTISTAKEHOLDER_MARKER in content:
        return "multistakeholder_v3"
    if V1_TEMPLATE_MARKER in content:
        return "v1_template"
    return "unknown"


def audit_split(split_name: str) -> dict:
    path = DATA_DIR / f"{split_name}.jsonl"
    if not path.exists():
        return {"error": f"{split_name}.jsonl not found"}

    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        examples.append(json.loads(line))

    task_dist = Counter()
    prompt_dist = Counter()
    stakeholders_count = Counter()
    has_multistakeholder = 0
    has_disagreement = 0
    has_general_policy = 0
    has_recommendation = 0
    source_only_examples = 0
    evidence_extraction_examples = 0

    for ex in examples:
        task_type = ex.get("task_type", "unknown")
        task_dist[task_type] += 1

        # System prompt classification
        msgs = ex.get("messages", [])
        sys_content = ""
        for m in msgs:
            if m.get("role") == "system":
                sys_content = m.get("content", "")
                break
        prompt_type = classify_prompt(sys_content)
        prompt_dist[prompt_type] += 1

        # Stakeholders
        stk = ex.get("stakeholders_mentioned", [])
        n_stk = len(stk) if stk else 0
        stakeholders_count[n_stk] += 1

        # Content analysis
        answer = ""
        for m in msgs:
            if m.get("role") == "assistant":
                answer = m.get("content", "").lower()

        if "multistakeholder" in answer or "multi-stakeholder" in answer:
            has_multistakeholder += 1
        if any(w in answer for w in ["disagree", "conflict", "tension", "trade-off",
                                       "tradeoff", "competing"]):
            has_disagreement += 1
        if any(w in answer for w in ["should", "recommend", "ought", "must consider"]):
            has_recommendation += 1
        if not stk or len(stk) == 0:
            source_only_examples += 1
        if task_type == "evidence_extraction":
            evidence_extraction_examples += 1

        # General policy reasoning (not document-grounded)
        question = ""
        for m in msgs:
            if m.get("role") == "user":
                question = m.get("content", "").lower()
        # Document-grounded questions typically reference "this document" or a title
        is_doc_grounded = ("this document" in question or
                          "the document" in question or
                          "does the document" in question)
        if not is_doc_grounded:
            has_general_policy += 1

    return {
        "split": split_name,
        "count": len(examples),
        "task_distribution": dict(task_dist.most_common()),
        "prompt_distribution": dict(prompt_dist),
        "stakeholders_per_example": dict(sorted(stakeholders_count.items())),
        "examples_with_multistakeholder_mention": has_multistakeholder,
        "examples_with_disagreement_language": has_disagreement,
        "examples_with_recommendation_language": has_recommendation,
        "examples_with_zero_stakeholders": source_only_examples,
        "evidence_extraction_examples": evidence_extraction_examples,
        "general_policy_examples": has_general_policy,
        "document_grounded_examples": len(examples) - has_general_policy,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit Arwen Policy SFT dataset")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args()

    results = {}
    for split in ["train", "validation", "test"]:
        results[split] = audit_split(split)

    # Summary
    total = sum(r.get("count", 0) for r in results.values())
    prompt_summary = Counter()
    for r in results.values():
        for k, v in r.get("prompt_distribution", {}).items():
            prompt_summary[k] += v

    task_summary = Counter()
    for r in results.values():
        for k, v in r.get("task_distribution", {}).items():
            task_summary[k] += v

    summary = {
        "total_examples": total,
        "splits": {k: v.get("count", 0) for k, v in results.items()},
        "prompt_distribution": dict(prompt_summary),
        "task_distribution": dict(task_summary.most_common()),
        "per_split": results,
        "deficiencies": {
            "multistakeholder_analysis_examples": sum(
                r.get("examples_with_multistakeholder_mention", 0)
                for r in results.values()
            ),
            "disagreement_examples": sum(
                r.get("examples_with_disagreement_language", 0)
                for r in results.values()
            ),
            "recommendation_examples": sum(
                r.get("examples_with_recommendation_language", 0)
                for r in results.values()
            ),
            "general_policy_examples": sum(
                r.get("general_policy_examples", 0)
                for r in results.values()
            ),
            "zero_stakeholder_examples": sum(
                r.get("examples_with_zero_stakeholders", 0)
                for r in results.values()
            ),
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("ARWEN POLICY — SFT DATASET AUDIT")
        print("=" * 60)
        print(f"\nTotal examples: {total}")
        print(f"  Train:      {results['train'].get('count', 0)}")
        print(f"  Validation: {results['validation'].get('count', 0)}")
        print(f"  Test:       {results['test'].get('count', 0)}")

        print(f"\n=== System Prompt Distribution ===")
        for ptype, count in prompt_summary.most_common():
            pct = 100 * count / total if total > 0 else 0
            print(f"  {ptype}: {count} ({pct:.1f}%)")

        print(f"\n=== Task Type Distribution ===")
        for task, count in task_summary.most_common():
            pct = 100 * count / total if total > 0 else 0
            print(f"  {task}: {count} ({pct:.1f}%)")

        print(f"\n=== Behavioral Coverage ===")
        defs = summary["deficiencies"]
        print(f"  Multistakeholder analysis mentions:  {defs['multistakeholder_analysis_examples']}")
        print(f"  Disagreement/conflict/trade-off:     {defs['disagreement_examples']}")
        print(f"  Recommendation language:             {defs['recommendation_examples']}")
        print(f"  General policy (not doc-grounded):   {defs['general_policy_examples']}")
        print(f"  Zero stakeholders mentioned:         {defs['zero_stakeholder_examples']}")

        print(f"\n=== Assessment ===")
        if prompt_summary.get("multistakeholder_v3", 0) == 0:
            print("  WARNING: No examples use the V3 multistakeholder prompt.")
            print("  The dataset teaches source-only behavior.")
        if defs["general_policy_examples"] == 0:
            print("  WARNING: No general-policy examples (all doc-grounded).")
            print("  Model may refuse to answer without a supplied source.")
        if defs["disagreement_examples"] < 5:
            print("  WARNING: Few disagreement/trade-off examples.")
            print("  Model may not learn to handle policy disagreement.")
        if defs["zero_stakeholder_examples"] > total * 0.5:
            print("  WARNING: Most examples mention zero stakeholders.")
            print("  Model may not learn multistakeholder analysis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
