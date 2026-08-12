#!/usr/bin/env python3
"""Deterministic dataset quality validator for Arwen Policy SFT data.

Validates that the training dataset teaches the intended multistakeholder
policy-analysis behavior — not merely source-extraction/RAG-QA.

Run AFTER dataset regeneration to verify coverage.
Run BEFORE training to gate on dataset quality.

Exit code 0 = dataset meets minimum quality standards.
Exit code 1 = dataset fails — do not train on this data.

Usage:
    uv run python scripts/validate_dataset.py
    uv run python scripts/validate_dataset.py --json
    uv run python scripts/validate_dataset.py --strict  # fail on any warning
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DATA_DIR = Path("datasets/sft_final")

# ===========================================================================
# Minimum acceptable thresholds
# ===========================================================================

# At least this fraction of examples must use the multistakeholder prompt
MIN_MULTISTAKEHOLDER_PROMPT_FRAC = 0.20

# At least this many unique task types (including policy-analysis types)
MIN_UNIQUE_TASK_TYPES = 10

# At least this fraction of examples must have 2+ stakeholders mentioned
MIN_MULTI_STAKEHOLDER_FRAC = 0.15

# At least this fraction of examples must be policy-analysis types
# (multistakeholder_analysis, stakeholder_disagreement, policy_recommendation,
#  perspective_vs_position, uncertainty_handling)
MIN_POLICY_TASK_FRAC = 0.05  # minimum for hybrid dataset; full regeneration targets 0.15

# Must have at least this many examples with disagreement/trade-off language
MIN_DISAGREEMENT_EXAMPLES = 10

# Must have at least this many examples with policy recommendation language
MIN_RECOMMENDATION_EXAMPLES = 10

# Must have at least this many examples that distinguish perspective vs position
MIN_PERSPECTIVE_VS_POSITION_EXAMPLES = 1

# Zero-tolerance: NO examples teaching refusal because "no source"
MAX_REFUSAL_EXAMPLES = 0

# HRIAM validation thresholds
MIN_HRIAM_EXAMPLES = 6
MAX_LEGAL_CONCLUSIONS = 2
# Document which HRIAM states should be represented in training data
HRIAM_STATES = ["HRIAM_NOT_MATERIAL", "HRIAM_RELEVANT", "HRIAM_CENTRAL"]
HRIAM_TASK_TYPES = {
    "rights_holder_identification", "rights_impact_analysis",
    "positive_negative_impacts", "disproportionate_impact",
    "stakeholder_rights_mapping", "participation_assessment",
    "accountability_remedy", "mitigation_safeguards",
    "rights_tradeoff_analysis", "lifecycle_hria", "panel_analysis",
}

# Policy task types (no source document required)
POLICY_TASK_TYPES = {
    "multistakeholder_analysis",
    "stakeholder_disagreement",
    "policy_recommendation",
    "perspective_vs_position",
    "uncertainty_handling",
    # HRIAM task types
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
}

# Document-grounded task types
DOCUMENT_TASK_TYPES = {
    "document_understanding",
    "evidence_extraction",
    "policy_question",
    "stakeholder_position",
    "argument_identification",
    "historical_context",
    "institutional_role",
    "policy_comparison",
    "tradeoff_analysis",
}


def load_examples(split_name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{split_name}.jsonl"
    if not path.exists():
        return []
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        examples.append(json.loads(line))
    return examples


def get_system_prompt(example: dict) -> str:
    for msg in example.get("messages", []):
        if msg.get("role") == "system":
            return msg.get("content", "")
    return ""


def classify_prompt(content: str) -> str:
    if "not a prerequisite for policy reasoning" in content:
        return "multistakeholder_v3"
    if "only the supplied source evidence" in content:
        return "restrictive_v3"
    if "using supplied evidence" in content:
        return "v1_template"
    return "unknown"


def validate_dataset(data_dir: str = "datasets/sft_final") -> dict[str, Any]:
    """Validate the dataset and return a quality report."""
    data_path = Path(data_dir)
    errors: list[str] = []
    warnings: list[str] = []

    # Load all data
    all_examples: list[dict] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = data_path / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"Missing {split}.jsonl")
            split_counts[split] = 0
            continue
        examples = load_examples(split)
        split_counts[split] = len(examples)
        all_examples.extend(examples)

    total = len(all_examples)
    if total == 0:
        errors.append("No examples found in dataset")
        return {"valid": False, "errors": errors, "warnings": warnings}

    # 1. Split counts
    if split_counts.get("test", 0) != 35:
        warnings.append(
            f"Test split has {split_counts.get('test', 0)} examples (expected 35)"
        )
    if split_counts.get("train", 0) < 100:
        errors.append(f"Train split too small: {split_counts.get('train', 0)}")

    # 2. Prompt distribution
    prompt_dist = Counter()
    for ex in all_examples:
        prompt_dist[classify_prompt(get_system_prompt(ex))] += 1

    ms_count = prompt_dist.get("multistakeholder_v3", 0)
    ms_frac = ms_count / total if total > 0 else 0
    if ms_frac < MIN_MULTISTAKEHOLDER_PROMPT_FRAC:
        errors.append(
            f"Multistakeholder prompt: {ms_count}/{total} ({ms_frac:.1%}) — "
            f"minimum {MIN_MULTISTAKEHOLDER_PROMPT_FRAC:.0%} required"
        )

    # 3. Task type distribution
    task_dist = Counter()
    for ex in all_examples:
        task_dist[ex.get("task_type", "unknown")] += 1

    unique_tasks = len(task_dist)
    if unique_tasks < MIN_UNIQUE_TASK_TYPES:
        warnings.append(
            f"Only {unique_tasks} unique task types — "
            f"minimum {MIN_UNIQUE_TASK_TYPES} recommended"
        )

    # 4. Policy task coverage
    policy_count = sum(task_dist.get(t, 0) for t in POLICY_TASK_TYPES)
    policy_frac = policy_count / total if total > 0 else 0
    if policy_frac < MIN_POLICY_TASK_FRAC:
        errors.append(
            f"Policy-analysis tasks: {policy_count}/{total} ({policy_frac:.1%}) — "
            f"minimum {MIN_POLICY_TASK_FRAC:.0%} required"
        )

    doc_count = sum(task_dist.get(t, 0) for t in DOCUMENT_TASK_TYPES)
    doc_frac = doc_count / total if total > 0 else 0

    # 5. Stakeholder coverage
    stakeholder_counts = Counter()
    zero_stk = 0
    multi_stk = 0  # 2+ stakeholders
    for ex in all_examples:
        stk = ex.get("stakeholders_mentioned", [])
        n = len(stk) if stk else 0
        stakeholder_counts[n] += 1
        if n == 0:
            zero_stk += 1
        if n >= 2:
            multi_stk += 1

    multi_stk_frac = multi_stk / total if total > 0 else 0
    if multi_stk_frac < MIN_MULTI_STAKEHOLDER_FRAC:
        warnings.append(
            f"Examples with 2+ stakeholders: {multi_stk}/{total} ({multi_stk_frac:.1%}) — "
            f"minimum {MIN_MULTI_STAKEHOLDER_FRAC:.0%} recommended"
        )

    # 6. Behavioral content analysis
    has_disagreement = 0
    has_recommendation = 0
    has_perspective_distinction = 0
    has_uncertainty = 0
    refusal_count = 0

    for ex in all_examples:
        answer = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant":
                answer = msg.get("content", "").lower()
                break

        if any(w in answer for w in ["disagree", "conflict", "tension",
                                       "trade-off", "tradeoff", "competing"]):
            has_disagreement += 1
        if any(w in answer for w in ["should", "recommend", "policy option",
                                       "policy approach"]):
            has_recommendation += 1
        if any(w in answer for w in ["general stakeholder perspective",
                                       "documented position",
                                       "perspective vs",
                                       "distinguish between"]):
            has_perspective_distinction += 1
        if any(w in answer for w in ["cannot be established", "cannot be verified",
                                       "evidence is incomplete",
                                       "not been comprehensively"]):
            has_uncertainty += 1
        if any(w in answer for w in ["cannot answer because no source",
                                       "cannot answer without",
                                       "no source was provided"]):
            refusal_count += 1

    if has_disagreement < MIN_DISAGREEMENT_EXAMPLES:
        errors.append(
            f"Disagreement/trade-off examples: {has_disagreement} — "
            f"minimum {MIN_DISAGREEMENT_EXAMPLES} required"
        )
    if has_recommendation < MIN_RECOMMENDATION_EXAMPLES:
        warnings.append(
            f"Recommendation examples: {has_recommendation} — "
            f"minimum {MIN_RECOMMENDATION_EXAMPLES} recommended"
        )
    if has_perspective_distinction < MIN_PERSPECTIVE_VS_POSITION_EXAMPLES:
        errors.append(
            f"Perspective vs. position examples: {has_perspective_distinction} — "
            f"minimum {MIN_PERSPECTIVE_VS_POSITION_EXAMPLES} required"
        )
    if refusal_count > MAX_REFUSAL_EXAMPLES:
        errors.append(
            f"Found {refusal_count} refusal-pattern examples — "
            f"maximum {MAX_REFUSAL_EXAMPLES} allowed"
        )

    # 7. Schema validation (sampled)
    schema_errors = 0
    for i, ex in enumerate(all_examples):
        msgs = ex.get("messages", [])
        task_type = ex.get("task_type", "")
        if not msgs:
            schema_errors += 1
            if schema_errors <= 3:
                errors.append(f"Example {i}: missing messages")
            continue
        roles = {m.get("role") for m in msgs}
        if "user" not in roles or "assistant" not in roles:
            schema_errors += 1
        if task_type in DOCUMENT_TASK_TYPES:
            if not ex.get("source_document_ids"):
                schema_errors += 1
                if schema_errors <= 5:
                    errors.append(
                        f"Example {i} (task={task_type}): "
                        "missing source_document_ids for doc-grounded task"
                    )
        # All examples must have non-empty content
        for msg in msgs:
            if not msg.get("content", "").strip():
                schema_errors += 1
                if schema_errors <= 5:
                    errors.append(f"Example {i}: empty {msg.get('role')} content")

    # 7. HRIAM coverage (advisory — warns on absence, errors on fabrication)
    hriam_task_count = sum(task_dist.get(t, 0) for t in HRIAM_TASK_TYPES)
    if hriam_task_count < MIN_HRIAM_EXAMPLES:
        warnings.append(
            f"HRIAM examples: {hriam_task_count} — "
            f"minimum {MIN_HRIAM_EXAMPLES} recommended"
        )
    # Check for fabricated violations or legal conclusions
    fabrication_count = 0
    legal_conclusion_count = 0
    for ex in all_examples:
        answer = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant":
                answer = msg.get("content", "").lower()
                break
        if any(w in answer for w in ["constitutes a violation of", "is a violation of",
                                       "violates international law", "is illegal under"]):
            fabrication_count += 1
        if any(w in answer for w in ["the law requires that", "is legally required to",
                                       "must by law", "is mandatory under"]):
            legal_conclusion_count += 1
    if fabrication_count > 0:
        errors.append(
            f"Found {fabrication_count} examples with potential human-rights violation "
            "claims — Arwen must not declare violations without evidence"
        )
    if legal_conclusion_count > MAX_LEGAL_CONCLUSIONS:
        warnings.append(
            f"Found {legal_conclusion_count} examples with legal-conclusion language — "
            f"Arwen should frame as policy recommendation, not legal requirement"
        )

    # 8. HRIAM behavioral content
    rights_holder_mention_count = 0
    for ex in all_examples:
        for msg in ex.get("messages", []):
            if msg.get("role") == "assistant" and "rights-holder" in msg.get("content", "").lower():
                rights_holder_mention_count += 1
                break
    hriam_behavior = {
        "hriam_task_examples": hriam_task_count,
        "rights_holder_mentions": rights_holder_mention_count,
        "potential_violation_claims": fabrication_count,
        "legal_conclusion_claims": legal_conclusion_count,
    }

    valid = len(errors) == 0

    return {
        "valid": valid,
        "total_examples": total,
        "split_counts": split_counts,
        "prompt_distribution": dict(prompt_dist),
        "task_distribution": dict(task_dist.most_common()),
        "unique_task_types": unique_tasks,
        "policy_task_count": policy_count,
        "policy_task_fraction": round(policy_frac, 4),
        "document_task_count": doc_count,
        "document_task_fraction": round(doc_frac, 4),
        "stakeholder_distribution": dict(sorted(stakeholder_counts.items())),
        "multi_stakeholder_count": multi_stk,
        "multi_stakeholder_fraction": round(multi_stk_frac, 4),
        "zero_stakeholder_count": zero_stk,
        "behavioral_content": {
            "disagreement_examples": has_disagreement,
            "recommendation_examples": has_recommendation,
            "perspective_vs_position_examples": has_perspective_distinction,
            "uncertainty_examples": has_uncertainty,
            "refusal_pattern_examples": refusal_count,
        },
        "schema_errors": schema_errors,
        "hriam_behavior": hriam_behavior,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate Arwen Policy SFT dataset quality"
    )
    parser.add_argument("--data-dir", default="datasets/sft_final")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    args = parser.parse_args()

    result = validate_dataset(args.data_dir)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print("=" * 60)
        print("ARWEN POLICY — DATASET QUALITY VALIDATOR")
        print("=" * 60)
        print(f"\nTotal examples: {result['total_examples']}")
        print(f"Splits: {result['split_counts']}")
        print(f"\nPrompt distribution:")
        for ptype, count in result['prompt_distribution'].items():
            pct = 100 * count / max(result['total_examples'], 1)
            print(f"  {ptype}: {count} ({pct:.1f}%)")

        print(f"\nTask distribution:")
        for task, count in result['task_distribution'].items():
            pct = 100 * count / max(result['total_examples'], 1)
            print(f"  {task}: {count} ({pct:.1f}%)")

        print(f"\nPolicy tasks: {result['policy_task_count']} "
              f"({result['policy_task_fraction']:.1%})")
        print(f"Document tasks: {result['document_task_count']} "
              f"({result['document_task_fraction']:.1%})")
        print(f"Multi-stakeholder examples: {result['multi_stakeholder_count']} "
              f"({result['multi_stakeholder_fraction']:.1%})")
        print(f"Zero-stakeholder examples: {result['zero_stakeholder_count']}")

        print(f"\nBehavioral content:")
        for k, v in result['behavioral_content'].items():
            print(f"  {k}: {v}")

        if result['errors']:
            print(f"\nERRORS ({len(result['errors'])}):")
            for e in result['errors']:
                print(f"  - {e}")
        if result['warnings']:
            print(f"\nWARNINGS ({len(result['warnings'])}):")
            for w in result['warnings']:
                print(f"  - {w}")

        if result['valid']:
            if args.strict and result['warnings']:
                print(f"\nRESULT: FAIL (strict mode — {len(result['warnings'])} warnings)")
                return 1
            print(f"\nRESULT: PASS — dataset meets minimum quality standards")
        else:
            print(f"\nRESULT: FAIL — {len(result['errors'])} errors found")
            return 1

    return 0 if result['valid'] else 1


if __name__ == "__main__":
    sys.exit(main())
