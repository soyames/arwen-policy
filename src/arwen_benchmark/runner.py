from __future__ import annotations

from .metrics import coverage_score, evidence_recall, reciprocal_rank
from .models import BenchmarkCase, BenchmarkResult


def evaluate_case(
    case: BenchmarkCase,
    retrieved_ids: list[str],
    represented_stakeholders: set[str],
) -> BenchmarkResult:
    relevant = set(case.relevant_record_ids)
    expected = set(case.expected_stakeholder_groups)
    return BenchmarkResult(
        case_id=case.case_id,
        reciprocal_rank=reciprocal_rank(retrieved_ids, relevant),
        evidence_recall=evidence_recall(retrieved_ids, relevant),
        stakeholder_coverage=coverage_score(represented_stakeholders, expected),
    )
