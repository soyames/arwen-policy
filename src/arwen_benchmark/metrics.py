from __future__ import annotations


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, record_id in enumerate(retrieved_ids, start=1):
        if record_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evidence_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids).intersection(relevant_ids)) / len(relevant_ids)


def coverage_score(represented: set[str], expected: set[str]) -> float:
    if not expected:
        return 1.0
    return len(represented.intersection(expected)) / len(expected)
