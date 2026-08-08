from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    relevant_record_ids: tuple[str, ...] = ()
    expected_stakeholder_groups: tuple[str, ...] = ()
    expected_position_record_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkResult:
    case_id: str
    reciprocal_rank: float
    evidence_recall: float
    stakeholder_coverage: float
