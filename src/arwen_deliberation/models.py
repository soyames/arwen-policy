from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyQuestion:
    question_id: str
    text: str
    topics: tuple[str, ...] = ()
    required_stakeholder_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class Perspective:
    stakeholder_group: str
    position: str
    arguments: tuple[str, ...] = ()
    evidence_record_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    attribution: str | None = None
    is_official_position: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeliberationResult:
    question_id: str
    perspectives: tuple[Perspective, ...]
    represented_groups: tuple[str, ...]
    missing_groups: tuple[str, ...]
    agreements: tuple[str, ...]
    disagreements: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    evidence_record_ids: tuple[str, ...]
    synthesis_constraints: tuple[str, ...]
