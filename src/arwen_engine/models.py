from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyRequest:
    question_id: str
    question: str
    topics: tuple[str, ...] = ()
    stakeholder_groups: tuple[str, ...] = ()
    top_k: int = 8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyAnswer:
    question_id: str
    question: str
    status: str
    evidence: tuple[dict[str, Any], ...]
    stakeholder_coverage: dict[str, Any]
    deliberation: dict[str, Any]
    synthesis_prompt: str
    limitations: tuple[str, ...]
