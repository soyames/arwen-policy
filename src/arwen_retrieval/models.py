from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CorpusRecord:
    """Canonical retrieval unit derived from the Arwen Policy Corpus."""

    record_id: str
    text: str
    source_id: str
    document_id: str
    segment_id: str | None = None
    title: str | None = None
    url: str | None = None
    language: str = "und"
    stakeholder_groups: tuple[str, ...] = ()
    organizations: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    document_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    top_k: int = 8
    stakeholder_groups: tuple[str, ...] = ()
    organizations: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedItem:
    record: CorpusRecord
    score: float
    rank: int
    matched_terms: tuple[str, ...] = ()
