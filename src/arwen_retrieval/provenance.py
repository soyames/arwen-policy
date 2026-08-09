from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceReference:
    record_id: str
    source_id: str
    document_id: str
    segment_id: str | None
    url: str | None
    retrieval_score: float
    stakeholder_groups: tuple[str, ...] = ()
    text_snippet: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "segment_id": self.segment_id,
            "url": self.url,
            "retrieval_score": self.retrieval_score,
            "stakeholder_groups": list(self.stakeholder_groups),
            "text_snippet": self.text_snippet,
        }
