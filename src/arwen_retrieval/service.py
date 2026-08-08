from __future__ import annotations

from .models import RetrievalQuery, RetrievedItem
from .provenance import EvidenceReference
from .retriever import InMemoryRetriever


class RetrievalService:
    """Application-facing retrieval service with explicit evidence references."""

    def __init__(self, retriever: InMemoryRetriever) -> None:
        self.retriever = retriever

    def search(self, query: RetrievalQuery) -> list[RetrievedItem]:
        return self.retriever.retrieve(query)

    def evidence(self, query: RetrievalQuery) -> list[EvidenceReference]:
        return [
            EvidenceReference(
                record_id=item.record.record_id,
                source_id=item.record.source_id,
                document_id=item.record.document_id,
                segment_id=item.record.segment_id,
                url=item.record.url,
                retrieval_score=item.score,
            )
            for item in self.search(query)
        ]
