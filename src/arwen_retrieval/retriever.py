from __future__ import annotations

from .index import InMemoryIndex, tokenize
from .models import CorpusRecord, RetrievalQuery, RetrievedItem


class InMemoryRetriever:
    """Metadata-aware lexical retriever used as the baseline retrieval layer."""

    def __init__(self, records: list[CorpusRecord] | None = None) -> None:
        self.index = InMemoryIndex(records)

    def add_records(self, records: list[CorpusRecord]) -> None:
        self.index.add_many(records)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedItem]:
        query_terms = tokenize(query.text)
        candidates = [
            record
            for record in self.index.all_records()
            if self._matches_filters(record, query)
        ]
        scored: list[tuple[CorpusRecord, float, tuple[str, ...]]] = []
        for record in candidates:
            score = self.index.score(record.record_id, query_terms)
            matched = self.index.matching_terms(record.record_id, query_terms)
            if score > 0:
                scored.append((record, score, matched))

        scored.sort(key=lambda item: (-item[1], item[0].record_id))
        return [
            RetrievedItem(record=record, score=score, rank=rank, matched_terms=matched)
            for rank, (record, score, matched) in enumerate(scored[: max(query.top_k, 0)], start=1)
        ]

    @staticmethod
    def _matches_filters(record: CorpusRecord, query: RetrievalQuery) -> bool:
        if query.stakeholder_groups and not set(query.stakeholder_groups).intersection(
            record.stakeholder_groups
        ):
            return False
        if query.organizations and not set(query.organizations).intersection(record.organizations):
            return False
        if query.topics and not set(query.topics).intersection(record.topics):
            return False
        if query.languages and record.language not in query.languages:
            return False
        return True
