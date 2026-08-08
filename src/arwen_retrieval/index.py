from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import CorpusRecord

_TOKEN_RE = re.compile(r"(?u)\b[\w][\w'-]*\b")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class InMemoryIndex:
    """Deterministic BM25-style lexical index for development and small corpora."""

    def __init__(self, records: list[CorpusRecord] | None = None) -> None:
        self._records: dict[str, CorpusRecord] = {}
        self._terms: dict[str, Counter[str]] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._avgdl = 0.0
        if records:
            self.add_many(records)

    def add(self, record: CorpusRecord) -> None:
        tokens = tokenize(record.text)
        self._records[record.record_id] = record
        self._terms[record.record_id] = Counter(tokens)
        for term in set(tokens):
            self._postings[term].add(record.record_id)
        self._recalculate_average_length()

    def add_many(self, records: list[CorpusRecord]) -> None:
        for record in records:
            self.add(record)

    def get(self, record_id: str) -> CorpusRecord | None:
        return self._records.get(record_id)

    def all_records(self) -> list[CorpusRecord]:
        return list(self._records.values())

    def document_frequency(self, term: str) -> int:
        return len(self._postings.get(term, set()))

    def score(self, record_id: str, query_terms: list[str]) -> float:
        if not query_terms or record_id not in self._records:
            return 0.0

        terms = self._terms[record_id]
        document_length = sum(terms.values()) or 1
        document_count = len(self._records) or 1
        k1 = 1.5
        b = 0.75
        score = 0.0

        for term in query_terms:
            frequency = terms.get(term, 0)
            if frequency == 0:
                continue
            df = self.document_frequency(term)
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * document_length / max(self._avgdl, 1.0)
            )
            score += idf * (frequency * (k1 + 1) / denominator)
        return score

    def matching_terms(self, record_id: str, query_terms: list[str]) -> tuple[str, ...]:
        terms = self._terms.get(record_id, Counter())
        return tuple(sorted({term for term in query_terms if term in terms}))

    def _recalculate_average_length(self) -> None:
        if not self._terms:
            self._avgdl = 0.0
            return
        self._avgdl = sum(sum(counter.values()) for counter in self._terms.values()) / len(
            self._terms
        )
