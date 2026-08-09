"""Stakeholder-aware retrieval backed by the real arwen_retrieval package."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from arwen_retrieval.models import CorpusRecord, RetrievalQuery
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService


def retrieve_evidence(
    query: str,
    source_filters: Optional[List[str]] = None,
    records: Optional[List[CorpusRecord]] = None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Retrieve evidence relevant to *query* from the corpus.

    If *records* are provided they are used directly (in-memory mode).
    Otherwise the function returns an empty result — long-term this should
    load the canonical corpus from disk or a vector store.

    Returns a list of dictionaries, each containing:
        - document_id: unique identifier
        - source_url: URL of the source
        - relevance_score: float 0-1
        - text: extracted snippet
        - provenance: meta-information (e.g., timestamp, author)
    """
    if not records:
        return []

    retriever = InMemoryRetriever(list(records))
    service = RetrievalService(retriever)
    retrieval_query = RetrievalQuery(
        text=query,
        top_k=top_k,
    )

    results: List[Dict[str, Any]] = []
    for item in service.search(retrieval_query):
        results.append(
            {
                "document_id": item.record.document_id,
                "segment_id": item.record.segment_id,
                "source_url": item.record.url or "",
                "relevance_score": item.score,
                "text": item.record.text,
                "provenance": {
                    "source_id": item.record.source_id,
                    "record_id": item.record.record_id,
                    "retrieval_method": "bm25_in_memory",
                    "stakeholder_groups": list(item.record.stakeholder_groups),
                    "topics": list(item.record.topics),
                },
            }
        )
    return results
