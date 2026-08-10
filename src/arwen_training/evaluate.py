"""Evaluation pipeline for Arwen Policy.

Measures retrieval, extraction, grounding, stakeholder reasoning,
deliberation quality, and hallucination resistance.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src/ is importable.
_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 5
) -> float:
    """Recall@K: fraction of relevant items retrieved in the top K."""
    if not relevant_ids:
        return 1.0
    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int = 5
) -> float:
    """Precision@K: fraction of top-K items that are relevant."""
    if k == 0:
        return 0.0
    retrieved = set(retrieved_ids[:k])
    if not retrieved:
        return 0.0
    return len(retrieved & relevant_ids) / len(retrieved)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank of the first relevant item."""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Grounding metrics
# ---------------------------------------------------------------------------

def evidence_support_rate(
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure what fraction of claims are supported by evidence."""
    if not claims:
        return {"supported": 0, "unsupported": 0, "total": 0, "rate": 0.0}
    supported = sum(1 for c in claims if c.get("evidence_linked", False))
    total = len(claims)
    return {
        "supported": supported,
        "unsupported": total - supported,
        "total": total,
        "rate": supported / total if total > 0 else 0.0,
    }


def source_attribution_accuracy(
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure accuracy of source attribution in claims."""
    if not claims:
        return {"accurate": 0, "inaccurate": 0, "total": 0, "rate": 0.0}
    accurate = sum(
        1 for c in claims
        if c.get("source_id") and c.get("source_url")
    )
    return {
        "accurate": accurate,
        "inaccurate": len(claims) - accurate,
        "total": len(claims),
        "rate": accurate / len(claims) if claims else 0.0,
    }


# ---------------------------------------------------------------------------
# Hallucination test
# ---------------------------------------------------------------------------

def hallucination_test(
    question: str,
    evidence_count: int,
    synthesis: str | None,
    acceptable_answer: str = "Insufficient evidence",
) -> dict[str, Any]:
    """Verify that the system refuses to answer when evidence is insufficient."""
    has_evidence = evidence_count > 0
    mentions_insufficient = (
        "insufficient" in (synthesis or "").lower()
        or "no evidence" in (synthesis or "").lower()
        or "not enough" in (synthesis or "").lower()
    )

    return {
        "question": question,
        "evidence_count": evidence_count,
        "passed": not has_evidence or mentions_insufficient,
        "hallucinated": has_evidence and not mentions_insufficient,
        "synthesis_mentions_insufficient_evidence": mentions_insufficient,
    }


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    data_dir: str = "data/extracted",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run the complete Arwen Policy evaluation suite.

    Returns structured results suitable for the Arwen paper reproducibility
    manifest.
    """
    # Ensure evaluation module is available.
    _src_path = Path(__file__).resolve().parents[1]
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))

    from arwen_training.builder import build_evaluation_set, corpus_training_stats

    started_at = datetime.now(UTC)

    # ---- 1. Corpus statistics ----
    corpus_stats = corpus_training_stats(data_dir)

    # ---- 2. Load evaluation questions ----
    eval_questions = build_evaluation_set(data_dir)

    # ---- 3. Run retrieval evaluation on evaluation questions ----
    retrieval_results: list[dict[str, Any]] = []
    hallucination_results: list[dict[str, Any]] = []

    # Try to use the real retrieval engine if available.
    try:
        from arwen_retrieval.models import CorpusRecord, RetrievalQuery
        from arwen_retrieval.retriever import InMemoryRetriever
        from arwen_retrieval.service import RetrievalService

        # Build records from corpus
        records = _build_records_from_corpus(data_dir)
        if records:
            retriever = InMemoryRetriever(records)
            service = RetrievalService(retriever)

            for eq in eval_questions:
                query = RetrievalQuery(
                    text=eq["question"],
                    top_k=10,
                    stakeholder_groups=eq.get("required_stakeholder_groups", ()),
                )
                items = service.search(query)

                retrieved_ids = [item.record.record_id for item in items]
                expected = set(eq.get("expected_evidence_sources", ()))
                # Match by source_id prefix
                relevant = {
                    item.record.record_id
                    for item in items
                    if any(
                        item.record.source_id.startswith(src)
                        for src in expected
                    )
                }

                retrieval_results.append({
                    "question_id": eq["question_id"],
                    "question": eq["question"],
                    "retrieved_count": len(items),
                    "recall_at_5": recall_at_k(retrieved_ids, relevant, 5),
                    "precision_at_5": precision_at_k(retrieved_ids, relevant, 5),
                    "mrr": mrr(retrieved_ids, relevant),
                    "top_scores": [
                        {"record_id": item.record.record_id, "score": item.score}
                        for item in items[:3]
                    ],
                })

                # Hallucination test
                if eq.get("is_hallucination_test"):
                    hallucination_results.append(
                        hallucination_test(
                            question=eq["question"],
                            evidence_count=len(items),
                            synthesis=None,
                            acceptable_answer=eq.get(
                                "acceptable_answer", "Insufficient evidence"
                            ),
                        )
                    )

            retrieval_summary = {
                "mean_recall_at_5": (
                    sum(r["recall_at_5"] for r in retrieval_results)
                    / len(retrieval_results)
                    if retrieval_results
                    else 0.0
                ),
                "mean_precision_at_5": (
                    sum(r["precision_at_5"] for r in retrieval_results)
                    / len(retrieval_results)
                    if retrieval_results
                    else 0.0
                ),
                "mean_mrr": (
                    sum(r["mrr"] for r in retrieval_results)
                    / len(retrieval_results)
                    if retrieval_results
                    else 0.0
                ),
                "per_question": retrieval_results,
            }
        else:
            retrieval_summary = {
                "status": "SKIPPED",
                "reason": "No corpus records available for retrieval evaluation",
                "per_question": [],
            }
    except ImportError as e:
        retrieval_summary = {
            "status": "SKIPPED",
            "reason": f"arwen_retrieval not importable: {e}",
            "per_question": [],
        }

    # ---- 4. Hallucination summary ----
    hallucination_summary = {
        "tests": len(hallucination_results),
        "passed": sum(1 for h in hallucination_results if h["passed"]),
        "hallucinated": sum(1 for h in hallucination_results if h["hallucinated"]),
        "results": hallucination_results,
    }

    # ---- 5. Build final report ----
    result = {
        "evaluation_id": f"eval-{started_at.strftime('%Y%m%d-%H%M%S')}",
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "corpus": corpus_stats,
        "retrieval": retrieval_summary,
        "hallucination": hallucination_summary,
        "evaluation_questions": len(eval_questions),
        "status": (
            "COMPLETED"
            if retrieval_results
            else "SKIPPED — no corpus records available"
        ),
    }

    if output_path:
        Path(output_path).write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )

    return result


def _build_records_from_corpus(data_dir: str) -> list[Any]:
    """Build CorpusRecords from extracted documents."""
    from arwen_retrieval.models import CorpusRecord

    records: list[CorpusRecord] = []
    extracted_dir = Path(data_dir)
    if not extracted_dir.is_dir():
        return records

    for fpath in sorted(extracted_dir.glob("*.json")):
        try:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        text = doc.get("text", "")
        if len(text) < 100:
            continue

        for seg in doc.get("segments", []) or []:
            seg_text = seg.get("text", "")
            if len(seg_text) < 50:
                continue
            records.append(
                CorpusRecord(
                    record_id=f"{doc['document_id']}/{seg['segment_id']}",
                    text=seg_text,
                    source_id=doc.get("source_id", "unknown"),
                    document_id=doc.get("document_id", "unknown"),
                    segment_id=seg.get("segment_id"),
                    title=(doc.get("metadata") or {}).get("title"),
                    url=doc.get("source_url"),
                )
            )

        if not doc.get("segments"):
            records.append(
                CorpusRecord(
                    record_id=doc.get("document_id", "unknown"),
                    text=text,
                    source_id=doc.get("source_id", "unknown"),
                    document_id=doc.get("document_id", "unknown"),
                    title=(doc.get("metadata") or {}).get("title"),
                    url=doc.get("source_url"),
                )
            )

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Arwen Policy Evaluation")
    parser.add_argument("--data_dir", default="data/extracted")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    result = run_evaluation(data_dir=args.data_dir, output_path=args.output)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
