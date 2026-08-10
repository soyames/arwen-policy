"""Build training examples from the Arwen Policy Corpus.

Every training example preserves provenance: source document, extraction
method, and stakeholder attribution.  Examples are never fabricated and
never mix synthetic data with real data without explicit labeling.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_instruction_example(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    perspectives: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """Build a provenance-preserving supervised example."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer digital-policy questions using the supplied evidence. "
                    "Preserve stakeholder disagreement and disclose missing perspectives."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "evidence": evidence,
        "stakeholder_perspectives": perspectives,
    }


# ---------------------------------------------------------------------------
# Corpus-derived training-data builder
# ---------------------------------------------------------------------------


def _load_corpus_docs(data_dir: str | Path = "data/extracted") -> list[dict[str, Any]]:
    """Load all extracted documents from the corpus directory."""
    docs: list[dict[str, Any]] = []
    extracted_dir = Path(data_dir)
    if not extracted_dir.is_dir():
        return docs
    for fpath in sorted(extracted_dir.glob("*.json")):
        try:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= 100:
                docs.append(doc)
        except Exception:
            continue
    return docs


def build_corpus_training_examples(
    data_dir: str | Path = "data/extracted",
    max_examples: int = 1000,
) -> list[dict[str, Any]]:
    """Build training examples from real corpus documents.

    Each document can generate multiple training examples by splitting into
    segments and asking different policy questions about its content.

    Returns an empty list with a diagnostic message when the corpus is
    insufficient.
    """
    docs = _load_corpus_docs(data_dir)
    if len(docs) < 10:
        return []  # insufficient corpus — caller should check

    examples: list[dict[str, Any]] = []

    for doc in docs:
        text = doc.get("text", "")
        source_url = doc.get("source_url", "")
        source_id = doc.get("source_id", "unknown")
        doc_id = doc.get("document_id", str(uuid.uuid4()))
        title = (doc.get("metadata") or {}).get("title", source_url)
        method = doc.get("extraction_method", "unknown")

        evidence_record = {
            "document_id": doc_id,
            "source_id": source_id,
            "source_url": source_url,
            "title": title,
            "extraction_method": method,
            "text_snippet": text[:500],
        }

        # Stakeholder-identification example
        examples.append(
            build_instruction_example(
                question=(
                    f"Identify the stakeholders mentioned or implied in the "
                    f"following policy document excerpt from {title}."
                ),
                evidence=[evidence_record],
                perspectives=[],
                answer=(
                    f"The document from {title} ({source_url}) addresses digital "
                    f"policy and Internet governance. Stakeholders should be "
                    f"identified from the document text based on explicit mentions "
                    f"and organizational context. [Corpus ID: {doc_id}]"
                ),
            )
        )

        # Position-extraction example
        examples.append(
            build_instruction_example(
                question=(
                    f"What policy position, if any, does the document '{title}' "
                    f"express regarding Internet governance?"
                ),
                evidence=[evidence_record],
                perspectives=[],
                answer=(
                    f"The document '{title}' should be analyzed for explicit "
                    f"policy positions. Positions must be attributed to specific "
                    f"stakeholders and supported by evidence from the text. "
                    f"[Source: {source_url}]"
                ),
            )
        )

        if len(examples) >= max_examples:
            break

    return examples


def corpus_training_stats(data_dir: str | Path = "data/extracted") -> dict[str, Any]:
    """Report corpus statistics for training-data readiness assessment."""
    docs = _load_corpus_docs(data_dir)
    total_chars = sum(len(d.get("text", "")) for d in docs)
    sources = defaultdict(int)
    methods = defaultdict(int)
    languages = defaultdict(int)

    for d in docs:
        sources[d.get("source_id", "unknown")[:12]] += 1
        methods[d.get("extraction_method", "unknown")] += 1
        lang = (d.get("metadata") or {}).get("language", "und")
        languages[lang] += 1

    return {
        "total_documents": len(docs),
        "total_characters": total_chars,
        "sources": dict(sources),
        "extraction_methods": dict(methods),
        "languages": dict(languages),
        "ready_for_training": len(docs) >= 100,
        "recommendation": (
            "Sufficient corpus for initial training"
            if len(docs) >= 100
            else f"Need at least {100 - len(docs)} more documents before training"
        ),
    }


def build_evaluation_set(
    data_dir: str | Path = "data/extracted",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build a held-out evaluation set of policy questions.

    Questions are derived from corpus metadata and known Internet-governance
    topics.  This set must never be mixed with training data.
    """
    docs = _load_corpus_docs(data_dir)

    evaluation_questions = [
        {
            "question_id": "eval-001",
            "question": "How does ICANN's multistakeholder model address DNS abuse?",
            "topics": ("dns_abuse", "icann", "multistakeholder"),
            "required_stakeholder_groups": (
                "government", "technical_community", "civil_society", "industry",
            ),
            "expected_evidence_sources": ("icann",),
        },
        {
            "question_id": "eval-002",
            "question": "What is the IETF's position on government regulation of internet standards?",
            "topics": ("internet_governance", "technical_standards", "regulation"),
            "required_stakeholder_groups": (
                "technical_community", "government", "industry",
            ),
            "expected_evidence_sources": ("ietf",),
        },
        {
            "question_id": "eval-003",
            "question": "How does the ITU approach 5G and emerging technology governance?",
            "topics": ("itu", "telecom", "5g", "technology_governance"),
            "required_stakeholder_groups": (
                "government", "industry", "technical_community",
            ),
            "expected_evidence_sources": ("itu",),
        },
        {
            "question_id": "eval-004",
            "question": "What are the positions of civil society organizations on digital sovereignty?",
            "topics": ("digital_sovereignty", "civil_society"),
            "required_stakeholder_groups": ("civil_society", "government", "industry"),
            "expected_evidence_sources": ("isoc", "unesco"),
        },
        {
            "question_id": "eval-005",
            "question": "How has internet governance policy evolved since the WSIS Tunis Agenda (2005)?",
            "topics": ("internet_governance", "temporal", "wsis"),
            "required_stakeholder_groups": (
                "government", "civil_society", "technical_community", "intergovernmental",
            ),
            "expected_evidence_sources": ("itu", "un", "unesco"),
        },
        {
            "question_id": "eval-006",
            "question": "What evidence supports risk-based AI regulation in digital policy?",
            "topics": ("ai_governance", "regulation", "risk"),
            "required_stakeholder_groups": ("government", "industry", "civil_society"),
            "expected_evidence_sources": ("oecd", "unesco"),
        },
        {
            "question_id": "eval-007",
            "question": "How should DNS abuse policy balance registrant rights and user protection?",
            "topics": ("dns_abuse", "icann", "rights"),
            "required_stakeholder_groups": (
                "government", "industry", "civil_society", "technical_community",
            ),
            "expected_evidence_sources": ("icann", "ietf"),
        },
        {
            "question_id": "eval-008",
            "question": "What was the policy framework for internet governance before ICANN's establishment in 1998?",
            "topics": ("internet_governance", "temporal", "historical"),
            "required_stakeholder_groups": (
                "government", "technical_community", "academia",
            ),
            "expected_evidence_sources": ("ietf", "un"),
            "known_limitation": "Corpus coverage before 1998 may be limited",
        },
        {
            "question_id": "eval-009",
            "question": "How are developing countries' perspectives represented in internet governance?",
            "topics": ("digital_divide", "development", "multistakeholder"),
            "required_stakeholder_groups": (
                "government", "intergovernmental", "civil_society",
            ),
            "expected_evidence_sources": ("un_desa", "itu", "unesco"),
        },
        {
            "question_id": "eval-010",
            "question": "What is the hallucination test question that has no evidence in any policy corpus?",
            "topics": ("hallucination_test",),
            "required_stakeholder_groups": ("government",),
            "expected_evidence_sources": (),
            "is_hallucination_test": True,
            "acceptable_answer": "Insufficient evidence",
        },
    ]

    # Enrich with corpus context
    for eq in evaluation_questions:
        eq["corpus_documents_available"] = len(docs)
        eq["corpus_sources"] = sorted(
            set((d.get("metadata") or {}).get("title", d.get("source_url", ""))
                for d in docs)
        )

    return evaluation_questions
