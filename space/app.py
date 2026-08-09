"""
Arwen Policy -- Evidence-grounded, model-backed digital-policy deliberation.

Uses the real Arwen retrieval + deliberation pipeline with optional
Qwen/Ollama model synthesis.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr

# Ensure the src/ directory is on the path.
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from arwen_deliberation.council import DeliberationCouncil
from arwen_deliberation.models import Perspective
from arwen_engine.models import PolicyRequest
from arwen_engine.pipeline import ArwenPolicyEngine
from arwen_retrieval.models import CorpusRecord
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService

# ---------------------------------------------------------------------------
# Model provider (optional -- Ollama)
# ---------------------------------------------------------------------------

_model_provider = None
try:
    from arwen_etl.engine.qwen_provider import QwenProvider

    _model_provider = QwenProvider()
    _model_backend = _model_provider.get_model_info().get("backend", "unknown")
except Exception:
    _model_backend = "unavailable"


# ---------------------------------------------------------------------------
# Load real corpus from data/extracted/
# ---------------------------------------------------------------------------

def _load_corpus() -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    extracted_dir = Path(__file__).resolve().parents[1] / "data" / "extracted"
    if not extracted_dir.is_dir():
        return records

    for fpath in sorted(extracted_dir.glob("*.json")):
        try:
            doc = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception:
            continue

        text = doc.get("text", "")
        if not text or len(text) < 100:
            continue

        # Build one retrieval record per segment for finer-grained retrieval,
        # falling back to the full document when there are no segments.
        segments = doc.get("segments", [])
        if segments:
            for seg in segments:
                seg_text = seg.get("text", "")
                if len(seg_text) < 50:
                    continue
                records.append(
                    CorpusRecord(
                        record_id=f"{doc['document_id']}/{seg['segment_id']}",
                        text=seg_text,
                        source_id=doc["source_id"],
                        document_id=doc["document_id"],
                        segment_id=seg["segment_id"],
                        title=(doc.get("metadata") or {}).get("title") or doc.get("source_url", ""),
                        url=doc.get("source_url"),
                        topics=(),
                    )
                )
        else:
            records.append(
                CorpusRecord(
                    record_id=doc["document_id"],
                    text=text,
                    source_id=doc["source_id"],
                    document_id=doc["document_id"],
                    title=(doc.get("metadata") or {}).get("title") or doc.get("source_url", ""),
                    url=doc.get("source_url"),
                    topics=(),
                )
            )
    return records


_corpus_records = _load_corpus()

# Fall back to sample data when no real corpus exists.
if not _corpus_records:
    _corpus_records = [
        CorpusRecord(
            record_id="sample-1",
            text=(
                "Internet governance requires transparent, inclusive, and accountable "
                "policy-development processes that enable the full participation of all "
                "stakeholders including governments, the private sector, civil society, "
                "the technical community, and international organisations."
            ),
            source_id="itu-wsis",
            document_id="d1",
            title="Tunis Agenda for the Information Society",
            url="https://www.itu.int/net/wsis/docs2/tunis/off/6rev1.html",
        ),
        CorpusRecord(
            record_id="sample-2",
            text=(
                "Risk-based regulatory frameworks for artificial intelligence should "
                "distinguish between high-risk applications that require mandatory "
                "conformity assessments and lower-risk applications that may rely on "
                "voluntary codes of conduct."
            ),
            source_id="oecd-ai",
            document_id="d2",
            title="OECD AI Principles",
            url="https://oecd.ai/en/ai-principles",
        ),
    ]

# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------

_retriever = InMemoryRetriever(_corpus_records)
_service = RetrievalService(_retriever)
_engine = ArwenPolicyEngine(
    retrieval=_service,
    council=DeliberationCouncil(),
    model_provider=_model_provider,
)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def analyse(question: str, top_k: int) -> str:
    if not question.strip():
        return "Please enter a policy question."

    request = PolicyRequest(
        question_id="query",
        question=question,
        top_k=max(1, min(top_k, len(_corpus_records))),
    )
    answer = _engine.analyze(request)

    lines = [
        "## Arwen Policy Analysis",
        "",
        f"**Question:** {question}",
        f"**Status:** {answer.status}",
        f"**Model backend:** {_model_backend}",
        "",
        "### Evidence Retrieved",
    ]

    for i, ev in enumerate(answer.evidence, 1):
        rid = ev.get("record_id", "?")
        score = ev.get("retrieval_score", 0)
        lines.append(f"{i}. `{rid}` (score: {score:.3f})")

    lines.extend([
        "",
        "### Stakeholder Coverage",
        f"Represented: {', '.join(answer.stakeholder_coverage.get('represented', [])) or 'none'}",
        f"Missing: {', '.join(answer.stakeholder_coverage.get('missing', [])) or 'none'}",
        "",
        "### Deliberation",
    ])

    for k, v in answer.deliberation.items():
        if v:
            lines.append(f"- **{k}:** {', '.join(str(x) for x in v)}")

    lines.append("")
    if answer.synthesis:
        lines.extend([
            "### Model Synthesis",
            "",
            answer.synthesis,
        ])
    else:
        lines.extend([
            "### Synthesis Prompt",
            "*(No model available; prompt ready for external synthesis)*",
            "",
            "```",
            answer.synthesis_prompt[:500],
            "...",
            "```",
        ])

    lines.extend([
        "",
        "### Limitations",
        *(f"- {lim}" for lim in answer.limitations),
        "",
        f"*Corpus: {len(_corpus_records)} records indexed.*",
    ])

    return "\n".join(lines)


with gr.Blocks(title="Arwen Policy") as demo:
    gr.Markdown(
        "# Arwen Policy\n"
        "**Evidence-grounded digital-policy deliberation with real retrieval "
        "and model synthesis.**\n\n"
        f"Corpus: {len(_corpus_records)} records. Model: {_model_backend}."
    )

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="Policy Question",
                placeholder="e.g. How should AI governance ensure accountability?",
                lines=3,
            )
            top_k = gr.Slider(
                label="Max Evidence Items",
                minimum=1,
                maximum=max(10, len(_corpus_records)),
                value=5,
                step=1,
            )
            submit = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown("*(Analysis will appear here)*")

    submit.click(fn=analyse, inputs=[question, top_k], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
