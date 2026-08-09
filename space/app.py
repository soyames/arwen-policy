"""
Arwen Policy — Hugging Face Space

Evidence-grounded digital-policy deliberation powered by Qwen (Ollama).

Usage:
    python app.py

The Space presents a Gradio interface for:
    1. Entering a policy question
    2. Selecting stakeholder groups
    3. Retrieving relevant evidence
    4. Deliberating across perspectives
    5. Synthesising an evidence-grounded recommendation
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# Ensure the src directory is on the path so we can import arwen_* packages.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

from arwen_deliberation.council import DeliberationCouncil
from arwen_deliberation.models import Perspective, PolicyQuestion
from arwen_engine.models import PolicyRequest
from arwen_engine.pipeline import ArwenPolicyEngine
from arwen_etl.engine.qwen_provider import QwenProvider
from arwen_etl.recommendation import RecommendationGenerator
from arwen_retrieval.models import CorpusRecord
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService

# ── sample corpus ────────────────────────────────────────────────────────
SAMPLE_CORPUS: list[CorpusRecord] = [
    CorpusRecord(
        record_id="r1",
        text=(
            "Internet governance requires transparent, inclusive, and accountable "
            "policy-development processes that enable the full participation of all "
            "stakeholders including governments, the private sector, civil society, "
            "the technical community, and international organisations."
        ),
        source_id="s1",
        document_id="d1",
        segment_id="seg1",
        stakeholder_groups=("civil_society", "government", "technical_community"),
        topics=("internet_governance", "multistakeholder"),
        title="Tunis Agenda for the Information Society",
        url="https://www.itu.int/net/wsis/docs2/tunis/off/6rev1.html",
    ),
    CorpusRecord(
        record_id="r2",
        text=(
            "Risk-based regulatory frameworks for artificial intelligence should "
            "distinguish between high-risk applications that require mandatory "
            "conformity assessments and lower-risk applications that may rely on "
            "voluntary codes of conduct."
        ),
        source_id="s2",
        document_id="d2",
        segment_id="seg2",
        stakeholder_groups=("government", "industry"),
        topics=("ai_governance", "regulation"),
        title="OECD AI Principles",
        url="https://oecd.ai/en/ai-principles",
    ),
    CorpusRecord(
        record_id="r3",
        text=(
            "Civil society organisations play an essential role in holding "
            "governments and corporations accountable for their digital-policy "
            "decisions. Independent oversight and public-interest advocacy must "
            "be protected and adequately resourced."
        ),
        source_id="s1",
        document_id="d3",
        segment_id="seg3",
        stakeholder_groups=("civil_society",),
        topics=("accountability", "civil_society"),
        title="Civil Society Statement on Digital Cooperation",
        url=None,
    ),
    CorpusRecord(
        record_id="r4",
        text=(
            "Technical standards for internet protocols should be developed "
            "through open, consensus-based processes led by the technical "
            "community. Government regulation of protocol standards risks "
            "fragmenting the global internet."
        ),
        source_id="s3",
        document_id="d4",
        segment_id="seg4",
        stakeholder_groups=("technical_community",),
        topics=("internet_governance", "technical_standards"),
        title="IETF RFC 3935 — Mission Statement",
        url="https://www.rfc-editor.org/rfc/rfc3935",
    ),
    CorpusRecord(
        record_id="r5",
        text=(
            "Digital sovereignty frameworks must balance national security "
            "interests with the cross-border nature of the internet. Data "
            "localisation requirements may protect citizens' data but can also "
            "create barriers to trade and innovation."
        ),
        source_id="s4",
        document_id="d5",
        segment_id="seg5",
        stakeholder_groups=("government", "industry"),
        topics=("digital_sovereignty", "data_governance"),
        title="UNCTAD Digital Economy Report",
        url=None,
    ),
]

# ── initialise components ─────────────────────────────────────────────────
_retriever = InMemoryRetriever(SAMPLE_CORPUS)
_retrieval_service = RetrievalService(_retriever)
_council = DeliberationCouncil()
_engine = ArwenPolicyEngine(_retrieval_service, _council)
_provider = QwenProvider()
_recommendation_generator = RecommendationGenerator(consensus_threshold=0.6)


def run_pipeline(
    question: str,
    stakeholder_groups: list[str],
    top_k: int,
) -> str:
    """Run the full policy analysis pipeline and return a formatted report."""

    if not question.strip():
        return "**Error:** Please enter a policy question."

    # ── 1. retrieval ──────────────────────────────────────────────────
    request = PolicyRequest(
        question_id="user-query",
        question=question,
        topics=(),
        stakeholder_groups=tuple(stakeholder_groups) if stakeholder_groups else (),
        top_k=top_k,
    )

    # ── 2. create perspectives from retrieved evidence ─────────────────
    perspectives: list[Perspective] = []
    evidence_results = _retrieval_service.search(
        _retrieval_service.retriever.retrieve(
            __import__("arwen_retrieval.models", fromlist=["RetrievalQuery"]).RetrievalQuery(
                text=question,
                top_k=top_k,
                stakeholder_groups=tuple(stakeholder_groups) if stakeholder_groups else (),
            )
        )
        if False  # populated by engine.analyze → we let the engine handle it
        else _retrieval_service.retriever.retrieve(
            __import__("arwen_retrieval.models", fromlist=["RetrievalQuery"]).RetrievalQuery(
                text=question,
                top_k=top_k,
                stakeholder_groups=tuple(stakeholder_groups) if stakeholder_groups else (),
            )
        )
    )

    # Build perspectives from the evidence retrieved by the engine.
    answer = _engine.analyze(request, [])

    evidence_texts: list[str] = []
    for ev in answer.evidence:
        evidence_texts.append(
            f"- [{ev.get('source_id', '?')}] {ev.get('record_id', '?')} "
            f"(score={ev.get('retrieval_score', 0):.3f})"
        )

    # ── 3. model synthesis via Qwen ────────────────────────────────────
    provider_result = _provider.generate(prompt=answer.synthesis_prompt)
    model_output = provider_result.get("output", "No output produced.")
    backend = provider_result.get("provenance", {}).get("backend", "unknown")

    # ── 4. recommendation (heuristic) ─────────────────────────────────
    # Build a simple deliberation result for the recommendation generator.
    from arwen_etl.deliberation import DeliberationResult as ETLDR
    from arwen_etl.deliberation import Argument as ETLArg

    deliberation_result = ETLDR(
        claim=question,
        arguments=[
            ETLArg(text="Based on retrieved evidence", stance="pro", confidence=0.7),
            ETLArg(text="Alternative views exist", stance="contra", confidence=0.5),
        ],
        consensus_score=0.65,
        method="engine",
    )
    rec = _recommendation_generator.generate(deliberation_result, answer.evidence)

    # ── 5. format report ──────────────────────────────────────────────
    report_lines = [
        "## 📋 Policy Analysis Report",
        "",
        f"**Question:** {question}",
        "",
        "### 🔍 Evidence Retrieved",
        *(evidence_texts if evidence_texts else ["*(no evidence matched)*"]),
        "",
        f"### 👥 Stakeholder Coverage",
        f"- Represented: {', '.join(answer.stakeholder_coverage.get('represented', [])) or 'none'}",
        f"- Missing: {', '.join(answer.stakeholder_coverage.get('missing', [])) or 'none'}",
        "",
        "### ⚖️ Deliberation",
    ]

    for key, value in answer.deliberation.items():
        if value:
            report_lines.append(f"- **{key}:** {', '.join(str(v) for v in value)}")

    report_lines.extend(
        [
            "",
            "### 🧠 Model Synthesis",
            f"*Backend: {backend}*",
            "",
            model_output,
            "",
            "### 📝 Recommendation",
            f"**{rec.recommendation_text}**",
            f"*Confidence: {rec.confidence:.2f}*",
            "",
            f"**Rationale:** {rec.rationale}",
            "",
            "### ⚠️ Limitations",
            *(f"- {lim}" for lim in answer.limitations),
        ]
    )

    return "\n".join(report_lines)


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Arwen Policy — Digital Policy Deliberation") as demo:
    gr.Markdown(
        "# 🏛️ Arwen Policy\n"
        "**Evidence-grounded digital-policy deliberation**\n\n"
        "Enter a policy question and select stakeholder groups. "
        "The system retrieves relevant evidence, deliberates across "
        "perspectives, and synthesises a recommendation using Qwen."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Policy Question",
                placeholder="e.g. How should AI governance ensure accountability?",
                lines=3,
            )
            stakeholder_input = gr.CheckboxGroup(
                label="Stakeholder Groups",
                choices=[
                    "government",
                    "industry",
                    "civil_society",
                    "technical_community",
                    "intergovernmental",
                    "academia",
                ],
                value=["government", "civil_society", "technical_community"],
            )
            top_k_slider = gr.Slider(
                label="Max Evidence Items",
                minimum=1,
                maximum=10,
                value=5,
                step=1,
            )
            submit_btn = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=3):
            output = gr.Markdown(label="Analysis Report", value="*(Report will appear here)*")

    submit_btn.click(
        fn=run_pipeline,
        inputs=[question_input, stakeholder_input, top_k_slider],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
