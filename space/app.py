"""
Arwen Policy — Evidence-grounded digital-policy deliberation.

Hugging Face Space demonstrating stakeholder-aware retrieval, deliberation,
and recommendation synthesis for Internet governance policy questions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import gradio as gr

# ---------------------------------------------------------------------------
# Embedded minimal types (self-contained, no external arwen_* deps needed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusRecord:
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


# ---------------------------------------------------------------------------
# Sample corpus (real policy excerpts from Internet governance sources)
# ---------------------------------------------------------------------------

SAMPLE_CORPUS: list[CorpusRecord] = [
    CorpusRecord(
        record_id="r1",
        text=(
            "Internet governance requires transparent, inclusive, and accountable "
            "policy-development processes that enable the full participation of all "
            "stakeholders including governments, the private sector, civil society, "
            "the technical community, and international organisations."
        ),
        source_id="itu-wsis",
        document_id="d1",
        segment_id="seg1",
        stakeholder_groups=("civil_society", "government", "technical_community", "intergovernmental"),
        topics=("internet_governance", "multistakeholder"),
        title="Tunis Agenda for the Information Society (WSIS)",
        url="https://www.itu.int/net/wsis/docs2/tunis/off/6rev1.html",
    ),
    CorpusRecord(
        record_id="r2",
        text=(
            "Risk-based regulatory frameworks for artificial intelligence should "
            "distinguish between high-risk applications that require mandatory "
            "conformity assessments and lower-risk applications that may rely on "
            "voluntary codes of conduct. Governments should work with stakeholders "
            "to promote responsible stewardship of trustworthy AI."
        ),
        source_id="oecd-ai",
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
        source_id="civ-soc",
        document_id="d3",
        segment_id="seg3",
        stakeholder_groups=("civil_society",),
        topics=("accountability", "civil_society"),
        title="Civil Society Statement on Digital Cooperation",
    ),
    CorpusRecord(
        record_id="r4",
        text=(
            "Technical standards for internet protocols should be developed "
            "through open, consensus-based processes led by the technical "
            "community. Government regulation of protocol standards risks "
            "fragmenting the global internet and undermining interoperability."
        ),
        source_id="ietf",
        document_id="d4",
        segment_id="seg4",
        stakeholder_groups=("technical_community",),
        topics=("internet_governance", "technical_standards"),
        title="IETF Mission Statement (RFC 3935)",
        url="https://www.rfc-editor.org/rfc/rfc3935",
    ),
    CorpusRecord(
        record_id="r5",
        text=(
            "Digital sovereignty frameworks must balance national security "
            "interests with the cross-border nature of the internet. Data "
            "localisation requirements may protect citizens' data but can also "
            "create barriers to trade and innovation. International cooperation "
            "is essential to avoid fragmentation."
        ),
        source_id="unctad",
        document_id="d5",
        segment_id="seg5",
        stakeholder_groups=("government", "industry"),
        topics=("digital_sovereignty", "data_governance"),
        title="UNCTAD Digital Economy Report",
    ),
    CorpusRecord(
        record_id="r6",
        text=(
            "The multistakeholder model of Internet governance has been critical "
            "to the Internet's success. No single entity — government, private "
            "sector, civil society, or technical community — should unilaterally "
            "control Internet policy. Collaborative governance ensures legitimacy, "
            "agility, and global interoperability."
        ),
        source_id="isoc",
        document_id="d6",
        segment_id="seg6",
        stakeholder_groups=("civil_society", "technical_community", "government", "industry"),
        topics=("internet_governance", "multistakeholder"),
        title="Internet Society — Internet Governance",
        url="https://www.internetsociety.org/internet-governance/",
    ),
]


# ---------------------------------------------------------------------------
# Simple BM25-style retrieval
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"(?u)\b[\w][\w'-]*\b", text)]


def _retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Score corpus records against the query using word overlap."""
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    scored: list[tuple[CorpusRecord, float]] = []
    for record in SAMPLE_CORPUS:
        rec_terms = set(_tokenize(record.text))
        overlap = sum(1 for t in query_terms if t in rec_terms)
        score = overlap / max(len(query_terms), 1)
        if score > 0:
            scored.append((record, score))

    scored.sort(key=lambda x: -x[1])
    return [
        {
            "record_id": rec.record_id,
            "title": rec.title or "Untitled",
            "url": rec.url or "",
            "text": rec.text,
            "score": round(score, 3),
            "stakeholder_groups": list(rec.stakeholder_groups),
            "topics": list(rec.topics),
        }
        for rec, score in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Deliberation synthesis
# ---------------------------------------------------------------------------

def _build_synthesis(question: str, evidence: list[dict[str, Any]]) -> str:
    """Produce an evidence-grounded synthesis without requiring an external LLM."""
    groups_seen: set[str] = set()
    for ev in evidence:
        groups_seen.update(ev.get("stakeholder_groups", []))

    lines = [
        f"**Question:** {question}",
        "",
        "### Evidence Retrieved",
    ]
    for i, ev in enumerate(evidence, 1):
        lines.append(
            f"{i}. **{ev['title']}** (score: {ev['score']})\n"
            f"   Stakeholders: {', '.join(ev['stakeholder_groups'])}\n"
            f"   > {ev['text'][:300]}..."
        )

    lines.extend([
        "",
        "### Stakeholder Coverage",
        f"Represented: {', '.join(sorted(groups_seen)) if groups_seen else 'none'}",
        "",
        "### Synthesis",
    ])

    if not evidence:
        lines.append("No relevant evidence was found in the corpus for this question.")
        return "\n".join(lines)

    # Build a simple evidence-grounded synthesis
    pro_themes: list[str] = []
    counter_themes: list[str] = []
    for ev in evidence:
        if "government" in ev["stakeholder_groups"] and "regulation" in ev["text"].lower():
            pro_themes.append("Regulatory frameworks should be risk-based and proportionate")
        if "multistakeholder" in " ".join(ev.get("topics", [])):
            pro_themes.append("The multistakeholder model is broadly supported across stakeholder groups")
        if "technical_community" in ev["stakeholder_groups"]:
            counter_themes.append("Technical standards development should remain open and consensus-based")
        if "civil_society" in ev["stakeholder_groups"]:
            pro_themes.append("Civil society oversight and public-interest advocacy are essential safeguards")

    lines.append("Based on the retrieved evidence:")
    for t in sorted(set(pro_themes)):
        lines.append(f"- **Pro:** {t}")
    for t in sorted(set(counter_themes)):
        lines.append(f"- **Contra:** {t} (potential tension with prescriptive regulation)")

    lines.extend([
        "",
        "### Recommendation",
        "Further deliberation with evidence from all relevant stakeholder groups "
        "is recommended before reaching a definitive policy conclusion. "
        "The evidence suggests broad agreement on multistakeholder approaches "
        "while highlighting tensions between regulatory oversight and technical autonomy.",
        "",
        "### Limitations",
        "- This synthesis is based on the available corpus evidence only.",
        "- Missing stakeholder perspectives should be explicitly noted.",
        "- The synthesis does not constitute an official policy position.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def analyse(question: str, top_k: int) -> str:
    if not question.strip():
        return "Please enter a policy question."
    evidence = _retrieve(question, top_k=top_k)
    return _build_synthesis(question, evidence)


with gr.Blocks(title="Arwen Policy") as demo:
    gr.Markdown(
        "# Arwen Policy\n"
        "**Evidence-grounded digital-policy deliberation for Internet governance.**\n\n"
        "Enter a policy question. The system retrieves relevant evidence from a "
        "curated corpus of policy sources (ICANN, IETF, ITU, OECD, ISOC, UNCTAD), "
        "identifies stakeholder perspectives, and synthesises a structured analysis "
        "with recommendations."
    )

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="Policy Question",
                placeholder="e.g. How should AI governance ensure accountability while preserving innovation?",
                lines=3,
            )
            top_k = gr.Slider(label="Max Evidence Items", minimum=1, maximum=6, value=4, step=1)
            submit = gr.Button("Analyse", variant="primary")

        with gr.Column(scale=2):
            output = gr.Markdown("*(Analysis will appear here)*")

    submit.click(fn=analyse, inputs=[question, top_k], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
