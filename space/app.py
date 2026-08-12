"""
Arwen Policy -- Evidence-grounded, model-backed digital-policy deliberation.

Uses the real Arwen retrieval + deliberation pipeline with optional
Qwen/Ollama model synthesis.  ZeroGPU compatible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
# Ensure the src/ directory is on the path.
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from arwen_deliberation.council import DeliberationCouncil
from arwen_engine.models import PolicyRequest
from arwen_engine.pipeline import ArwenPolicyEngine
from arwen_retrieval.models import CorpusRecord
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService

# ---------------------------------------------------------------------------
# Model provider — HF Inference API (configurable via MODEL_ID / HF_TOKEN)
# ---------------------------------------------------------------------------

_model_provider = None
_model_backend = "unavailable"
_model_error = None

# Try HF Inference API first (works on Spaces with HF_TOKEN secret)
try:
    from arwen_etl.engine.hf_inference_provider import HfInferenceProvider

    _model_provider = HfInferenceProvider()
    _info = _model_provider.get_model_info()
    _model_backend = _info.get("backend", "unknown")
    if not _info.get("token_configured"):
        _model_backend = "unconfigured"
        _model_error = "HF_TOKEN not set. Add it as a Space Secret."
except Exception as exc:
    _model_backend = "unavailable"
    _model_error = str(exc)


# ---------------------------------------------------------------------------
# Load real corpus from data/extracted/
# ---------------------------------------------------------------------------

def _load_corpus() -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    # Try the HF Space path first (/app/data/extracted), then local paths.
    app_dir = Path(__file__).resolve().parent
    candidates = [
        app_dir / "data" / "extracted",          # HF Space
        app_dir.parent / "data" / "extracted",    # local dev (space/ dir)
        Path("data") / "extracted",               # repo root
    ]
    extracted_dir = None
    for cand in candidates:
        if cand.is_dir():
            extracted_dir = cand
            break
    if extracted_dir is None:
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


# ---------------------------------------------------------------------------
# Build engine
# ---------------------------------------------------------------------------

_retriever = InMemoryRetriever(_corpus_records)
_service = RetrievalService(_retriever)
# Build engine WITHOUT model provider  --  synthesis is done via the GPU fn.
_engine = ArwenPolicyEngine(
    retrieval=_service,
    council=DeliberationCouncil(),
    model_provider=None,
)


# ---------------------------------------------------------------------------
# Model inference — runs on HF Inference Providers (serverless).
# No local GPU required. Retrieval, deliberation, and prompt construction
# stay on CPU.  Synthesis prompt is sent to HF-hosted model.
# ---------------------------------------------------------------------------

def _model_synthesize(synthesis_prompt: str) -> dict[str, Any]:
    """Run model inference on ZeroGPU. Returns dict with output or error."""
    if _model_provider is None:
        return {"output": None, "error": "No model provider available. Check Space configuration."}
    try:
        result = _model_provider.generate(prompt=synthesis_prompt)
        if isinstance(result, dict):
            err = result.get("error")
            if err:
                return {"output": None, "error": f"{err.get('code', 'unknown')}: {err.get('message', '')}"}
            return {"output": result.get("output"), "error": None}
        return {"output": str(result), "error": None}
    except Exception as exc:
        return {"output": None, "error": str(exc)}


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
    # Retrieval + deliberation (CPU only)
    answer = _engine.analyze(request)

    # Model synthesis (ZeroGPU)  --  called separately
    synthesis_result: dict[str, Any] | None = None
    if _model_provider is not None and answer.synthesis_prompt:
        synthesis_result = _model_synthesize(answer.synthesis_prompt)
    synthesis = synthesis_result.get("output") if synthesis_result else None
    synthesis_error = synthesis_result.get("error") if synthesis_result else None

    # -- Header -----------------------------------------------------------
    lines = [
        "## Policy Analysis",
        "",
        f"> **{question}**",
        "",
    ]

    # -- Evidence ---------------------------------------------------------
    lines.append("### Evidence")
    lines.append("")
    if not answer.evidence:
        lines.append("*No relevant evidence found in the corpus.*")
    else:
        for i, ev in enumerate(answer.evidence, 1):
            ev.get("record_id", "?")
            score = ev.get("retrieval_score", 0)
            url = ev.get("url", "")
            doc_id = ev.get("document_id", "?")
            source_id = ev.get("source_id", "?")

            source_link = f" [source]({url})" if url else ""
            lines.append(
                f"**{i}.** `{doc_id[:12]}...` "
                f"(relevance: {score:.3f}, source: `{source_id[:12]}...`)"
                f"{source_link}"
            )
    lines.append("")

    # -- Stakeholders -----------------------------------------------------
    coverage = answer.stakeholder_coverage
    represented = coverage.get("represented", [])
    missing = coverage.get("missing", [])

    lines.append("### Stakeholders")
    lines.append("")
    if represented:
        for group in represented:
            lines.append(f"- **{group}**  --  represented in evidence")
    else:
        lines.append("- *No stakeholder groups identified in evidence*")
    if missing:
        for group in missing:
            lines.append(f"- **{group}**  --  *no evidence found*")
    lines.append("")

    # -- Deliberation -----------------------------------------------------
    deliberation = answer.deliberation
    agreements = deliberation.get("agreements", [])
    disagreements = deliberation.get("disagreements", [])

    if agreements or disagreements:
        lines.append("### Positions")
        lines.append("")
        if agreements:
            lines.append("**Agreements:**")
            for a in agreements:
                lines.append(f"- {a}")
        if disagreements:
            lines.append("**Disagreements:**")
            for d in disagreements:
                lines.append(f"- {d}")
        lines.append("")

    # -- Analysis ---------------------------------------------------------
    lines.append("### Analysis")
    lines.append("")
    if synthesis:
        synth = synthesis
        if synth.strip().startswith("{"):
            try:
                parsed = json.loads(synth)
                analysis_text = parsed.get("analysis", "")
                pro = parsed.get("pro_argument", "")
                contra = parsed.get("contra_argument", "")
                if pro:
                    lines.append(f"**Pro:** {pro}")
                    lines.append("")
                if contra:
                    lines.append(f"**Contra:** {contra}")
                    lines.append("")
                if analysis_text:
                    lines.append(analysis_text)
                else:
                    lines.append(synth)
            except json.JSONDecodeError:
                lines.append(synth)
        else:
            lines.append(synth)
    elif synthesis_error:
        lines.append(f":warning: **Model synthesis unavailable:** {synthesis_error}")
        lines.append("")
        lines.append("*Evidence retrieval is working. The model backend needs configuration.*")
    elif _model_backend == "unconfigured":
        lines.append(":warning: **Model not configured.** Add `HF_TOKEN` as a Space Secret.")
    elif _model_backend == "unavailable":
        lines.append(":warning: **Model backend unavailable.** Check Space logs for details.")
    else:
        lines.append("*(Model synthesis pending — provider did not return output)*")
    lines.append("")

    # -- Limitations ------------------------------------------------------
    if answer.limitations:
        lines.append("### Limitations")
        lines.append("")
        for lim in answer.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    # -- Footer -----------------------------------------------------------
    lines.append(
        "---\n"
        f"*Corpus: {len(_corpus_records)} records. "
        f"Model: {_model_backend}. "
        f"Status: {answer.status}.*"
    )

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
