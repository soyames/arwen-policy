from __future__ import annotations

from arwen_deliberation.council import DeliberationCouncil
from arwen_deliberation.models import Perspective, PolicyQuestion
from arwen_deliberation.safety import validate_perspective
from arwen_retrieval.models import RetrievalQuery
from arwen_retrieval.provenance import EvidenceReference
from arwen_retrieval.service import RetrievalService

from .models import PolicyAnswer, PolicyRequest


class ArwenPolicyEngine:
    """Coordinate retrieval, deliberation, and optional model synthesis.

    When a *model_provider* is supplied, ``analyze()`` will call the provider
    to synthesise the final answer from the deliberation result and retrieved
    evidence.  Without a provider the engine still produces a structured
    ``PolicyAnswer`` with a ``synthesis_prompt`` ready for external consumption.
    """

    def __init__(
        self,
        retrieval: RetrievalService,
        council: DeliberationCouncil | None = None,
        model_provider: object | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.council = council or DeliberationCouncil()
        self.model_provider = model_provider

    def analyze(
        self,
        request: PolicyRequest,
        perspectives: list[Perspective] | None = None,
    ) -> PolicyAnswer:
        retrieval_query = RetrievalQuery(
            text=request.question,
            top_k=request.top_k,
            stakeholder_groups=request.stakeholder_groups,
            topics=request.topics,
        )
        evidence = self.retrieval.evidence(retrieval_query)
        perspectives = perspectives or []

        # Auto-generate perspectives from evidence when none are supplied.
        if not perspectives and evidence:
            perspectives = _perspectives_from_evidence(evidence)

        perspective_errors = [
            error
            for perspective in perspectives
            for error in validate_perspective(perspective)
        ]

        question = PolicyQuestion(
            question_id=request.question_id,
            text=request.question,
            topics=request.topics,
            required_stakeholder_groups=request.stakeholder_groups,
        )
        deliberation = self.council.deliberate(question, perspectives)

        limitations = list(deliberation.unresolved_questions)
        limitations.extend(perspective_errors)
        if not evidence:
            limitations.append("No retrieval evidence matched the question.")

        prompt = build_synthesis_prompt(request, deliberation, evidence)
        synthesis: str | None = None
        model_provenance: dict[str, object] | None = None

        if self.model_provider is not None and hasattr(self.model_provider, "generate"):
            try:
                result = self.model_provider.generate(prompt=prompt, context=list(evidence))
                synthesis = result.get("output") if isinstance(result, dict) else str(result)
                model_provenance = (
                    result.get("provenance") if isinstance(result, dict) else None
                )
            except Exception:
                synthesis = None

        status = "ready_for_model_synthesis" if not perspective_errors else "needs_review"
        if synthesis:
            status = "model_synthesis_complete"

        return PolicyAnswer(
            question_id=request.question_id,
            question=request.question,
            status=status,
            evidence=tuple(_evidence_dict(item) for item in evidence),
            stakeholder_coverage={
                "represented": list(deliberation.represented_groups),
                "missing": list(deliberation.missing_groups),
            },
            deliberation={
                "agreements": list(deliberation.agreements),
                "disagreements": list(deliberation.disagreements),
                "evidence_record_ids": list(deliberation.evidence_record_ids),
            },
            synthesis_prompt=prompt,
            limitations=tuple(dict.fromkeys(limitations)),
            synthesis=synthesis,
            model_provenance=model_provenance,
        )


def _evidence_dict(reference: EvidenceReference) -> dict[str, object]:
    return reference.as_dict()


def _perspectives_from_evidence(
    evidence: list[EvidenceReference],
) -> list[Perspective]:
    """Generate perspectives from evidence records' stakeholder metadata.

    Each unique stakeholder group found in the evidence produces one
    Perspective whose position text is drawn from the evidence records
    associated with that group.
    """
    from collections import defaultdict

    grouped: dict[str, list[EvidenceReference]] = defaultdict(list)
    for ref in evidence:
        for group in getattr(ref, "stakeholder_groups", ()) or ():
            grouped[group].append(ref)

    perspectives: list[Perspective] = []
    for group, refs in sorted(grouped.items()):
        position_text = "; ".join(
            getattr(r, "text_snippet", "") or ""
            for r in refs[:3]
        )[:500]
        if not position_text:
            position_text = f"Evidence from {len(refs)} record(s) associated with {group}"

        perspectives.append(
            Perspective(
                stakeholder_group=group,
                position=position_text,
                evidence_record_ids=tuple(
                    r.record_id for r in refs if hasattr(r, "record_id")
                ),
                confidence=min(0.8, 0.4 + 0.1 * len(refs)),
                attribution=f"Auto-generated from {len(refs)} evidence record(s)",
                is_official_position=False,
            )
        )
    return perspectives


def build_synthesis_prompt(request: PolicyRequest, deliberation, evidence) -> str:
    evidence_lines = [
        f"- {item.document_id}/{item.segment_id or item.record_id}: "
        f"score={item.retrieval_score:.4f}"
        for item in evidence
    ]
    groups = ", ".join(deliberation.represented_groups) or "none documented"
    missing = ", ".join(deliberation.missing_groups) or "none"
    return "\n".join(
        [
            "You are preparing a multistakeholder, human-rights-aware policy synthesis.",
            f"Question: {request.question}",
            f"Documented stakeholder groups: {groups}",
            f"Missing requested stakeholder groups: {missing}",
            "Rules:",
            "1. Combine policy reasoning with evidence when available.",
            "2. Identify stakeholders AND rights-holders where materially relevant.",
            "3. Consider human-rights impacts where the policy has material human-rights implications.",
            "4. Attribute specific claims to documented sources or perspectives.",
            "5. Preserve disagreements; never manufacture consensus.",
            "6. Distinguish official positions from individual statements.",
            "7. Distinguish general stakeholder perspectives from documented organizational positions.",
            "8. State uncertainty and missing evidence explicitly.",
            "9. Do not invent evidence, positions, facts, or human-rights impacts.",
            "10. For general policy questions, provide reasoned multistakeholder and HRIAM analysis even when retrieved evidence is limited.",
            "Retrieved evidence:",
            *evidence_lines,
        ]
    )
