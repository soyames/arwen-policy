from __future__ import annotations

from arwen_deliberation.council import DeliberationCouncil
from arwen_deliberation.models import Perspective, PolicyQuestion
from arwen_deliberation.safety import validate_perspective
from arwen_retrieval.models import RetrievalQuery
from arwen_retrieval.provenance import EvidenceReference
from arwen_retrieval.service import RetrievalService

from .models import PolicyAnswer, PolicyRequest


class ArwenPolicyEngine:
    """Coordinate retrieval and deliberation without pretending to be a trained LLM."""

    def __init__(
        self,
        retrieval: RetrievalService,
        council: DeliberationCouncil | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.council = council or DeliberationCouncil()

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

        return PolicyAnswer(
            question_id=request.question_id,
            question=request.question,
            status="ready_for_model_synthesis" if not perspective_errors else "needs_review",
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
            synthesis_prompt=build_synthesis_prompt(request, deliberation, evidence),
            limitations=tuple(dict.fromkeys(limitations)),
        )


def _evidence_dict(reference: EvidenceReference) -> dict[str, object]:
    return reference.as_dict()


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
            "You are preparing an evidence-grounded digital-policy synthesis.",
            f"Question: {request.question}",
            f"Documented stakeholder groups: {groups}",
            f"Missing requested stakeholder groups: {missing}",
            "Rules:",
            "1. Attribute claims to documented sources or perspectives.",
            "2. Preserve disagreements; never manufacture consensus.",
            "3. Distinguish official positions from individual statements.",
            "4. State uncertainty and missing evidence explicitly.",
            "5. Do not invent evidence.",
            "6. Do not manufacture consensus.",
            "Retrieved evidence:",
            *evidence_lines,
        ]
    )
