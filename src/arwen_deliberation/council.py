from __future__ import annotations

from collections import defaultdict

from .models import DeliberationResult, Perspective, PolicyQuestion


class DeliberationCouncil:
    """Deterministic council coordinator; model adapters can replace its heuristics later."""

    def deliberate(
        self,
        question: PolicyQuestion,
        perspectives: list[Perspective],
    ) -> DeliberationResult:
        grouped: dict[str, list[Perspective]] = defaultdict(list)
        for perspective in perspectives:
            grouped[perspective.stakeholder_group].append(perspective)

        represented = tuple(sorted(grouped))
        required = set(question.required_stakeholder_groups)
        missing = tuple(sorted(required - set(represented)))

        agreements = self._find_shared_positions(perspectives)
        disagreements = self._find_disagreements(perspectives)
        evidence_ids = tuple(
            sorted(
                {
                    record_id
                    for perspective in perspectives
                    for record_id in perspective.evidence_record_ids
                }
            )
        )

        constraints = [
            "Do not treat silence as neutrality.",
            "Do not treat a participant statement as an organizational position "
            "without attribution.",
            "Do not manufacture consensus from absent disagreement.",
        ]
        if missing:
            constraints.append(
                "Explicitly disclose stakeholder groups for which evidence is missing."
            )

        unresolved = list(missing)
        if not evidence_ids:
            unresolved.append("No linked corpus evidence was supplied for the deliberation.")

        return DeliberationResult(
            question_id=question.question_id,
            perspectives=tuple(perspectives),
            represented_groups=represented,
            missing_groups=missing,
            agreements=tuple(agreements),
            disagreements=tuple(disagreements),
            unresolved_questions=tuple(unresolved),
            evidence_record_ids=evidence_ids,
            synthesis_constraints=tuple(constraints),
        )

    @staticmethod
    def _find_shared_positions(perspectives: list[Perspective]) -> list[str]:
        normalized: dict[str, set[str]] = defaultdict(set)
        for perspective in perspectives:
            key = " ".join(perspective.position.lower().split())
            if key:
                normalized[key].add(perspective.stakeholder_group)
        return sorted(
            position
            for position, groups in normalized.items()
            if len(groups) >= 2
        )

    @staticmethod
    def _find_disagreements(perspectives: list[Perspective]) -> list[str]:
        positions = {
            " ".join(p.position.lower().split())
            for p in perspectives
            if p.position.strip()
        }
        if len(positions) <= 1:
            return []
        return ["Documented stakeholder positions differ; preserve the disagreement in synthesis."]
