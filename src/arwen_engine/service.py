from __future__ import annotations

from arwen_deliberation.models import Perspective

from .models import PolicyAnswer, PolicyRequest
from .pipeline import ArwenPolicyEngine


class PolicyService:
    """Stable application boundary for CLI, API and Space integrations."""

    def __init__(self, engine: ArwenPolicyEngine) -> None:
        self.engine = engine

    def answer(
        self,
        request: PolicyRequest,
        perspectives: list[Perspective] | None = None,
    ) -> PolicyAnswer:
        return self.engine.analyze(request, perspectives)
