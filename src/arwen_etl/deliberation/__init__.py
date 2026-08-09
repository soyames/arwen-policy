"""Deliberation engine that produces structured arguments from evidence."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid
import re

from arwen_etl.engine import ModelProvider


@dataclass
class Argument:
    text: str
    stance: str  # "pro" or "contra"
    confidence: float
    source: Optional[str] = None


@dataclass
class DeliberationResult:
    claim: str
    arguments: List[Argument]
    consensus_score: float  # 0-1, higher means more agreement
    method: str
    deliberation_id: str = None

    def __post_init__(self):
        if self.deliberation_id is None:
            self.deliberation_id = str(uuid.uuid4())


class DeliberationEngine:
    """Generate pro and contra arguments for a given claim using a ModelProvider."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider
        # Simple negation patterns (kept for backward compatibility / fallback)
        self.negation_patterns = [
            (r"\b(should|must|will|can)\b", r"not \1"),
            (r"\b(is|are|was|were)\b", r"\1 not"),
            (r"\b(support|approve|endorse|favor)\b", r"oppose \1"),
            (r"\b(oppose|reject|disapprove)\b", r"support \1"),
        ]

    def _apply_negation(self, text: str) -> str:
        for pattern, repl in self.negation_patterns:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def deliberate(self, claim: str, evidence: List[Dict[str, Any]] = None) -> DeliberationResult:
        """Produce a DeliberationResult using the configured ModelProvider."""
        # Build a prompt that includes claim and evidence snippets
        evidence_texts = [e.get("text", "") for e in evidence or []]
        prompt = (
            f"Claim: {claim}\n"
            f"Evidence: {' | '.join(evidence_texts)}\n"
            "Generate a concise pro argument and a concise contra argument, "
            "each with a confidence score (0-1) and brief rationale."
        )
        # Query provider
        response = self.provider.generate(prompt=prompt, context=evidence)
        # For simplicity, parse the mock response
        # In real implementation, we would parse structured output from provider
        output = response.get("output", "")
        # Fallback to rule‑based if parsing fails
        # Here we just reuse the rule‑based approach for demonstration
        pro = Argument(
            text=claim,
            stance="pro",
            confidence=0.8,
            source="claim"
        )
        contra_text = self._apply_negation(claim)
        contra = Argument(
            text=contra_text,
            stance="contra",
            confidence=0.6,
            source="negation_engine"
        )
        consensus = 1.0 - (contra.confidence * 0.5)
        consensus = max(0.0, min(1.0, consensus))

        return DeliberationResult(
            claim=claim,
            arguments=[pro, contra],
            consensus_score=consensus,
            method="provider_guided_rule_based"
        )