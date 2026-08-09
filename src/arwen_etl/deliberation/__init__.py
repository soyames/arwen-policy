"""Deliberation engine that produces structured arguments from evidence."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    deliberation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class DeliberationEngine:
    """Generate pro and contra arguments for a given claim using a ModelProvider."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.negation_patterns = [
            (r"\b(should|must|will|can)\b", r"not \1"),
            (r"\b(is|are|was|were)\b", r"\1 not"),
            (r"\b(support|approve|endorse|favor)\b", r"oppose \1"),
            (r"\b(oppose|reject|disapprove)\b", r"support \1"),
        ]

    def _apply_negation(self, text: str) -> str:
        result = text
        for pattern, repl in self.negation_patterns:
            result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", result).strip()

    def _parse_model_output(self, output: str) -> Dict[str, Any] | None:
        """Try to extract a JSON object from the model output."""
        # Find the first JSON object in the output
        try:
            start = output.index("{")
            end = output.rindex("}") + 1
            return json.loads(output[start:end])
        except (ValueError, json.JSONDecodeError):
            return None

    def deliberate(
        self,
        claim: str,
        evidence: List[Dict[str, Any]] | None = None,
    ) -> DeliberationResult:
        """Produce a DeliberationResult using the configured ModelProvider."""
        evidence = evidence or []
        evidence_texts = [e.get("text", "") for e in evidence]
        prompt = (
            f"Claim: {claim}\n"
            f"Evidence: {' | '.join(evidence_texts)}\n"
            "Analyse the claim. Return a JSON object with keys: "
            "pro_argument (string), pro_confidence (float 0-1), "
            "contra_argument (string), contra_confidence (float 0-1), "
            "analysis (string)."
        )

        response = self.provider.generate(prompt=prompt, context=evidence)
        output = response.get("output", "")
        parsed = self._parse_model_output(output)

        if parsed and all(
            k in parsed
            for k in ("pro_argument", "pro_confidence", "contra_argument", "contra_confidence")
        ):
            pro = Argument(
                text=parsed["pro_argument"],
                stance="pro",
                confidence=float(parsed["pro_confidence"]),
                source="model",
            )
            contra = Argument(
                text=parsed["contra_argument"],
                stance="contra",
                confidence=float(parsed["contra_confidence"]),
                source="model",
            )
            consensus = 1.0 - abs(pro.confidence - contra.confidence) * 0.5
            method = "provider_model"
        else:
            # Fallback to rule-based heuristics
            pro = Argument(text=claim, stance="pro", confidence=0.8, source="claim")
            contra_text = self._apply_negation(claim)
            contra = Argument(
                text=contra_text, stance="contra", confidence=0.6, source="negation_engine"
            )
            consensus = 1.0 - (contra.confidence * 0.5)
            method = "rule_based_fallback"

        consensus = max(0.0, min(1.0, consensus))

        return DeliberationResult(
            claim=claim,
            arguments=[pro, contra],
            consensus_score=consensus,
            method=method,
        )
