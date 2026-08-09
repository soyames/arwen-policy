from __future__ import annotations

from typing import List, Dict, Any, Optional
import re

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

class DeliberationEngine:
    """
    Generate pro and contra arguments for a given claim using simple heuristics.
    For Phase 6, we implement a lightweight rule-based engine.
    """

    def __init__(self):
        # Simple negation patterns
        self.negation_patterns = [
            (r"\b(should|must|will|can)\b", r"not \1"),
            (r"\b(is|are|was|were)\b", r"\1 not"),
            (r"\b(support|approve|endorse|favor)\b", r"oppose \1"),
            (r"\b(oppose|reject|disapprove)\b", r"support \1"),
        ]

    def _apply_negation(self, text: str) -> str:
        for pattern, repl in self.negation_patterns:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        # Clean up double spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def deliberate(self, claim: str, evidence: List[Dict[str, Any]] = None) -> DeliberationResult:
        # Generate pro argument (simplified: use claim as is with high confidence)
        pro = Argument(
            text=claim,
            stance="pro",
            confidence=0.8,
            source="claim"
        )
        # Generate contra by applying negation
        contra_text = self._apply_negation(claim)
        contra = Argument(
            text=contra_text,
            stance="contra",
            confidence=0.6,
            source="negation_engine"
        )
        # Simple consensus: if contra confidence low, higher consensus
        consensus = 1.0 - (contra.confidence * 0.5)  # heuristic
        consensus = max(0.0, min(1.0, consensus))

        return DeliberationResult(
            claim=claim,
            arguments=[pro, contra],
            consensus_score=consensus,
            method="rule_based_negation"
        )