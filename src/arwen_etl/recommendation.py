from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Recommendation:
    policy_area: str
    recommendation_text: str
    rationale: str
    confidence: float
    supporting_arguments: List[str]
    dissenting_arguments: List[str]
    metadata: Dict[str, Any]

class RecommendationGenerator:
    """
    Generate policy recommendations based on deliberation results.
    Simple rule-based: if consensus > threshold, recommend action; else recommend further study.
    """

    def __init__(self, consensus_threshold: float = 0.6):
        self.consensus_threshold = consensus_threshold

    def generate(self, deliberation: 'DeliberationResult', evidence: List[Dict[str, Any]] = None) -> Recommendation:
        pro_args = [arg.text for arg in deliberation.arguments if arg.stance == "pro"]
        contra_args = [arg.text for arg in deliberation.arguments if arg.stance == "contra"]

        if deliberation.consensus_score >= self.consensus_threshold:
            # High consensus -> recommend
            rec_text = f"Implement policy aligned with: {pro_args[0] if pro_args else 'the claim'}"
            rationale = f"Consensus score {deliberation.consensus_score:.2f} exceeds threshold {self.consensus_threshold}. Pro arguments: {', '.join(pro_args)}. Contra arguments: {', '.join(contra_args)}."
            confidence = deliberation.consensus_score
        else:
            # Low consensus -> further study
            rec_text = "Commission further analysis and stakeholder consultation."
            rationale = f"Consensus score {deliberation.consensus_score:.2f} below threshold {self.consensus_threshold}. Divergent views: pro={len(pro_args)}, contra={len(contra_args)}."
            confidence = 1.0 - deliberation.consensus_score

        return Recommendation(
            policy_area="digital_policy",
            recommendation_text=rec_text,
            rationale=rationale,
            confidence=confidence,
            supporting_arguments=pro_args,
            dissenting_arguments=contra_args,
            metadata={
                "consensus_score": deliberation.consensus_score,
                "method": "threshold_based"
            }
        )