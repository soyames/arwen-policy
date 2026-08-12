from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass

from arwen_deliberation.models import DeliberationResult

@dataclass
class DeliberationMetrics:
    consensus_score: float
    argument_balance: float  # ratio of pro to contra (0-1, 0.5 = balanced)
    coverage_score: float    # proportion of claim addressed
    diversity_score: float   # variety of sources
    overall_quality: float   # weighted average

class DeliberationEvaluator:
    """
    Evaluate the quality of a deliberation result.
    """

    def evaluate(self, deliberation: 'DeliberationResult', evidence: List[Dict[str, Any]] = None) -> DeliberationMetrics:
        args = deliberation.arguments
        if not args:
            return DeliberationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        pro = [a for a in args if a.stance == "pro"]
        contra = [a for a in args if a.stance == "contra"]
        pro_conf = sum(a.confidence for a in pro) / len(pro) if pro else 0.0
        contra_conf = sum(a.confidence for a in contra) / len(contra) if contra else 0.0

        # Argument balance: closer to 0.5 is better
        total_conf = pro_conf + contra_conf
        if total_conf > 0:
            balance = min(pro_conf, contra_conf) / max(pro_conf, contra_conf) if max(pro_conf, contra_conf) > 0 else 0.0
        else:
            balance = 0.0

        # Coverage: proportion of claim words appearing in arguments (simple)
        import re
        claim_words = set(re.findall(r'\b\w+\b', deliberation.claim.lower()))
        arg_text = " ".join([a.text for a in args])
        arg_words = set(re.findall(r'\b\w+\b', arg_text.lower()))
        coverage = len(claim_words & arg_words) / len(claim_words) if claim_words else 0.0

        # Diversity: placeholder based on number of distinct sources
        sources = set(a.source for a in args if a.source)
        diversity = min(len(sources) / 3.0, 1.0)  # assume up to 3 sources is diverse

        # Overall quality: weighted average
        overall = (
            0.4 * deliberation.consensus_score +
            0.2 * balance +
            0.2 * coverage +
            0.2 * diversity
        )

        return DeliberationMetrics(
            consensus_score=deliberation.consensus_score,
            argument_balance=balance,
            coverage_score=coverage,
            diversity_score=diversity,
            overall_quality=overall
        )