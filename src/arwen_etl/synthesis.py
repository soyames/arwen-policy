from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class EvidenceSnippet:
    text: str
    source_id: str
    confidence: float
    provenance: Dict[str, Any]

@dataclass
class SynthesisResult:
    claim: str
    synthesized_text: str
    supporting_evidence: List[EvidenceSnippet]
    confidence: float
    method: str

class EvidenceGroundedSynthesizer:
    """
    Generate evidence-grounded policy syntheses from retrieved evidence snippets.
    Simple implementation: extractive summary based on sentence scoring.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def _score_sentence(self, sentence: str, claim: str) -> float:
        # Simple heuristic: overlap of nouns/verbs with claim
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        sent_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        if not claim_words:
            return 0.0
        overlap = len(claim_words & sent_words) / len(claim_words)
        return overlap

    def synthesize(self, claim: str, evidence: List[EvidenceSnippet]) -> SynthesisResult:
        if not evidence:
            return SynthesisResult(
                claim=claim,
                synthesized_text="No evidence provided.",
                supporting_evidence=[],
                confidence=0.0,
                method="fallback"
            )

        # Filter by confidence
        filtered = [e for e in evidence if e.confidence >= self.min_confidence]
        if not filtered:
            filtered = evidence  # fallback

        # Extract sentences
        sentences: List[tuple[str, EvidenceSnippet]] = []
        for ev in filtered:
            # Simple sentence split
            for sent in re.split(r'(?<=[.!?])\s+', ev.text):
                if sent.strip():
                    sentences.append((sent.strip(), ev))

        # Score sentences
        scored = [(self._score_sentence(sent, claim), sent, ev) for sent, ev in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top 3 sentences
        top_n = scored[:3]
        synthesized = " ".join([sent for _, sent, _ in top_n])
        supporting = [ev for _, _, ev in top_n]
        avg_conf = sum(ev.confidence for _, _, ev in top_n) / len(top_n) if top_n else 0.0

        return SynthesisResult(
            claim=claim,
            synthesized_text=synthesized,
            supporting_evidence=supporting,
            confidence=avg_conf,
            method="extractive_top3"
        )