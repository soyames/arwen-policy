from __future__ import annotations

import re
from uuid import uuid4

from .models import Candidate, Segment

# Counterargument indicators
COUNTERARGUMENT_PATTERNS = (
    r"\bbut\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\bthough\b",
    r"\bdespite\b",
    r"\byet\b",
    r"\bunfortunately\b",
    r"\bregardless\b",
    r"\bnot\b",
    r"\bno\b",
    r"\bopposed\b",
    r"\bagainst\b",
    r"\bcontary\b"  # common misspelling of 'contrary'
)

# Enhanced argument patterns with confidence weights
ARGUMENT_PATTERNS = {
    # Strong argument indicators
    r"\bbecause\b": 0.9,
    r"\btherefore\b": 0.85,
    r"\bconsequently\b": 0.85,
    r"\bas a result\b": 0.8,
    r"\bthe rationale\b": 0.9,
    r"\bthe justification\b": 0.9,
    # Contrast/qualification indicators
    r"\bhowever\b": 0.7,
    r"\bnevertheless\b": 0.75,
    r"\bon the other hand\b": 0.75,
    # Evidence/support indicators
    r"\bevidence\b": 0.8,
    r"\bstudies show\b": 0.85,
    r"\bresearch indicates\b": 0.85,
    r"\bdata suggests\b": 0.8,
    # Concern/objection indicators
    r"\bconcern(?:s)?\b": 0.75,
    r"\bobject(?:s|ion)?\b": 0.8,
    r"\bdisagree(?:s)?\b": 0.8,
    # Argument structural indicators
    r"\bargument\b": 0.7,
    r"\bclaim\b": 0.7,
    r"\bassertion\b": 0.7,
    r"\bpremise\b": 0.75,
    r"\bconclusion\b": 0.8,
}

# Position indicators
POSITION_PATTERNS = (
    r"\bwe support\b",
    r"\bwe oppose\b",
    r"\bwe recommend\b",
    r"\bwe propose\b",
    r"\bwe call for\b",
    r"\bwe believe\b",
    r"\bsupports?\b",
    r"\bopposes?\b",
    r"\brecommends?\b",
    r"\bshould\b",
)

# Evidence patterns with confidence weights
EVIDENCE_PATTERNS = {
    # Core evidence indicators
    r"\bbecause\b": 0.9,
    r"\btherefore\b": 0.85,
    r"\bconsequently\b": 0.85,
    r"\bas a result\b": 0.8,
    r"\bthe rationale\b": 0.9,
    r"\bthe justification\b": 0.9,
    r"\bdata suggests\b": 0.85,
    r"\bevidence\b": 0.9,
    r"\bstudies show\b": 0.9,
    r"\bresearch indicates\b": 0.85,
    r"\bevidence suggests\b": 0.8,
    r"\baccording to\b": 0.88,
    r"\bstudy shows\b": 0.85,
    r"\bdata indicates\b": 0.82,
    r"\bresults indicate\b": 0.8,
    r"\bfindings suggest\b": 0.85,
    r"\bor observations show\b": 0.8,
    # Citation patterns
    r"\bstudy \(.*\)\b": 0.9,
    r"\bwork by \w+\b": 0.88,
    r"\breference\s+\d+\)\b": 0.9,
    r"\bcitation\s+\d+\)\b": 0.85,
    r"\bChen et al\.\b": 0.9,
    r"\bSmith, et al\.\b": 0.9,
    # Quantitative evidence markers
    r"\b%\s+": 0.85,
    r"\b\d+\s+figures\b": 0.88,
    r"\b\d+\(\d+\)\b": 0.9,
    r"\bstatistical analysis\b": 0.82,
    r"\bcorrelation analysis\b": 0.85,
    r"\bregression model\b": 0.88,
    r"\bmeta-analysis\b": 0.9,
    r"\bcomparative study\b": 0.8,
    # Document-specific strength indicators
    r"\bstrong evidence\b": 0.8,
    r"\bcompelling data\b": 0.83,
    r"\bboxers ifields results\b": 0.88,
    r"\bconvincing finding\b": 0.9,
    r"\bnotable discovery\b": 0.95,
    r"\bempirical support\b": 0.9,
    r"\bor surprising outcome\b": 0.82,
    r"\bfinal note\b": 0.9
}


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _extract_arguments_with_confidence(text: str) -> list[dict]:
    """Extract argument phrases with confidence scores based on pattern matching."""
    arguments = []
    lowered = text.lower()

    for pattern, confidence in ARGUMENT_PATTERNS.items():
        matches = list(re.finditer(pattern, lowered))
        for match in matches:
            # Extract a window around the matched argument keyword
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            argument_text = text[start:end].strip()

            arguments.append({
                "text": argument_text,
                "confidence": confidence,
                "keyword": match.group(0),
                "position": match.start(),
                "full_text": text
            })

    return arguments


def _extract_evidence_with_confidence(text: str) -> list[dict]:
    """Extract evidence phrases with confidence scores based on EVIDENCE_PATTERNS."""
    evidence_matches = []
    for pattern, weight in EVIDENCE_PATTERNS.items():
        for match in re.finditer(pattern, text.lower()):
            evidence_matches.append({
                "segment": match.group(0),
                "pattern": pattern,
                "confidence": weight,
                "position": match.start()
            })
    return evidence_matches


def extract_candidates(segments: list[Segment]) -> list[Candidate]:
    candidates: list[Candidate] = []

    for segment in segments:
        # Position detection
        if _matches(segment.text, POSITION_PATTERNS):
            candidates.append(
                Candidate(
                    candidate_id=str(uuid4()),
                    candidate_type="position",
                    segment_id=segment.segment_id,
                    text=segment.text,
                    extraction_method="rule_based_candidate_v1",
                )
            )

        # Argument detection
        if _matches(segment.text, tuple(ARGUMENT_PATTERNS.keys())):
            candidates.append(
                Candidate(
                    candidate_id=str(uuid4()),
                    candidate_type="argument",
                    segment_id=segment.segment_id,
                    text=segment.text,
                    extraction_method="rule_based_candidate_v1",
                )
            )

        # Counterargument detection
        if _matches(segment.text, COUNTERARGUMENT_PATTERNS):
            candidates.append(
                Candidate(
                    candidate_id=str(uuid4()),
                    candidate_type="counterargument",
                    segment_id=segment.segment_id,
                    text=segment.text,
                    extraction_method="rule_based_candidate_v1",
                )
            )

        # Evidence detection
        if _matches(segment.text, tuple(EVIDENCE_PATTERNS.keys())):
            candidates.append(
                Candidate(
                    candidate_id=str(uuid4()),
                    candidate_type="evidence",
                    segment_id=segment.segment_id,
                    text=segment.text,
                    extraction_method="rule_based_candidate_v1",
                )
            )

    return candidates


def extract_stakeholders(segments: list[Segment]) -> list[Candidate]:
    candidates: list[Candidate] = []
    ORG_PATTERN = (
        r"\b([A-Z][A-Za-z0-9&,. ]{2,50}(?:Society|Institute|Association|Council|"
        r"Company|Inc|LLC|Corporation|University))\b"
    )

    for segment in segments:
        m = re.search(ORG_PATTERN, segment.text)
        if m:
            candidates.append(
                Candidate(
                    candidate_id=str(uuid4()),
                    candidate_type="stakeholder",
                    segment_id=segment.segment_id,
                    text=m.group(1),
                    extraction_method="org_regex_v1",
                )
            )

    return candidates


def extract_candidates_with_model(
    segments: list[Segment], model_callable=None
) -> list[Candidate]:
    """Use an external model callable to extract candidates.

    The `model_callable` should accept a list of segment texts and return a list
    of dicts with keys: `candidate_type`, `segment_index`, and `text`.

    If `model_callable` is None, fall back to rule-based extraction.
    """
    if model_callable is None:
        # fallback to existing heuristics
        out = []
        out.extend(extract_candidates(segments))
        out.extend(extract_stakeholders(segments))
        return out

    texts = [s.text for s in segments]
    suggestions = model_callable(texts)
    results: list[Candidate] = []
    for s in suggestions:
        seg_idx = s.get("segment_index")
        if seg_idx is not None and 0 <= seg_idx < len(segments):
            segment_id = segments[seg_idx].segment_id
        else:
            segment_id = ""
        results.append(
            Candidate(
                candidate_id=str(uuid4()),
                candidate_type=s.get("candidate_type", "unknown"),
                segment_id=segment_id,
                text=s.get("text", ""),
                extraction_method="ml_model",
            )
        )

    return results