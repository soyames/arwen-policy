from __future__ import annotations

import re
from uuid import uuid4

from .models import Candidate, Segment

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

ARGUMENT_PATTERNS = (
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bhowever\b",
    r"\bthe rationale\b",
    r"\bevidence\b",
    r"\bconcern(?:s)?\b",
)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def extract_candidates(segments: list[Segment]) -> list[Candidate]:
    candidates: list[Candidate] = []

    for segment in segments:
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

        if _matches(segment.text, ARGUMENT_PATTERNS):
            candidates.append(
                Candidate(
                    candidate_id=str(uuid4()),
                    candidate_type="argument",
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
