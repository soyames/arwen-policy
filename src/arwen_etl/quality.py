from __future__ import annotations


def extraction_quality(text: str, *, minimum_chars: int = 20) -> dict[str, float | str]:
    if not text.strip():
        return {"status": "failed", "score": 0.0}

    length_score = min(len(text) / max(minimum_chars * 10, 1), 1.0)
    replacement_penalty = min(text.count("\ufffd") / max(len(text), 1) * 100, 0.5)
    score = max(0.0, length_score - replacement_penalty)

    return {
        "status": "passed" if len(text.strip()) >= minimum_chars else "failed",
        "score": round(score, 4),
    }


def provenance_quality(*, source_url: bool, source_hash: bool, retrieval_time: bool) -> float:
    return round(sum([source_url, source_hash, retrieval_time]) / 3, 4)
