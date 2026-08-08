from arwen_etl.candidates import extract_candidates
from arwen_etl.normalization import normalize_text
from arwen_etl.segmentation import segment_text


def test_segment_and_candidates():
    text = normalize_text("We support this policy because it improves interoperability.")
    segments = segment_text(text, "doc-1", target_chars=100)
    candidates = extract_candidates(segments)
    assert segments
    assert any(candidate.candidate_type == "position" for candidate in candidates)
    assert any(candidate.candidate_type == "argument" for candidate in candidates)
