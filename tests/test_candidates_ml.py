from arwen_etl.candidates import extract_candidates_with_model
from arwen_etl.models import Segment


def test_ml_candidate_interface():
    segs = [
        Segment(
            segment_id="s1",
            document_id="d",
            ordinal=0,
            text="We support this.",
            char_start=0,
            char_end=14,
        ),
        Segment(
            segment_id="s2",
            document_id="d",
            ordinal=1,
            text="Because it helps.",
            char_start=15,
            char_end=31,
        ),
    ]

    def mock_model(texts):
        # pretend it finds a position in segment 0
        return [{"candidate_type": "position", "segment_index": 0, "text": "We support this."}]

    candidates = extract_candidates_with_model(segs, model_callable=mock_model)
    assert len(candidates) == 1
    assert candidates[0].candidate_type == "position"
