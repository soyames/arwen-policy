from arwen_etl.provenance import provenance_event


def test_provenance_event():
    event = provenance_event("capture", entity_id="doc-1", agent="test")
    assert event["event_type"] == "capture"
    assert event["entity_id"] == "doc-1"
