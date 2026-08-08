from __future__ import annotations

import json
from pathlib import Path

import httpx

from arwen_etl import capture


def test_capture_timeout_records_provenance(tmp_path, monkeypatch):
    # Fake policy that allows capture
    class AllowPolicy:
        def check(self, url):
            from arwen_etl.policy import PolicyDecision

            return PolicyDecision(allowed=True, reason="ROBOTS_ALLOWED")

    monkeypatch.setattr(capture, "SourcePolicy", lambda timeout: AllowPolicy())

    # Fake client whose stream context raises a timeout
    class TimeoutCtx:
        def __enter__(self):
            raise httpx.ReadTimeout("simulated timeout")

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def stream(self, *args, **kwargs):
            return TimeoutCtx()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(capture.httpx, "Client", lambda **kwargs: FakeClient())

    monkeypatch.chdir(tmp_path)

    # Attempt capture; CaptureError should be raised
    try:
        capture.capture_url("https://example.org/")
    except Exception:
        # Ensure provenance file exists for the source_id
        sid = capture.source_id_for_url("https://example.org/")
        rev = Path("data/rejections") / f"{sid}.json"
        assert rev.exists()
        event = json.loads(rev.read_text(encoding="utf-8"))
        assert event["event_type"] == "CAPTURE_REJECTED"
        assert event["attributes"]["action"] == "NO_CAPTURE"
    else:
        raise AssertionError("Expected capture to fail with timeout")
