from __future__ import annotations

import json

import pytest

from arwen_etl import capture


def test_capture_rejected_by_robots(
    tmp_path,
    monkeypatch,
) -> None:
    class FakePolicy:
        def check(self, url):
            from arwen_etl.policy import PolicyDecision

            return PolicyDecision(
                allowed=False,
                reason="ROBOTS_DISALLOWED",
                policy_source="https://example.org/robots.txt",
            )

    monkeypatch.setattr(
        capture,
        "SourcePolicy",
        lambda timeout: FakePolicy(),
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        PermissionError,
        match="ROBOTS_DISALLOWED",
    ):
        capture.capture_url(
            "https://example.org/private"
        )

    rejection_files = list(
        (tmp_path / "data/rejections").glob("*.json")
    )

    assert len(rejection_files) == 1

    event = json.loads(
        rejection_files[0].read_text(
            encoding="utf-8"
        )
    )

    assert event["event_type"] == "CAPTURE_REJECTED"
    assert event["attributes"]["reason"] == "ROBOTS_DISALLOWED"
    assert event["attributes"]["action"] == "NO_CAPTURE"