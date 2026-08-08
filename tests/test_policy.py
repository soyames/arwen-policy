from __future__ import annotations

import httpx

from arwen_etl.policy import SourcePolicy


def test_robots_allows_when_robots_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        policy = SourcePolicy(client=client)
        decision = policy.check(
            "https://example.org/public-document"
        )

    assert decision.allowed is True
    assert decision.reason == "ROBOTS_NOT_FOUND"


def test_robots_rejects_disallowed_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "User-agent: ArwenPolicyETL\n"
                "Disallow: /private\n"
            ),
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        policy = SourcePolicy(client=client)
        decision = policy.check(
            "https://example.org/private/document"
        )

    assert decision.allowed is False
    assert decision.reason == "ROBOTS_DISALLOWED"


def test_robots_allows_public_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "User-agent: ArwenPolicyETL\n"
                "Disallow: /private\n"
            ),
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        policy = SourcePolicy(client=client)
        decision = policy.check(
            "https://example.org/public/document"
        )

    assert decision.allowed is True
    assert decision.reason == "ROBOTS_ALLOWED"