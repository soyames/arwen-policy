from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    policy_source: str | None = None
    details: dict[str, str] | None = None


class SourcePolicy:
    """
    Acquisition policy gate.

    This gate executes BEFORE downloading the source artifact.

    It does not attempt to bypass:
    - robots.txt restrictions
    - authentication
    - paywalls
    - access controls

    A rejection is a valid pipeline outcome and must be recorded
    as provenance.
    """

    USER_AGENT = "ArwenPolicyETL/0.1 (+https://github.com/soyames/arwen-policy-etl)"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.timeout = timeout
        self._client = client

    def _robots_url(self, url: str) -> str:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def check_robots(self, url: str) -> PolicyDecision:
        robots_url = self._robots_url(url)

        parser = RobotFileParser()
        parser.set_url(robots_url)

        try:
            client = self._client or httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.USER_AGENT},
            )

            try:
                response = client.get(robots_url)
            finally:
                if self._client is None:
                    client.close()

        except httpx.HTTPError as exc:
            return PolicyDecision(
                allowed=False,
                reason="ROBOTS_UNAVAILABLE",
                policy_source=robots_url,
                details={"error": str(exc)},
            )

        if response.status_code == 404:
            # No robots.txt means there is no robots exclusion
            # file to apply.
            return PolicyDecision(
                allowed=True,
                reason="ROBOTS_NOT_FOUND",
                policy_source=robots_url,
            )

        if response.status_code >= 400:
            return PolicyDecision(
                allowed=False,
                reason="ROBOTS_UNAVAILABLE",
                policy_source=robots_url,
                details={"status_code": str(response.status_code)},
            )

        parser.parse(response.text.splitlines())

        allowed = parser.can_fetch(self.USER_AGENT, url)

        return PolicyDecision(
            allowed=allowed,
            reason="ROBOTS_ALLOWED" if allowed else "ROBOTS_DISALLOWED",
            policy_source=robots_url,
        )

    def check(self, url: str) -> PolicyDecision:
        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return PolicyDecision(
                allowed=False,
                reason="UNSUPPORTED_SCHEME",
                details={"scheme": parsed.scheme},
            )

        if not parsed.netloc:
            return PolicyDecision(
                allowed=False,
                reason="INVALID_URL",
            )

        return self.check_robots(url)