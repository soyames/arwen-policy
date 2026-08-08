from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .hashing import sha256_bytes
from .policy import SourcePolicy
from .provenance import capture_rejection_event

DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_DOWNLOAD_MB = DEFAULT_MAX_BYTES / (1024 * 1024)
DEFAULT_TIMEOUT_SECONDS = 30.0

USER_AGENT = "ArwenPolicyETL/0.1 (+https://github.com/soyames/arwen-policy-etl)"


@dataclass(frozen=True)
class CapturedArtifact:
    source_id: str
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    data: bytes
    sha256: str
    etag: str | None = None
    last_modified: str | None = None


def source_id_for_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


class CaptureError(Exception):
    """Raised when a capture attempt fails and should be recorded.

    Attributes:
        reason: short code describing the failure
        details: optional dict with extra information
    """

    def __init__(self, reason: str, details: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _effective_timeout(
    timeout: float | None,
    timeout_seconds: float | None,
) -> float:
    if timeout is not None and timeout_seconds is not None:
        raise ValueError(
            "Specify either 'timeout' or 'timeout_seconds', not both."
        )

    return (
        timeout_seconds
        if timeout_seconds is not None
        else timeout
        if timeout is not None
        else DEFAULT_TIMEOUT_SECONDS
    )


def _effective_max_bytes(
    max_bytes: int | None,
    max_download_mb: float | None,
) -> int:
    if max_bytes is not None and max_download_mb is not None:
        raise ValueError(
            "Specify either 'max_bytes' or 'max_download_mb', not both."
        )

    if max_download_mb is not None:
        if max_download_mb <= 0:
            raise ValueError("max_download_mb must be greater than zero.")

        return int(max_download_mb * 1024 * 1024)

    if max_bytes is not None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be greater than zero.")

        return max_bytes

    return DEFAULT_MAX_BYTES


def _record_capture_rejection(
    *,
    source_id: str,
    url: str,
    reason: str,
    policy_source: str | None,
    details: dict[str, str] | None,
) -> Path:
    event = capture_rejection_event(
        source_id=source_id,
        url=url,
        reason=reason,
        policy_source=policy_source,
        details=details,
    )

    rejection_dir = Path("data/rejections")
    rejection_dir.mkdir(parents=True, exist_ok=True)

    rejection_path = rejection_dir / f"{source_id}.json"

    rejection_path.write_text(
        json.dumps(
            event,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return rejection_path


def capture_url(
    url: str,
    *,
    output_dir: str | Path = "data/raw",
    max_bytes: int | None = None,
    max_download_mb: float | None = None,
    timeout: float | None = None,
    timeout_seconds: float | None = None,
    max_redirects: int | None = None,
    user_agent: str | None = None,
    respect_robots: bool = True,
) -> CapturedArtifact:
    source_id = source_id_for_url(url)

    effective_timeout = _effective_timeout(
        timeout,
        timeout_seconds,
    )

    effective_max_bytes = _effective_max_bytes(
        max_bytes,
        max_download_mb,
    )

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme}"
        )

    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    if respect_robots:
        policy = SourcePolicy(
            timeout=effective_timeout,
        )

        decision = policy.check(url)

        if not decision.allowed:
            rejection_path = _record_capture_rejection(
                source_id=source_id,
                url=url,
                reason=decision.reason,
                policy_source=decision.policy_source,
                details=decision.details,
            )

            raise PermissionError(
                f"Source capture rejected: {decision.reason}. "
                f"URL={url}. "
                f"Provenance={rejection_path}"
            )

    headers = {
        "User-Agent": user_agent or USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/pdf,text/plain,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
    }
    client_kwargs: dict = {
        "timeout": effective_timeout,
        "follow_redirects": True,
        "headers": headers,
    }

    if max_redirects is not None:
        client_kwargs["max_redirects"] = max_redirects

    try:
        with httpx.Client(**client_kwargs) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()

                content_length = response.headers.get(
                    "content-length"
                )

                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        # Record and surface invalid header as capture rejection
                        raise CaptureError("INVALID_CONTENT_LENGTH", {"error": str(exc)}) from exc

                    if declared_size > effective_max_bytes:
                        _record_capture_rejection(
                            source_id=source_id,
                            url=url,
                            reason="REMOTE_TOO_LARGE",
                            policy_source=None,
                            details={
                                "declared_size": str(declared_size),
                                "max_bytes": str(effective_max_bytes),
                            },
                        )

                        raise CaptureError(
                            "REMOTE_TOO_LARGE",
                            {"declared_size": declared_size, "max_bytes": effective_max_bytes},
                        )

                chunks: list[bytes] = []
                total = 0

                for chunk in response.iter_bytes():
                    total += len(chunk)

                    if total > effective_max_bytes:
                            _record_capture_rejection(
                                source_id=source_id,
                                url=url,
                                reason="TOO_LARGE",
                                policy_source=None,
                                details={
                                    "total": str(total),
                                    "max_bytes": str(effective_max_bytes),
                                },
                            )

                            raise CaptureError(
                                "TOO_LARGE",
                                {"total": total, "max_bytes": effective_max_bytes},
                            )

                    chunks.append(chunk)

                data = b"".join(chunks)

                content_type = response.headers.get(
                    "content-type",
                    "application/octet-stream",
                ).split(";")[0].strip()

                final_url = str(response.url)
                status_code = response.status_code
                etag = response.headers.get("etag")
                last_modified = response.headers.get("last-modified")
    except httpx.ReadTimeout as exc:
        rejection = capture_rejection_event(
            source_id=source_id,
            url=url,
            reason="TIMEOUT",
            policy_source=None,
            details={"message": str(exc)},
        )

        rejection_dir = Path("data/rejections")
        rejection_dir.mkdir(parents=True, exist_ok=True)
        rejection_path = rejection_dir / f"{source_id}.json"
        rejection_path.write_text(
            json.dumps(rejection, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        raise CaptureError("TIMEOUT", {"message": str(exc)}) from exc
    except httpx.HTTPError as exc:
        # Record provenance for HTTP-level failures and raise CaptureError
        resp = getattr(exc, "response", None)
        status_code = getattr(resp, "status_code", "ERROR")
        final_url = str(getattr(resp, "url", url))

        rejection = capture_rejection_event(
            source_id=source_id,
            url=url,
            reason=f"HTTP_{status_code}",
            policy_source=None,
            details={
                "message": str(exc),
                "final_url": final_url,
                "status_code": status_code,
            },
        )

        rejection_dir = Path("data/rejections")
        rejection_dir.mkdir(parents=True, exist_ok=True)
        rejection_path = rejection_dir / f"{source_id}.json"
        rejection_path.write_text(
            json.dumps(rejection, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        raise CaptureError("HTTP_ERROR", {"message": str(exc)}) from exc
    except CaptureError:
        # propagate CaptureError without modification
        raise
    except Exception as exc:
        # Any other unexpected errors (including timeout) should be recorded
        rejection = capture_rejection_event(
            source_id=source_id,
            url=url,
            reason="CAPTURE_EXCEPTION",
            policy_source=None,
            details={"error": str(exc)},
        )

        rejection_dir = Path("data/rejections")
        rejection_dir.mkdir(parents=True, exist_ok=True)
        rejection_path = rejection_dir / f"{source_id}.json"
        rejection_path.write_text(
            json.dumps(rejection, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        raise CaptureError("CAPTURE_EXCEPTION", {"error": str(exc)}) from exc

    digest = sha256_bytes(data)

    artifact = CapturedArtifact(
        source_id=source_id,
        requested_url=url,
        final_url=final_url,
        status_code=status_code,
        content_type=content_type,
        data=data,
        sha256=digest,
        etag=etag,
        last_modified=last_modified,
    )

    write_artifact(artifact, output_dir)

    return artifact


def write_artifact(
    artifact: CapturedArtifact,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Persist a captured artifact using its content hash as the filename."""
    output_path = Path(output_dir) / artifact.sha256
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not output_path.exists():
        output_path.write_bytes(artifact.data)

    return output_path