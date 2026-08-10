from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from ..capture import CaptureError, capture_url
from ..config import load_pipeline_config
from ..extraction import extract, extract_metadata
from ..normalization import normalize_text
from ..registry import SourceDefinition
from .generic import (
    DiscoveredURL,
    discover_from_rss_xml,
    discover_from_sitemap_xml,
    discover_links_from_html_content,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SourceHealthRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_name: str
    source_family: str
    adapter: str
    source_url: str
    canonical_url: str | None = None
    retrieved_at: datetime = Field(default_factory=_utc_now)
    http_status: int | None = None
    content_type: str | None = None
    robots_allowed: bool | None = None
    source_status: str = "unknown"
    content_hash: str | None = None
    extraction_status: str = "not_extracted"
    document_identifier: str | None = None
    title: str | None = None
    publisher: str | None = None
    published_at: datetime | str | None = None
    language: str | None = None
    jurisdiction: str | None = None
    policy_topics: list[str] = Field(default_factory=list)
    license: str | None = None
    access_conditions: str | None = None
    discovery_urls: list[str] = Field(default_factory=list)
    error: str | None = None


class SourceDiscoveryBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_name: str
    source_family: str
    adapter: str
    health: SourceHealthRecord
    discovered_urls: list[DiscoveredURL] = Field(default_factory=list)


def _dedupe_urls(urls: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _dedupe_discovered(items: Iterable[DiscoveredURL]) -> list[DiscoveredURL]:
    seen: set[str] = set()
    out: list[DiscoveredURL] = []
    for item in items:
        key = item.normalized_url or item.url
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _root_url(source: SourceDefinition) -> str:
    return f"https://{source.domains[0].rstrip('/')}/"


def _config_urls(source: SourceDefinition, key: str) -> list[str]:
    discovery = source.discovery or {}
    values = discovery.get(key, [])
    if isinstance(values, str):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


def _candidate_urls(source: SourceDefinition) -> list[str]:
    root = _root_url(source)
    discovery = source.discovery or {}

    candidates: list[str] = []
    candidates.extend(_config_urls(source, "seed_urls"))
    candidates.extend(_config_urls(source, "start_urls"))
    candidates.extend(_config_urls(source, "sitemap_urls"))
    candidates.extend(_config_urls(source, "feed_urls"))
    candidates.extend(_config_urls(source, "urls"))

    if not candidates:
        candidates.extend(
            [
                root,
                urljoin(root, "sitemap.xml"),
                urljoin(root, "feed"),
                urljoin(root, "rss.xml"),
                urljoin(root, "atom.xml"),
            ]
        )

    paths = discovery.get("paths", [])
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        candidates.append(urljoin(root, str(path).lstrip("/")))

    return _dedupe_urls(candidates)


def _capture_settings() -> dict[str, Any]:
    config = load_pipeline_config()
    capture_cfg = config["capture"]
    return {
        "timeout_seconds": capture_cfg["timeout_seconds"],
        "max_download_mb": capture_cfg["max_download_mb"],
        "max_redirects": capture_cfg["max_redirects"],
        "user_agent": capture_cfg["user_agent"],
        "respect_robots": capture_cfg.get("respect_robots", True),
    }


class GenericSourceAdapter:
    adapter_name = "generic-web-adapter"

    def __init__(self, source: SourceDefinition):
        self.source = source

    def discover(self) -> SourceDiscoveryBundle:
        settings = _capture_settings()
        discovered: list[DiscoveredURL] = []
        health = SourceHealthRecord(
            source_id=self.source.id,
            source_name=self.source.name,
            source_family=self.source.family,
            adapter=self.adapter_name,
            source_url=_root_url(self.source),
            publisher=self.source.name,
            language=self.source.default_language,
        )
        saw_success = False
        last_error: str | None = None

        import time as _time
        _last_domain: str | None = None
        for candidate_url in _candidate_urls(self.source):
            # Rate-limit: delay 1s between requests to the same domain.
            from urllib.parse import urlparse as _urlparse
            _domain = (_urlparse(candidate_url).hostname or "")[:30]
            if _domain == _last_domain:
                _time.sleep(1.0)
            _last_domain = _domain
            try:
                artifact = capture_url(candidate_url, **settings)
            except PermissionError as exc:
                last_error = str(exc)
                if not saw_success:
                    health.source_status = "blocked"
                    health.robots_allowed = False
                    health.error = last_error
                continue
            except CaptureError as exc:
                last_error = exc.reason
                if not saw_success:
                    health.source_status = "degraded"
                    health.error = last_error
                continue

            extracted = extract(artifact.data, artifact.content_type)
            metadata = extract_metadata(artifact.data, artifact.content_type, extracted)

            saw_success = True
            health.canonical_url = artifact.final_url
            health.retrieved_at = _utc_now()
            health.http_status = artifact.status_code
            health.content_type = artifact.content_type
            health.robots_allowed = True
            health.content_hash = artifact.sha256
            health.extraction_status = "extracted" if extracted.text.strip() else "failed"
            health.document_identifier = artifact.source_id
            health.title = metadata.get("title") or health.title
            health.language = metadata.get("language") or health.language
            if metadata.get("published_at"):
                health.published_at = metadata["published_at"]
            health.source_status = "reachable"
            health.error = None

            text = normalize_text(extracted.text)
            if not text:
                continue

            content_type = (artifact.content_type or "").lower()
            decoded = artifact.data.decode("utf-8", errors="replace")
            if "html" in content_type:
                discovered.extend(
                    discover_links_from_html_content(artifact.final_url, decoded)
                )
            elif content_type.endswith("xml") or candidate_url.endswith((".xml", ".rss", ".atom")):
                if "sitemap" in candidate_url.lower():
                    discovered.extend(discover_from_sitemap_xml(artifact.final_url, decoded))
                else:
                    discovered.extend(discover_from_rss_xml(artifact.final_url, decoded))

        discovered = _dedupe_discovered(discovered)
        health.discovery_urls = [item.url for item in discovered]
        if health.source_status == "unknown":
            health.source_status = "unreachable"
            health.error = last_error

        return SourceDiscoveryBundle(
            source_id=self.source.id,
            source_name=self.source.name,
            source_family=self.source.family,
            adapter=self.adapter_name,
            health=health,
            discovered_urls=discovered,
        )


def build_source_adapter(source: SourceDefinition) -> GenericSourceAdapter:
    return GenericSourceAdapter(source)
