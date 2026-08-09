from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..sources.adapter import GenericSourceAdapter, _root_url
from ..sources.generic import DiscoveredURL, discover_from_sitemap_xml, discover_from_rss_xml, discover_links_from_html_content
from core.config import load_pipeline_config


class ICANNAdapter(GenericSourceAdapter):
    """ICANN-specific adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source)
        self.specific_domains = ["icann.org", "www.icann.org"]
        self.domain_patterns = ["rdf", "policy", "governance"]

    def _discover_docs(self, domain: str, patterns: Optional[List[str]] = None) -> List[str]:
        """Discover documents specific to ICANN with pattern matching"""
        root = _root_url(self.source)
        candidates = [root]

        if patterns:
            for pattern in patterns:
                candidates.extend([f"{root}{pattern}", f"{root}{pattern}/"])

        candidates.extend([
            f"{root}sitemap.xml",
            f"{root}resources",
            f"{root}policy",
            f"{root}governance",
            f"{root}about",
            f"{root}registry"
        ])

        # Filter by domain patterns
        return [url for url in candidates if url.startswith(f"https://{domain}")]

    def discover(self) -> "SourceDiscoveryBundle":
        """Enhanced discovery for ICANN sources"""
        from ..sources.health import SourceHealthRecord
        from .bundle import SourceDiscoveryBundle

        settings = self._capture_settings()
        discovered: List[DiscoveredURL] = []
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

        # Domain-specific discovery paths
        domain_specific_paths = [
            f"{_root_url(self.source)}resources",
            f"{_root_url(self.source)}policy",
            f"{_root_url(self.source)}governance",
            f"{_root_url(self.source)}about",
            f"{_root_url(self.source)}registry"
        ]

        # Add sitemap and feed discovery
        candidates = list(set(
            self._candidate_urls(self.source) +
            domain_specific_paths +
            ["https://www.icann.org/annual-reports/",
             "https://www.icann.org/financial-reports/"]
        ))

        for candidate_url in candidates[:50]:  # Limit to first 50 for efficiency
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
    return ICANNAdapter(source)