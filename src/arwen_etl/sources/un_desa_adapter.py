from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from ..sources.adapter import GenericSourceAdapter, _root_url
from ..sources.generic import DiscoveredURL, discover_from_sitemap_xml, discover_from_rss_xml, discover_links_from_html_content
from core.config import load_pipeline_config


class UNDESAAdapter(GenericSourceAdapter):
    """UN DESA (Department of Economic and Social Affairs) specific adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source)
        self.specific_domains = ["un.org", "desa.un.org", "publications.un.org"]
        self.domain_patterns = ["sdg", "development", "economic", "social", "statistics"]

    def _discover_docs(self, domain: str) -> List[str]:
        """Discover documents specific to UN DESA with pattern matching"""
        root = _root_url(self.source)
        candidates = [root]

        # UN DESA specific discovery paths
        domain_specific_paths = [
            f"{root}publications",
            f"{root}sdgs",
            f"{root}sustainable-development",
            f"{root}development-data",
            f"{root}statistics",
            f"{root}policy-briefs",
            f"{root}working-papers"
        ]

        # Add SDG and development pathways
        candidates.extend([
            f"{root}sdg-report",
            f"{root}world-population",
            f"{root}financing-development",
            f"{root}urban",
            f"{root}energy"
        ])

        current_year = datetime.now().year
        candidates.extend([
            f"{root}{current_year}/sdg-progress",
            f"{root}{current_year}/development-report",
            f"{root}{current_year}/population-report"
        ])

        return [url for url in candidates if any(d in url for d in ["desa", "publications", "un.org"])]

    def discover(self) -> "SourceDiscoveryBundle":
        """Enhanced discovery for UN DESA sources"""
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

        # Domain-specific discovery paths for UN DESA
        domain_specific_paths = self._discover_docs(self.source.domain)

        # Add thematic and data-focused paths
        candidates = list(set(
            self._candidate_urls(self.source) +
            domain_specific_paths +
            [
                f"{root}data",
                f"{root}indicators",
                f"{root}capacity-building",
                f"{root}intergovernmental-processes"
            ]
        ))

        candidates = candidates[:75]

        for candidate_url in candidates:
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
    return UNDESAAdapter(source)