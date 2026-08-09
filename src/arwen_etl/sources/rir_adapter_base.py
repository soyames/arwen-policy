from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime

from ..sources.adapter import GenericSourceAdapter, _root_url
from ..sources.generic import DiscoveredURL, discover_from_sitemap_xml, discover_from_rss_xml, discover_links_from_html_content
from core.config import load_pipeline_config


class BaseRIRAdapter(GenericSourceAdapter):
    """Base adapter for Regional Internet Registries (RIRs)"""

    def __init__(self, source: SourceDefinition, rir_name: str, regions: List[str]):
        super().__init__(source)
        self.rir_name = rir_name
        self.regions = regions
        self.specific_domains = self._get_rir_domains()
        self.domain_patterns = [
            "statistics", "delegated", "allocations", "assignments",
            "policy", "ripe", "arin", "apnic", "lacnic", "afrinic",
            "whois", "database", "ftp", "reports"
        ]

    def _get_rir_domains(self) -> List[str]:
        """Get domain patterns based on RIR name"""
        rir_domains = {
            "ARIN": ["arin.net", "www.arin.net"],
            "RIPE NCC": ["ripe.net", "www.ripe.net", "db.ripe.net"],
            "APNIC": ["apnic.net", "www.apnic.net"],
            "LACNIC": ["lacnic.net", "www.lacnic.net"],
            "AFRINIC": ["afrinic.net", "www.afrinic.net"]
        }
        return rir_domains.get(self.rir_name, ["example.net"])

    def _discover_docs(self, domain: str) -> List[str]:
        """Discover documents specific to RIRs with pattern matching"""
        root = _root_url(self.source)
        candidates = [root]

        # RIR specific discovery paths
        domain_specific_paths = [
            f"{root}stats/",
            f"{root}ftp/",
            f"{root}db/",
            f"{root}whois/",
            f"{root}publications/",
            f"{root}reports/",
            f"{root}policy/",
            f"{root}documents/"
        ]

        # Add specific data endpoints
        candidates.extend([
            f"{root}ftp/stats/",
            f"{root}db/{self.rir_name.lower()}-db/",
            f"{root}whois/{self.rir_name.lower()}/",
            f"{root}statistics/",
            f"{root}delegated-{self.rir_name.lower()}-extended-latest",
            f"{root}delegated-{self.rir_name.lower()}-latest",
            f"{root}ftp/publications/",
            f"{root}ftp/reports/"
        ])

        # Add yearly archives
        current_year = datetime.now().year
        for year in range(current_year - 5, current_year + 1):
            candidates.extend([
                f"{root}stats/{year}/",
                f"{root}reports/{year}/",
                f"{root}ftp/stats/{year}/"
            ])

        candidates.extend(domain_specific_paths)

        return [url for url in candidates if any(domain in url for domain in self.specific_domains) or any(r in url.lower() for r in self.regions)]

    def discover(self) -> "SourceDiscoveryBundle":
        """Enhanced discovery for RIR sources"""
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

        # Domain-specific discovery paths for RIR
        domain_specific_paths = self._discover_docs(self.source.domain)

        # Add standard RIR data endpoints
        candidates = list(set(
            self._candidate_urls(self.source) +
            domain_specific_paths +
            [
                f"{root}statistics/",
                f"{root}reports/",
                f"{root}ftp/publications/",
                f"{root}whois/",
                f"{root}db/",
                f"{root}ftp/stats/"
            ]
        ))

        candidates = candidates[:100]  # RIRs can have many endpoints

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
    """Factory method - to be overridden by specific RIR adapters"""
    # This should not be called directly
    return BaseRIRAdapter(source, "Unknown", [])