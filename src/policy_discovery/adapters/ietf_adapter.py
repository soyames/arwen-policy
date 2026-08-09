#!/usr/bin/env python3
"""
IETF (Internet Engineering Task Force) source adapter for policy discovery.

Discovers URLs from RFCs, Internet-Drafts, meeting minutes, and working group outputs.
"""

from typing import List
from datetime import datetime, timezone
from .base_adapter import BaseAdapter, DiscoveredURL


class IETFAdapter(BaseAdapter):
    """
    Adapter for IETF policy and technical standards sources.

    Known seed URLs (configurable via adapter config):
    - RFC repository: https://www.rfc-editor.org/
    - Internet-Drafts: https://datatracker.ietf.org/doc/
    - Meeting materials: https://www.ietf.org/how/meetings/
    - Working group charters: https://datatracker.ietf.org/wg/
    - IETF Statements: https://www.ietf.org/about/statements/
    """

    DEFAULT_SEEDS = [
        "https://www.rfc-editor.org/rfc-index.xml",
        "https://datatracker.ietf.org/doc/",
        "https://www.ietf.org/how/meetings/",
        "https://datatracker.ietf.org/wg/",
        "https://www.ietf.org/about/statements/",
    ]

    def __init__(self, source_id: str = "ietf", config: dict = None):
        super().__init__(source_id, config)
        self.seeds = self.config.get("seeds", self.DEFAULT_SEEDS)
        self.max_pages = self.config.get("max_pages", 80)

    def discover_urls(self) -> List[DiscoveredURL]:
        """
        Discover IETF policy and technical standards URLs.
        Focuses on RFCs, Internet-Drafts, and policy-relevant documents.
        """
        from ...sources.generic import discover_from_sitemap, discover_links_from_html
        import httpx

        discovered = []
        for seed in self.seeds[:self.max_pages]:
            try:
                if seed.endswith(".xml"):
                    urls = discover_from_sitemap(seed)
                else:
                    resp = httpx.get(seed, follow_redirects=True, timeout=30.0)
                    resp.raise_for_status()
                    urls = discover_links_from_html(seed, resp.text)

                for u in urls:
                    discovered.append(DiscoveredURL(
                        url=u.url,
                        source_id=self.source_id,
                        discovered_at=datetime.now(timezone.utc).isoformat(),
                        normalized_url=u.normalized_url,
                    ))
            except Exception as e:
                print(f"IETF adapter error for {seed}: {e}")

        # Deduplicate by normalized URL
        seen = set()
        unique = []
        for d in discovered:
            if d.normalized_url not in seen:
                seen.add(d.normalized_url)
                unique.append(d)

        return unique

    def capture(self, url: str) -> List:
        """
        Capture full content from an IETF URL.
        Returns a list of Segment objects (integration point for ETL pipeline).
        """
        raise NotImplementedError("IETF capture requires integration with ETL pipeline")

    def provenance(self) -> dict:
        return {
            "source": "IETF (Internet Engineering Task Force)",
            "type": "standards_organization",
            "adapter": "IETFAdapter",
            "seeds": self.seeds,
        }