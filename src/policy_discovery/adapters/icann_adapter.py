#!/usr/bin/env python3
"""
ICANN source adapter for policy discovery.

Discovers URLs from ICANN's public policy pages, meeting transcripts, and public comment forums.
"""

from typing import List
from datetime import datetime, timezone
from .base_adapter import BaseAdapter, DiscoveredURL


class ICANNAdapter(BaseAdapter):
    """
    Adapter for ICANN policy sources.

    Known seed URLs (configurable via adapter config):
    - Sitemap: https://www.icann.org/sitemap.xml
    - Policy development pages: https://www.icann.org/resources/pages/policy-2012-02-25-en
    - Public comment forums: https://www.icann.org/public-comments
    """

    DEFAULT_SEEDS = [
        "https://www.icann.org/sitemap.xml",
        "https://www.icann.org/resources/pages/policy-2012-02-25-en",
        "https://www.icann.org/public-comments",
        "https://www.icann.org/resources/pages/governance-2012-02-25-en",
    ]

    def __init__(self, source_id: str = "icann", config: dict = None):
        super().__init__(source_id, config)
        self.seeds = self.config.get("seeds", self.DEFAULT_SEEDS)
        self.max_pages = self.config.get("max_pages", 50)

    def discover_urls(self) -> List[DiscoveredURL]:
        """
        Discover ICANN policy URLs by crawling seed sitemaps and known policy pages.
        Uses generic web discovery helpers.
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
                        url=u.normalized_url,
                        source_id=self.source_id,
                        discovered_at=datetime.now(timezone.utc).isoformat(),
                        normalized_url=u.normalized_url,
                    ))
            except Exception as e:
                # log and continue
                print(f"ICANN adapter error for {seed}: {e}")

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
        Capture full content from an ICANN URL.
        Returns a list of Segment objects (to be implemented using extraction pipeline).
        """
        # This would call the existing extraction pipeline.
        # For now, we raise NotImplementedError to indicate integration point.
        raise NotImplementedError("ICANN capture requires integration with ETL pipeline")

    def provenance(self) -> dict:
        return {
            "source": "ICANN",
            "type": "policy_organization",
            "adapter": "ICANNAdapter",
            "seeds": self.seeds,
        }