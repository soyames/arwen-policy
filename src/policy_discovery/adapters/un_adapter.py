#!/usr/bin/env python3
"""
UN Agencies source adapter for policy discovery.

Discovers URLs from UN DESA, UNESCO, UNDP, and other UN agency publications.
"""

from typing import List
from datetime import datetime, timezone
from .base_adapter import BaseAdapter, DiscoveredURL


class UNAgenciesAdapter(BaseAdapter):
    """
    Adapter for UN agencies policy sources.

    Known seed URLs (configurable via adapter config):
    - UN DESA publications: https://www.un.org/development/desa/publications/
    - UNESCO publications: https://unesdoc.unesco.org/
    - UNDP publications: https://www.undp.org/publications
    - UN Digital Library: https://digitallibrary.un.org/
    - ITU (already separate) but include for completeness
    """

    DEFAULT_SEEDS = [
        "https://www.un.org/sitemap.xml",
        "https://www.un.org/development/desa/publications/",
        "https://unesdoc.unesco.org/ark:/48223/pf0000123456",  # placeholder
        "https://www.undp.org/publications",
        "https://digitallibrary.un.org/record/",
    ]

    def __init__(self, source_id: str = "un", config: dict = None):
        super().__init__(source_id, config)
        self.seeds = self.config.get("seeds", self.DEFAULT_SEEDS)
        self.max_pages = self.config.get("max_pages", 50)

    def discover_urls(self) -> List[DiscoveredURL]:
        """
        Discover UN agencies policy URLs by crawling seed sitemaps and publication pages.
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
                print(f"UN adapter error for {seed}: {e}")

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
        Capture full content from a UN agency URL.
        Returns a list of Segment objects (integration point for ETL pipeline).
        """
        raise NotImplementedError("UN capture requires integration with ETL pipeline")

    def provenance(self) -> dict:
        return {
            "source": "UN Agencies (UN DESA, UNESCO, UNDP, etc.)",
            "type": "intergovernmental_organization",
            "adapter": "UNAgenciesAdapter",
            "seeds": self.seeds,
        }