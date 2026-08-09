#!/usr/bin/env python3
"""
ITU (International Telecommunication Union) source adapter for policy discovery.

Discovers URLs from ITU-T recommendations, ITU-R reports, and policy publications.
"""

from typing import List
from datetime import datetime, timezone
from .base_adapter import BaseAdapter, DiscoveredURL


class ITUAdapter(BaseAdapter):
    """
    Adapter for ITU policy and technical standards sources.

    Known seed URLs (configurable via adapter config):
    - ITU-T recommendations: https://www.itu.int/rec/T-REC/
    - ITU-R recommendations: https://www.itu.int/rec/R-REC/
    - ITU-D publications: https://www.itu.int/en/ITU-D/Pages/default.aspx
    - WSIS Forum outcomes: https://www.itu.int/net4/wsis/
    - ITU Council documents: https://www.itu.int/en/council/Pages/default.aspx
    """

    DEFAULT_SEEDS = [
        "https://www.itu.int/sitemap.xml",
        "https://www.itu.int/rec/T-REC/",
        "https://www.itu.int/rec/R-REC/",
        "https://www.itu.int/en/ITU-D/Pages/default.aspx",
        "https://www.itu.int/net4/wsis/",
        "https://www.itu.int/en/council/Pages/default.aspx",
    ]

    def __init__(self, source_id: str = "itu", config: dict = None):
        super().__init__(source_id, config)
        self.seeds = self.config.get("seeds", self.DEFAULT_SEEDS)
        self.max_pages = self.config.get("max_pages", 50)

    def discover_urls(self) -> List[DiscoveredURL]:
        """
        Discover ITU policy and technical standards URLs.
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
                print(f"ITU adapter error for {seed}: {e}")

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
        Capture full content from an ITU URL.
        Returns a list of Segment objects (integration point for ETL pipeline).
        """
        raise NotImplementedError("ITU capture requires integration with ETL pipeline")

    def provenance(self) -> dict:
        return {
            "source": "ITU (International Telecommunication Union)",
            "type": "intergovernmental_organization",
            "adapter": "ITUAdapter",
            "seeds": self.seeds,
        }