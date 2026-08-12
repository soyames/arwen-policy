from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

import httpx

from .base_adapter import BaseAdapter, DiscoveredURL

class ITUTAdapter(BaseAdapter):
    DEFAULT_SEEDS = [
        "https://www.itu.int/en",
        "https://www.itu.int/en/ITU-D/main",
        "https://www.itu.int/en/ITU-T/studygroups",
        "https://www.itu.int/en/ITU-T/expert-groups"
    ]

    def __init__(self, source_id: str = "itut", config: dict = None):
        super().__init__(source_id, config)
        self.seeds = self.config.get("seeds", self.DEFAULT_SEEDS)
        self.max_pages = self.config.get("max_pages", 50)

    def discover_urls(self) -> List[DiscoveredURL]:
        """
        Discover policy-related URLs from ITU-T and ITU-D domains.
        """
        from ...sources.generic import discover_from_sitemap, discover_links_from_html
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
                print(f"ITUT adapter error for {seed}: {e}")
        # Deduplication
        seen = set()
        unique = [d for d in discovered if d.normalized_url not in seen if not seen.add(d.normalized_url)]
        return unique

    def capture(self, url: str) -> List:
        """
        Capture ITU policy content and format for ETL.
        """
        try:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
            segments = [
                {
                    "text": "Policy content from ITU: {url}",
                    "extraction_method": "itu-html-parsing",
                    "source_id": self.source_id,
                    "provenance": self.provenance(url)
                }
            ]
            return segments
        except Exception as e:
            print(f"ITUT capture failed for {url}: {e}")
            return []

    def provenance(self, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Return ITU-specific provenance metadata.
        """
        base = {
            "source": "ITU-T/ITU-D",
            "type": "telecom_standards",
            "adapter": "ITUTAdapter"
        }
        if url:
            base["url"] = url
        return base