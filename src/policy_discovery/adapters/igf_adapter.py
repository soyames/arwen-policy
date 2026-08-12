from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from .base_adapter import BaseAdapter, DiscoveredURL
import httpx
from ..utils import parse_html_content  # Assuming this util exists for content parsing

class IGFAdapter(BaseAdapter):
    def __init__(self, source_id: str = "igf", config: dict = None):
        super().__init__(source_id, config)
        self.seeds = self.config.get("seeds", self.DEFAULT_SEEDS)
        self.max_pages = self.config.get("max_pages", 60)

    def discover_urls(self) -> List[DiscoveredURL]:
        # Existing discovery logic remains unchanged
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
                print(f"IGF adapter error for {seed}: {e}")
        # Deduplicate and return
        seen = set()
        unique = [d for d in discovered if d.normalized_url not in seen if not seen.add(d.normalized_url)]
        return unique

    def capture(self, url: str) -> List:
        """Capture content from an IGF URL and return ETL-compatible segments."""
        try:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
            # Parse HTML content into policy segments
            segments = parse_html_content(resp.text)
            # Format segments for ETL
            etl_segments = [
                {
                    "text": seg.text,
                    "extraction_method": "igf-html-parsing",
                    "source_id": self.source_id,
                    "provenance": self.provenance(url)
                } for seg in segments
            ]
            return etl_segments
        except Exception as e:
            print(f"IGF capture failed for {url}: {e}")
            return []

    def provenance(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Return source provenance metadata."""
        base_provenance = {
            "source": "IGF (Internet Governance Forum)",
            "type": "intergovernmental_forum",
            "adapter": "IGFAdapter"
        }
        if url:
            base_provenance["url"] = url
        return base_provenance