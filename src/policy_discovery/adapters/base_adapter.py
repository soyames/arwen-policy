#!/usr/bin/env python3
"""
Base adapter interface for policy source discovery.

All specific adapters must implement:
- discover_urls(): Returns a list of DiscoveredURL objects
- capture() -> List[Segment] (optional, for direct capture)
- provenance() -> dict
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from dataclasses import dataclass

from ...models import Segment as SegmentModel


@dataclass
class DiscoveredURL:
    """Immutable record of a discovered URL."""
    url: str
    source_id: str
    discovered_at: str
    normalized_url: str


class BaseAdapter(ABC):
    """
    Abstract base class for all source adapters.
    Provides shared utilities like provenance tracking and pattern matching.
    """

    def __init__(self, source_id: str, config: dict):
        self.source_id = source_id
        self.config = config or {}
        self._provenance_cache: Dict[str, str] = {}

    @abstractmethod
    def discover_urls(self) -> List[DiscoveredURL]:
        """Fetch and return discovered URLs for this source."""
        raise NotImplementedError

    def capture(self, url: str) -> List[SegmentModel]:
        """Capture and return extracted segments from a URL."""
        raise NotImplementedError

    def provenance(self) -> Dict[str, str]:
        """Return structured provenance data for this adapter."""
        return {}

    def _normalize_url(self, url: str) -> str:
        """Canonical URL representation."""
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            '', parsed.query, parsed.fragment
        ))