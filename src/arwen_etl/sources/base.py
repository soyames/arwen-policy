from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

from .generic import DiscoveredURL


@dataclass
class ProvenanceRecord:
    """Record of data provenance for a captured source."""
    source_id: str
    url: str
    captured_at: str
    content_sha256: str
    source_type: str
    metadata: dict


class BaseSourceAdapter(ABC):
    """Base class for all source adapters."""

    def __init__(self, source_type: str):
        self.source_type = source_type

    @abstractmethod
    def discover(self, config: dict) -> List[DiscoveredURL]:
        """Discover URLs to capture from the source."""
        pass

    @abstractmethod
    def capture(self, url: str, output_dir: Path) -> ProvenanceRecord:
        """Capture content from a URL and return provenance record."""
        pass

    def _generate_provenance(
        self,
        url: str,
        content_path: Path,
        metadata: Optional[dict] = None
    ) -> ProvenanceRecord:
        """Generate a provenance record for captured content."""
        import hashlib

        # Calculate SHA-256 of content
        sha256_hash = hashlib.sha256()
        with content_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        content_sha256 = sha256_hash.hexdigest()

        return ProvenanceRecord(
            source_id=self._source_id_from_url(url),
            url=url,
            captured_at=datetime.now(UTC).isoformat(),
            content_sha256=content_sha256,
            source_type=self.source_type,
            metadata=metadata or {},
        )

    @staticmethod
    def _source_id_from_url(url: str) -> str:
        """Generate a source ID from a URL (consistent with generic.py)."""
        from hashlib import sha256
        return sha256(url.encode("utf-8")).hexdigest()[:24]