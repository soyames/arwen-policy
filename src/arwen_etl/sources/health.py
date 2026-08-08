from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..storage import write_json
from .adapter import SourceDiscoveryBundle


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_source_health_index(bundles: Iterable[SourceDiscoveryBundle]) -> dict[str, Any]:
    items = list(bundles)
    summary = Counter()
    family_counts = Counter()
    adapter_counts = Counter()
    discovered_url_count = 0

    sources: list[dict[str, Any]] = []
    for bundle in items:
        health = bundle.health.model_dump(mode="json")
        health["discovered_count"] = len(bundle.discovered_urls)
        sources.append(health)

        summary[bundle.health.source_status] += 1
        family_counts[bundle.health.source_family] += 1
        adapter_counts[bundle.health.adapter] += 1
        discovered_url_count += len(bundle.discovered_urls)

    return {
        "generated_at": _utc_now().isoformat(),
        "source_count": len(items),
        "discovered_url_count": discovered_url_count,
        "status_counts": dict(summary),
        "family_counts": dict(family_counts),
        "adapter_counts": dict(adapter_counts),
        "sources": sources,
    }


def write_source_health_index(
    path: str | Path,
    bundles: Iterable[SourceDiscoveryBundle],
) -> Path:
    return write_json(path, build_source_health_index(bundles))
