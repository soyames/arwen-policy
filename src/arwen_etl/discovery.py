from __future__ import annotations

"""Public discovery helpers.

Phase 2 moved content-level discovery into `sources.generic` so adapters can
reuse the same parsers under capture policy. This module re-exports the public
API so existing imports keep working.
"""

from .sources.generic import (
    DiscoveredURL,
    discover_from_rss,
    discover_from_rss_xml,
    discover_from_sitemap,
    discover_from_sitemap_xml,
    discover_from_urls,
    discover_links_from_html,
    discover_links_from_html_content,
    source_id_from_url,
    validate_public_url,
)

__all__ = [
    "DiscoveredURL",
    "discover_from_rss",
    "discover_from_rss_xml",
    "discover_from_sitemap",
    "discover_from_sitemap_xml",
    "discover_from_urls",
    "discover_links_from_html",
    "discover_links_from_html_content",
    "source_id_from_url",
    "validate_public_url",
]
