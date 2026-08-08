from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


def validate_public_url(url: str, allowed_schemes: set[str] | None = None) -> None:
    allowed = allowed_schemes or {"https"}
    parsed = urlparse(url)
    if parsed.scheme not in allowed:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    if not parsed.netloc:
        raise ValueError("URL must contain a host")


def source_id_from_url(url: str) -> str:
    from hashlib import sha256

    return sha256(url.encode("utf-8")).hexdigest()[:24]


@dataclass
class DiscoveredURL:
    discovery_source: str
    discovered_at: str
    parent_url: str | None
    source_id: str
    url: str
    normalized_url: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def discover_from_sitemap(sitemap_url: str) -> list[DiscoveredURL]:
    """Fetch a sitemap.xml and return discovered URLs."""
    resp = httpx.get(sitemap_url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")
    urls = []
    for loc in soup.find_all("loc"):
        url = loc.get_text().strip()
        urls.append(
            DiscoveredURL(
                discovery_source=sitemap_url,
                discovered_at=_now_iso(),
                parent_url=sitemap_url,
                source_id=source_id_from_url(url),
                url=url,
                normalized_url=url,
            )
        )

    return urls


def discover_from_rss(feed_url: str) -> list[DiscoveredURL]:
    resp = httpx.get(feed_url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")
    urls = []
    for item in soup.find_all("item"):
        link = item.find("link")
        if link:
            url = link.get_text().strip()
        else:
            # sometimes <link> is a tag with href attribute
            link_tag = item.find("link", href=True)
            url = link_tag["href"].strip() if link_tag else None

        if not url:
            continue

        urls.append(
            DiscoveredURL(
                discovery_source=feed_url,
                discovered_at=_now_iso(),
                parent_url=feed_url,
                source_id=source_id_from_url(url),
                url=url,
                normalized_url=url,
            )
        )

    return urls


def discover_links_from_html(page_url: str, html: str) -> list[DiscoveredURL]:
    """Extract links from an HTML page and normalize relative URLs against page_url."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[DiscoveredURL] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#"):
            continue
        normalized = urljoin(page_url, href)
        results.append(
            DiscoveredURL(
                discovery_source=page_url,
                discovered_at=_now_iso(),
                parent_url=page_url,
                source_id=source_id_from_url(normalized),
                url=normalized,
                normalized_url=normalized,
            )
        )

    return results


def discover_from_urls(urls: Iterable[str]) -> list[DiscoveredURL]:
    out: list[DiscoveredURL] = []
    for u in urls:
        out.append(
            DiscoveredURL(
                discovery_source="config",
                discovered_at=_now_iso(),
                parent_url=None,
                source_id=source_id_from_url(u),
                url=u,
                normalized_url=u,
            )
        )

    return out

