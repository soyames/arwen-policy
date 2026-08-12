#!/usr/bin/env python3
"""PDF corpus ingestion.

Ingests legitimate public PDF documents from policy sources.
Verifies content-type, extraction, and adds to canonical corpus.

Usage:
    .venv/Scripts/python.exe scripts/ingest_pdfs.py [--dry-run]
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

USER_AGENT = "ArwenPolicyETL/0.1 (+https://github.com/soyames/arwen-policy)"

# ---------------------------------------------------------------------------
# PDF sources — legitimate public policy PDFs
# ---------------------------------------------------------------------------
PDF_SOURCES = [
    # IETF RFCs as PDF (good test of PDF pipeline with real content)
    {"url": "https://www.rfc-editor.org/rfc/rfc1591.pdf",
     "source": "IETF", "title": "RFC 1591: Domain Name System Structure and Delegation (PDF)",
     "pub_year": 1994},
    {"url": "https://www.rfc-editor.org/rfc/rfc2026.pdf",
     "source": "IETF", "title": "RFC 2026: The Internet Standards Process (PDF)",
     "pub_year": 1996},
    {"url": "https://www.rfc-editor.org/rfc/rfc3935.pdf",
     "source": "IETF", "title": "RFC 3935: A Mission Statement for the IETF (PDF)",
     "pub_year": 2004},
    {"url": "https://www.rfc-editor.org/rfc/rfc7258.pdf",
     "source": "IETF", "title": "RFC 7258: Pervasive Monitoring Is an Attack (PDF)",
     "pub_year": 2014},
    {"url": "https://www.rfc-editor.org/rfc/rfc8280.pdf",
     "source": "IETF", "title": "RFC 8280: Research into Human Rights Protocol Considerations (PDF)",
     "pub_year": 2017},

    # ITU publications (PDF)
    {"url": "https://www.itu.int/en/ITU-D/Statistics/Documents/facts/ICTFactsFigures2017.pdf",
     "source": "ITU", "title": "ITU ICT Facts and Figures 2017",
     "pub_year": 2017},

    # OECD digital policy papers
    {"url": "https://www.oecd.org/content/dam/oecd/en/publications/reports/2024/06/oecd-digital-economy-outlook-2024-volume-1_5a654a09/24d2704b-en.pdf",
     "source": "OECD", "title": "OECD Digital Economy Outlook 2024",
     "pub_year": 2024},

    # UNESCO AI ethics
    {"url": "https://unesdoc.unesco.org/ark:/48223/pf0000381137",
     "source": "UNESCO",
     "title": "UNESCO Recommendation on the Ethics of Artificial Intelligence",
     "pub_year": 2021},

    # EU digital strategy
    {"url": "https://digital-strategy.ec.europa.eu/en/library/european-declaration-digital-rights-and-principles-digital-decade",
     "source": "EU",
     "title": "European Declaration on Digital Rights and Principles",
     "pub_year": 2022},

    # ISOC reports
    {"url": "https://www.internetsociety.org/resources/doc/2024/2024-internet-impact-brief-global-digital-compact/",
     "source": "ISOC",
     "title": "2024 Internet Impact Brief: Global Digital Compact",
     "pub_year": 2024},

    # IGF
    {"url": "https://intgovforum.org/en/content/igf-2023-leadership-panel-report",
     "source": "IGF",
     "title": "IGF 2023 Leadership Panel Report",
     "pub_year": 2023},
]


def fetch_url(url: str) -> dict[str, Any] | None:
    """Fetch a URL and return content + metadata."""
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=60.0,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            # Limit to 20MB for PDFs
            content = resp.content[:20 * 1024 * 1024]
            return {
                "final_url": str(resp.url),
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type", "unknown"),
                "data": content,
            }
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None


def extract_pdf_text(raw: bytes) -> tuple[str, str]:
    """Extract text from PDF bytes. Returns (text, method)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        if any(p.strip() for p in pages):
            return text, "pypdf"
        return "", "pdf_without_text"
    except Exception as e:
        return "", f"pypdf_error: {e}"


def extract_html_text(raw: bytes) -> str:
    """Extract text from HTML bytes."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for e in soup(["script", "style", "noscript", "nav", "footer"]):
            e.decompose()
        text = soup.get_text("\n", strip=True)
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if title and title not in text[:500]:
            text = f"{title}\n\n{text}"
        return text
    except Exception:
        return raw.decode("utf-8", errors="replace")


def ingest_document(url: str, source: str, title: str, pub_year: int | None,
                    raw: bytes, final_url: str, content_type: str,
                    corpus_dir: str = "corpus") -> dict[str, Any] | None:
    """Ingest a document into the corpus."""
    content_hash = hashlib.sha256(raw).hexdigest()

    # Check for existing duplicate
    corpus = Path(corpus_dir)
    for f in corpus.glob("*.json"):
        try:
            existing = json.loads(f.read_text(encoding="utf-8"))
            if existing.get("artifact_sha256") == content_hash:
                print(f"    SKIP: duplicate of {existing.get('document_id')}")
                return None
        except Exception:
            continue

    # Extract based on content type
    ct_lower = content_type.lower()
    is_pdf = raw[:5] == b"%PDF-" or "pdf" in ct_lower

    if is_pdf:
        text, extraction_method = extract_pdf_text(raw)
        ct = "application/pdf"
    elif "html" in ct_lower:
        text = extract_html_text(raw)
        extraction_method = "html_bs4"
        ct = content_type
    else:
        text = raw.decode("utf-8", errors="replace")[:100000]
        extraction_method = "utf8_text"
        ct = content_type

    if len(text) < 100:
        print(f"    SKIP: text too short ({len(text)} chars), method={extraction_method}")
        return None

    doc_id = str(uuid4())

    meta: dict[str, Any] = {"title": title, "language": "en"}
    if pub_year:
        meta["published_at"] = date(pub_year, 7, 1).isoformat()

    doc = {
        "document_id": doc_id,
        "source_id": content_hash[:24],
        "source_url": url,
        "final_url": final_url,
        "artifact_sha256": content_hash,
        "content_type": ct,
        "extraction_method": extraction_method,
        "byte_size": len(raw),
        "text": text,
        "text_length": len(text),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "http_status": 200,
        "metadata": meta,
        "provenance_events": [],
        "policy_topics": [],
    }

    target = corpus / f"{doc_id}.json"
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def run(dry_run: bool = False):
    total = 0
    pdf_total = 0

    print("=" * 60)
    print("PDF AND SOURCE EXPANSION INGESTION")
    print("=" * 60)

    for src in PDF_SOURCES:
        print(f"\n  [{src['source']}] {src['title'][:70]}")
        if dry_run:
            print("    [DRY RUN]")
            continue

        result = fetch_url(src["url"])
        if not result:
            continue

        doc = ingest_document(
            url=src["url"], source=src["source"],
            title=src["title"], pub_year=src["pub_year"],
            raw=result["data"], final_url=result["final_url"],
            content_type=result["content_type"],
        )
        if doc:
            is_pdf = "pdf" in doc.get("content_type", "").lower()
            total += 1
            if is_pdf:
                pdf_total += 1
            print(f"    INGESTED {'PDF' if is_pdf else 'HTML'} [{doc['extraction_method']}] -> {doc['document_id']} ({doc['text_length']} chars)")
        time.sleep(0.5)

    print(f"\nTotal new: {total} ({pdf_total} PDFs)")
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PDF and source expansion ingestion")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
