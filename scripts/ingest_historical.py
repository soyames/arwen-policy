#!/usr/bin/env python3
"""Historical corpus expansion: 1990-1993 and PDF ingestion.

Ingests governance-relevant IETF RFCs from 1990-1993, plus additional
historical material from ITU, UN, and academic sources.

Also handles PDF ingestion from sources that provide PDF documents.

Usage:
    .venv/Scripts/python.exe scripts/ingest_historical.py [--dry-run] [--source SOURCE]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

USER_AGENT = "ArwenPolicyETL/0.1 (+https://github.com/soyames/arwen-policy)"
HTTP_TIMEOUT = 30.0

# ---------------------------------------------------------------------------
# Historical IETF RFCs 1990-1993 — governance/policy-relevant selection
# ---------------------------------------------------------------------------
HISTORICAL_RFCS = [
    # 1990
    {"rfc": "1160", "year": 1990, "month": "May",
     "title": "The Internet Activities Board",
     "authors": "V. Cerf"},
    {"rfc": "1166", "year": 1990, "month": "July",
     "title": "Internet Numbers",
     "authors": "S. Kirkpatrick, M.K. Stahl, M. Recker"},
    {"rfc": "1174", "year": 1990, "month": "August",
     "title": "IAB Recommended Policy on Distributing Internet Identifier Assignment and IAB Recommended Policy Change to Internet 'Connected' Status",
     "authors": "V. Cerf"},
    {"rfc": "1181", "year": 1990, "month": "September",
     "title": "RIPE Terms of Reference",
     "authors": "R. Blokzijl"},
    {"rfc": "1192", "year": 1990, "month": "November",
     "title": "Commercialization of the Internet Summary Report",
     "authors": "B. Kahin"},
    # 1991
    {"rfc": "1207", "year": 1991, "month": "February",
     "title": "FYI on Questions and Answers: Answers to Commonly asked 'Experienced Internet User' Questions",
     "authors": "G. Malkin, A. Marine, J. Reynolds"},
    {"rfc": "1211", "year": 1991, "month": "March",
     "title": "Problems with the Maintenance of Large Mailing Lists",
     "authors": "A. Westine, J. Postel"},
    {"rfc": "1250", "year": 1991, "month": "August",
     "title": "IAB Official Protocol Standards",
     "authors": "Internet Activities Board"},
    {"rfc": "1261", "year": 1991, "month": "September",
     "title": "Transition of NIC Services",
     "authors": "S. Williamson, L. Nobile"},
    {"rfc": "1262", "year": 1991, "month": "September",
     "title": "Guidelines for Internet Measurement Activities",
     "authors": "V. Cerf"},
    {"rfc": "1280", "year": 1991, "month": "November",
     "title": "IAB Official Protocol Standards",
     "authors": "Internet Activities Board"},
    # 1992
    {"rfc": "1296", "year": 1992, "month": "January",
     "title": "Internet Growth (1981-1991)",
     "authors": "M. Lottor"},
    {"rfc": "1310", "year": 1992, "month": "March",
     "title": "The Internet Standards Process",
     "authors": "L. Chapin"},
    {"rfc": "1327", "year": 1992, "month": "May",
     "title": "X.400 1988 to 1984 downgrading",
     "authors": "S. Hardcastle-Kille"},
    {"rfc": "1358", "year": 1992, "month": "August",
     "title": "Charter of the Internet Architecture Board (IAB)",
     "authors": "L. Chapin"},
    {"rfc": "1360", "year": 1992, "month": "September",
     "title": "IAB Official Protocol Standards",
     "authors": "Internet Architecture Board"},
    {"rfc": "1375", "year": 1992, "month": "October",
     "title": "Suggestion for New Classes of IP Addresses",
     "authors": "P. Robinson"},
    {"rfc": "1380", "year": 1992, "month": "November",
     "title": "IESG Deliberations on Routing and Addressing",
     "authors": "P. Gross, P. Almquist"},
    {"rfc": "1386", "year": 1992, "month": "December",
     "title": "The US Domain",
     "authors": "A. Cooper, J. Postel"},
    # 1993
    {"rfc": "1400", "year": 1993, "month": "March",
     "title": "Transition and Modernization of the Internet Registration Service",
     "authors": "S. Williamson"},
    {"rfc": "1410", "year": 1993, "month": "March",
     "title": "IAB Official Protocol Standards",
     "authors": "Internet Architecture Board"},
    {"rfc": "1436", "year": 1993, "month": "March",
     "title": "The Internet Gopher Protocol",
     "authors": "F. Anklesaria, M. McCahill, P. Lindner, D. Johnson, D. Torrey, B. Alberti"},
    {"rfc": "1454", "year": 1993, "month": "May",
     "title": "Comparison of Proposals for Next Version of IP",
     "authors": "T. Dixon"},
    {"rfc": "1462", "year": 1993, "month": "May",
     "title": "FYI on 'What is the Internet?'",
     "authors": "E. Krol, E. Hoffman"},
    {"rfc": "1480", "year": 1993, "month": "June",
     "title": "The US Domain",
     "authors": "A. Cooper, J. Postel"},
    {"rfc": "1491", "year": 1993, "month": "July",
     "title": "A Survey of Advanced Usages of X.500",
     "authors": "C. Weider, R. Wright"},
]

# Additional ITU/UN/UNESCO historical URLs 1990-1993
HISTORICAL_OTHER = [
    # ITU - early digital/information society
    {"url": "https://www.itu.int/en/history/Pages/ConstitutionAndConvention.aspx",
     "year": 1992, "source": "ITU",
     "title": "ITU Constitution and Convention (1992)"},
    {"url": "https://www.itu.int/osg/spu/wtpf/wtpf1996/",
     "year": 1996, "source": "ITU",
     "title": "World Telecommunication Policy Forum 1996"},
    # UN - early information/internet policy
    {"url": "https://www.un.org/en/global-issues/internet-governance",
     "year": None, "source": "UN",
     "title": "UN Internet Governance Overview"},
    # UNESCO
    {"url": "https://www.unesco.org/en/communication-information",
     "year": None, "source": "UNESCO",
     "title": "UNESCO Communication and Information"},
    # WTO - telecom services
    {"url": "https://www.wto.org/english/tratop_e/serv_e/telecom_e/telecom_e.htm",
     "year": None, "source": "other",
     "title": "WTO Telecommunications Services"},
]


def fetch_url(url: str, max_bytes: int = 5 * 1024 * 1024) -> dict[str, Any] | None:
    """Fetch a URL and return content + metadata."""
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            # Limit content size
            content = resp.content[:max_bytes]
            return {
                "final_url": str(resp.url),
                "status_code": resp.status_code,
                "content_type": resp.headers.get("content-type", "unknown"),
                "data": content,
                "etag": resp.headers.get("etag"),
                "last_modified": resp.headers.get("last-modified"),
            }
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}", file=sys.stderr)
        return None


def classify_source(url: str) -> str:
    u = url.lower()
    for domain, name in [
        ("icann.org", "ICANN"), ("ietf.org", "IETF"), ("rfc-editor.org", "IETF"),
        ("datatracker.ietf.org", "IETF"), ("itu.int", "ITU"),
        ("intgovforum.org", "IGF"), ("internetsociety.org", "ISOC"),
        ("oecd.org", "OECD"), ("un.org", "UN"), ("unesco.org", "UNESCO"),
        ("arin.net", "ARIN"), ("ripe.net", "RIPE"),
        ("apnic.net", "APNIC"), ("lacnic.net", "LACNIC"),
        ("afrinic.net", "AFRINIC"), ("europa.eu", "EU"),
        ("arxiv.org", "Academic"), ("iana.org", "IANA"),
        ("wto.org", "WTO"),
    ]:
        if domain in u:
            return name
    return "other"


def extract_html_text(html_bytes: bytes) -> str:
    """Extract readable text from HTML."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_bytes, "html.parser")
        for element in soup(["script", "style", "noscript", "template", "nav", "footer"]):
            element.decompose()
        text = soup.get_text("\n", strip=True)
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if title and title not in text[:500]:
            text = f"{title}\n\n{text}"
        return text
    except Exception:
        return html_bytes.decode("utf-8", errors="replace")


def ingest_rfc(rfc_num: str, year: int, month: str, title: str, authors: str,
               corpus_dir: str = "corpus") -> dict[str, Any] | None:
    """Ingest a specific RFC from rfc-editor.org."""
    url = f"https://www.rfc-editor.org/rfc/rfc{rfc_num}"
    print(f"  Fetching RFC {rfc_num}: {title[:70]}...")

    result = fetch_url(url)
    if not result or result["status_code"] != 200:
        print(f"  FAILED to fetch RFC {rfc_num}")
        return None

    raw = result["data"]
    content_hash = hashlib.sha256(raw).hexdigest()
    content_type = result.get("content_type", "text/html")

    # Extract text
    text = extract_html_text(raw)

    # Check for existing duplicate by hash
    corpus = Path(corpus_dir)
    for f in corpus.glob("*.json"):
        try:
            existing = json.loads(f.read_text(encoding="utf-8"))
            if existing.get("artifact_sha256") == content_hash:
                print(f"  SKIP: RFC {rfc_num} already in corpus as {existing.get('document_id')}")
                return None
        except Exception:
            continue

    doc_id = str(uuid4())
    pub_date = date(year, {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }.get(month, 1), 1)

    doc = {
        "document_id": doc_id,
        "source_id": content_hash[:24],
        "source_url": url,
        "final_url": result["final_url"],
        "artifact_sha256": content_hash,
        "content_type": content_type,
        "extraction_method": "html_bs4",
        "byte_size": len(raw),
        "text": text,
        "text_length": len(text),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "http_status": result["status_code"],
        "metadata": {
            "title": f"RFC {rfc_num}: {title}",
            "authors": authors,
            "published_at": pub_date.isoformat(),
            "language": "en",
        },
        "provenance_events": [],
        "policy_topics": ["internet-governance", "technical-standards"],
    }

    # Write to corpus
    target = corpus / f"{doc_id}.json"
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  INGESTED RFC {rfc_num} [{year}] -> {doc_id} ({len(text)} chars)")
    return doc


def ingest_generic(url: str, year: int | None, source: str, title: str,
                   corpus_dir: str = "corpus") -> dict[str, Any] | None:
    """Ingest a generic URL into the corpus."""
    print(f"  Fetching: {title[:70]}...")

    result = fetch_url(url)
    if not result or result["status_code"] != 200:
        print(f"  FAILED: HTTP {result.get('status_code') if result else 'error'}")
        return None

    raw = result["data"]
    content_hash = hashlib.sha256(raw).hexdigest()

    # Check for duplicates
    corpus = Path(corpus_dir)
    for f in corpus.glob("*.json"):
        try:
            existing = json.loads(f.read_text(encoding="utf-8"))
            if existing.get("artifact_sha256") == content_hash:
                print(f"  SKIP: already in corpus")
                return None
        except Exception:
            continue

    text = extract_html_text(raw)
    if len(text) < 200:
        print(f"  SKIP: too short ({len(text)} chars)")
        return None

    doc_id = str(uuid4())

    meta: dict[str, Any] = {
        "title": title,
        "language": "en",
    }
    if year:
        meta["published_at"] = date(year, 7, 1).isoformat()

    doc = {
        "document_id": doc_id,
        "source_id": content_hash[:24],
        "source_url": url,
        "final_url": result["final_url"],
        "artifact_sha256": content_hash,
        "content_type": result.get("content_type", "text/html"),
        "extraction_method": "html_bs4",
        "byte_size": len(raw),
        "text": text,
        "text_length": len(text),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "http_status": result["status_code"],
        "metadata": meta,
        "provenance_events": [],
        "policy_topics": [],
    }

    target = corpus / f"{doc_id}.json"
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  INGESTED [{source}] -> {doc_id} ({len(text)} chars)")
    return doc


def ingest_pdf(url: str, source: str, title: str, pub_year: int | None,
               corpus_dir: str = "corpus") -> dict[str, Any] | None:
    """Download and ingest a PDF document."""
    print(f"  Fetching PDF: {title[:70]}...")

    result = fetch_url(url)
    if not result or result["status_code"] != 200:
        print(f"  FAILED: HTTP {result.get('status_code') if result else 'error'}")
        return None

    raw = result["data"]
    ct = result.get("content_type", "").lower()

    # Verify it's actually a PDF
    if raw[:5] != b"%PDF-" and "pdf" not in ct:
        print(f"  SKIP: Not a PDF (content-type={ct})")
        return None

    content_hash = hashlib.sha256(raw).hexdigest()

    # Check for duplicates
    corpus = Path(corpus_dir)
    for f in corpus.glob("*.json"):
        try:
            existing = json.loads(f.read_text(encoding="utf-8"))
            if existing.get("artifact_sha256") == content_hash:
                print(f"  SKIP: already in corpus")
                return None
        except Exception:
            continue

    # Extract PDF text using pypdf
    text = ""
    extraction_method = "pypdf"
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(raw))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages_text)
        if not any(pages_text):
            text = ""
            extraction_method = "pdf_without_text"

        # Also try to get PDF metadata
        pdf_meta: dict[str, Any] = {}
        if reader.metadata:
            if reader.metadata.title:
                pdf_meta["title"] = str(reader.metadata.title)
            if hasattr(reader.metadata, "creation_date") and reader.metadata.creation_date:
                cd = reader.metadata.creation_date
                if isinstance(cd, datetime):
                    pdf_meta["published_at"] = cd.isoformat()
    except Exception as e:
        print(f"  PDF extraction error: {e}")
        text = ""
        extraction_method = "pdf_extraction_failed"

    if len(text) < 100:
        print(f"  SKIP: PDF text too short ({len(text)} chars)")
        # Still save PDF if it seems like it could be OCRed later
        if extraction_method == "pypdf":
            print(f"  Saving PDF metadata record (extractable text insufficient)")

    doc_id = str(uuid4())

    meta: dict[str, Any] = {
        "title": title,
        "language": "en",
    }
    if pub_year:
        meta["published_at"] = date(pub_year, 7, 1).isoformat()

    doc = {
        "document_id": doc_id,
        "source_id": content_hash[:24],
        "source_url": url,
        "final_url": result["final_url"],
        "artifact_sha256": content_hash,
        "content_type": "application/pdf",
        "extraction_method": extraction_method,
        "byte_size": len(raw),
        "text": text,
        "text_length": len(text),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "http_status": result["status_code"],
        "metadata": meta,
        "provenance_events": [],
        "policy_topics": [],
    }

    target = corpus / f"{doc_id}.json"
    target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  INGESTED PDF [{source}] -> {doc_id} ({len(text)} chars)")
    return doc


def run(dry_run: bool = False):
    corpus_dir = "corpus"
    total = 0

    print("=" * 60)
    print("HISTORICAL CORPUS EXPANSION: 1990-1993")
    print("=" * 60)

    # Phase 1: IETF RFCs 1990-1993
    print("\n--- IETF RFCs 1990-1993 ---")
    for rfc_info in HISTORICAL_RFCS:
        if dry_run:
            print(f"  [DRY RUN] Would fetch RFC {rfc_info['rfc']}: {rfc_info['title'][:60]}")
            continue
        result = ingest_rfc(
            rfc_num=rfc_info["rfc"], year=rfc_info["year"],
            month=rfc_info["month"], title=rfc_info["title"],
            authors=rfc_info["authors"], corpus_dir=corpus_dir,
        )
        if result:
            total += 1
        time.sleep(0.5)  # Rate limiting

    # Phase 2: Other historical sources
    print("\n--- Other Historical Sources ---")
    for info in HISTORICAL_OTHER:
        if dry_run:
            print(f"  [DRY RUN] Would fetch: {info['title'][:60]}")
            continue
        result = ingest_generic(
            url=info["url"], year=info["year"],
            source=info["source"], title=info["title"],
            corpus_dir=corpus_dir,
        )
        if result:
            total += 1
        time.sleep(0.5)

    print(f"\nTotal new documents ingested: {total}")
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Historical corpus expansion")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source", choices=["ietf", "other", "all"], default="all")
    args = parser.parse_args()

    run(dry_run=args.dry_run)
