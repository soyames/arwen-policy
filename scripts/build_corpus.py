#!/usr/bin/env python3
"""Build the canonical Arwen Policy Corpus from ETL outputs.

Deterministic synchronization from ETL registry + extracted documents
to the tracked corpus/ directory.

Usage:
    python scripts/build_corpus.py [--validate] [--report]

The canonical corpus lives in corpus/ and is tracked in Git.
ETL working state in data/ is ephemeral (gitignored).
This script bridges the two.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def classify_source(url: str) -> str:
    if not url:
        return "unknown"
    u = url.lower()
    for domain, name in [
        ("icann.org", "ICANN"), ("ietf.org", "IETF"), ("rfc-editor.org", "IETF"),
        ("datatracker.ietf.org", "IETF"), ("itu.int", "ITU"),
        ("intgovforum.org", "IGF"), ("internetsociety.org", "ISOC"),
        ("isocfoundation.org", "ISOC"), ("oecd.org", "OECD"),
        ("un.org", "UN"), ("unesco.org", "UNESCO"),
        ("arin.net", "ARIN"), ("ripe.net", "RIPE"),
        ("apnic.net", "APNIC"), ("lacnic.net", "LACNIC"),
        ("afrinic.net", "AFRINIC"), ("europa.eu", "EU"),
        ("arxiv.org", "Academic"), ("iana.org", "IANA"),
    ]:
        if domain in u:
            return name
    return "other"


def extract_year(doc: dict) -> int | None:
    """Extract publication year from metadata, falling back to text patterns."""
    meta = doc.get("metadata") or {}
    pub = meta.get("published_at")
    if pub:
        try:
            if isinstance(pub, str) and len(pub) >= 4:
                return int(pub[:4])
            if hasattr(pub, "year"):
                return pub.year
        except (ValueError, TypeError):
            pass

    # Fallback to text patterns
    import re
    text = doc.get("text", "")[:3000]
    # Look for explicit date patterns: 1990-2027
    matches = re.findall(r"\b((?:19[9]\d|20[0-2]\d))(?:[-/](?:0[1-9]|1[0-2]))?\b", text)
    for m in matches:
        try:
            year = int(m)
            if 1990 <= year <= 2027:
                return year
        except ValueError:
            continue
    return None


def load_corpus(corpus_dir: str = "corpus") -> list[dict[str, Any]]:
    """Load all valid documents from the canonical corpus."""
    docs = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= 100:
                docs.append(doc)
        except Exception:
            continue
    return docs


def validate_corpus(corpus_dir: str = "corpus") -> dict[str, Any]:
    """Validate the canonical corpus and return a quality report."""
    docs = load_corpus(corpus_dir)
    if not docs:
        return {"error": "No valid documents found"}

    doc_ids = [d.get("document_id", "") for d in docs]
    hashes = [d.get("artifact_sha256", "") for d in docs if d.get("artifact_sha256")]
    texts = [d.get("text", "") for d in docs]

    # Duplicates
    id_dupes = len(doc_ids) - len(set(doc_ids))
    hash_dupes = len(hashes) - len(set(hashes))

    # Sources
    sources = Counter()
    for d in docs:
        url = d.get("source_url", "") or d.get("final_url", "")
        sources[classify_source(url)] += 1

    # Years
    years = []
    missing_years = 0
    for d in docs:
        y = extract_year(d)
        if y:
            years.append(y)
        else:
            missing_years += 1

    # Types
    types = Counter()
    methods = Counter()
    for d in docs:
        ct = (d.get("content_type") or "").lower()
        types["PDF" if "pdf" in ct else "HTML" if "html" in ct else "TEXT" if "text" in ct else "other"] += 1
        methods[d.get("extraction_method", "?")] += 1

    total_chars = sum(len(t) for t in texts)
    empty_docs = sum(1 for t in texts if len(t) < 20)
    short_docs = sum(1 for t in texts if len(t) < 200)

    # Decade distribution
    decades = Counter()
    for y in years:
        decades[f"{(y//10)*10}s"] += 1

    return {
        "total_documents": len(docs),
        "unique_ids": len(set(doc_ids)),
        "id_duplicates": id_dupes,
        "hash_duplicates": hash_dupes,
        "sources": len(sources),
        "source_distribution": dict(sources.most_common()),
        "document_types": dict(types),
        "extraction_methods": dict(methods),
        "total_characters": total_chars,
        "empty_documents": empty_docs,
        "short_documents": short_docs,
        "documents_with_dates": len(years),
        "missing_dates": missing_years,
        "earliest_year": min(years) if years else None,
        "latest_year": max(years) if years else None,
        "decade_distribution": dict(sorted(decades.items())),
    }


def sync_etl_to_corpus(
    extracted_dir: str = "data/extracted",
    corpus_dir: str = "corpus",
    dry_run: bool = False,
) -> dict[str, int]:
    """Sync ETL extracted documents to the canonical corpus directory."""
    extracted = Path(extracted_dir)
    corpus = Path(corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0
    if not extracted.is_dir():
        return {"added": 0, "skipped": 0, "note": "No extracted data to sync"}

    for f in sorted(extracted.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if not doc.get("text") or len(doc["text"]) < 100:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        doc_id = doc.get("document_id", "")
        target = corpus / f"{doc_id}.json"
        if target.exists():
            skipped += 1
            continue

        if not dry_run:
            target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        added += 1

    return {"added": added, "skipped": skipped}


def build_manifest(corpus_dir: str = "corpus", output: str | None = None) -> list[dict]:
    """Build a machine-readable manifest of the canonical corpus."""
    docs = load_corpus(corpus_dir)
    manifest = []
    for d in docs:
        mani = {
            "document_id": d.get("document_id"),
            "source": classify_source(d.get("source_url", "") or d.get("final_url", "")),
            "source_url": d.get("source_url"),
            "final_url": d.get("final_url"),
            "title": (d.get("metadata") or {}).get("title"),
            "year": extract_year(d),
            "language": (d.get("metadata") or {}).get("language", "und"),
            "content_type": d.get("content_type"),
            "extraction_method": d.get("extraction_method"),
            "artifact_sha256": d.get("artifact_sha256"),
            "text_length": len(d.get("text", "")),
            "segments": len(d.get("segments", [])),
        }
        manifest.append(mani)

    if output:
        Path(output).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arwen Policy Canonical Corpus Builder")
    parser.add_argument("--sync", action="store_true", help="Sync ETL output to corpus/")
    parser.add_argument("--validate", action="store_true", help="Validate corpus quality")
    parser.add_argument("--report", action="store_true", help="Print quality report")
    parser.add_argument("--manifest", type=str, help="Generate manifest JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()

    if args.sync:
        result = sync_etl_to_corpus(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))

    if args.validate or args.report:
        report = validate_corpus()
        if args.report:
            print("=" * 55)
            print("ARWEN POLICY — CANONICAL CORPUS REPORT")
            print("=" * 55)
            print(f"Documents:      {report['total_documents']}")
            print(f"Unique IDs:     {report['unique_ids']}")
            print(f"ID duplicates:  {report['id_duplicates']}")
            print(f"Hash duplicates:{report['hash_duplicates']}")
            print(f"Sources:        {report['sources']}")
            print(f"Total chars:    {report['total_characters']:,}")
            print(f"Empty docs:     {report['empty_documents']}")
            print(f"Short docs:     {report['short_documents']}")
            print(f"With dates:     {report['documents_with_dates']}")
            print(f"Missing dates:  {report['missing_dates']}")
            print(f"Earliest year:  {report['earliest_year']}")
            print(f"Latest year:    {report['latest_year']}")
            print()
            print("Source Distribution:")
            for s, c in report["source_distribution"].items():
                print(f"  {s:12} {c:>4}")
            print()
            print("Document Types:")
            for t, c in report["document_types"].items():
                print(f"  {t:10} {c:>4}")
            print()
            print("Extraction Methods:")
            for m, c in report["extraction_methods"].items():
                print(f"  {m:15} {c:>4}")
            print()
            print("Decade Distribution:")
            for d, c in report["decade_distribution"].items():
                print(f"  {d}: {c}")
        else:
            print(json.dumps(report, indent=2, default=str))

    if args.manifest:
        mani = build_manifest(output=args.manifest)
        print(f"Manifest: {len(mani)} records -> {args.manifest}")

    if not any([args.sync, args.validate, args.report, args.manifest]):
        parser.print_help()
