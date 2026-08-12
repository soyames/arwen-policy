#!/usr/bin/env python3
"""Build the canonical Arwen Policy Corpus from ETL outputs.

Deterministic synchronization from ETL registry + extracted documents
to the tracked corpus/ directory.

Usage:
    python scripts/build_corpus.py [--validate] [--report] [--sync] [--export]

The canonical corpus lives in corpus/ and is tracked in Git.
ETL working state in data/ is ephemeral (gitignored).
This script bridges the two.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from arwen_etl.date_provenance import (
    DateConfidence,
    EARLIEST_YEAR,
    extract_year_safe,
)

# Today's date for future-date rejection
TODAY = date.today()
FUTURE_CUTOFF = TODAY.year + 1


def classify_source(url: str) -> str:
    """Classify document source from URL."""
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


def load_corpus(corpus_dir: str = "corpus", min_text_length: int = 200) -> list[dict[str, Any]]:
    """Load all valid documents from the canonical corpus.

    min_text_length: minimum character count to consider a document
    valid/substantive. Default 200 excludes navigation pages, stubs, etc.
    """
    docs = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= min_text_length:
                docs.append(doc)
        except Exception:
            continue
    return docs


def find_invalid_docs(corpus_dir: str = "corpus") -> list[dict[str, Any]]:
    """Return all documents below the validity threshold with reasons."""
    invalid = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            text = doc.get("text") or ""
            meta = doc.get("metadata") or {}
            text_len = len(text)
            if text_len < 200:
                reason = "empty" if text_len < 20 else (
                    "navigation_page" if text_len < 60 else
                    "insufficient_extraction"
                )
                # Heuristic: detect auth pages
                if "signing in" in text.lower() or "sign in" in text.lower():
                    reason = "authentication_page"
                elif "cookie" in text.lower() and text_len < 200:
                    reason = "cookie_page"
                invalid.append({
                    "document_id": doc.get("document_id"),
                    "text_length": text_len,
                    "reason": reason,
                    "title": meta.get("title"),
                    "source_url": doc.get("source_url"),
                    "final_url": doc.get("final_url"),
                    "text_preview": text[:200],
                })
        except Exception:
            continue
    return invalid


def find_duplicate_groups(
    corpus_dir: str = "corpus", min_text_length: int = 200,
) -> dict[str, list[dict[str, Any]]]:
    """Find groups of documents with identical SHA-256 hashes.

    Only considers valid documents (above min_text_length).
    """
    from collections import defaultdict

    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in sorted(Path(corpus_dir).glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            text = doc.get("text") or ""
            if len(text) < min_text_length:
                continue
            h = doc.get("artifact_sha256") or doc.get("content_hash") or ""
            if h:
                hash_groups[h].append({
                    "document_id": doc.get("document_id"),
                    "source_url": doc.get("source_url"),
                    "final_url": doc.get("final_url"),
                    "title": (doc.get("metadata") or {}).get("title"),
                    "filename": f.name,
                })
        except Exception:
            continue

    return {h: docs for h, docs in hash_groups.items() if len(docs) > 1}


def resolve_duplicates(
    corpus_dir: str = "corpus",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resolve content duplicates by keeping canonical, recording alternates."""
    groups = find_duplicate_groups(corpus_dir)
    if not groups:
        return {"status": "no_duplicates"}

    Path(corpus_dir) / "canonical"
    aliases_dir = Path(corpus_dir) / "aliases"
    removed = []
    kept = []

    for h, docs in groups.items():
        # Keep the first document (sorted by ID for determinism)
        docs_sorted = sorted(docs, key=lambda d: d["document_id"])
        canonical = docs_sorted[0]
        alternates = docs_sorted[1:]

        kept.append({
            "document_id": canonical["document_id"],
            "hash": h[:16] + "...",
            "title": canonical["title"],
        })

        for alt in alternates:
            removed.append({
                "document_id": alt["document_id"],
                "duplicate_of": canonical["document_id"],
                "reason": "content_duplicate",
                "hash": h[:16] + "...",
            })
            # Move alternate out of corpus
            if not dry_run:
                alt_path = Path(corpus_dir) / alt["filename"]
                if alt_path.exists():
                    aliases_dir.mkdir(parents=True, exist_ok=True)
                    # Write alias record
                    alias_record = {
                        "alias_id": alt["document_id"],
                        "canonical_id": canonical["document_id"],
                        "reason": "content_duplicate",
                        "source_url": alt["source_url"],
                        "final_url": alt["final_url"],
                    }
                    alias_target = aliases_dir / f"{alt['document_id']}.json"
                    alias_target.write_text(
                        json.dumps(alias_record, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    # Remove duplicate from main corpus
                    alt_path.unlink()

    return {
        "status": "resolved",
        "duplicate_groups": len(groups),
        "documents_removed": len(removed),
        "documents_kept": len(kept),
        "removed": removed,
        "kept": kept,
    }


def classify_invalid_reason(text: str) -> str:
    """Classify why a document is invalid/short."""
    t = text.lower()
    if "signing in" in t or "sign in" in t:
        return "authentication_page"
    if len(text) < 20:
        return "empty_or_near_empty"
    if "cookie" in t and len(text) < 200:
        return "cookie_page"
    if len(text) < 60:
        return "navigation_stub"
    if "privacy policy" in t and len(text) < 200:
        return "legal_stub"
    if len(text) < 200:
        return "insufficient_extraction"
    return "unknown"


def validate_corpus(corpus_dir: str = "corpus") -> dict[str, Any]:
    """Validate the canonical corpus and return a quality report with date provenance."""
    # Load both valid and invalid docs
    all_docs = []
    valid_docs = []
    invalid_docs = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        # Skip alias files
        if "aliases" in str(f):
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            all_docs.append(doc)
            if doc.get("text") and len(doc["text"]) >= 200:
                valid_docs.append(doc)
            else:
                invalid_docs.append({
                    "document_id": doc.get("document_id"),
                    "text_length": len(doc.get("text") or ""),
                    "reason": classify_invalid_reason(doc.get("text") or ""),
                    "title": (doc.get("metadata") or {}).get("title"),
                    "source_url": doc.get("source_url"),
                })
        except Exception:
            continue

    if not valid_docs:
        return {"error": "No valid documents found"}

    doc_ids = [d.get("document_id", "") for d in valid_docs]
    hashes = [d.get("artifact_sha256", "") for d in valid_docs if d.get("artifact_sha256")]
    texts = [d.get("text", "") for d in valid_docs]

    # Duplicates
    id_dupes = len(doc_ids) - len(set(doc_ids))
    hash_dupes = len(hashes) - len(set(hashes))

    # Sources
    sources = Counter()
    for d in valid_docs:
        url = d.get("source_url", "") or d.get("final_url", "")
        sources[classify_source(url)] += 1

    # Date extraction with provenance
    years_verified = []  # high/medium confidence
    years_weak = []      # low confidence (URL year only)
    years_all = []       # all extracted years
    missing_dates = 0
    future_dates = []
    date_provenance = Counter()
    date_sources = Counter()

    for d in valid_docs:
        meta = d.get("metadata") or {}
        source_url = d.get("source_url", "") or d.get("final_url", "")
        text = d.get("text", "")

        dr = extract_year_safe(
            doc_metadata=meta,
            source_url=source_url,
            extracted_text=text,
        )

        if dr.published_at and dr.date_confidence != DateConfidence.none:
            year = dr.published_at.year
            years_all.append(year)
            date_provenance[dr.date_confidence.value] += 1
            date_sources[dr.date_source.value] += 1

            if dr.date_confidence in (DateConfidence.high, DateConfidence.medium):
                years_verified.append(year)
            elif dr.date_confidence == DateConfidence.low:
                years_weak.append(year)

            # Flag future dates
            if dr.published_at.year >= FUTURE_CUTOFF:
                future_dates.append({
                    "document_id": d.get("document_id"),
                    "year": dr.published_at.year,
                    "date_source": dr.date_source.value,
                    "date_confidence": dr.date_confidence.value,
                    "title": meta.get("title"),
                })
        else:
            missing_dates += 1

    # Types
    types = Counter()
    methods = Counter()
    for d in valid_docs:
        ct = (d.get("content_type") or "").lower()
        types["PDF" if "pdf" in ct else "HTML" if "html" in ct else "TEXT" if "text" in ct else "other"] += 1
        methods[d.get("extraction_method", "?")] += 1

    total_chars = sum(len(t) for t in texts)

    # Decade distribution (verified dates only)
    decades = Counter()
    for y in years_verified:
        decades[f"{(y // 10) * 10}s"] += 1

    # 1990-1993 coverage
    early_years = {y for y in years_all if 1990 <= y <= 1993}
    historical_coverage = {
        "years_found": sorted(early_years),
        "docs_1990_1993": sum(1 for y in years_all if 1990 <= y <= 1993),
    }

    # Source detail with year ranges and date quality
    source_detail = {}
    for src_name in sorted(sources.keys()):
        src_docs = [
            d for d in valid_docs
            if classify_source(d.get("source_url", "") or d.get("final_url", "")) == src_name
        ]
        src_years = []
        src_undated = 0
        for d in src_docs:
            meta = d.get("metadata") or {}
            source_url = d.get("source_url", "") or d.get("final_url", "")
            dr = extract_year_safe(doc_metadata=meta, source_url=source_url, extracted_text=d.get("text", ""))
            if dr.published_at and dr.date_confidence != DateConfidence.none:
                src_years.append(dr.published_at.year)
            else:
                src_undated += 1

        source_detail[src_name] = {
            "valid_documents": len(src_docs),
            "dated": len(src_years),
            "undated": src_undated,
            "earliest_verified": min(src_years) if src_years else None,
            "latest_verified": max(src_years) if src_years else None,
            "years": sorted(set(src_years)) if src_years else [],
        }

    return {
        "total_documents": len(valid_docs),
        "invalid_documents": len(invalid_docs),
        "invalid_details": invalid_docs,
        "unique_ids": len(set(doc_ids)),
        "id_duplicates": id_dupes,
        "hash_duplicates": hash_dupes,
        "sources": len(sources),
        "source_distribution": dict(sources.most_common()),
        "source_detail": source_detail,
        "document_types": dict(types),
        "extraction_methods": dict(methods),
        "total_characters": total_chars,
        "documents_with_verified_dates": len(years_verified),
        "documents_with_weak_dates": len(years_weak),
        "missing_dates": missing_dates,
        "earliest_verified_year": min(years_verified) if years_verified else None,
        "latest_verified_year": max(years_verified) if years_verified else None,
        "earliest_any_year": min(years_all) if years_all else None,
        "latest_any_year": max(years_all) if years_all else None,
        "future_dates_flagged": future_dates,
        "date_provenance": dict(date_provenance),
        "date_sources": dict(date_sources),
        "decade_distribution_verified": dict(sorted(decades.items())),
        "historical_coverage_1990_1993": historical_coverage,
        "collection_boundary": EARLIEST_YEAR,
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


def build_manifest(corpus_dir: str = "corpus", output: str | None = None) -> list[dict[str, Any]]:
    """Build a machine-readable manifest of the canonical corpus with date provenance."""
    docs = load_corpus(corpus_dir)
    manifest = []
    for d in docs:
        meta = d.get("metadata") or {}
        source_url = d.get("source_url", "") or d.get("final_url", "")
        dr = extract_year_safe(
            doc_metadata=meta,
            source_url=source_url,
            extracted_text=d.get("text", ""),
        )

        mani = {
            "document_id": d.get("document_id"),
            "source": classify_source(source_url),
            "source_url": d.get("source_url"),
            "final_url": d.get("final_url"),
            "title": meta.get("title"),
            "year": dr.year,
            "published_at": dr.published_at.isoformat() if dr.published_at else None,
            "date_source": dr.date_source.value,
            "date_confidence": dr.date_confidence.value,
            "date_type": dr.date_type.value,
            "language": meta.get("language", "und"),
            "content_type": d.get("content_type"),
            "extraction_method": d.get("extraction_method"),
            "artifact_sha256": d.get("artifact_sha256"),
            "text_length": len(d.get("text", "")),
            "segments": len(d.get("segments", [])),
        }
        manifest.append(mani)

    if output:
        Path(output).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    return manifest


def export_canonical(
    corpus_dir: str = "corpus",
    output_dir: str = "data/canonical",
    format: str = "jsonl",
) -> dict[str, Any]:
    """Export the canonical corpus with full date provenance.

    Only exports valid documents (>= 200 characters).
    Excludes duplicates that have been resolved to aliases.
    """
    docs = load_corpus(corpus_dir, min_text_length=200)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for d in docs:
        meta = d.get("metadata") or {}
        source_url = d.get("source_url", "") or d.get("final_url", "")
        dr = extract_year_safe(
            doc_metadata=meta,
            source_url=source_url,
            extracted_text=d.get("text", ""),
        )

        record = {
            "document_id": d.get("document_id"),
            "source_id": d.get("source_id", ""),
            "source": classify_source(source_url),
            "source_url": d.get("source_url"),
            "final_url": d.get("final_url"),
            "artifact_sha256": d.get("artifact_sha256"),
            "content_type": d.get("content_type"),
            "extraction_method": d.get("extraction_method"),
            "byte_size": d.get("byte_size"),
            "title": meta.get("title"),
            "language": meta.get("language", "und"),
            "text_length": len(d.get("text", "")),
            "characters": len(d.get("text", "")),
            "text": d.get("text"),
            "segments": d.get("segments", []),
            # Date with full provenance
            "published_at": dr.published_at.isoformat() if dr.published_at else None,
            "date_source": dr.date_source.value,
            "date_confidence": dr.date_confidence.value,
            "date_type": dr.date_type.value,
            "no_date_reason": dr.no_date_reason if not dr.published_at else "",
            "date_evidence": [
                {
                    "date_value": e.date_value.isoformat() if e.date_value else None,
                    "date_type": e.date_type.value,
                    "source": e.source.value,
                    "confidence": e.confidence.value,
                    "raw_text": e.raw_text,
                    "notes": e.notes,
                }
                for e in dr.evidence
            ],
            # Metadata
            "retrieved_at": str(d.get("retrieved_at") or ""),
            "http_status": d.get("http_status"),
            "jurisdiction": meta.get("jurisdiction"),
            "policy_topics": d.get("policy_topics", []),
            "license": d.get("license") or meta.get("license"),
            "access_conditions": d.get("access_conditions"),
        }
        records.append(record)

    # Write JSONL
    jsonl_path = output / "canonical_documents.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write Parquet if pyarrow available
    parquet_path = None
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Remove 'text' and 'segments' for parquet (too large as columns)
        pq_records = [{k: v for k, v in rec.items() if k not in ("text", "segments", "date_evidence")} for rec in records]
        # Convert date_evidence to string for parquet
        for rec in pq_records:
            rec["date_evidence_json"] = json.dumps(
                [e for e in (next((r["date_evidence"] for r in records if r["document_id"] == rec["document_id"]), []))]
            )

        table = pa.Table.from_pylist(pq_records)
        parquet_path = output / "canonical_documents.parquet"
        pq.write_table(table, parquet_path)
    except ImportError:
        pass

    # Write full manifest
    manifest_path = output / "canonical_manifest.json"
    manifest = [
        {
            k: v for k, v in rec.items()
            if k not in ("text", "segments", "date_evidence")
        }
        for rec in records
    ]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "canonical_documents": len(records),
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path) if parquet_path else None,
        "manifest": str(manifest_path),
        "total_chars": sum(len(d.get("text", "")) for d in docs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arwen Policy Canonical Corpus Builder")
    parser.add_argument("--sync", action="store_true", help="Sync ETL output to corpus/")
    parser.add_argument("--validate", action="store_true", help="Validate corpus quality")
    parser.add_argument("--report", action="store_true", help="Print full quality report")
    parser.add_argument("--manifest", type=str, help="Generate manifest JSON file")
    parser.add_argument("--export", action="store_true", help="Export canonical corpus (JSONL + Parquet)")
    parser.add_argument("--resolve-duplicates", action="store_true", help="Resolve content duplicates")
    parser.add_argument("--find-invalid", action="store_true", help="Find and report invalid documents")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    args = parser.parse_args()

    if args.sync:
        result = sync_etl_to_corpus(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))

    if args.resolve_duplicates:
        result = resolve_duplicates(dry_run=args.dry_run)
        print(json.dumps(result, indent=2, default=str))

    if args.find_invalid:
        invalid = find_invalid_docs()
        print(f"Invalid documents: {len(invalid)}")
        for doc in invalid:
            print(f"  {doc['document_id']}: {doc['reason']} ({doc['text_length']} chars)")
            print(f"    title: {doc['title']}")
            print(f"    url: {doc['source_url']}")
            print(f"    preview: {doc['text_preview'][:120]}")
            print()

    if args.validate or args.report:
        report = validate_corpus()
        if args.report:
            print("=" * 60)
            print("ARWEN POLICY — CANONICAL CORPUS REPORT")
            print("=" * 60)
            print(f"Valid documents:       {report['total_documents']}")
            print(f"Invalid documents:     {report['invalid_documents']}")
            print(f"Unique IDs:            {report['unique_ids']}")
            print(f"ID duplicates:         {report['id_duplicates']}")
            print(f"Hash duplicates:       {report['hash_duplicates']}")
            print(f"Sources:               {report['sources']}")
            print(f"Total chars:           {report['total_characters']:,}")
            print(f"Verified dates:        {report['documents_with_verified_dates']}")
            print(f"Weak dates (URL):      {report['documents_with_weak_dates']}")
            print(f"Missing dates:         {report['missing_dates']}")
            print(f"Earliest verified:     {report['earliest_verified_year']}")
            print(f"Latest verified:       {report['latest_verified_year']}")
            print(f"Earliest any:          {report['earliest_any_year']}")
            print(f"Latest any:            {report['latest_any_year']}")
            print(f"Future dates flagged:  {len(report['future_dates_flagged'])}")
            print(f"Collection boundary:   {report['collection_boundary']}")
            print()
            if report.get("future_dates_flagged"):
                print("--- Future Dates Flagged ---")
                for fd in report["future_dates_flagged"]:
                    print(f"  {fd['document_id']}: {fd['year']} ({fd['date_source']}, {fd['date_confidence']})")
                    print(f"    {fd['title']}")
                print()
            print("--- Source Distribution ---")
            for s, c in report["source_distribution"].items():
                detail = report.get("source_detail", {}).get(s, {})
                extras = ""
                if detail:
                    extras = f"  dated={detail.get('dated', '?')}/{detail.get('valid_documents', '?')}  range={detail.get('earliest_verified', '?')}-{detail.get('latest_verified', '?')}"
                print(f"  {s:12} {c:>4}{extras}")
            print()
            print("--- Date Provenance ---")
            for k, v in report.get("date_provenance", {}).items():
                print(f"  {k:12} {v:>4}")
            print()
            print("--- Date Sources ---")
            for k, v in report.get("date_sources", {}).items():
                print(f"  {k:20} {v:>4}")
            print()
            print("--- Decade Distribution (verified dates) ---")
            for d, c in report.get("decade_distribution_verified", {}).items():
                print(f"  {d}: {c}")
            print()
            hist = report.get("historical_coverage_1990_1993", {})
            print("--- 1990–1993 Historical Coverage ---")
            print(f"  Documents: {hist.get('docs_1990_1993', 0)}")
            print(f"  Years: {hist.get('years_found', [])}")
            print()
            if report.get("invalid_details"):
                print("--- Invalid/Short Documents ---")
                for inv in report["invalid_details"]:
                    print(f"  {inv['document_id']}: {inv['reason']} ({inv['text_length']} chars)")
                    print(f"    {inv.get('title', 'N/A')}")
                    print()
        else:
            # Remove verbose details for JSON output
            summary = {k: v for k, v in report.items() if k not in ("invalid_details", "source_detail")}
            print(json.dumps(summary, indent=2, default=str))

    if args.manifest:
        mani = build_manifest(output=args.manifest)
        print(f"Manifest: {len(mani)} records -> {args.manifest}")

    if args.export:
        result = export_canonical()
        print(json.dumps(result, indent=2, default=str))

    if not any([args.sync, args.validate, args.report, args.manifest, args.export,
                args.resolve_duplicates, args.find_invalid]):
        parser.print_help()
