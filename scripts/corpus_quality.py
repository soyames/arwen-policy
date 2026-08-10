"""Arwen Policy Corpus Quality Diagnostics.

Uses the date provenance model to distinguish verified dates from
weak URL-based inferences. Future dates are flagged, not accepted.

Usage: python scripts/corpus_quality.py [--corpus-dir corpus/] [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from arwen_etl.date_provenance import (
    DateConfidence,
    EARLIEST_YEAR,
    extract_year_safe,
)

TODAY_YEAR = 2026
FUTURE_CUTOFF = TODAY_YEAR + 1


def classify_source(url: str) -> str:
    """Classify document source from URL."""
    if not url:
        return "unknown"
    u = url.lower()
    for domain, name in [
        ("icann.org", "ICANN"),
        ("ietf.org", "IETF"),
        ("rfc-editor.org", "IETF"),
        ("datatracker.ietf.org", "IETF"),
        ("itu.int", "ITU"),
        ("intgovforum.org", "IGF"),
        ("internetsociety.org", "ISOC"),
        ("isocfoundation.org", "ISOC"),
        ("oecd.org", "OECD"),
        ("un.org", "UN"),
        ("unesco.org", "UNESCO"),
        ("arin.net", "ARIN"),
        ("ripe.net", "RIPE"),
        ("apnic.net", "APNIC"),
        ("lacnic.net", "LACNIC"),
        ("afrinic.net", "AFRINIC"),
        ("europa.eu", "EU"),
        ("arxiv.org", "Academic"),
        ("iana.org", "IANA"),
    ]:
        if domain in u:
            return name
    return "other"


def run(corpus_dir: str = "corpus", json_output: bool = False) -> dict:
    """Run corpus quality diagnostics with date provenance."""
    corpus = Path(corpus_dir)
    docs = []
    for f in sorted(corpus.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= 100:
                docs.append(doc)
        except Exception:
            continue

    if not docs:
        print("No documents found.")
        return {}

    # Source distribution
    sources = Counter()
    for d in docs:
        url = d.get("source_url", "") or d.get("final_url", "")
        sources[classify_source(url)] += 1

    # Date extraction with provenance
    years_verified = []
    years_weak = []
    years_all = []
    missing_years = 0
    future_flags = []
    date_sources = Counter()
    year_counts_verified = Counter()

    for d in docs:
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
            date_sources[dr.date_source.value] += 1

            if dr.date_confidence in (DateConfidence.high, DateConfidence.medium):
                years_verified.append(year)
                year_counts_verified[year] += 1
            else:
                years_weak.append(year)

            if year >= FUTURE_CUTOFF:
                future_flags.append({
                    "document_id": d.get("document_id"),
                    "year": year,
                    "title": meta.get("title"),
                    "date_source": dr.date_source.value,
                })
        else:
            missing_years += 1

    # Document types
    content_types = Counter()
    extraction_methods = Counter()
    languages = Counter()
    short_docs = 0
    total_chars = 0
    missing_titles = 0
    empty_content = 0

    for d in docs:
        ct = (d.get("content_type") or "").lower()
        if "pdf" in ct:
            content_types["PDF"] += 1
        elif "html" in ct:
            content_types["HTML"] += 1
        elif "text" in ct:
            content_types["TEXT"] += 1
        else:
            content_types["other"] += 1

        method = d.get("extraction_method", "unknown")
        extraction_methods[method] += 1

        lang = (d.get("metadata") or {}).get("language") or "und"
        languages[lang] += 1

        text_len = len(d.get("text", ""))
        total_chars += text_len
        if text_len < 200:
            short_docs += 1
        if text_len < 20:
            empty_content += 1

        title = (d.get("metadata") or {}).get("title")
        if not title:
            missing_titles += 1

    # Duplicate detection via SHA-256
    hashes = Counter()
    for d in docs:
        h = d.get("artifact_sha256", "")
        if h:
            hashes[h] += 1

    duplicates = sum(c - 1 for c in hashes.values() if c > 1)
    unique_docs = len(hashes)

    # 1990-1993 coverage
    early_years = sorted(set(y for y in years_all if EARLIEST_YEAR <= y <= 1993))

    result = {
        "total_documents": len(docs),
        "unique_documents": unique_docs,
        "duplicates": duplicates,
        "sources": len(sources),
        "source_distribution": dict(sources.most_common()),
        "content_types": dict(content_types),
        "extraction_methods": dict(extraction_methods),
        "languages": dict(languages.most_common(10)),
        "total_characters": total_chars,
        "short_documents": short_docs,
        "empty_documents": empty_content,
        "missing_titles": missing_titles,
        # Date provenance
        "verified_dates": len(years_verified),
        "weak_dates": len(years_weak),
        "missing_dates": missing_years,
        "earliest_verified": min(years_verified) if years_verified else None,
        "latest_verified": max(years_verified) if years_verified else None,
        "earliest_any": min(years_all) if years_all else None,
        "latest_any": max(years_all) if years_all else None,
        "year_distribution_verified": dict(sorted(year_counts_verified.items())),
        "date_sources": dict(date_sources),
        "future_dates_flagged": len(future_flags),
        "temporal_coverage_1990": years_all and min(years_all) >= EARLIEST_YEAR,
        "historical_1990_1993": early_years,
    }

    if not json_output:
        print("=" * 60)
        print("ARWEN POLICY CORPUS — QUALITY DIAGNOSTICS")
        print("=" * 60)
        print(f"Documents:                {len(docs)}")
        print(f"Unique (by SHA-256):      {unique_docs}")
        print(f"Duplicates:               {duplicates}")
        print(f"Sources:                  {len(sources)}")
        print(f"Total characters:         {total_chars:,}")
        print(f"Short docs (<200c):       {short_docs}")
        print(f"Empty docs (<20c):        {empty_content}")
        print(f"Missing titles:           {missing_titles}")
        print()
        print("--- Dates (with provenance) ---")
        print(f"Verified dates:           {len(years_verified)}")
        print(f"Weak dates (URL-based):   {len(years_weak)}")
        print(f"Missing dates:            {missing_years}")
        print(f"Earliest verified:        {result['earliest_verified']}")
        print(f"Latest verified:          {result['latest_verified']}")
        print(f"Earliest any:             {result['earliest_any']}")
        print(f"Latest any:               {result['latest_any']}")
        print(f"Future dates flagged:     {len(future_flags)}")
        print(f"Date sources:             {dict(date_sources)}")
        if future_flags:
            for ff in future_flags:
                print(f"  FUTURE: {ff['document_id']} year={ff['year']} src={ff['date_source']} title={ff.get('title','?')[:60]}")
        print()
        print("--- 1990–1993 Historical Coverage ---")
        print(f"Documents found:          {len(early_years)}")
        print(f"Years covered:            {early_years}")
        print()
        print("--- Source Distribution ---")
        for s, c in sources.most_common():
            print(f"  {s:12} {c:>4}")
        print()
        print("--- Document Types ---")
        for ct, c in content_types.most_common():
            print(f"  {ct:10} {c:>4}")
        print()
        print("--- Extraction Methods ---")
        for m, c in extraction_methods.most_common():
            print(f"  {m:15} {c:>4}")
        print()
        print("--- Year Distribution (verified + weak) ---")
        all_year_counts = Counter(years_all)
        for y, c in sorted(all_year_counts.items()):
            marker = ""
            if y >= FUTURE_CUTOFF:
                marker = " [FUTURE - FLAGGED]"
            elif y <= 1993:
                marker = " [1990-1993 target]"
            print(f"  {y}: {c}{marker}")
        print()
        print("--- Blocked/Absent Sources ---")
        configured = [
            "ICANN", "IETF", "ITU", "IGF", "ISOC", "OECD", "UN", "UNESCO",
            "ARIN", "RIPE", "APNIC", "LACNIC", "AFRINIC", "EU", "Academic",
        ]
        for src_name in configured:
            count = sources.get(src_name, 0)
            threshold = 5 if src_name in ("Academic", "EU", "OECD") else 10
            if count == 0:
                status = "MISSING"
            elif count < threshold:
                status = f"LOW (<{threshold})"
            else:
                status = "OK"
            print(f"  {src_name:12} {count:>4} {status}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arwen Policy Corpus Quality Diagnostics")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run(args.corpus_dir, args.json)
