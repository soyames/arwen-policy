"""Arwen Policy Corpus Quality Diagnostics.

Usage: python scripts/corpus_quality.py [--corpus-dir corpus/] [--json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


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


def extract_year(doc: dict) -> int | None:
    """Extract year from document metadata."""
    meta = doc.get("metadata") or {}
    published = meta.get("published_at")
    if published:
        try:
            if isinstance(published, str):
                return int(published[:4])
            if hasattr(published, "year"):
                return published.year
        except (ValueError, TypeError):
            pass
    # Try title for common year patterns (e.g., "IGF 2024")
    title = meta.get("title", "")
    import re
    years = re.findall(r"\b(19[9]\d|20[0-2]\d)\b", str(title))
    if years:
        return int(years[0])
    return None


def run(corpus_dir: str = "corpus", json_output: bool = False) -> dict:
    """Run corpus quality diagnostics."""
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

    # Year distribution
    years = []
    missing_years = 0
    for d in docs:
        y = extract_year(d)
        if y:
            years.append(y)
        else:
            missing_years += 1

    year_counts = Counter(years)

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
        "missing_years": missing_years,
        "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
        "year_distribution": dict(sorted(year_counts.items())),
        "temporal_coverage_1990": years and min(years) >= 1990,
    }

    if not json_output:
        print("=" * 60)
        print("ARWEN POLICY CORPUS — QUALITY DIAGNOSTICS")
        print("=" * 60)
        print(f"Documents:           {len(docs)}")
        print(f"Unique (by SHA-256): {unique_docs}")
        print(f"Duplicates:          {duplicates}")
        print(f"Sources:             {len(sources)}")
        print(f"Total characters:    {total_chars:,}")
        print(f"Short docs (<200c):  {short_docs}")
        print(f"Empty docs (<20c):   {empty_content}")
        print(f"Missing titles:      {missing_titles}")
        print(f"Missing years:       {missing_years}")
        print(f"Year range:          {result['year_range']}")
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
        print("--- Year Distribution ---")
        for y, c in sorted(year_counts.items()):
            print(f"  {y}: {c}")
        print()
        print("--- Blocked/Absent Sources ---")
        configured = [
            "ICANN", "IETF", "ITU", "IGF", "ISOC", "OECD", "UN", "UNESCO",
            "ARIN", "RIPE", "APNIC", "LACNIC", "AFRINIC", "EU", "Academic",
        ]
        for src_name in configured:
            count = sources.get(src_name, 0)
            status = "OK" if count >= 20 else ("LOW" if count > 0 else "MISSING")
            print(f"  {src_name:12} {count:>4} {status}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arwen Policy Corpus Quality Diagnostics")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run(args.corpus_dir, args.json)
