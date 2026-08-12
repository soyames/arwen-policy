#!/usr/bin/env python3
"""Sync the canonical Arwen Policy Corpus to the Hugging Face dataset format.

Converts the canonical corpus (corpus/ + date provenance) into the HF
dataset schema and writes updated data files.

The HF dataset distinguishes:
  - data/documents.jsonl      (canonical document corpus)
  - data/arguments.jsonl       (annotations layer — preserved as-is)
  - data/evidence.jsonl        (evidence layer — preserved as-is)
  - data/meetings.jsonl        (deliberation layer — preserved as-is)
  - data/stakeholder_positions.jsonl (stakeholder layer — preserved as-is)

The document corpus and later annotation layers remain conceptually separate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from arwen_etl.date_provenance import (
    extract_year_safe,
)


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


def classify_doc_type(doc: dict) -> str:
    """Classify document type from metadata/title."""
    title = str((doc.get("metadata") or {}).get("title") or "").lower()
    url = str(doc.get("source_url") or doc.get("final_url") or "").lower()

    if "rfc" in title and any(c.isdigit() for c in title):
        return "rfc"
    if "arxiv" in url or "arxiv" in title:
        return "academic_paper"
    if any(w in title for w in ["report", "annual", "review"]):
        return "report"
    if any(w in title for w in ["policy", "procedure", "guideline", "framework"]):
        return "policy_document"
    if any(w in title for w in ["meeting", "minutes", "agenda", "transcript"]):
        return "meeting_document"
    if any(w in title for w in ["recommendation", "standard"]):
        return "standard"
    if any(w in title for w in ["blog", "news", "press", "announcement"]):
        return "news_article"
    if any(w in title for w in ["about", "overview", "history"]):
        return "overview_page"
    return "web_page"


def convert_to_hf_document(doc: dict) -> dict[str, Any]:
    """Convert a canonical corpus document to the HF dataset schema."""
    meta = doc.get("metadata") or {}
    source_url = doc.get("source_url", "") or doc.get("final_url", "")
    text = doc.get("text", "")

    # Date with provenance
    dr = extract_year_safe(
        doc_metadata=meta,
        source_url=source_url,
        extracted_text=text,
    )

    pub_date = None
    if dr.published_at:
        pub_date = dr.published_at.isoformat()

    # Title
    title = meta.get("title") or doc.get("title") or ""
    if not title:
        # Generate from first line
        first_line = text.strip().split("\n")[0][:120] if text else ""
        title = first_line or "Untitled"

    # Source
    source = classify_source(source_url)

    # Language
    lang = meta.get("language") or "und"
    if lang and len(lang) > 10:
        lang = lang[:10]

    # Summary
    summary = meta.get("summary") or ""
    if not summary and text:
        summary = text[:500].strip()

    # Topics — from policy_topics or empty
    topics = doc.get("policy_topics", [])

    # Stakeholder groups — empty (annotation layer)
    stakeholder_groups: list[str] = []

    # Authors
    authors = meta.get("authors") or []

    # Content hash
    content_hash = doc.get("artifact_sha256") or doc.get("content_hash") or ""

    # Provenance
    provenance = {
        "source_url": doc.get("source_url"),
        "final_url": doc.get("final_url"),
        "retrieved_at": str(doc.get("retrieved_at") or ""),
        "http_status": doc.get("http_status"),
        "content_hash": content_hash,
        "extraction_method": doc.get("extraction_method", "unknown"),
        "content_type": doc.get("content_type"),
        "byte_size": doc.get("byte_size"),
        # Date provenance
        "date_value": pub_date,
        "date_source": dr.date_source.value,
        "date_confidence": dr.date_confidence.value,
        "date_type": dr.date_type.value,
    }

    return {
        "id": doc.get("document_id", ""),
        "title": title,
        "source": source,
        "document_type": classify_doc_type(doc),
        "language": lang,
        "publication_date": pub_date,
        "content": text,
        "summary": summary,
        "topics": topics,
        "stakeholder_groups": stakeholder_groups,
        "authors": authors,
        "provenance": provenance,
    }


def load_canonical_docs(corpus_dir: str = "corpus") -> list[dict[str, Any]]:
    """Load all valid canonical documents."""
    docs = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        # Skip alias directories
        if "aliases" in str(f) or "excluded" in str(f) or "canonical" in str(f):
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if doc.get("text") and len(doc["text"]) >= 200:
                docs.append(doc)
        except Exception:
            continue
    return docs


def sync_hf_dataset(
    corpus_dir: str = "corpus",
    hf_data_dir: str = "arwen-policy-corpus/data",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync canonical corpus to HF dataset format."""
    docs = load_canonical_docs(corpus_dir)
    hf_docs = [convert_to_hf_document(d) for d in docs]

    data_dir = Path(hf_data_dir)
    if not data_dir.exists():
        return {"error": f"HF data directory not found: {hf_data_dir}"}

    # Write documents.jsonl
    docs_path = data_dir / "documents.jsonl"
    if not dry_run:
        with open(docs_path, "w", encoding="utf-8") as f:
            for d in hf_docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Create train parquet (full corpus for reproducibility)
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = train_dir / "documents.parquet"
    parquet_written = False
    if not dry_run:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            def _safe_str(v, default=""):
                if v is None:
                    return default
                if isinstance(v, (list, dict)):
                    return json.dumps(v, ensure_ascii=False)[:500]
                return str(v)[:500]

            def _safe_int(v, default=0):
                if v is None:
                    return default
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, list):
                    return len(v)
                try:
                    return int(v)
                except (ValueError, TypeError):
                    return default

            # Build flat schema suitable for Parquet — no nested structures
            pq_records = []
            for d in hf_docs:
                prov = d.get("provenance", {}) if isinstance(d.get("provenance"), dict) else {}
                rec = {
                    "document_id": _safe_str(d.get("id")),
                    "source_id": _safe_str(prov.get("content_hash", "")[:24] if prov.get("content_hash") else ""),
                    "source": _safe_str(d.get("source")),
                    "source_url": _safe_str(prov.get("source_url") or ""),
                    "final_url": _safe_str(prov.get("final_url") or ""),
                    "title": _safe_str(d.get("title")),
                    "language": _safe_str(d.get("language"), "und"),
                    "extraction_method": _safe_str(prov.get("extraction_method")),
                    "content_type": _safe_str(prov.get("content_type")),
                    "artifact_sha256": _safe_str(prov.get("content_hash")),
                    "byte_size": _safe_int(prov.get("byte_size")),
                    "text_length": len(d.get("content", "") or ""),
                    "published_at": _safe_str(prov.get("date_value")),
                    "date_source": _safe_str(prov.get("date_source")),
                    "date_confidence": _safe_str(prov.get("date_confidence")),
                }
                pq_records.append(rec)

            table = pa.Table.from_pylist(pq_records)
            pq.write_table(table, parquet_path)
            parquet_written = True
        except ImportError:
            pass
        except Exception as e:
            print(f"Parquet generation warning: {e}")

    # Write split JSONL files (train = all current docs)
    train_jsonl = train_dir / "documents.jsonl"
    if not dry_run:
        with open(train_jsonl, "w", encoding="utf-8") as f:
            for d in hf_docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Count by source
    from collections import Counter
    source_counts = Counter(d["source"] for d in hf_docs)
    date_counts = Counter(
        "dated_verified" if (
            d.get("provenance", {}).get("date_confidence") in ("high", "medium")
        ) else "dated_weak" if d.get("publication_date") else "undated"
        for d in hf_docs
    )

    return {
        "documents_written": len(hf_docs),
        "documents_path": str(docs_path),
        "parquet_path": str(parquet_path) if parquet_written else None,
        "parquet_written": parquet_written,
        "train_jsonl": str(train_jsonl),
        "source_distribution": dict(source_counts),
        "date_status": dict(date_counts),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync canonical corpus to HF dataset")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--hf-data-dir", default="arwen-policy-corpus/data")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = sync_hf_dataset(
        corpus_dir=args.corpus_dir,
        hf_data_dir=args.hf_data_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, default=str))
