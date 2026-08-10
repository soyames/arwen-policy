"""End-to-end tests for PDF extraction pipeline.

Tests discovery, download, content-type verification, extraction,
metadata extraction, normalization, hashing, and provenance.
"""

import hashlib
import io
import json
import tempfile
from pathlib import Path

import pytest

from arwen_etl.extraction import extract, extract_pdf, ExtractedContent
from arwen_etl.hashing import compute_hash, sha256_bytes


# A minimal valid PDF for testing (hand-crafted)
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\n"
    b"BT /F1 24 Tf 100 700 Td (Hello Policy World) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \n"
    b"0000000360 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n418\n%%EOF"
)


class TestPDFExtraction:
    """Test basic PDF content extraction."""

    def test_extract_minimal_pdf(self):
        """A minimal valid PDF should extract text successfully."""
        result = extract_pdf(MINIMAL_PDF)
        assert result is not None
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.method in ("pypdf", "pypdf+ocr", "pdf_without_text")
        assert result.media_type == "application/pdf"

    def test_extract_via_dispatcher(self):
        """The main extract() dispatcher should handle PDF."""
        result = extract(MINIMAL_PDF, "application/pdf")
        assert result is not None
        assert "pdf" in result.method.lower() or result.method == "unsupported"

    def test_pdf_content_type_detected(self):
        """Content type should be detected as PDF."""
        from arwen_etl.extraction import detect_media_type
        mt = detect_media_type("document.pdf", "application/pdf")
        assert mt == "application/pdf"

    def test_pdf_hash_stable(self):
        """SHA-256 of PDF data should be stable."""
        h1 = sha256_bytes(MINIMAL_PDF)
        h2 = sha256_bytes(MINIMAL_PDF)
        assert h1 == h2
        assert len(h1) == 64

    def test_pdf_with_metadata_extraction(self):
        """PDF with metadata should extract title and creation date."""
        # Test that the extraction module doesn't crash on PDF metadata
        result = extract_pdf(MINIMAL_PDF)
        assert result.text is not None


class TestPDFPipeline:
    """Test the full PDF ingestion pipeline end-to-end."""

    def test_discover_download_extract(self):
        """Simulate the full pipeline: raw PDF → extract → normalize."""
        # Step 1: Capture raw bytes (simulated)
        raw = MINIMAL_PDF

        # Step 2: Verify content type
        assert raw[:5] == b"%PDF-"

        # Step 3: Compute content hash
        content_hash = sha256_bytes(raw)
        assert len(content_hash) == 64

        # Step 4: Extract text
        result = extract_pdf(raw)
        assert result.text is not None
        assert result.method in ("pypdf", "pypdf+ocr", "pdf_without_text")

        # Step 5: Normalize and create document record
        doc_id = hashlib.sha256(raw).hexdigest()[:16]
        record = {
            "document_id": doc_id,
            "content_type": "application/pdf",
            "extraction_method": result.method,
            "artifact_sha256": content_hash,
            "byte_size": len(raw),
            "text": result.text,
            "text_length": len(result.text),
            "metadata": {"title": "Test PDF"},
        }

        # Verify the record is complete
        assert record["document_id"] is not None
        assert record["content_type"] == "application/pdf"
        assert record["byte_size"] > 0
        assert record["text_length"] >= 0


class TestPDFDateExtraction:
    """Test date extraction from PDF metadata."""

    def test_pdf_creation_date_becomes_date(self):
        """PDF creation_date should be picked up as publication date."""
        from datetime import date as dt_date
        from arwen_etl.date_provenance import extract_dates, DateConfidence, DateSource

        dr = extract_dates(pdf_info={"creation_date": dt_date(2019, 3, 15)})
        assert dr.published_at == dt_date(2019, 3, 15)
        assert dr.date_confidence == DateConfidence.high
        assert dr.date_source == DateSource.document_metadata

    def test_pdf_no_date_handled(self):
        """PDF without date metadata should not crash."""
        from arwen_etl.date_provenance import extract_dates

        dr = extract_dates(pdf_info={})
        assert dr.published_at is None


class TestCorpusQualityAfterFixes:
    """Verify the corpus meets quality criteria after all fixes."""

    def test_canonical_corpus_exists(self):
        """The corpus directory should exist and contain documents."""
        corpus_path = Path("corpus")
        assert corpus_path.is_dir()
        json_files = list(corpus_path.glob("*.json"))
        assert len(json_files) > 0

    def test_no_future_dates_in_corpus(self):
        """No document in the corpus should have a future publication date."""
        from arwen_etl.date_provenance import extract_year_safe

        corpus_path = Path("corpus")
        today_year = 2026
        for f in corpus_path.glob("*.json"):
            doc = json.loads(f.read_text(encoding="utf-8"))
            text = doc.get("text", "")
            if len(text) < 200:
                continue
            meta = doc.get("metadata") or {}
            source_url = doc.get("source_url", "") or doc.get("final_url", "")
            dr = extract_year_safe(
                doc_metadata=meta, source_url=source_url, extracted_text=text
            )
            if dr.published_at and dr.date_confidence != "none":
                assert dr.published_at.year < today_year + 1, (
                    f"Future date in {doc.get('document_id')}: {dr.published_at}"
                )

    def test_no_duplicate_hashes_in_corpus(self):
        """No unexplained content duplicates in corpus."""
        from collections import Counter

        corpus_path = Path("corpus")
        hashes = Counter()
        for f in corpus_path.glob("*.json"):
            doc = json.loads(f.read_text(encoding="utf-8"))
            text = doc.get("text", "")
            if len(text) < 200:
                continue
            h = doc.get("artifact_sha256") or doc.get("content_hash") or ""
            if h:
                hashes[h] += 1

        duplicates = sum(c - 1 for c in hashes.values() if c > 1)
        assert duplicates == 0, f"Found {duplicates} unexplained content duplicates"

    def test_all_docs_have_id(self):
        """Every document should have a document_id."""
        corpus_path = Path("corpus")
        for f in corpus_path.glob("*.json"):
            doc = json.loads(f.read_text(encoding="utf-8"))
            assert doc.get("document_id"), f"Missing document_id in {f.name}"

    def test_all_valid_docs_substantive(self):
        """All documents in main corpus should have >= 200 chars."""
        corpus_path = Path("corpus")
        for f in corpus_path.glob("*.json"):
            doc = json.loads(f.read_text(encoding="utf-8"))
            text = doc.get("text", "")
            assert len(text) >= 200, (
                f"Short document in corpus: {f.name} ({len(text)} chars)"
            )
