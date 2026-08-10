"""Tests for date provenance extraction and validation."""

import json
from datetime import date
from pathlib import Path

import pytest

from arwen_etl.date_provenance import (
    DateConfidence,
    DateRecord,
    DateSource,
    DateType,
    EARLIEST_YEAR,
    extract_dates,
    extract_year_safe,
)


class TestDateExtraction:
    """Test basic date extraction with provenance."""

    def test_structured_meta_date(self):
        """Article published_time meta tag should produce high-confidence date."""
        html = b"""<html><head>
        <meta property="article:published_time" content="2012-06-15">
        </head><body>Some content</body></html>"""
        dr = extract_dates(html_bytes=html)
        assert dr.published_at == date(2012, 6, 15)
        assert dr.date_confidence == DateConfidence.high
        assert dr.date_source == DateSource.structured_meta
        assert dr.date_type == DateType.published_at

    def test_dc_date_meta(self):
        """DC.date meta tag should produce high-confidence date."""
        html = b"""<html><head>
        <meta name="dc.date" content="2018-03-22">
        </head><body>Content</body></html>"""
        dr = extract_dates(html_bytes=html)
        assert dr.published_at == date(2018, 3, 22)
        assert dr.date_confidence == DateConfidence.high

    def test_pdf_creation_date(self):
        """PDF creation_date metadata should produce high-confidence date."""
        dr = extract_dates(pdf_info={"creation_date": date(2015, 1, 15)})
        assert dr.published_at == date(2015, 1, 15)
        assert dr.date_confidence == DateConfidence.high
        assert dr.date_source == DateSource.document_metadata

    def test_url_full_date(self):
        """Full YYYY/MM/DD in URL should produce medium-confidence date."""
        dr = extract_dates(source_url="https://example.com/2020/03/15/report")
        assert dr.published_at == date(2020, 3, 15)
        assert dr.date_confidence == DateConfidence.medium
        assert dr.date_source == DateSource.url_pattern

    def test_url_year_only_low_confidence(self):
        """Bare year in URL should be LOW confidence only."""
        dr = extract_dates(source_url="https://example.com/events/2024/summary")
        assert dr.published_at is not None
        assert dr.published_at.year == 2024
        assert dr.date_confidence == DateConfidence.low
        assert dr.date_source == DateSource.url_year_only

    def test_no_date_returns_none(self):
        """When no evidence exists, no date should be returned."""
        dr = extract_dates(source_url="https://example.com/about")
        assert dr.published_at is None
        assert dr.date_confidence == DateConfidence.none
        assert dr.no_date_reason != ""

    def test_extract_year_safe_with_metadata(self):
        """extract_year_safe should use metadata published_at."""
        meta = {"published_at": "2019-08-01T12:00:00Z"}
        dr = extract_year_safe(doc_metadata=meta)
        assert dr.published_at == date(2019, 8, 1)
        assert dr.date_confidence == DateConfidence.high

    def test_extract_year_safe_from_url(self):
        """extract_year_safe should fall back to URL patterns."""
        dr = extract_year_safe(source_url="https://icann.org/pages/report-2012-02-25-en")
        assert dr.published_at is not None
        if dr.published_at:
            assert dr.published_at.year == 2012


class TestFutureDateRejection:
    """Future dates must NEVER be accepted as publication dates."""

    def test_future_year_in_meta_rejected(self):
        """A meta tag with a year > current+1 should be rejected."""
        html = b"""<html><head>
        <meta property="article:published_time" content="2028-01-15">
        </head><body></body></html>"""
        dr = extract_dates(html_bytes=html)
        # The structured meta extractor checks _is_future, rejects this
        assert dr.published_at is None or dr.published_at.year < 2027

    def test_study_period_not_confused_with_pub_date(self):
        """Study period references (2025-2028) should not become publication dates."""
        dr = extract_dates(
            source_url="https://itu.int/en/study-groups/2025-2028/",
            extracted_text="ITU-T Study Groups (Study Period 2025 - 2028)",
        )
        # The study period range 2025-2028 should not produce a verified date
        if dr.published_at:
            assert dr.published_at.year < 2027
            assert dr.date_confidence == DateConfidence.low

    def test_future_text_year_not_accepted(self):
        """A year 2027+ found in body text should not be accepted."""
        dr = extract_dates(
            extracted_text="Plans for the 2028 conference are underway."
        )
        assert dr.published_at is None
        assert dr.no_date_reason != ""


class TestHistoricalBoundary:
    """1990 is the collection boundary."""

    def test_pre_1990_rejected(self):
        """Dates before 1990 should not be extracted as publication dates."""
        dr = extract_dates(
            source_url="https://example.com/1985/report"
        )
        # No year before 1990 should be returned
        if dr.published_at:
            assert dr.published_at.year >= EARLIEST_YEAR

    def test_early_collection_boundary(self):
        """The EARLIEST_YEAR should be 1990."""
        assert EARLIEST_YEAR == 1990


class TestDateRecordSerialization:
    """DateRecord should serialize correctly."""

    def test_to_dict(self):
        dr = DateRecord(
            published_at=date(2020, 6, 15),
            date_source=DateSource.structured_meta,
            date_confidence=DateConfidence.high,
            date_type=DateType.published_at,
        )
        d = dr.to_dict()
        assert d["published_at"] == "2020-06-15"
        assert d["date_source"] == "structured_meta"
        assert d["date_confidence"] == "high"
        assert d["date_type"] == "published_at"

    def test_null_date_to_dict(self):
        dr = DateRecord(no_date_reason="Not found")
        d = dr.to_dict()
        assert d["published_at"] is None
        assert d["no_date_reason"] == "Not found"


class TestDateConfidenceLevels:
    """Verify the confidence modeling is correct."""

    def test_high_confidence_is_verified(self):
        dr = DateRecord(
            published_at=date(2020, 1, 1),
            date_confidence=DateConfidence.high,
        )
        assert dr.is_verified
        assert not dr.is_weak

    def test_medium_confidence_is_verified(self):
        dr = DateRecord(
            published_at=date(2020, 1, 1),
            date_confidence=DateConfidence.medium,
        )
        assert dr.is_verified
        assert not dr.is_weak

    def test_low_confidence_is_weak(self):
        dr = DateRecord(
            published_at=date(2020, 1, 1),
            date_confidence=DateConfidence.low,
        )
        assert not dr.is_verified
        assert dr.is_weak

    def test_none_confidence_is_neither(self):
        dr = DateRecord(date_confidence=DateConfidence.none)
        assert not dr.is_verified
        assert not dr.is_weak
