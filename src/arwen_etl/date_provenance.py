"""Date extraction with provenance tracking.

Distinguishes multiple date types and records confidence/evidence
so that weak signals (URL years, copyright notices, future-plan
references) are never silently treated as verified publication dates.

Core principles:
- A publication date is only assigned when supported by reliable evidence.
- URL-based year extraction is recorded as a weak fallback, not a verified date.
- Future dates (after the current year) are rejected as publication dates.
- Every assigned date tracks its source and confidence level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class DateSource(str, Enum):
    """Where a date value was found."""
    structured_meta = "structured_meta"        # article:published_time, dc.date
    document_metadata = "document_metadata"    # meta tags, PDF info
    page_content = "page_content"              # explicit date in visible text
    http_header = "http_header"               # Last-Modified, etc.
    url_pattern = "url_pattern"               # /YYYY/MM/DD in URL
    url_year_only = "url_year_only"           # bare /YYYY/ in URL (weak)
    archive_context = "archive_context"        # from archive/listing page
    copyright_notice = "copyright_notice"      # copyright line in footer
    event_reference = "event_reference"        # refers to an event date
    inferred = "inferred"                     # best-guess heuristics
    unknown = "unknown"


class DateConfidence(str, Enum):
    """How trustworthy the date assignment is."""
    high = "high"        # verified structured metadata
    medium = "medium"    # good evidence but not authoritative
    low = "low"          # weak/inferred evidence
    none = "none"        # no date evidence at all


class DateType(str, Enum):
    """What kind of date this represents."""
    published_at = "published_at"        # actual publication/release date
    event_date = "event_date"           # date of a meeting/conference
    updated_at = "updated_at"           # last modification date
    retrieved_at = "retrieved_at"       # when the document was captured
    archive_year = "archive_year"       # year from an archive listing
    url_year = "url_year"               # year extracted from URL
    copyright_year = "copyright_year"   # year from copyright notice
    study_period = "study_period"       # a future study/planning period
    unknown = "unknown"


# Earliest and latest acceptable publication years.
EARLIEST_YEAR = 1990
# Reject publication dates after this year (inclusive boundary for rejection)
FUTURE_REJECTION_YEAR = date.today().year + 1  # 2027 for today=2026


@dataclass
class DateEvidence:
    """A single piece of date evidence with provenance."""
    date_value: date | None = None
    date_type: DateType = DateType.unknown
    source: DateSource = DateSource.unknown
    confidence: DateConfidence = DateConfidence.none
    raw_text: str | None = None
    notes: str = ""


@dataclass
class DateRecord:
    """Complete date information for a document, with provenance."""
    published_at: date | None = None
    date_source: DateSource = DateSource.unknown
    date_confidence: DateConfidence = DateConfidence.none
    date_type: DateType = DateType.unknown
    # All evidence found during extraction
    evidence: list[DateEvidence] = field(default_factory=list)
    # Explanation when no date could be assigned
    no_date_reason: str = ""

    @property
    def year(self) -> int | None:
        if self.published_at:
            return self.published_at.year
        return None

    @property
    def is_verified(self) -> bool:
        """True if the publication date has high or medium confidence."""
        return (
            self.published_at is not None
            and self.date_confidence in (DateConfidence.high, DateConfidence.medium)
        )

    @property
    def is_weak(self) -> bool:
        """True if the date is only weakly supported."""
        return self.date_confidence == DateConfidence.low

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "date_source": self.date_source.value,
            "date_confidence": self.date_confidence.value,
            "date_type": self.date_type.value,
        }
        if self.no_date_reason:
            result["no_date_reason"] = self.no_date_reason
        if self.evidence:
            result["evidence"] = [
                {
                    "date_value": e.date_value.isoformat() if e.date_value else None,
                    "date_type": e.date_type.value,
                    "source": e.source.value,
                    "confidence": e.confidence.value,
                    "raw_text": e.raw_text,
                    "notes": e.notes,
                }
                for e in self.evidence
            ]
        return result


def _parse_date(text: str) -> date | None:
    """Parse a date string into a date object. Returns None on failure."""
    text = text.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%B %Y", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(
                text[:len("January 2000")] if fmt == "%B %Y" else text[:10],
                fmt,
            )
            return dt.date()
        except (ValueError, IndexError):
            continue
    # ISO format
    try:
        dt_str = text.replace("Z", "+00:00")
        return datetime.fromisoformat(dt_str).date()
    except (ValueError, Exception):
        pass
    return None


def _year_only_date(year: int) -> date:
    """Create a date object for a year-only value (use July 1 as mid-year)."""
    return date(year, 7, 1)


def _is_future(d: date) -> bool:
    """Check if a date is unacceptably far in the future for publication."""
    return d.year >= FUTURE_REJECTION_YEAR


def _is_historic(d: date) -> bool:
    """Check if a date is before our collection boundary."""
    return d.year < EARLIEST_YEAR


def extract_dates(
    html_bytes: bytes | None = None,
    pdf_info: dict[str, Any] | None = None,
    source_url: str = "",
    extracted_text: str = "",
    http_headers: dict[str, str] | None = None,
) -> DateRecord:
    """Extract dates from all available sources with provenance.

    Priority order:
    1. Structured metadata (article:published_time, dc.date) — high confidence
    2. PDF metadata (creation_date) — high confidence
    3. HTTP headers (Last-Modified) — medium confidence
    4. Explicit publication date in document body — medium confidence
    5. URL pattern with full YYYY/MM/DD — low/medium confidence
    6. URL year only — low confidence (weak fallback)

    Future dates are NEVER accepted as publication dates.
    Study period references are tracked but not treated as publication dates.
    """
    record = DateRecord()
    evidence_list: list[DateEvidence] = []

    # 1. HTML structured metadata (highest priority)
    if html_bytes:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_bytes, "html.parser")
            meta_selectors = [
                ("article:published_time", "property"),
                ("article:published_time", "name"),
                ("og:article:published_time", "property"),
                ("date", "name"),
                ("dc.date", "name"),
                ("dc:date", "name"),
                ("citation_date", "name"),
                ("citation_publication_date", "name"),
                ("prism.publicationDate", "name"),
            ]
            for attr_val, attr_name in meta_selectors:
                tag = soup.find("meta", attrs={attr_name: attr_val})
                if tag and tag.get("content"):
                    d = _parse_date(tag["content"].strip())
                    if d and not _is_future(d):
                        evidence_list.append(DateEvidence(
                            date_value=d,
                            date_type=DateType.published_at,
                            source=DateSource.structured_meta,
                            confidence=DateConfidence.high,
                            raw_text=tag["content"].strip(),
                        ))
                        record.published_at = d
                        record.date_source = DateSource.structured_meta
                        record.date_confidence = DateConfidence.high
                        record.date_type = DateType.published_at
                        record.evidence = evidence_list
                        return record
        except Exception:
            pass

    # 2. PDF metadata
    if pdf_info:
        creation = pdf_info.get("creation_date") or pdf_info.get("CreationDate")
        if creation:
            d = None
            if isinstance(creation, datetime):
                d = creation.date()
            elif isinstance(creation, date):
                d = creation
            elif isinstance(creation, str):
                d = _parse_date(creation)
            if d and not _is_future(d) and not _is_historic(d):
                evidence_list.append(DateEvidence(
                    date_value=d,
                    date_type=DateType.published_at,
                    source=DateSource.document_metadata,
                    confidence=DateConfidence.high,
                    raw_text=str(creation),
                ))
                record.published_at = d
                record.date_source = DateSource.document_metadata
                record.date_confidence = DateConfidence.high
                record.date_type = DateType.published_at
                record.evidence = evidence_list
                return record

    # 3. HTTP headers
    if http_headers:
        last_modified = http_headers.get("last-modified", "")
        if last_modified:
            try:
                from email.utils import parsedate_to_datetime
                d = parsedate_to_datetime(last_modified).date()
                if not _is_future(d):
                    evidence_list.append(DateEvidence(
                        date_value=d,
                        date_type=DateType.updated_at,
                        source=DateSource.http_header,
                        confidence=DateConfidence.medium,
                        raw_text=last_modified,
                        notes="HTTP Last-Modified header",
                    ))
            except Exception:
                pass

    # 4. Explicit publication date in document body
    # Look for clear date labels near the beginning of the text
    if extracted_text and not record.published_at:
        text_head = extracted_text[:3000]
        pub_patterns = [
            r'(?:published|posted|released|issued)(?:\s+(?:on|date))?\s*:?\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(?:published|posted|released|issued)(?:\s+(?:on|date))?\s*:?\s*(\d{4}-\d{2}-\d{2})',
            r'(?:Date|DATE)\s*:\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        ]
        for pattern in pub_patterns:
            match = re.search(pattern, text_head, re.IGNORECASE)
            if match:
                d = _parse_date(match.group(1))
                if d and not _is_future(d) and not _is_historic(d):
                    evidence_list.append(DateEvidence(
                        date_value=d,
                        date_type=DateType.published_at,
                        source=DateSource.page_content,
                        confidence=DateConfidence.medium,
                        raw_text=match.group(0),
                    ))
                    record.published_at = d
                    record.date_source = DateSource.page_content
                    record.date_confidence = DateConfidence.medium
                    record.date_type = DateType.published_at
                    record.evidence = evidence_list
                    return record

    # 5. URL patterns — full date
    if source_url and not record.published_at:
        # YYYY/MM/DD or YYYY-MM-DD in URL
        url_match = re.search(
            r'/((?:19[9]\d|20[0-2]\d))[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])',
            source_url,
        )
        if url_match:
            try:
                y, m, d = int(url_match.group(1)), int(url_match.group(2)), int(url_match.group(3))
                parsed = date(y, m, d)
                if not _is_future(parsed):
                    evidence_list.append(DateEvidence(
                        date_value=parsed,
                        date_type=DateType.url_year,
                        source=DateSource.url_pattern,
                        confidence=DateConfidence.medium,
                        raw_text=url_match.group(0),
                        notes="Full date pattern in URL",
                    ))
                    record.published_at = parsed
                    record.date_source = DateSource.url_pattern
                    record.date_confidence = DateConfidence.medium
                    record.date_type = DateType.url_year
                    record.evidence = evidence_list
                    return record
            except (ValueError, IndexError):
                pass

    # 6. URL year only — WEAK fallback
    if source_url and not record.published_at:
        # Look for a standalone year in URL paths: /2024/ or -2024- or rfcNNNN
        # But NOT for ranges like 2025-2028 or study-period indicators
        url_without_ranges = re.sub(r'\d{4}-\d{4}', '', source_url)
        year_match = re.search(
            r'(?:/|-)((?:199\d|20[0-2]\d))(?:/|-|\.|$|_)',
            url_without_ranges,
        )
        if year_match:
            y = int(year_match.group(1))
            if EARLIEST_YEAR <= y < FUTURE_REJECTION_YEAR:
                d = _year_only_date(y)
                evidence_list.append(DateEvidence(
                    date_value=d,
                    date_type=DateType.url_year,
                    source=DateSource.url_year_only,
                    confidence=DateConfidence.low,
                    raw_text=year_match.group(0),
                    notes="Year pattern in URL — weak evidence, not a verified publication date",
                ))
                # Record as LOW confidence — do NOT treat as verified
                record.published_at = d
                record.date_source = DateSource.url_year_only
                record.date_confidence = DateConfidence.low
                record.date_type = DateType.url_year
                record.evidence = evidence_list
                return record

    # 7. Look for reference to a year in the first portion of text
    # BUT: exclude study periods, copyrights, footers, future events
    if extracted_text and not record.published_at:
        text_head = extracted_text[:2000]
        # Find explicit publication mentions
        pub_year_match = re.search(
            r'(?:^|\n)(?:Published|Posted|Released|Issued)\s+(?:in\s+)?(\d{4})',
            text_head,
            re.IGNORECASE,
        )
        if pub_year_match:
            y = int(pub_year_match.group(1))
            if EARLIEST_YEAR <= y < FUTURE_REJECTION_YEAR:
                d = _year_only_date(y)
                evidence_list.append(DateEvidence(
                    date_value=d,
                    date_type=DateType.published_at,
                    source=DateSource.page_content,
                    confidence=DateConfidence.medium,
                    raw_text=pub_year_match.group(0),
                ))
                record.published_at = d
                record.date_source = DateSource.page_content
                record.date_confidence = DateConfidence.medium
                record.date_type = DateType.published_at
                record.evidence = evidence_list
                return record

    # If we reach here, no reliable date was found.
    if not record.published_at:
        record.no_date_reason = (
            "No structured metadata, no explicit publication date in content, "
            "and no reliable date pattern in URL."
        )

    record.evidence = evidence_list
    return record


def extract_year_safe(
    doc_metadata: dict[str, Any] | None = None,
    source_url: str = "",
    extracted_text: str = "",
) -> DateRecord:
    """Convenience function that works with the existing document model.

    Returns a DateRecord with provenance, never silently treating a URL year
    as a verified publication date.
    """
    if doc_metadata is None:
        doc_metadata = {}

    record = DateRecord()
    evidence_list: list[DateEvidence] = []

    # 1. Check structured published_at in metadata
    pub = doc_metadata.get("published_at")
    if pub:
        d = None
        raw = str(pub)
        if isinstance(pub, datetime):
            d = pub.date()
        elif isinstance(pub, date):
            d = pub
        elif isinstance(pub, str) and len(pub) >= 4:
            d = _parse_date(pub)
        if d and not _is_future(d) and not _is_historic(d):
            evidence_list.append(DateEvidence(
                date_value=d,
                date_type=DateType.published_at,
                source=DateSource.document_metadata,
                confidence=DateConfidence.high,
                raw_text=raw,
            ))
            record.published_at = d
            record.date_source = DateSource.document_metadata
            record.date_confidence = DateConfidence.high
            record.date_type = DateType.published_at
            record.evidence = evidence_list
            return record

    # 2. Check title for year pattern (e.g., "IGF 2024 Report")
    # IMPORTANT: Do NOT extract dates from RFC numbers like "RFC 2026"
    title = doc_metadata.get("title", "")
    if title and not re.search(r'\bRFC\s*\d{4}\b', str(title), re.IGNORECASE):
        year_match = re.search(r'\b((?:199[0-9]|20[0-2]\d))\b', str(title))
        if year_match:
            y = int(year_match.group(1))
            # Skip if the year is part of an identifier (like arxiv ID, RFC number, etc.)
            # and the title has another indicator it's an identifier
            context = str(title)
            if not re.search(r'\b(?:RFC|arxiv|arXiv|doc\s*id)\s*' + str(y), context, re.IGNORECASE):
                if not (FUTURE_REJECTION_YEAR <= y):
                    d = _year_only_date(y)
                    evidence_list.append(DateEvidence(
                        date_value=d,
                        date_type=DateType.published_at,
                        source=DateSource.document_metadata,
                        confidence=DateConfidence.medium,
                        raw_text=title,
                        notes=f"Year found in title: '{title[:100]}'",
                    ))
                    record.published_at = d
                    record.date_source = DateSource.document_metadata
                    record.date_confidence = DateConfidence.medium
                    record.date_type = DateType.published_at
                    record.evidence = evidence_list
                    return record

    # 3. URL patterns — with provenance
    if source_url:
        # Full date in URL: YYYY/MM/DD or YYYY-MM-DD
        url_match = re.search(
            r'/((?:19[9]\d|20[0-2]\d))[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])',
            source_url,
        )
        if url_match:
            try:
                y, m, d = int(url_match.group(1)), int(url_match.group(2)), int(url_match.group(3))
                parsed = date(y, m, d)
                if not _is_future(parsed):
                    evidence_list.append(DateEvidence(
                        date_value=parsed,
                        date_type=DateType.url_year,
                        source=DateSource.url_pattern,
                        confidence=DateConfidence.medium,
                        raw_text=url_match.group(0),
                        notes="Full date pattern in URL",
                    ))
                    record.published_at = parsed
                    record.date_source = DateSource.url_pattern
                    record.date_confidence = DateConfidence.medium
                    record.date_type = DateType.url_year
                    record.evidence = evidence_list
                    return record
            except (ValueError, IndexError):
                pass

        # Year only in URL — LOW confidence
        url_without_ranges = re.sub(r'\d{4}-\d{4}', '', source_url)
        year_match = re.search(
            r'(?:/|-)((?:199\d|20[0-2]\d))(?:/|-|\.|$|_)',
            url_without_ranges,
        )
        if year_match:
            y = int(year_match.group(1))
            if EARLIEST_YEAR <= y < FUTURE_REJECTION_YEAR:
                d = _year_only_date(y)
                evidence_list.append(DateEvidence(
                    date_value=d,
                    date_type=DateType.url_year,
                    source=DateSource.url_year_only,
                    confidence=DateConfidence.low,
                    raw_text=year_match.group(0),
                    notes="Year pattern in URL — weak evidence, not a verified publication date",
                ))
                record.published_at = d
                record.date_source = DateSource.url_year_only
                record.date_confidence = DateConfidence.low
                record.date_type = DateType.url_year
                record.evidence = evidence_list
                return record

    # 4. RFC-style date pattern in text: "Month YYYY" near "Request for Comments:"
    if extracted_text:
        text_head = extracted_text[:2000]
        rfc_date_match = re.search(
            r'Request\s+for\s+Comments\s*:\s*\d+.*?((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            text_head,
            re.IGNORECASE | re.DOTALL,
        )
        if rfc_date_match:
            d = _parse_date(rfc_date_match.group(1))
            if d and not _is_future(d) and not _is_historic(d):
                evidence_list.append(DateEvidence(
                    date_value=d,
                    date_type=DateType.published_at,
                    source=DateSource.page_content,
                    confidence=DateConfidence.high,
                    raw_text=rfc_date_match.group(0)[:200],
                    notes="RFC publication date from document header",
                ))
                record.published_at = d
                record.date_source = DateSource.page_content
                record.date_confidence = DateConfidence.high
                record.date_type = DateType.published_at
                record.evidence = evidence_list
                return record

    # 5. Generic "Month YYYY" date near document start (common in policy docs)
    if extracted_text:
        text_head = extracted_text[:1500]
        month_year_match = re.search(
            r'(?:^|\n)\s*(?:Published|Issued|Released|Approved|Adopted|Revised|Updated|Created|Date)[\s:]*(?:[\s\w,]*?)?'
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            text_head,
            re.IGNORECASE,
        )
        if not month_year_match:
            # Fallback: standalone "Month YYYY" near beginning (RFC style)
            month_year_match = re.search(
                r'(?:^|\n)[^\n]{0,100}?((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})(?:[^\w]|$)',
                text_head,
                re.IGNORECASE,
            )
        if month_year_match:
            d = _parse_date(month_year_match.group(1))
            if d and not _is_future(d) and not _is_historic(d):
                # Only accept if at least 150 chars into the document (past navigation)
                # and not in a footer/copyright context
                match_pos = month_year_match.start()
                context_before = text_head[max(0, match_pos - 50):match_pos].lower()
                if not any(w in context_before for w in ('copyright', '©', 'all rights reserved')):
                    confidence = DateConfidence.medium
                    evidence_list.append(DateEvidence(
                        date_value=d,
                        date_type=DateType.published_at,
                        source=DateSource.page_content,
                        confidence=confidence,
                        raw_text=month_year_match.group(0),
                        notes="Month Year date pattern in document text",
                    ))
                    record.published_at = d
                    record.date_source = DateSource.page_content
                    record.date_confidence = confidence
                    record.date_type = DateType.published_at
                    record.evidence = evidence_list
                    return record

    # 6. Text content — explicit publication mentions only
    if extracted_text:
        text_head = extracted_text[:2000]
        pub_year_match = re.search(
            r'(?:^|\n)(?:Published|Posted|Released|Issued)\s+(?:in\s+)?(\d{4})',
            text_head,
            re.IGNORECASE,
        )
        if pub_year_match:
            y = int(pub_year_match.group(1))
            if EARLIEST_YEAR <= y < FUTURE_REJECTION_YEAR:
                d = _year_only_date(y)
                evidence_list.append(DateEvidence(
                    date_value=d,
                    date_type=DateType.published_at,
                    source=DateSource.page_content,
                    confidence=DateConfidence.medium,
                    raw_text=pub_year_match.group(0),
                ))
                record.published_at = d
                record.date_source = DateSource.page_content
                record.date_confidence = DateConfidence.medium
                record.date_type = DateType.published_at
                record.evidence = evidence_list
                return record

    if not record.published_at:
        record.no_date_reason = "No reliable date evidence found."

    record.evidence = evidence_list
    return record
