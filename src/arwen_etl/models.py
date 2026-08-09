from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_url: str
    final_url: str | None = None
    source_family: str | None = None
    source_adapter: str | None = None
    source_status: str | None = None
    title: str | None = None
    publisher: str | None = None
    content_type: str | None = None
    media_type: str | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    published_at: datetime | None = None
    artifact_sha256: str
    byte_size: int
    license: str | None = None
    access_conditions: str | None = None
    extraction_status: str = "not_extracted"


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: str
    document_id: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    language: str = "und"
    attributes: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    candidate_type: str
    segment_id: str
    text: str
    extraction_method: str
    human_verified: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_linked: bool = False
    evidence_confidence: float | None = None
    evidence_required: bool = True
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """Normalized document after extraction and processing."""
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_id: str
    canonical_url: str | None = None
    retrieved_at: datetime | None = None
    http_status: int | None = None
    content_type: str | None = None
    content_hash: str | None = None
    extraction_status: str = "not_extracted"
    text: str = ""
    title: str | None = None
    language: str = "und"
    jurisdiction: str | None = None
    policy_topics: list[str] = Field(default_factory=list)
    license: str | None = None
    access_conditions: str | None = None
    discovery_urls: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance_events: list[Any] = Field(default_factory=list)