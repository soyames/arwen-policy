from __future__ import annotations

import math
import re
from uuid import uuid4

from .models import Segment


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def _estimate_reading_time(text: str, words_per_min: int = 200) -> int:
    words = len(text.split())
    return max(1, math.ceil(words / words_per_min))


def segment_text(
    text: str,
    document_id: str,
    *,
    target_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[Segment]:
    """Produce segments that prefer paragraph boundaries and include simple metadata.

    Each Segment contains `text`, `char_start`, `char_end`, `ordinal` and we set
    `metadata` fields like `reading_time_minutes` when available.
    """
    if not text.strip():
        return []

    segments: list[Segment] = []
    ordinal = 0
    pos = 0

    paragraphs = _split_paragraphs(text)

    buffer = ""
    buffer_start = 0

    for para in paragraphs:
        if not buffer:
            buffer_start = text.find(para, pos)
        buffer += ("\n\n" if buffer else "") + para
        if len(buffer) >= target_chars:
            start = buffer_start
            end = start + len(buffer)
            chunk = text[start:end].strip()

            if chunk:
                rel_index = text[start:end].find(chunk) if chunk in text[start:end] else 0
                actual_start = start + rel_index
                actual_end = actual_start + len(chunk)
                segments.append(
                    Segment(
                        segment_id=str(uuid4()),
                        document_id=document_id,
                        ordinal=ordinal,
                        text=chunk,
                        char_start=actual_start,
                        char_end=actual_end,
                        attributes={
                            "reading_time_minutes": _estimate_reading_time(chunk),
                            "word_count": len(chunk.split()),
                        },
                    )
                )
                ordinal += 1

            # reset buffer
            pos = end
            buffer = ""

    # flush remainder
    if buffer:
        start = text.find(buffer)
        end = start + len(buffer)
        chunk = text[start:end].strip()
        if chunk:
            actual_start = start + (text[start:end].find(chunk) if chunk in text[start:end] else 0)
            actual_end = actual_start + len(chunk)
            segments.append(
                Segment(
                    segment_id=str(uuid4()),
                    document_id=document_id,
                    ordinal=ordinal,
                    text=chunk,
                    char_start=actual_start,
                    char_end=actual_end,
                    attributes={
                        "reading_time_minutes": _estimate_reading_time(chunk),
                        "word_count": len(chunk.split()),
                    },
                )
            )

    return segments
