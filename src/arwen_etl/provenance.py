from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def provenance_event(
    event_type: str,
    *,
    entity_id: str,
    agent: str,
    input_ids: list[str] | None = None,
    attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": str(uuid4()),
        "event_type": event_type,
        "entity_id": entity_id,
        "agent": agent,
        "input_ids": input_ids or [],
        "attributes": attributes or {},
        "timestamp": datetime.now(UTC).isoformat(),
    }


def capture_rejection_event(
    *,
    source_id: str,
    url: str,
    reason: str,
    policy_source: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "url": url,
        "reason": reason,
        "action": "NO_CAPTURE",
    }

    if policy_source:
        attributes["policy_source"] = policy_source

    if details:
        attributes["details"] = details

    return provenance_event(
        "CAPTURE_REJECTED",
        entity_id=source_id,
        agent="arwen-policy-etl",
        attributes=attributes,
    )