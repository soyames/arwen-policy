from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .storage import read_json, write_json


def _queue_path() -> Path:
    p = Path("data") / "review"
    p.mkdir(parents=True, exist_ok=True)
    return p / "queue.json"


def enqueue_review(document_id: str, reason: str, metadata: dict[str, Any] | None = None) -> dict:
    path = _queue_path()
    try:
        queue = read_json(path) if path.exists() else []
    except Exception:
        queue = []

    item = {
        "review_id": str(uuid.uuid4()),
        "document_id": document_id,
        "reason": reason,
        "metadata": metadata or {},
    }
    queue.append(item)
    write_json(path, queue)
    return item


def list_queue() -> list[dict]:
    path = _queue_path()
    try:
        return read_json(path) if path.exists() else []
    except Exception:
        return []


def pop_review() -> dict | None:
    path = _queue_path()
    q = list_queue()
    if not q:
        return None
    item = q.pop(0)
    write_json(path, q)
    return item
