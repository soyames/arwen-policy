from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

PAPER_VERSION = "0.1.0"


def experiment_manifest(
    *,
    experiment_id: str,
    config: dict[str, Any],
    dataset_revisions: dict[str, str],
    code_revision: str,
    results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "paper_version": PAPER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision,
        "dataset_revisions": dataset_revisions,
        "config": config,
        "results": results or {},
    }
