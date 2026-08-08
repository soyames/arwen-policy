from __future__ import annotations

from .models import Perspective


def evidence_coverage(perspectives: list[Perspective]) -> dict[str, int]:
    """Return evidence counts by stakeholder group without interpreting silence."""
    return {
        group: sum(len(p.evidence_record_ids) for p in perspectives if p.stakeholder_group == group)
        for group in sorted({p.stakeholder_group for p in perspectives})
    }
