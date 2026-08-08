from __future__ import annotations

from .models import Perspective


def validate_perspective(perspective: Perspective) -> list[str]:
    errors: list[str] = []
    if not perspective.stakeholder_group.strip():
        errors.append("stakeholder_group is required")
    if not perspective.position.strip():
        errors.append("position is required")
    if perspective.confidence < 0 or perspective.confidence > 1:
        errors.append("confidence must be between 0 and 1")
    if perspective.is_official_position and not perspective.attribution:
        errors.append("official positions require explicit attribution")
    return errors
