from __future__ import annotations


def validate_training_example(example: dict) -> list[str]:
    errors: list[str] = []
    if not example.get("messages"):
        errors.append("messages is required")
    if not example.get("evidence"):
        errors.append("evidence is required")
    if not example.get("stakeholder_perspectives"):
        errors.append("stakeholder_perspectives is required")
    return errors
