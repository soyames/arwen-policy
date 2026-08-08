from __future__ import annotations

from typing import Any


def build_instruction_example(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    perspectives: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """Build a provenance-preserving supervised example; it never fabricates evidence."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer digital-policy questions using the supplied evidence. "
                    "Preserve stakeholder disagreement and disclose missing perspectives."
                ),
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "evidence": evidence,
        "stakeholder_perspectives": perspectives,
    }
