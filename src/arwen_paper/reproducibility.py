from __future__ import annotations

import json
from pathlib import Path

from .manifest import experiment_manifest


def write_experiment_manifest(path: str | Path, **kwargs) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(experiment_manifest(**kwargs), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output
