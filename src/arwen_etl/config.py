from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    return load_yaml(path)


def load_sources_config(path: str | Path = "configs/sources.yaml") -> dict[str, Any]:
    return load_yaml(path)
