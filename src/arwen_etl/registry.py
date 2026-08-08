from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from .config import load_sources_config


class SourceDefinition(BaseModel):
    id: str
    name: str
    family: str
    domains: list[str]
    enabled: bool = True
    default_language: str | None = None
    allowed_url_patterns: list[str] | None = None
    robots: dict[str, Any] | None = None
    rate_limit: dict[str, Any] | None = None
    discovery: dict[str, Any] | None = None

    @field_validator("domains", mode="after")
    def domain_must_be_nonempty(cls, v: list[str]) -> list[str]:
        for item in v:
            if not item or "/" in item:
                raise ValueError("Invalid domain")
        return v


class SourceRegistry(BaseModel):
    sources: list[SourceDefinition]

    @classmethod
    def load(cls, path: str | Path = "configs/sources.yaml") -> SourceRegistry:
        data = load_sources_config(path)
        try:
            return cls(**data)
        except ValidationError:
            raise

    def find_by_domain(self, domain: str) -> SourceDefinition | None:
        domain = domain.lower()
        for s in self.sources:
            if any(d.lower() == domain for d in s.domains):
                return s
        return None


def load_registry(path: str | Path = "configs/sources.yaml") -> SourceRegistry:
    return SourceRegistry.load(path)
