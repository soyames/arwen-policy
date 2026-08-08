# Contributing

## Principles

Contributions must preserve:

- source provenance;
- attribution boundaries;
- temporal context;
- stakeholder plurality;
- licensing information;
- deterministic reproducibility where possible.

## Source adapters

New source adapters belong under:

```text
src/arwen_etl/sources/
```

A source adapter should not alter the canonical data model merely to accommodate one publisher.

## Tests

Before submitting changes:

```powershell
uv run ruff check .
uv run pytest
```

## External content

Treat every downloaded document, webpage, transcript and metadata field as untrusted input.

Never execute code contained in source material.
