# Changelog

## 0.2.0 — 2026-08-08

Phase 2 foundation: real policy sources and source health.

- generic web source adapter for registry sources
- capture-policy-safe discovery (HTML links, sitemaps, feeds)
- source health records and aggregate health index
- registry-linked source family / publisher / adapter metadata on ingest
- publication-time metadata extraction for HTML (and PDF when available)
- discovery seed configuration for initial policy sources
- dataset card / release layout alignment for Hugging Face corpus

## 0.1.0 — 2026-08-08

Initial engineering foundation.

- deterministic source capture
- SHA-256 content hashing
- MIME and format detection
- HTML and text extraction
- optional PDF/DOCX extraction adapters
- normalization
- segmentation
- candidate generation
- provenance events
- quality scoring
- JSON Schema validation
- release manifests
- source registry
- automated tests
- uv-based development workflow
