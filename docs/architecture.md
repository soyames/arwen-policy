# Architecture

Arwen Policy ETL is the reproducible data-engineering layer between public policy sources and the Arwen Policy Corpus.

```text
public sources
    ↓
source adapters + discovery
    ↓
source-health records
    ↓
capture (policy-enforced)
    ↓
extraction
    ↓
normalization
    ↓
segmentation
    ↓
candidate extraction
    ↓
quality + validation
    ↓
release
    ↓
arwen-policy-corpus
```

The pipeline distinguishes deterministic processing, AI-assisted processing and human verification.

## Source adapters

Phase 2 introduces a shared adapter layer under `src/arwen_etl/sources/`:

* `GenericSourceAdapter` discovers URLs from configured seeds, sitemaps, and feeds;
* every capture path still uses `capture_url` so robots, redirects, size, and timeout policy apply;
* discovery emits per-source health records and an aggregate health index;
* ingest enriches canonical source records with family, publisher, adapter, and publication metadata when the URL matches the registry.
