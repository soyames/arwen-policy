# Architecture

Arwen Policy ETL is the reproducible data-engineering layer between public policy sources and the Arwen Policy Corpus.

```text
public sources
    ↓
capture
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
