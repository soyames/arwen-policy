# Arwen Policy Monorepo Architecture

Arwen Policy is maintained as one GitHub repository and a set of versioned Hugging Face artifacts.

## Layers

```text
Public sources
    -> arwen_etl
    -> Hugging Face arwen-policy-corpus
    -> retrieval
    -> deliberation
    -> engine
    -> benchmark / training
    -> model and public Space
```

The existing `src/arwen_etl` package remains the ingestion system. New packages live beside it under `src/` and communicate through explicit data contracts rather than importing internal implementation details from each other.

## Artifact boundary

GitHub contains code, tests, configurations, documentation and reproducibility material. Hugging Face contains released datasets, models and Spaces.

## Trust boundary

Every evidence-bearing object must retain source identifiers. Deliberation may summarize perspectives, but it must not erase attribution, disagreement or missing evidence.
