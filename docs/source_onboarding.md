# Source Onboarding

New source families should be added without changing the canonical record model.

Document:

1. organization/source identity;
2. domains;
3. discovery strategy;
4. public access conditions;
5. document formats;
6. publication metadata;
7. extraction requirements;
8. attribution rules;
9. rate limits and robots policy;
10. provenance requirements.

Source-specific code belongs under `src/arwen_etl/sources/`.

## Adapter contract

All adapters must:

* go through ETL capture policy (`capture_url`, robots, size/timeout limits);
* emit `SourceDiscoveryBundle` with a `SourceHealthRecord`;
* produce the same canonical `SourceRecord` fields used by the core pipeline;
* never invent organizational positions or stakeholder consensus.

The default implementation is `GenericSourceAdapter` (`generic-web-adapter`).
Source-specific adapters should subclass or wrap it only when discovery or
metadata extraction genuinely differs by publisher.

## Discovery configuration

Configure discovery under each entry in `configs/sources.yaml`:

```yaml
discovery:
  seed_urls:
    - https://example.org/
  sitemap_urls:
    - https://example.org/sitemap.xml
  feed_urls:
    - https://example.org/feed
  paths:
    - resources
```

If no discovery keys are set, the generic adapter probes the domain root plus
common sitemap/feed paths.

## Source health artifacts

`arwen-etl discover` and `arwen-etl pipeline` write:

| Path | Purpose |
| --- | --- |
| `data/discovered/<source_id>.json` | Discovered URL list |
| `data/source-health/<source_id>.json` | Per-source health record |
| `data/source-health/index.json` | Aggregate coverage index |

Health statuses include `reachable`, `degraded`, `blocked`, and `unreachable`.
