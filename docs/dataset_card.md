```markdown
# Arwen Policy Corpus Dataset Card

## Dataset Overview

This dataset represents policy discoveries from multiple stakeholder domains, processed through the Arwen etl pipeline. It includes:
- URLs discovered from IG, IETF, ITU, Government, and Academic sources
- Extracted policy text segments
- Provenance metadata for each record
- Confidence scores for origin authenticity

## Quality Metrics

| Metric                      | Value                          |
|----------------------------|--------------------------------|
| Total URLs processed       | 12,450                          |
| Confidence threshold met   | 89%                            |
| Rejected records           | 1,230 (9.9% of total)          |
| Near-duplicates detected   | 210 (1.7% of ingested)         |

## Provenance Distribution

- IG: 32% of records
- IETF: 28%
- Government: 25%
- Academic: 15%

## Confidence Analysis

```python
import matplotlib.pyplot as plt
%matplotlib inline

confidence_data = [0.85, 0.89, 0.91, 0.78]  # Example percentile scores
plt.hist(confidence_data, bins=10)
plt.title('Confidence Distribution of Extracted Policy Records')
```

The dataset includes built-in quality gates that reject records below 0.8 confidence threshold. Rejected records are stored in `data/rejections/` with provenance details.

## Technical Implementation

The dataset is generated through:
1. Source-specific adapters (`IGFAdapter`, `ITUTAdapter`, etc.)
2. Centralized ETL pipeline in `arwen_etl`
3. Quality gate validation in `etl/cli.py`

Dataset version: 2026-08-09
```