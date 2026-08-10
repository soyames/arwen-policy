# Arwen Policy

**Evidence-grounded, multistakeholder AI deliberation for digital policy and Internet governance.**

Arwen Policy is a provenance-preserving data-engineering pipeline that builds the [Arwen Policy Corpus](https://huggingface.co/datasets/soyames/arwen-policy-corpus) — a multimodal dataset of policy documents from Internet governance sources (ICANN, IETF, ITU, OECD, UN agencies, ISOC, and others). The project supports AI-assisted policy analysis that preserves stakeholder disagreement, attributes claims to sources, and discloses missing perspectives.

## What the Project Produces

| Asset | Description |
|---|---|
| [**Arwen Policy Corpus**](https://huggingface.co/datasets/soyames/arwen-policy-corpus) | Canonical dataset of extracted and normalised policy documents with provenance metadata |
| [**Arwen Policy Base**](https://huggingface.co/soyames/arwen-policy-base) | Base language model configured for evidence-grounded policy analysis |
| [**Arwen Policy LoRA**](https://huggingface.co/soyames/arwen-policy-lora) | Task-adapted LoRA weights for policy deliberation |
| [**Arwen Policy Space**](https://huggingface.co/spaces/soyames/arwen-policy) | Interactive demo: policy question → evidence retrieval → synthesis |

## Research Purpose

The project enables:

- Evidence-grounded retrieval from real policy sources
- Stakeholder-aware analysis of policy positions
- Argument mining and position extraction
- Multistakeholder deliberation without manufacturing consensus
- Reproducible, provenance-preserving corpus construction

The system records documented perspectives with attribution, provenance, evidence, and uncertainty — rather than making definitive policy assertions.

## Installation

```bash
git clone https://github.com/soyames/arwen-policy.git
cd arwen-policy
uv sync
```

Optional extras: `uv sync --extra pdf --extra ocr --extra audio --extra video --extra semantic --extra hf`

Or for development:

```bash
uv sync --extra dev
```

## Quick Start

```bash
uv run arwen-etl validate-config
uv run arwen-etl ingest-url https://www.icann.org/resources/pages/governance/bylaws-en
uv run arwen-etl discover
uv run arwen-etl build-release --version 0.1.0
```

## Design Principles

- **Provenance first**: every derived record traceable to a source artifact
- **Multistakeholder by construction**: explicit modelling of stakeholder representation
- **Absence is not neutrality**: missing perspectives are recorded without interpretation
- **Candidate is not verified fact**: automated extraction produces candidates requiring validation
- **Reproducible releases**: versioned metadata and processing history

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
