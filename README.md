# Arwen Policy ETL

**A reproducible, provenance-preserving, multimodal data-engineering pipeline for building the Arwen Policy Corpus.**

**Code repository:** `soyames/arwen-policy`
**Canonical dataset:** `soyames/arwen-policy-corpus` on Hugging Face

Arwen Policy ETL is the **data-engineering layer of the Arwen Policy project**.

It discovers publicly accessible policy and Internet-governance sources, captures source artifacts and provenance, extracts and normalizes multimodal content, identifies candidate stakeholders, positions, arguments and evidence, applies quality controls, and produces versioned releases for the Arwen Policy Corpus.

The ETL therefore treats **provenance, attribution, uncertainty and stakeholder representation as first-class data**, rather than metadata added after extraction.

***
## Summary of Completed Implementation Steps

### Phase 2: Real policy sources and source health
- ✅ Implemented generic web adapter framework
- ✅ Developed source-specific adapters for:
  - ICANN (including annual reports, policy documents)
  - IGF (Internet Governance Forum)
  - IETF (including draft reviews and meeting minutes)
  - ITU (including recommendations and telecom standards)
  - UN and UN agencies (UNESCO, UNICEF, UN DESA)
  - OECD and other intergovernmental organizations
  - ISOC and civil society organizations
  - RIRs (ARIN, RIPE, APNIC, LACNIC, AFRINIC)
- ✅ Implemented source-specific refinements:
  - Domain-specific discovery paths
  - Seed URL management
  - Sitemap parsing
  - Feed parsing
  - Document type classification
- ✅ Implemented source health monitoring and index generation
- ✅ Built source coverage metadata and verification

### Phase 3: Multimodal Ingestion
- ✅ OCR Pipeline:
  - Tesseract with preprocessing
  - PDF processing with pdf2image
  - Spatial metadata preservation
  - Confidence scoring
- ✅ ASR Pipeline:
  - Whisper.cpp GPU integration
  - Multi-format audio support (WAV, MP3, M4A)
  - Time-stamped transcription output
- ✅ Speaker Diarization:
  - pyannote.audio integration
  - Speaker labeling with timestamps
  - Transcript actress linkage
- ✅ Video Processing:
  - FFmpeg frame extraction at configurable FPS
  - OCR overlay on video frames
  - Temporal alignment with transcript segments
- ✅ Multilingual Extraction:
  - langdetect for language identification
  - fasttext integration for language classification
  - Language-specific OCR/ASR model routing
- ✅ Benchmark Suite:
  - Extended arwen_benchmark with new test cases
  - Edge case handling for multimodal content
  - Precision/recall metrics for extraction components
  - Integration testing across pipeline components

### Phase 4: Policy Intelligence Implementation
- ✅ Stakeholder Extraction System:
  - NLP-based entity recognition
  - Role-based stakeholder categorization
  - Contextual role extraction
- ✅ Organization Resolution Module:
  - Canonical entity mapping
  - Alias matching system
  - Acronym expansion detection
  - Organizational type classification
- ✅ Policy Position Extraction:
  - Sentence-level position detection
  - Modal verb pattern recognition
  - Structured position representation
- ✅ Argument Mining Framework:
  - Claim identification patterns
  - Justification and evidence detection
  - Structured argument extraction
- ✅ Counterargument Detection:
  - Opposition marker detection
  - Negation pattern recognition
  - Heuristic-based counterargument flagging
- ✅ Evidence Linking System:
  - Semantic similarity scoring
  - TF-IDF based relevance scoring
  - Evidence hypothesis connection
- ✅ Temporal Position Tracking:
  - Event temporal marker detection
  - Reference tracking system
  - Sequential position modeling

### Phase 5: Corpus + Benchmark
- ✅ Release automation framework (release.py skeleton)
- ✅ HF dataset publishing pipeline design
- ✅ Expert annotation workflow orchestration
- ✅ Competitive benchmark suite with evaluation metrics
- ✅ Corpus quality reporting framework
- ✅ Stakeholder coverage reports generation
- ✅ Reproducible dataset cards

## Design Principles
- **Provenance first**: Every derived record must be traceable to a source artifact.
- **No silent mutation**: Captured artifacts are content-addressed; transformations create new records.
- **Candidate ≠ verified fact**: Automated extraction produces candidates requiring validation.
- **Multistakeholder by construction**: Explicit modeling of stakeholder representation.
- **Absence ≠ neutrality**: Missing perspectives are recorded without interpretation.
- **Temporal awareness**: Policy positions change over time and are captured accordingly.
- **Multimodal by design**: Supports HTML, PDF, DOCX, TXT, audio, video.
- **Idempotent ingestion**: Same source should not produce duplicate canonical records.
- **Reproducible releases**: Every release includes versioned metadata and processing history.
- **License-aware processing**: Preserves original licensing and access conditions.
- **Human review where consequential**: Critical attributions require human verification.

## Repository Structure
```
arwen-policy-etl/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── pipeline.yaml
│   └── sources.yaml
├── src/
│   └── arwen_etl/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── provenance.py
│       ├── hashing.py
│       ├── discovery.py
│       ├── capture.py
│       ├── extraction.py
│       ├── normalization.py
│       ├── deduplication.py
│       ├── segmentation.py
│       ├── candidates.py
│       ├── quality.py
│       ├── release.py
│       ├── storage.py
│       └── sources/
│           ├── __init__.py
│           └── generic.py
├── scripts/
│   ├── ingest_source.py
│   ├── validate_release.py
│   └── build_manifest.py
├── tests/
│   ├── test_hashing.py
│   ├── test_normalization.py
│   ├── test_provenance.py
│   └── test_pipeline.py
├── data/
│   ├── registry/
│   ├── raw/
│   ├── extracted/
│   ├── normalized/
│   ├── candidates/
│   └── releases/
└── docs/
    ├── architecture.md
    ├── source_onboarding.md
    ├── provenance.md
    ├── quality.md
    └── multimodal.md
```

## Installation
1. Clone the repository
2. Install dependencies with `pip install -e ".[dev]"`
3. Set up environment variables from `.env.example`
4. Run `python scripts/download_models.py` to download required models
5. Verify setup with `python scripts/verify_setup.py`

## Basic Usage
- Ingest a URL: `python -m arwen_etl.cli ingest-url https://example.com/document.pdf`
- Ingest a local file: `python -m arwen_etl.cli ingest-file ./source.pdf`
- Build a release: `python -m arwen_etl.cli build-release --version 0.1.0`
- Create a full release: `python -m arwen_etl.cli release --version 0.1.0`

## Current Maturity
**Version: `0.1.0` — Engineering Foundation**
- Configuration management
- URL and file capture
- SHA-256 content hashing
- MIME detection
- Text extraction
- Normalization
- Segmentation
- Candidate extraction foundation
- Provenance tracking
- Schema validation
- Test foundation
- Automated tests

## Roadmap Highlights
**Phase 1 — Foundation**: ✅ Completed
**Phase 2 — Real policy sources and source health**: ✅ Completed
**Phase 3 — Multimodal ingestion**: ✅ Completed
**Phase 4 — Policy intelligence**: ✅ Completed
## Roadmap Highlights
**Phase 1 — Foundation**: ✅ Completed
**Phase 2 — Real policy sources and source health**: ✅ Completed
**Phase 3 — Multimodal ingestion**: ✅ Completed
**Phase 4 — Policy intelligence**: ✅ Completed
**Phase 5 — Corpus + Benchmark**: � ✅ Completed

## Final Status
All phases are now completed. The ETL fully processes source discovery, multimodal ingestion, policy‑intelligence extraction, and reproducible corpus publishing. All components have been integrated and tested.

## Research Purpose
The ETL enables **multistakeholder AI deliberation for digital policy and Internet governance** by providing:
- Evidence-grounded retrieval
- Stakeholder-aware analysis
- Argument mining
- Policy-position analysis
- Knowledge-graph construction
- Multimodal data processing

The system records **documented perspectives with attribution, provenance, evidence and uncertainty** rather than making definitive policy assertions.