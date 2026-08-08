# Arwen Policy ETL

**A reproducible, provenance-preserving, multimodal data-engineering pipeline for building the Arwen Policy Corpus.**

**Code repository:** `soyames/arwen-policy`
**Canonical dataset:** `soyames/arwen-policy-corpus` on Hugging Face

Arwen Policy ETL is the **data-engineering layer of the Arwen Policy project**.

It discovers publicly accessible policy and Internet-governance sources, captures source artifacts and provenance, extracts and normalizes multimodal content, identifies candidate stakeholders, positions, arguments and evidence, applies quality controls, and produces versioned releases for the Arwen Policy Corpus.

The ETL system and the corpus are intentionally separate:

> **`arwen-policy-etl` = reproducible pipeline and engineering infrastructure**
> **`arwen-policy-corpus` = canonical research dataset and released data**

The canonical corpus is published on Hugging Face:

https://huggingface.co/datasets/soyames/arwen-policy-corpus

---

## Why this pipeline exists

Arwen Policy is intended to support **multistakeholder AI deliberation for digital policy and Internet governance**.

That requires more than collecting documents.

A policy corpus must preserve:

* where information came from;
* when it was published or captured;
* who made a statement;
* whether a statement represents an individual or an organization;
* which stakeholder perspective is represented;
* what evidence supports a claim;
* how an argument was extracted;
* whether an interpretation was automated or human-verified;
* what information was unavailable;
* and how the final dataset record was derived.

The ETL therefore treats **provenance, attribution, uncertainty and stakeholder representation as first-class data**, rather than metadata added after extraction.

---

# Design principles

### 1. Provenance first

Every derived record must be traceable to a source artifact and, where possible, to a precise source segment.

```text
derived record
    ↓
source segment
    ↓
source artifact
    ↓
capture event
    ↓
source URL / identifier
```

### 2. No silent mutation

Captured source artifacts are content-addressed using cryptographic hashes.

Transformations create new derived records rather than silently modifying the captured source.

### 3. Candidate ≠ verified fact

Automated extraction produces **candidates**.

A model identifying:

> "Organization X supports policy Y"

does not automatically make that statement an authoritative organizational position.

Attribution and interpretation remain subject to provenance and, where required, human review.

### 4. Multistakeholder by construction

The pipeline explicitly models stakeholder representation.

It does not assume that the most frequently published perspective represents consensus.

It records:

* represented stakeholders;
* stakeholder groups;
* documented positions;
* supporting arguments;
* opposing arguments;
* uncertainty;
* missing perspectives;
* temporal changes;
* and unresolved disagreement.

### 5. Absence ≠ neutrality

The absence of a stakeholder perspective from the collected corpus must never be interpreted as:

* neutrality;
* agreement;
* disagreement;
* consent;
* or lack of interest.

The corpus records **what is documented**, not what is assumed.

### 6. Temporal awareness

Policy positions change.

A position is therefore treated as an observation associated with a temporal context rather than as a permanent attribute of a stakeholder.

The pipeline preserves:

* publication date;
* event date;
* capture date;
* effective dates where available;
* supersession relationships;
* and source versions.

### 7. Multimodal by design

The architecture supports:

* HTML
* PDF
* DOCX
* TXT
* structured web content
* scanned documents
* audio
* video

Multimodal processing may involve:

```text
audio → ASR → speaker diarization → segments
video → ASR + visual metadata → segments
PDF → text extraction / OCR → segments
scan → OCR → text → segments
```

### 8. Idempotent ingestion

The same source should not produce duplicate canonical records.

Content identity is based on hashes and stable source identifiers where available.

### 9. Reproducible releases

Every release records:

* source identifiers;
* content hashes;
* extraction methods;
* pipeline version;
* configuration;
* schema versions;
* quality results;
* transformation history;
* and release metadata.

A dataset release should therefore be reproducible and auditable.

### 10. License-aware processing

The ETL preserves source licensing, attribution and access conditions.

The presence of a URL in the corpus does **not** transfer copyright to Arwen.

The pipeline does not attempt to bypass:

* authentication;
* paywalls;
* access controls;
* robots restrictions;
* technical restrictions;
* or terms governing restricted material.

### 11. Human review where consequential

Automated systems may assist with:

* stakeholder detection;
* argument extraction;
* position extraction;
* evidence linking;
* topic classification;
* language identification;
* quality assessment.

They do not automatically establish authoritative stakeholder positions.

---

# Pipeline architecture

```text
                         PUBLIC SOURCES
                              │
             ┌────────────────┼────────────────┐
             │                │                │
            HTML             PDF             DOCX
             │                │                │
           AUDIO            VIDEO            TEXT
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                       SOURCE DISCOVERY
                              │
                              ▼
                    CAPTURE + CONTENT HASH
                              │
                     ┌────────┴────────┐
                     │                 │
                     ▼                 ▼
                RAW ARTIFACT       PROVENANCE
                     │
                     ▼
                  FORMAT
                 DETECTION
                     │
                     ▼
             CONTENT EXTRACTION
                     │
        ┌────────────┼────────────┐
        │            │            │
       OCR          ASR       DOCUMENT PARSING
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
                 NORMALIZATION
                     │
                     ▼
             LANGUAGE + METADATA
                     │
                     ▼
                DEDUPLICATION
                     │
                     ▼
                SEGMENTATION
                     │
          ┌──────────┼───────────┐
          │          │           │
          ▼          ▼           ▼
     STAKEHOLDERS  POSITIONS   ARGUMENTS
          │          │           │
          └──────────┼───────────┘
                     │
                     ▼
                   EVIDENCE
                     │
                     ▼
              PROVENANCE LINKING
                     │
                     ▼
               QUALITY GATES
                     │
                     ▼
             SCHEMA VALIDATION
                     │
                     ▼
               HUMAN REVIEW
                     │
                     ▼
              RELEASE MANIFEST
                     │
                     ▼
        ┌────────────────────────────┐
        │ arwen-policy-corpus        │
        │ Hugging Face               │
        └────────────────────────────┘
```

---

# Core data lineage

A central requirement of Arwen Policy is that a substantive policy output should be traceable to its underlying evidence.

The intended lineage is:

```text
policy synthesis
      ↓
argument
      ↓
stakeholder position
      ↓
stakeholder
      ↓
source segment
      ↓
source artifact
      ↓
capture event
      ↓
original source
```

This lineage is fundamental to the project's trust model.

---

# Repository structure

```text
arwen-policy-etl/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── configs/
│   ├── pipeline.yaml
│   └── sources.yaml
│
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
│       │
│       └── sources/
│           ├── __init__.py
│           └── generic.py
│
├── schemas/
│   ├── source_record.schema.json
│   ├── extracted_document.schema.json
│   └── provenance_event.schema.json
│
├── scripts/
│   ├── ingest_source.py
│   ├── validate_release.py
│   └── build_manifest.py
│
├── tests/
│   ├── test_hashing.py
│   ├── test_normalization.py
│   ├── test_provenance.py
│   └── test_pipeline.py
│
├── data/
│   ├── registry/
│   ├── raw/
│   ├── extracted/
│   ├── normalized/
│   ├── candidates/
│   └── releases/
│
└── docs/
    ├── architecture.md
    ├── source_onboarding.md
    ├── provenance.md
    ├── quality.md
    └── multimodal.md
```

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e ".[dev]"
```

Validate the configuration:

```bash
python -m arwen_etl.cli validate-config
```

Run the tests:

```bash
pytest
```

---

# Basic ingestion

Ingest a publicly accessible URL:

```bash
python -m arwen_etl.cli ingest-url \
  https://www.icann.org/resources/pages/rdap-operational-profile-2019-06-25-en
```

Ingest a local document:

```bash
python -m arwen_etl.cli ingest-file ./source.pdf
```

Build a release:

```bash
python -m arwen_etl.cli build-release --version 0.1.0
```

Or use the complete release command:

```bash
python -m arwen_etl.cli release --version 0.1.0
```

---

# Configuration

Pipeline configuration lives in:

```text
configs/pipeline.yaml
```

Source definitions live in:

```text
configs/sources.yaml
```

The configuration layer is intentionally separate from implementation code so that new source families and processing policies can be introduced without rewriting the pipeline.

---

# Environment variables

Copy:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Optional external services may be configured through environment variables.

The core ETL should remain functional without external AI services wherever deterministic processing is sufficient.

---

# Processing lifecycle

A source record may move through the following lifecycle:

```text
DISCOVERED
    ↓
CAPTURED
    ↓
EXTRACTED
    ↓
NORMALIZED
    ↓
SEGMENTED
    ↓
CANDIDATE
    ↓
VALIDATED
    ↓
HUMAN_REVIEW
    ↓
RELEASED
```

Alternative terminal states include:

```text
REJECTED
SUPERSEDED
```

Not every record must pass through human review.

However, **claims with consequential attribution or interpretation requirements should not be promoted to verified research records solely because an automated model produced them.**

---

# Quality model

The ETL distinguishes several dimensions of quality rather than assigning a single unexplained confidence number.

Relevant dimensions include:

* source reliability;
* extraction quality;
* provenance completeness;
* attribution quality;
* temporal completeness;
* linguistic quality;
* structural completeness;
* stakeholder attribution;
* evidence linkage;
* human verification status.

A high-quality document extraction does not automatically imply a high-confidence stakeholder position.

---

# Multistakeholder representation

The corpus is designed around the principle that Internet governance and digital policy are inherently multistakeholder domains.

The ETL therefore supports representation across stakeholder categories including, where applicable:

* governments;
* intergovernmental organizations;
* private sector;
* civil society;
* technical community;
* academia;
* users and individual participants;
* media;
* standards communities;
* regional and national Internet governance communities;
* marginalized or underrepresented groups.

These categories are analytical constructs.

They must not be used to infer that every individual or organization within a category shares the same position.

For example:

```text
stakeholder group
      │
      ├── organization A
      │      ├── position X
      │      └── position Y at different times
      │
      ├── organization B
      │      └── position Z
      │
      └── individual participant
             └── documented view
```

The system therefore models **plurality within stakeholder groups**, not merely differences between groups.

---

# Multistakeholder safety rules

The ETL must never infer:

* that silence means neutrality;
* that silence means agreement;
* that participation means endorsement;
* that an employee's statement represents an organization's official position;
* that a meeting participant represents an entire stakeholder group;
* that one organization represents an entire stakeholder category;
* that two differently worded arguments are equivalent;
* that disagreement does not exist merely because it was not detected;
* that repeated publication equals correctness;
* that frequency of appearance equals legitimacy;
* that majority representation equals consensus;
* that a generated synthesis constitutes a negotiated policy outcome.

The system records **documented perspectives with attribution, provenance, evidence and uncertainty**.

---

# Source families

The initial source architecture is designed to support public materials from:

* ICANN;
* Internet Governance Forum / United Nations;
* IETF;
* ITU;
* UN agencies and programmes;
* UN DESA;
* UNESCO;
* OECD;
* Internet Society;
* Regional Internet Registries;
* national and regional Internet Governance initiatives;
* public government sources;
* standards organizations;
* academic institutions;
* research organizations;
* public-interest organizations.

Additional source adapters can be introduced without changing the canonical data model.

---

# Source onboarding

A source adapter should define, where applicable:

```text
source identity
    ↓
discovery mechanism
    ↓
URL/document acquisition
    ↓
publication metadata
    ↓
content extraction
    ↓
pagination / section handling
    ↓
source-specific metadata
    ↓
provenance
```

Source-specific logic belongs under:

```text
src/arwen_etl/sources/
```

rather than being embedded in the core pipeline.

---

# Multimodal processing

Multimodal processing is designed as a sequence of deterministic and optional AI-assisted stages.

### Documents

```text
PDF/DOCX/HTML
      ↓
parser
      ↓
text
      ↓
normalization
      ↓
segmentation
```

### Scanned documents

```text
image/PDF
    ↓
OCR
    ↓
text
    ↓
quality assessment
    ↓
normalization
```

### Audio

```text
audio
  ↓
ASR
  ↓
speaker diarization
  ↓
speaker-labelled segments
  ↓
normalization
```

### Video

```text
video
  ↓
audio extraction
  ↓
ASR + diarization
  ↓
visual/event metadata
  ↓
timestamped segments
```

AI-assisted multimodal processing must preserve the original artifact and processing metadata.

---

# AI-assisted extraction

AI models may be used to generate candidates for:

* stakeholder identification;
* organization identification;
* policy topics;
* claims;
* positions;
* arguments;
* counterarguments;
* evidence relationships;
* questions;
* policy proposals;
* disagreement;
* uncertainty.

The output must remain explicitly marked as machine-generated until validated.

For example:

```json
{
  "extraction_status": "candidate",
  "extraction_method": "llm",
  "human_verified": false
}
```

rather than silently storing the result as an authoritative fact.

---

# Reproducibility

Every release should be associated with:

```text
dataset version
pipeline version
schema version
configuration version
source hashes
processing timestamps
processing methods
quality results
release manifest
```

A release should therefore answer:

> What data did we process?

> From which sources?

> When?

> With which pipeline?

> With which configuration?

> Which transformations occurred?

> Which records were rejected?

> Which records require review?

---

# Release model

The ETL produces versioned release artifacts that can be published to:

```text
soyames/arwen-policy-corpus
```

The Hugging Face repository is the canonical public dataset.

The ETL repository remains the authoritative location for the **code and reproducible processing pipeline**.

The two repositories therefore have distinct responsibilities:

| Repository                    | Role                                                              |
| ----------------------------- | ----------------------------------------------------------------- |
| `soyames/arwen-policy-etl`    | Pipeline, code, schemas, configuration, tests and reproducibility |
| `soyames/arwen-policy-corpus` | Canonical research corpus and dataset releases                    |

---

# Research and benchmarking

The resulting corpus is intended to support:

* policy research;
* evidence-grounded retrieval;
* stakeholder-aware retrieval;
* argument mining;
* policy-position analysis;
* knowledge-graph construction;
* multistakeholder deliberation experiments;
* model fine-tuning;
* retrieval-augmented generation;
* evaluation and benchmarking;
* reproducibility research.

The ETL itself is **not** a policy decision-making authority.

---

# Current maturity

**Version: `0.1.0` — Engineering Foundation**

The current foundation includes:

* configuration management;
* URL and file capture;
* SHA-256 content hashing;
* MIME detection;
* text extraction;
* normalization;
* segmentation;
* candidate generation;
* provenance tracking;
* schema validation;
* quality controls;
* release manifests;
* automated tests.

The following are being developed as extensible adapters or subsequent pipeline stages:

* source-specific crawlers;
* OCR;
* automatic speech recognition;
* speaker diarization;
* multilingual processing;
* semantic embeddings;
* vector indexing;
* knowledge-graph persistence;
* advanced stakeholder extraction;
* argument mining;
* evidence linking;
* human annotation workflows;
* Hugging Face release automation.

---

# Roadmap

### Phase 1 — Foundation

* [x] Canonical data model
* [x] Provenance model
* [x] Source capture
* [x] Content hashing
* [x] Normalization
* [x] Segmentation
* [x] Candidate extraction foundation
* [x] Validation
* [x] Release manifests
* [x] Test foundation

### Phase 2 — Real policy sources

* [ ] ICANN source adapter
* [ ] IGF / UN source adapter
* [ ] IETF source adapter
* [ ] ITU source adapter
* [ ] UN / UN DESA adapter
* [ ] Internet Society adapter
* [ ] RIR source adapters
* [ ] Government-source framework
* [ ] Academic/public-interest source framework

### Phase 3 — Multimodal ingestion

* [ ] OCR pipeline
* [ ] ASR pipeline
* [ ] speaker diarization
* [ ] timestamped video processing
* [ ] multilingual extraction
* [ ] extraction-quality benchmarks

### Phase 4 — Policy intelligence

* [ ] stakeholder extraction
* [ ] organization resolution
* [ ] position extraction
* [ ] argument extraction
* [ ] counterargument detection
* [ ] evidence linking
* [ ] temporal position tracking
* [ ] disagreement modelling

### Phase 5 — Corpus production

* [ ] automated HF dataset releases
* [ ] expert annotation
* [ ] benchmark suite
* [ ] corpus quality reports
* [ ] stakeholder coverage reports
* [ ] reproducible dataset cards

### Phase 6 — Arwen Policy

* [ ] stakeholder-aware retrieval
* [ ] evidence-grounded synthesis
* [ ] multistakeholder deliberation
* [ ] structured disagreement analysis
* [ ] policy recommendation generation
* [ ] deliberation evaluation
* [ ] Arwen Policy model

---

# Relationship to the Arwen Policy research project

Arwen Policy ETL is one component of the broader research project:

> **Build a Multistakeholder AI Deliberation System for Digital Policy and Internet Governance.**

The broader architecture is envisioned as:

```text
                    PUBLIC POLICY INFORMATION
                              │
                              ▼
                    ARWEN POLICY ETL
                              │
                              ▼
                   ARWEN POLICY CORPUS
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              KNOWLEDGE GRAPH     POLICY RETRIEVAL
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    STAKEHOLDER MODELS
                              │
                              ▼
                    DELIBERATION ENGINE
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
          GOVERNMENT       CIVIL SOCIETY    INDUSTRY
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    EVIDENCE-GROUNDED
                     POLICY SYNTHESIS
```

The objective is **not** to build an AI that decides what policy should be.

The objective is to build infrastructure that helps people understand:

* what different stakeholders have said;
* why they hold particular positions;
* what evidence they rely upon;
* where positions agree;
* where they disagree;
* which perspectives are missing;
* how positions change over time;
* and what possible areas of convergence exist.

---

# Citation

If you use the ETL or resulting research artifacts in academic work, please cite the project using `CITATION.cff`.

The canonical dataset should be cited separately from the ETL software.

---

# License

The software, schemas and original project metadata are released under the license specified in `LICENSE`, unless otherwise stated.

**Source material remains subject to its original copyright, license, access and attribution conditions.**

The ETL does not claim ownership of third-party source material merely because it is processed by the pipeline.

---

# Status

**Active research and engineering project**

The pipeline is under active development toward the first reproducible Arwen Policy Corpus release.

**Code:** `soyames/arwen-policy-etl`
**Dataset:** `soyames/arwen-policy-corpus`
**Project:** Arwen Policy — Multistakeholder AI Deliberation for Digital Policy and Internet Governance
