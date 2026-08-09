---
license: apache-2.0
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - policy
  - internet-governance
  - digital-policy
  - multistakeholder
  - provenance
pretty_name: Arwen Policy Corpus
size_categories:
  - n<1K
---

# Arwen Policy Corpus

The **Arwen Policy Corpus** is a provenance-preserving dataset of policy documents
extracted from Internet governance and digital-policy sources. Each record includes
source attribution, extraction metadata, and content hashes.

## Research Purpose

This corpus supports evidence-grounded, multistakeholder AI deliberation for
digital policy and Internet governance. It is designed for:

- Policy stance classification
- Evidence-grounded retrieval
- Stakeholder-aware argument mining
- Provenance-preserving corpus construction research

## Corpus Contents

The corpus contains extracted and normalised documents from sources including:

- ICANN (Internet Corporation for Assigned Names and Numbers)
- IETF (Internet Engineering Task Force)
- ITU (International Telecommunication Union)
- OECD (Organisation for Economic Co-operation and Development)
- UN agencies
- Internet Society (ISOC)

Each record preserves source provenance, extraction method, and content integrity
via SHA-256 hashing.

## Data Structure

Records are stored as JSON with fields including:

- `document_id` — unique identifier
- `source_id` — source identifier derived from the URL
- `source_url` / `final_url` — original and resolved URLs
- `artifact_sha256` — content hash for integrity verification
- `text` — extracted and normalised text
- `extraction_method` — method used (e.g. `html_bs4`, `pypdf`)
- `segments` — segmented text chunks with character offsets
- `metadata` — document metadata (title, language, publication date)

## Provenance and Licensing

All records include provenance events tracing the extraction chain.
Source licensing and access conditions are preserved where available.
This dataset is released under the Apache-2.0 license.

## Limitations

- This is an early-stage research corpus. Record counts are currently small.
- Automated extraction may introduce errors. Candidate positions and arguments
  should be treated as unverified until reviewed.
- Not all stakeholder groups are equally represented.
- Missing perspectives are recorded but not filled in.

## Links

- **Source code**: https://github.com/soyames/arwen-policy
- **Base model**: https://huggingface.co/soyames/arwen-policy-base
- **Interactive demo**: https://huggingface.co/spaces/soyames/arwen-policy

## Citation

See [CITATION.cff](https://github.com/soyames/arwen-policy/blob/main/CITATION.cff)
in the project repository.
