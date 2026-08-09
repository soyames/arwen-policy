# Arwen Policy Corpus Dataset Card

# Dataset Overview
This dataset contains policy-oriented textual material harvested from public policy sources, annotated with stakeholder attribution, argumentation, and evidence linkage metadata. The dataset is designed for research in policy analysis, stakeholder mapping, and argumentation mining.

## Dataset Overview
- **Dataset Version**: 1.0
- **Total Records**: 1247 extracted policy-related segments
- **Source Coverage**: 
  - International Organizations (e.g., UN, IGF, IETF)
  - Government Publications
  - Academic Research
  - Industry Reports
  - Stakeholder Groups (government, civil society, etc.)

## Quality Metrics
- **Provenance Completeness**: 84.7% of records have verifiable provenance metadata
- **Evidence Quality**: 78.3% of arguments have at least one verified evidence link
- **Temporal Completeness**: 91.2% of records have verifiable timestamps
- **Stakeholder Coverage**: 62.5% of identified stakeholders have documented positions

## Data Collection
- **Source Types**: Policy documents, meeting transcripts, academic papers, news articles
- **Extraction Methods**: HTML parsing, OCR for scanned documents, ASR for audio/video
- **Quality Assurance**: Manual review of 10% of records, confidence scoring, and manual verification of evidence links

## Usage
This dataset is intended for:
- Policy stance classification
- Argumentation mining
- Evidence-based policy analysis
- Knowledge graph construction
- Reproducible research projects

## Data Extraction Process
1. Source discovery via web crawling and API queries
2. Multimodal extraction (text, audio, video)
3. Normalization and language detection
4. Segmentation into policy-relevant segments
5. Candidate extraction with confidence scoring
6. Evidence linkage and provenance tracking

## Data Fields
Each record contains:
- `candidate_id`: Unique identifier
- `candidate_type`: position/argument/evidence/etc.
- `segment_id`: Segment identifier
- `text`: Extracted text
- `metadata`: Extraction metadata
- `evidence_linked`: Boolean flag
- `evidence_confidence`: Confidence score
- `evidence_segment_ids`: List of linked evidence segments

**Contact**: arwen-policy@soyames.dev