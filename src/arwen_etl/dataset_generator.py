from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .storage import write_json
from .models import SourceRecord, ExtractedDocument


class DatasetGenerator:
    """Generate Hugging Face dataset from processed policy corpus."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.extracted_dir = self.data_dir / "extracted"
        self.normalized_dir = self.data_dir / "normalized"
        self.candidates_dir = self.data_dir / "candidates"
        self.releases_dir = self.data_dir / "releases"

        # Ensure directories exist
        for dir_path in [self.extracted_dir, self.normalized_dir, self.candidates_dir, self.releases_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_source_records(self) -> List[SourceRecord]:
        """Load all source records from the extracted data."""
        records = []
        for record_file in self.extracted_dir.glob("*.json"):
            record = SourceRecord.model_validate_json(record_file.read_text())
            records.append(record)
        return records

    def load_extracted_documents(self) -> List[ExtractedDocument]:
        """Load all extracted documents."""
        documents = []
        for doc_file in self.normalized_dir.glob("*.json"):
            doc = ExtractedDocument.model_validate_json(doc_file.read_text())
            documents.append(doc)
        return documents

    def build_dataset(self, version: str = "0.1.0") -> Dict[str, Any]:
        """Build a dataset dictionary for Hugging Face."""
        source_records = self.load_source_records()
        extracted_docs = self.load_extracted_documents()

        # Build dataset entries
        dataset_entries = []

        for doc in extracted_docs:
            # Find corresponding source record
            source_record = next((sr for sr in source_records if sr.source_id == doc.source_id), None)
            if not source_record:
                continue

            entry = {
                "id": doc.document_id,
                "source_id": doc.source_id,
                "source_name": source_record.publisher,
                "source_family": source_record.source_family,
                "source_url": source_record.source_url,
                "canonical_url": doc.canonical_url,
                "retrieved_at": doc.retrieved_at.isoformat() if doc.retrieved_at else None,
                "http_status": doc.http_status,
                "content_type": doc.content_type,
                "content_hash": doc.content_hash,
                "extraction_status": doc.extraction_status,
                "text": doc.text,
                "title": doc.title,
                "language": doc.language,
                "jurisdiction": doc.jurisdiction,
                "policy_topics": doc.policy_topics,
                "license": doc.license,
                "access_conditions": doc.access_conditions,
                "discovery_urls": doc.discovery_urls,
                "metadata": doc.metadata,
                "provenance": [
                    {
                        "event_id": event["event_id"],
                        "event_type": event["event_type"],
                        "timestamp": event["timestamp"],
                        "actor": event["agent"],
                        "description": event["attributes"].get("description", ""),
                        "details": event["attributes"]
                    }
                    for event in doc.provenance_events
                ] if hasattr(doc, 'provenance_events') and isinstance(doc.provenance_events, list) else []
            }
            dataset_entries.append(entry)

        # Build dataset info
        dataset_info = {
            "dataset_name": "arwen-policy-corpus",
            "version": version,
            "description": "Multimodal policy corpus for Internet governance and digital policy research",
            "homepage": "https://huggingface.co/datasets/soyames/arwen-policy-corpus",
            "license": "CC-BY-4.0",
            "citation": "@article{arwen2026policy, title={Arwen Policy Corpus}, author={Arwen Policy Team}, journal={Journal of Digital Policy}, year={2026}}",
            "size_in_bytes": self._calculate_dataset_size(dataset_entries),
            "num_examples": len(dataset_entries),
            "last_updated": datetime.now(datetime.UTC).isoformat(),
            "sources_processed": len(set([e["source_id"] for e in dataset_entries]))
        }

        return {
            "info": dataset_info,
            "data": dataset_entries
        }

    def _calculate_dataset_size(self, entries: List[Dict[str, Any]]) -> int:
        """Calculate approximate size of dataset in bytes."""
        total_size = 0
        for entry in entries:
            total_size += len(json.dumps(entry).encode('utf-8'))
        return total_size

    def save_dataset(self, dataset: Dict[str, Any], output_dir: Optional[str] = None) -> Path:
        """Save dataset to disk in Hugging Face format."""
        if output_dir is None:
            output_dir = self.releases_dir / f"dataset_v{dataset['info']['version'].replace('.', '_')}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save dataset info
        info_path = output_path / "dataset_info.json"
        write_json(info_path, dataset["info"])

        # Save data as JSONL
        data_path = output_path / "data.jsonl"
        with data_path.open('w') as f:
            for entry in dataset["data"]:
                f.write(json.dumps(entry) + '\n')

        # Save README
        readme_path = output_path / "README.md"
        readme_content = self._generate_readme(dataset["info"])
        readme_path.write_text(readme_content)

        return output_path

    def _generate_readme(self, info: Dict[str, Any]) -> str:
        """Generate a README.md for the dataset."""
        return f"""# {info['dataset_name']}

{info['description']}

## Dataset Information

- **Version**: {info['version']}
- **Size**: {info['size_in_bytes'] / (1024*1024):.2f} MB
- **Number of Examples**: {info['num_examples']}
- **Sources Processed**: {info['sources_processed']}
- **Last Updated**: {info['last_updated']}
- **License**: {info['license']}

## Citation

{info['citation']}

## Homepage

{info['homepage']}

## Data Structure

Each example in the dataset is a JSON line with the following fields:

- `id`: Unique document identifier
- `source_id`: Source identifier
- `source_name`: Name of the source organization
- `source_family`: Source family (e.g., internet_governance, intergovernmental)
- `source_url`: Original source URL
- `canonical_url`: Final URL after redirects
- `retrieved_at`: Timestamp when the source was captured
- `http_status`: HTTP status code from capture
- `content_type`: MIME type of the captured content
- `content_hash`: SHA-256 hash of the content
- `extraction_status`: Status of content extraction (extracted, failed, not_extracted)
- `text`: Extracted and normalized text content
- `title`: Document title
- `language`: Language code (ISO 639-1)
- `jurisdiction`: Geographic jurisdiction
- `policy_topics`: List of policy topics
- `license`: Document license
- `access_conditions`: Access conditions or restrictions
- `discovery_urls`: URLs discovered during ingestion
- `metadata`: Additional metadata from extraction
- `provenance`: List of provenance events

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data.jsonl")
```

## License

This dataset is licensed under the {info['license']} - see the [LICENSE](LICENSE) file for details.
"""
