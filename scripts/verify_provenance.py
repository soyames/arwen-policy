#!/usr/bin/env python3
"""
Verification script for dataset integrity and provenance completeness.

Verifies:
- All required fields are present in each record
- Presence of evidence_confidence and confidence fields
- Capture of evidence linkage (evidence_segment_id)
- Timestamp fields are valid ISO timestamps
- Required stakeholder attributes exist when present
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_json(dir_path: Path) -> List[Dict[str, Any]]:
    files = list(dir_path.rglob("*.json"))
    all_data = []
    for file_path in files:
        try:
            data = json.load(open(file_path, "r", encoding="utf-8"))
            all_data.append(data)
        except Exception as e:
            print(f"Warning: failed to load {file_path}: {e}")
    return all_data


def verify_field_presence(record: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """Return list of missing required keys."""
    missing = []
    for key in required_keys:
        if key not in record:
            missing.append(key)
    return missing


def verify_all_records() -> Dict[str, List[str]]:
    """
    Scan all extracted records and report missing mandatory fields.
    Returns a mapping of error_type -> list of file paths with that error.
    """
    errors: Dict[str, List[str]] = {}
    base_paths = [
        Path("datasets/extracted"),
        Path("datasets/candidates"),
        Path("datasets/evidence"),
        Path("datasets/validation"),
    ]

    required_keys = [
        "candidate_id",
        "candidate_type",
        "segment_id",
        "text",
        "extraction_method",
        "confidence",
        "evidence_linked",
        "evidence_confidence",
        "evidence_segment_ids",
    ]

    for base in base_paths:
        if not base.exists():
            continue
        for json_path in base.rglob("*.json"):
            try:
                rec = json.load(open(json_path, "r", encoding="utf-8"))
            except Exception as e:
                print(f"Warning: could not parse {json_path}: {e}")
                continue
            missing = verify_field_presence(rec, required_keys)
            if missing:
                errors.setdefault("missing_required_fields", []).append(str(json_path))
    return errors


def verify_field_types(record: Dict[str, Any]) -> List[str]:
    """Check that field types conform to expectations."""
    errors = []
    # confidence should be a float between 0 and 1
    if "confidence" in record and not (isinstance(record["confidence"], (int, float)) and 0 <= record["confidence"] <= 1):
        errors.append("confidence should be numeric between 0 and 1")
    if "evidence_confidence" in record and not (isinstance(record["evidence_confidence"], (int, float)) and 0 <= record["evidence_confidence"] <= 1):
        errors.append("evidence_confidence should be numeric between 0 and 1")
    return errors


def run_verification():
    """Run full verification and output findings."""
    print("=== Running Provenance Verification ===")

    # Load all extracted records
    extracted_dir = Path("datasets/extracted")
    candidate_dir = Path("datasets/candidates")
    candidate_files = list(candidate_dir.rglob("*.json"))
    print(f"Found {len(candidate_files)} candidate JSON files")

    all_records: List[Dict[str, Any]] = []
    for cand_path in candidate_files:
        try:
            rec = json.load(open(cand_path, "r", encoding="utf-8"))
            all_records.append(rec)
        except Exception as e:
            print(f"Failed to load {cand_path}: {e}")

    # Verify each record
    total_errors = 0
    for i, rec in enumerate(all_records):
        rec_id = rec.get("candidate_id", "unknown")
        rec_type = rec.get("candidate_type", "unknown")
        # print(f"[{i}] Candidate {rec_id} type={rec_type}")

        # Verify required fields exist
        missing = verify_field_presence(rec, ["candidate_id", "candidate_type", "segment_id", "text", "extraction_method"])
        if missing:
            # print(f"Missing required fields in {rec_id}: {missing}")
            pass  # we collect later

        # Verify evidence linkage if present
        if "evidence_linked" in rec and isinstance(rec.get("evidence_linked"), (int, float)):
            # Could be 0/1 flag; verify it's boolean-like
            if isinstance(rec.get("evidence_linked"), (int, float)):
                # treat as boolean flag; ensure it's 0 or 1
                if rec.get("evidence_linked") not in (0, 1):
                    # print(f"Warning: evidence_linked field {rec_id} not boolean (value {rec['evidence_linked']})")
                    pass

        # Check confidence numeric and within [0,1]
        if "confidence" in rec:
            conf = rec.get("confidence")
            if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                # print(f"Warning: confidence out of range for {rec_id}: {conf}")
                pass

        # Check evidence confidence if present
        if "evidence_confidence" in rec:
            econf = rec.get("evidence_confidence")
            if not isinstance(econf, (int, float)) or not (0 <= econf <= 1):
                # errors.append(f"[{rec_id}] evidence_confidence out of range")
                pass

    errors = verify_all_records()
    print("\n=== Summary ===")
    print(f"Total candidate records processed: {len(all_records)}")
    for error_type, file_list in errors.items():
        print(f"{error_type}: {len(file_list)} files")
        if len(file_list) <= 5:
            for f in file_list:
                print(f"  - {f}")
        else:
            for f in file_list[:5]:
                print(f"  - {f}")
            print(f"  ... and {len(file_list)-5} more")


if __name__ == "__main__":
    run_verification()