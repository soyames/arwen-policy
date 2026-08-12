"""Verify remote HF dataset state independently."""
from collections import Counter

from huggingface_hub import hf_hub_download, dataset_info
import pyarrow.parquet as pq

print("=== REMOTE HF DATASET ===")
info = dataset_info("soyames/arwen-policy-corpus")
print(f"Dataset: {info.id}")
print(f"SHA: {info.sha[:16]}...")
print(f"Last modified: {info.last_modified}")

# Download parquet with force refresh
path = hf_hub_download(
    "soyames/arwen-policy-corpus",
    "data/train/documents.parquet",
    repo_type="dataset",
    force_download=True,
)
table = pq.read_table(path)
print(f"\nParquet rows: {table.num_rows}")
print(f"Columns ({len(table.column_names)}): {list(table.column_names)}")

# Content types
cts = Counter()
methods = Counter()
srcs = Counter()
for r in table.to_pylist():
    ct = str(r.get("content_type", "")).lower()
    label = "PDF" if "pdf" in ct else "HTML" if "html" in ct else "TEXT"
    cts[label] += 1
    methods[r.get("extraction_method", "?")] += 1
    srcs[r.get("source", "?")] += 1

print(f"Content types: {dict(cts)}")
print(f"PDF rows: {cts.get('PDF', 0)}")
print(f"Extraction: {dict(methods)}")
print(f"Sources: {dict(sorted(srcs.items()))}")

# IDs and hashes
ids = [r["document_id"] for r in table.to_pylist()]
hashes = [r["artifact_sha256"] for r in table.to_pylist() if r.get("artifact_sha256")]
print(f"Unique IDs: {len(set(ids))}")
print(f"Unique hashes: {len(set(hashes))}")
print(f"ID duplicates: {len(ids) - len(set(ids))}")

# Check sample
print("\nSample rows (first 3):")
for r in table.to_pylist()[:3]:
    print(f"  {r['document_id'][:20]} | {r.get('source','?')} | {str(r.get('title',''))[:60]} | ct={r.get('content_type','')[:20]} | method={r.get('extraction_method','')}")

# PDF records
pdf_rows = [r for r in table.to_pylist() if "pdf" in str(r.get("content_type", "")).lower()]
print(f"\n=== PDF RECORDS ON REMOTE ({len(pdf_rows)}) ===")
for r in pdf_rows:
    print(f"  {r['document_id'][:20]} | {r.get('source','?')} | {str(r.get('title',''))[:70]}")
    print(f"    hash={r.get('artifact_sha256','')[:16]} len={r.get('text_length',0)} method={r.get('extraction_method','')}")

# Compare with local export
print("\n=== LOCAL CANONICAL EXPORT ===")
import json as _json
from pathlib import Path

canonical_path = Path("data/canonical/canonical_documents.jsonl")
if canonical_path.exists():
    local_docs = []
    with open(canonical_path, "r", encoding="utf-8") as f:
        for line in f:
            local_docs.append(_json.loads(line))
    print(f"Local canonical docs: {len(local_docs)}")

    local_ids = {d["document_id"] for d in local_docs}
    local_hashes = {d.get("artifact_sha256", "") for d in local_docs if d.get("artifact_sha256")}
    remote_ids = set(ids)
    remote_hashes = set(h for h in hashes if h)

    id_matches = local_ids & remote_ids
    hash_matches = local_hashes & remote_hashes

    print(f"Exact ID matches: {len(id_matches)}")
    print(f"Hash matches: {len(hash_matches)}")
    print(f"Local-only IDs: {len(local_ids - remote_ids)}")
    print(f"Remote-only IDs: {len(remote_ids - local_ids)}")
    print(f"Local-only hashes: {len(local_hashes - remote_hashes)}")
    print(f"Remote-only hashes: {len(remote_hashes - local_hashes)}")
else:
    print("Canonical export not found at data/canonical/canonical_documents.jsonl")
    print("Attempting to regenerate...")
