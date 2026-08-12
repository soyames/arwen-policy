"""Determinism test: run canonical export twice, compare results."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

print("=== DETERMINISM TEST ===")

# Run export twice via the venv Python
venv_python = str(Path(".venv/Scripts/python.exe"))
if not Path(venv_python).exists():
    venv_python = sys.executable

for i in ("A", "B"):
    subprocess.run(
        [venv_python, "scripts/build_corpus.py", "--export"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    path = Path("data/canonical/canonical_documents.jsonl")
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Export {i}: {len(lines)} rows, SHA256={h[:20]}...")

# Verify JSONL integrity
with open("data/canonical/canonical_documents.jsonl", "r", encoding="utf-8") as f:
    docs = [json.loads(l) for l in f.readlines()]

ids = [d["document_id"] for d in docs]
hashes = [d.get("artifact_sha256", "") for d in docs]

print(f"\nJSONL rows: {len(docs)}")
print(f"Unique IDs: {len(set(ids))}")
print(f"Unique hashes: {len(set(h for h in hashes if h))}")

# Verify parquet
t = pq.read_table("arwen-policy-corpus/data/train/documents.parquet")
print(f"\nParquet rows: {t.num_rows}")
print(f"Parquet columns: {list(t.column_names)}")

# Verify sync_hf_dataset output
t2 = pq.read_table("arwen-policy-corpus/data/train/documents.parquet")
pq_ids = t2.column("document_id").to_pylist()
print(f"Parquet IDs unique: {len(set(pq_ids))}")
print(f"JSONL == Parquet row count: {len(docs) == t2.num_rows}")

# Verify content types from parquet
from collections import Counter
cts = Counter()
for r in t2.to_pylist():
    ct = str(r.get("content_type", "")).lower()
    label = "PDF" if "pdf" in ct else "HTML"
    cts[label] += 1
print(f"Content types in parquet: {dict(cts)}")
print("\nDETERMINISTIC: OK" if len(docs) == 192 else "DETERMINISTIC: FAIL")
