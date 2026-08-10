#!/bin/bash
# Fix HF dataset repository: canonical files and clean history
set -e

SRC=/mnt/c/Users/soyames/Documents/GitHub/arwen-policy/data/canonical/hf_backup_20260810
DST=/tmp/hf-clean

echo "=== Copying canonical files ==="
cp "$SRC/canonical_documents.jsonl" "$DST/data/documents.jsonl"
cp "$SRC/canonical_documents.jsonl" "$DST/data/canonical_documents.jsonl"
cp "$SRC/canonical_documents.jsonl" "$DST/data/train/documents.jsonl"
cp "$SRC/documents.parquet" "$DST/data/train/documents.parquet"
cp "$SRC/README.md" "$DST/README.md"

echo "=== File verification ==="
for f in data/documents.jsonl data/canonical_documents.jsonl data/train/documents.jsonl; do
    echo "$f: $(wc -l < "$DST/$f") lines"
done
echo "Parquet: $(wc -c < "$DST/data/train/documents.parquet") bytes"

echo "=== Parquet verification ==="
cd "$DST"
python3 -c "
import pyarrow.parquet as pq
from collections import Counter
t = pq.read_table('data/train/documents.parquet')
print(f'Rows: {t.num_rows}')
cts = Counter()
for r in t.to_pylist():
    ct = str(r.get('content_type','')).lower()
    cts['PDF' if 'pdf' in ct else 'HTML'] += 1
print(f'PDF: {cts.get(\"PDF\",0)}, HTML: {cts.get(\"HTML\",0)}')
ids = [r['document_id'] for r in t.to_pylist()]
print(f'Unique IDs: {len(set(ids))}')
"

echo "=== Git status ==="
cd "$DST"
git status --short

echo ""
echo "All files in place. Ready to commit."
