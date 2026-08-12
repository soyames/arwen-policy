"""Upload canonical corpus files to Hugging Face dataset."""
from pathlib import Path

from huggingface_hub import HfApi, upload_file

REPO = "soyames/arwen-policy-corpus"
REPO_TYPE = "dataset"
BASE = Path("arwen-policy-corpus")

api = HfApi()

files_to_upload = [
    "README.md",
    "data/README.md",
    "data/documents.jsonl",
    "data/canonical_documents.jsonl",
    "data/train/documents.jsonl",
    "data/train/documents.parquet",
]

for fname in files_to_upload:
    path = BASE / fname
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    print(f"Uploading {fname} ({path.stat().st_size:,} bytes)...")
    upload_file(
        path_or_fileobj=str(path),
        path_in_repo=fname,
        repo_id=REPO,
        repo_type=REPO_TYPE,
        commit_message=f"Update {fname} — canonical corpus v2, 192 docs",
    )
    print("  Done.")

print("\nAll uploads complete.")
print(f"Dataset: https://huggingface.co/datasets/{REPO}")
