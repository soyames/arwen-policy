"""GitHub <-> Hugging Face reconciliation audit."""
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, dataset_info, space_info
import pyarrow.parquet as pq

api = HfApi()

# ===================================================================
# 1. REMOTE HF CORPUS
# ===================================================================
print("=" * 60)
print("1. REMOTE: soyames/arwen-policy-corpus")
print("=" * 60)
try:
    dinfo = dataset_info("soyames/arwen-policy-corpus")
    print(f"  SHA: {dinfo.sha[:16]}...")
    print(f"  Last modified: {dinfo.last_modified}")

    path = hf_hub_download("soyames/arwen-policy-corpus", "data/train/documents.parquet",
                           repo_type="dataset")
    t = pq.read_table(path)
    print(f"  Parquet rows: {t.num_rows}")
    print(f"  Columns ({t.num_columns}): {list(t.column_names)[:8]}...")

    cts = Counter()
    methods = Counter()
    sources = Counter()
    years = []
    for r in t.to_pylist():
        ct = str(r.get("content_type", "")).lower()
        cts["PDF" if "pdf" in ct else "HTML"] += 1
        methods[r.get("extraction_method", "?")] += 1
        sources[r.get("source", "?")] += 1
        pub = r.get("published_at", "")
        if pub and len(str(pub)) >= 4:
            try:
                years.append(int(str(pub)[:4]))
            except ValueError:
                pass

    ids = [r["document_id"] for r in t.to_pylist()]
    print(f"  Unique IDs: {len(set(ids))}")
    print(f"  Content: {dict(cts)}")
    print(f"  Extraction: {dict(methods)}")
    print(f"  Sources: {len(sources)}")
    print(f"  Year range: {min(years)}-{max(years)}" if years else "  Year range: N/A")
    print(f"  1990-1999: {len([y for y in years if y <= 1999])}")
    print(f"  PDF records: {cts.get('PDF', 0)}")

    # Source distribution
    for src, cnt in sources.most_common():
        print(f"    {src}: {cnt}")

except Exception as e:
    print(f"  ERROR: {e}")

# ===================================================================
# 2. REMOTE BASE
# ===================================================================
print()
print("=" * 60)
print("2. REMOTE: soyames/arwen-policy-base")
print("=" * 60)
try:
    files = list_repo_files("soyames/arwen-policy-base")
    print(f"  Files ({len(files)}): {files}")
    if len(files) <= 2:
        print("  STATUS: Placeholder only — no model weights")
except Exception as e:
    print(f"  ERROR: {e}")

# ===================================================================
# 3. REMOTE LORA
# ===================================================================
print()
print("=" * 60)
print("3. REMOTE: soyames/arwen-policy-lora")
print("=" * 60)
try:
    files = list_repo_files("soyames/arwen-policy-lora")
    print(f"  Files ({len(files)}): {files}")
    if len(files) <= 2:
        print("  STATUS: Placeholder only — no adapter weights")
except Exception as e:
    print(f"  ERROR: {e}")

# ===================================================================
# 4. REMOTE SPACE
# ===================================================================
print()
print("=" * 60)
print("4. REMOTE: soyames/arwen-policy (Space)")
print("=" * 60)
try:
    sinfo = space_info("soyames/arwen-policy")
    print(f"  SHA: {sinfo.sha[:16] if sinfo.sha else '?'}")
    files = list_repo_files("soyames/arwen-policy", repo_type="space")
    print(f"  Files ({len(files)})")
except Exception as e:
    print(f"  ERROR: {e}")

# ===================================================================
# 5. LOCAL CANONICAL CORPUS
# ===================================================================
print()
print("=" * 60)
print("5. LOCAL: Canonical Corpus (corpus/)")
print("=" * 60)
corpus = Path("corpus")
local_docs = []
for f in sorted(corpus.glob("*.json")):
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
        if doc.get("text") and len(doc["text"]) >= 200:
            local_docs.append(doc)
    except Exception:
        continue
print(f"  Documents: {len(local_docs)}")
local_ids = {d["document_id"] for d in local_docs}
local_hashes = {d.get("artifact_sha256", "") for d in local_docs}
print(f"  Unique IDs: {len(local_ids)}")
print(f"  Unique hashes: {len(local_hashes)}")

local_sources = Counter()
for d in local_docs:
    url = d.get("source_url", "") or d.get("final_url", "")
    u = url.lower()
    for domain, name in [
        ("icann.org", "ICANN"), ("ietf.org", "IETF"), ("rfc-editor.org", "IETF"),
        ("itu.int", "ITU"), ("intgovforum.org", "IGF"), ("internetsociety.org", "ISOC"),
        ("oecd.org", "OECD"), ("un.org", "UN"), ("unesco.org", "UNESCO"),
        ("arin.net", "ARIN"), ("ripe.net", "RIPE"), ("apnic.net", "APNIC"),
        ("lacnic.net", "LACNIC"), ("afrinic.net", "AFRINIC"), ("europa.eu", "EU"),
        ("arxiv.org", "Academic"), ("iana.org", "IANA"), ("wto.org", "WTO"),
    ]:
        if domain in u:
            local_sources[name] += 1
            break
    else:
        local_sources["other"] += 1
for src, cnt in local_sources.most_common():
    print(f"    {src}: {cnt}")

# ===================================================================
# 6. LOCAL SFT
# ===================================================================
print()
print("=" * 60)
print("6. LOCAL: SFT Final (datasets/sft_final/)")
print("=" * 60)
sft = Path("datasets/sft_final")
for split in ("train", "validation", "test"):
    path = sft / f"{split}.jsonl"
    if path.exists():
        count = len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])
        print(f"  {split}: {count}")
    else:
        print(f"  {split}: MISSING")

# ===================================================================
# 7. RECONCILIATION
# ===================================================================
print()
print("=" * 60)
print("7. RECONCILIATION (Local vs HF)")
print("=" * 60)
if t.num_rows > 0:
    remote_ids = set(ids)
    id_matches = local_ids & remote_ids
    hash_matches = local_hashes & set(h for h in [r.get("artifact_sha256", "") for r in t.to_pylist()] if h)
    print(f"  Local IDs: {len(local_ids)}")
    print(f"  Remote IDs: {len(remote_ids)}")
    print(f"  ID matches: {len(id_matches)}")
    print(f"  Hash matches: {len(hash_matches)}")
    print(f"  Local-only IDs: {len(local_ids - remote_ids)}")
    print(f"  Remote-only IDs: {len(remote_ids - local_ids)}")

    if local_ids == remote_ids:
        print("  STATUS: FULLY SYNCHRONIZED")
    else:
        print("  STATUS: DIFFERENCES DETECTED")
