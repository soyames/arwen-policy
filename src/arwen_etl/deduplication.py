from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .storage import read_json, write_json

# process-local lock to serialize index updates
_write_lock = threading.Lock()

# defaults (configurable via function args in calls)
DEFAULT_SHINGLE_K = 5
DEFAULT_THRESHOLD = 0.9


def content_identity(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def unique_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for text in texts:
        key = content_identity(text)
        if key not in seen:
            seen.add(key)
            output.append(text)
    return output


def canonicalize_url(url: str) -> str:
    """Return a simple canonical form of a URL for deduplication.

    - Lowercases scheme and host
    - Removes fragment
    - Strips common analytics query params (utm_*)
    - Removes default ports
    """
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    p = urlparse(url)
    scheme = p.scheme.lower()
    netloc = p.hostname.lower() if p.hostname else ""
    if p.port and (p.port not in (80, 443)):
        netloc = f"{netloc}:{p.port}"

    # filter query params
    qs = [(k, v) for k, v in parse_qsl(p.query) if not k.startswith("utm_")]
    query = urlencode(sorted(qs))

    normalized = urlunparse((scheme, netloc, p.path or "", "", query, ""))
    return normalized


def _ensure_index_dir() -> Path:
    p = Path("data") / "dedup"
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_duplicate_by_sha256(sha256_hex: str) -> str | None:
    idx = _ensure_index_dir() / "index.json"
    if not idx.exists():
        return None
    try:
        data = read_json(idx)
    except Exception:
        return None
    return data.get(sha256_hex)


def _shingle(text: str, k: int = DEFAULT_SHINGLE_K) -> set[str]:
    s = set()
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    joined = cleaned.replace("\n", " ")
    if len(joined) < k:
        return {joined}
    for i in range(len(joined) - k + 1):
        s.add(joined[i : i + k])
    return s


def find_near_duplicate(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    k: int = DEFAULT_SHINGLE_K,
) -> str | None:
    """Return an existing document_id if text is near-duplicate of an indexed document.

    `k` is the shingle size; `threshold` is Jaccard similarity cutoff.
    """
    sigs_file = _ensure_index_dir() / "signatures.json"
    if not sigs_file.exists():
        return None
    try:
        sigs = read_json(sigs_file)
    except Exception:
        return None

    s1 = _shingle(text, k=k)
    for doc_id, shingles in sigs.items():
        s2 = set(shingles)
        if not s2:
            continue
        inter = len(s1 & s2)
        union = len(s1 | s2)
        if union == 0:
            continue
        j = inter / union
        if j >= threshold:
            return doc_id
    return None


def register_document(
    sha256_hex: str,
    document_id: str,
    text: str,
    k: int = DEFAULT_SHINGLE_K,
) -> None:
    idx = _ensure_index_dir() / "index.json"
    sigs_file = _ensure_index_dir() / "signatures.json"

    # helper to perform atomic write with optimistic retry
    def _atomic_update(path: Path, update_fn, max_attempts: int = 8) -> None:
        for _attempt in range(max_attempts):
            try:
                current = read_json(path) if path.exists() else {}
            except Exception:
                current = {}

            new = update_fn(current)

            # merge with latest on disk to avoid overwriting concurrent updates
            try:
                latest = read_json(path) if path.exists() else {}
            except Exception:
                latest = {}

            merged = dict(latest)
            merged.update(new)

            # write to unique temp then replace to avoid tmp collisions
            tmp = path.with_suffix(f".tmp.{uuid4().hex}")
            try:
                write_json(tmp, merged)
                tmp.replace(path)
                return
            except Exception:
                # small backoff and retry
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
                time.sleep(0.02 * (_attempt + 1))
                continue

        # final fallback: overwrite directly
        write_json(path, new)

    # update index.json
    def _upd_index(cur: dict) -> dict:
        cur = dict(cur)
        cur[sha256_hex] = document_id
        return cur

    # serialize updates in-process to avoid races on tmp files
    with _write_lock:
        _atomic_update(idx, _upd_index)

    # update signatures using supplied k
    def _upd_sigs(cur: dict) -> dict:
        cur = dict(cur)
        cur[document_id] = list(_shingle(text, k=k))
        return cur

    with _write_lock:
        _atomic_update(sigs_file, _upd_sigs)
