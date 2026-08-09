from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def compute_hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()