from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

from .storage import write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def build_release_manifest(
    release_dir: str | Path,
    release_version: str,
) -> Path:
    root = Path(release_dir)
    root.mkdir(parents=True, exist_ok=True)

    entries = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "RELEASE_MANIFEST.json":
            continue

        entries.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

    manifest = {
        "release_version": release_version,
        "etl_version": package_version("arwen-policy-etl"),
        "generated_at": datetime.now(UTC).isoformat(),
        "files": entries,
    }

    return write_json(root / "RELEASE_MANIFEST.json", manifest)


def validate_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    errors = []

    for entry in manifest.get("files", []):
        target = manifest_path.parent / entry["path"]

        if not target.exists():
            errors.append(f"Missing file: {entry['path']}")
            continue

        actual = file_sha256(target)

        if actual != entry["sha256"]:
            errors.append(f"Hash mismatch: {entry['path']}")

    return errors