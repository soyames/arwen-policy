import os
import json
import hashlib
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Dict, Any
from datetime import UTC, datetime as dt_utc
from importlib.metadata import version as package_version

from .storage import write_json
from .dataset_generator import DatasetGenerator

# ---------------------------------------------------------------------------
# Utility: Load HF token (now fully environment-agnostic)
# ---------------------------------------------------------------------------
def get_hf_token() -> str:
    """
    Retrieve the Hugging Face authentication token.

    The function follows this precedence:
    1. Environment variable ``HF_TOKEN`` (standard approach)
    2. File path specified via ``TOKEN_FILE_PATH`` environment variable
    3. Raise an informative error if neither source provides a token

    Returns:
        str: The raw token string
    """
    # 1��️��⃣  Environment variable (standard)
    token = os.getenv("HF_TOKEN")
    if token:
        return token.strip()

    # 2��️��⃣  Token file path from environment variable
    token_path = os.getenv("TOKEN_FILE_PATH")
    if token_path:
        try:
            return Path(token_path).read_text(encoding="utf-8").strip()
        except Exception as exc:
            raise RuntimeError("Failed to read token from specified file path") from exc

    # 3��️��⃣  No token found
    raise RuntimeError(
        "Hugging Face token not found. Set HF_TOKEN or TOKEN_FILE_PATH environment variable and retry."
    )

# ---------------------------------------------------------------------------
# Quality‑gate verification helpers
# ---------------------------------------------------------------------------
def _assert_min(value: Any, min_val: float, field_name: str, error_msg: str) -> None:
    """Raise an error if a numeric field does not meet the minimum value."""
    if isinstance(value, (int, float)):
        if value < min_val:
            raise ValueError(f"{field_name} must be ≥ {min_val} (got {value})")
    else:
        raise ValueError(f"{field_name} must be a number")

def _verify_record(record: Dict[str, Any]) -> None:
    """
    Verify that a record has all mandatory provenance fields and that
    confidence values are within [0, 1].
    """
    required = {
        "candidate_id": str,
        "candidate_type": str,
        "segment_id": str,
        "text": str,
        "extraction_method": str,
    }
    missing = [k for k in required if k not in record]
    for key in missing:
        raise ValueError(f"Missing required field '{key}'")

    # confidence must be between 0 and 1
    if "confidence" in record:
        if not (isinstance(record["confidence"], (int, float)) and
                0.0 <= record["confidence"] <= 1.0):
            raise ValueError("confidence must be a float in [0,1]")

    if "evidence_linked" in record and not isinstance(record["evidence_linked"], bool):
        raise ValueError("evidence_linked must be a boolean")

    if "evidence_confidence" in record:
        if not (isinstance(record["evidence_confidence"], (int, float)) and
                0.0 <= record["evidence_confidence"] <= 1.0):
            raise ValueError("evidence_confidence must be a float in [0,1]")


def _verify_manifest(manifest_path: Path) -> List[str]:
    """
    Verify every record referenced in the manifest passes the quality gate.
    Returns a list of error messages (empty if all records pass).
    """
    errors = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"Failed to read manifest: {e}"]

    for entry in manifest.get("files", []):
        target = manifest_path.parent / entry["path"]
        if not target.is_file():
            errors.append(f"Missing file: {entry['path']}")
            continue
        try:
            rec = json.loads(target.read_text(encoding="utf-8"))
            _verify_record(rec)
        except Exception as e:
            errors.append(f"Record validation failed for {entry['path']}: {e}")
    return errors


def generate_dataset(version: str = "0.1.0", data_dir: str = "data") -> Path:
    """
    Generate a Hugging Face dataset from the processed corpus.

    Args:
        version: Dataset version string
        data_dir: Directory containing processed data

    Returns:
        Path to the generated dataset directory
    """
    generator = DatasetGenerator(data_dir=data_dir)
    dataset = generator.build_dataset(version=version)
    dataset_path = generator.save_dataset(dataset)
    return dataset_path


# ---------------------------------------------------------------------------
# Upload with retry and email on failure
# ---------------------------------------------------------------------------
def upload_manifest_to_hf(
    manifest_path: str | Path,
    repo_id: str,
    token: str | None = None,
    dry_run: bool = False,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
) -> dict:
    """
    Upload a RELEASE_MANIFEST.json file to a Hugging Face repository.

    Performs:
      * Quality‑gate verification of the manifest (provenance, confidence, etc.)
      * Exponential back‑off retry on upload failures
      * Optional email alert on final failure (if SMTP config is present).

    Args:
        manifest_path: Path to the manifest file to upload
        repo_id: Hugging Face repo identifier (e.g., "soyames/arwen-policy-corpus")
        token: HF authentication token (uses HF_TOKEN env var if None)
        dry_run: If True, only validate but don't upload
        max_retries: Number of upload attempts before giving up
        backoff_factor: Multiplicative factor for wait time between retries

    Returns:
        Response dictionary from HF API or error information.
    """
    from huggingface_hub import upload_file, HfApi

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # ----- 1��️��⃣  Quality gate -------------------------------------------------
    verification_errors = _verify_manifest(manifest_path)
    if verification_errors:
        return {
            "success": False,
            "stage": "quality_gate",
            "errors": verification_errors,
        }

    if dry_run:
        return {"success": True, "message": "Dry run validated – quality gate passed"}

    # ----- 2��️��⃣  Prepare HF API ------------------------------------------------
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise EnvironmentError("HF_TOKEN environment variable not set")

    HfApi()
    if "/" in repo_id:
        repo_name, branch = repo_id.split("/", 1)
    else:
        branch = "main"

    # ----- 3��️��⃣  Upload with exponential back‑off -------------------------------
    attempt = 0
    wait_time = 1.0  # start with 1 second
    last_error = None

    while attempt <= max_retries:
        try:
            response = upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo="RELEASE_MANIFEST.json",
                repo_id=repo_id,
                token=token,
                commit_message=f"Release manifest update for version {manifest_path.stem}",
                branch=branch,
            )
            return {"success": True, "response": response}
        except Exception as exc:
            attempt += 1
            last_error = exc
            if attempt > max_retries:
                break
            # exponential back‑off
            time.sleep(wait_time)
            wait_time *= backoff_factor

    # ----- 4��️��⃣  Final failure – send email if configured -----------------------
    # Optional email alert
    try:
        from .utils import load_config
        cfg = load_config()
        if cfg.get("email_notifications", {}).get("enabled", False):
            # Build and send email
            msg = MIMEMultipart()
            msg["From"] = cfg.get("email_notifications", {}).get("from", "noreply@arwen-policy.org")
            msg["To"] = cfg.get("email_notifications", {}).get("to", "admin@arwen-policy.org")
            msg["Subject"] = "Arwen Policy – Release Upload Failed"
            body = f"""
                The release upload to {repo_id} failed after {max_retries} attempts.
                Last error: {last_error}
                Manifest: {manifest_path}
                Please investigate and retry manually if needed.
                """
            msg.attach(MIMEText(body, "plain"))

            smtp_cfg = cfg.get("email_notifications", {}).get("smtp", {})
            host = smtp_cfg.get("host", "smtp.office365.com")
            port = smtp_cfg.get("port", 587)
            user = smtp_cfg.get("user")
            password = smtp_cfg.get("password")

            with smtplib.SMTP(host, port) as server:
                server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
    except Exception as mail_exc:
        # If emailing fails, just log it; the primary error is still the upload failure.
        print(f"Warning: failed to send failure‑notification email: {mail_exc}")

    return {
        "success": False,
        "stage": "upload",
        "error": str(last_error),
        "attempts": attempt,
    }


# ---------------------------------------------------------------------------
# Release manifest builder and validator
# ---------------------------------------------------------------------------


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
        "generated_at": dt_utc.now(UTC).isoformat(),
        "files": entries,
    }

    return write_json(root / "RELEASE_MANIFEST.json", manifest)


def validate_manifest(path: str | Path) -> list[str]:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"Failed to read manifest: {e}"]

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