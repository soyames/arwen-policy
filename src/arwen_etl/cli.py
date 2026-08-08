from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from uuid import uuid4

import httpx

from . import __version__
from .capture import CaptureError, capture_url, write_artifact
from .config import load_pipeline_config, load_sources_config
from .deduplication import (
    find_near_duplicate,
    is_duplicate_by_sha256,
    register_document,
)
from .discovery import (
    discover_links_from_html,
    source_id_from_url,
    validate_public_url,
)
from .extraction import extract, extract_metadata
from .models import SourceRecord
from .normalization import normalize_text
from .provenance import provenance_event
from .publish import publish_release, publish_release_git_lfs
from .quality import extraction_quality
from .registry import load_registry
from .release import build_release_manifest, validate_manifest
from .review import enqueue_review
from .segmentation import segment_text
from .storage import write_json


def _pipeline_config() -> dict:
    return load_pipeline_config()


def _load_registry() -> dict:
    return load_registry()


def ingest_url(url: str) -> dict:
    config = _pipeline_config()
    capture_cfg = config["capture"]
    validate_public_url(url, set(capture_cfg["allowed_schemes"]))
    try:
        artifact = capture_url(
            url,
            timeout_seconds=capture_cfg["timeout_seconds"],
            max_download_mb=capture_cfg["max_download_mb"],
            max_redirects=capture_cfg["max_redirects"],
            user_agent=capture_cfg["user_agent"],
            respect_robots=capture_cfg.get("respect_robots", True),
        )
    except CaptureError as exc:
        # capture_url already recorded a provenance rejection; return structured response
        source_id = source_id_from_url(url)
        rejection_path = Path("data") / "rejections" / f"{source_id}.json"
        return {
            "error": True,
            "reason": getattr(exc, "reason", "CAPTURE_ERROR"),
            "details": getattr(exc, "details", {}),
            "provenance": str(rejection_path) if rejection_path.exists() else None,
        }
    except PermissionError as exc:
        # SourcePolicy rejection already recorded by capture; surface structured result
        source_id = source_id_from_url(url)
        data_dir = Path("data")
        rejection_path = data_dir / "rejections" / f"{source_id}.json"
        return {
            "error": True,
            "reason": "POLICY_REJECTION",
            "message": str(exc),
            "provenance": str(rejection_path) if rejection_path.exists() else None,
        }

    data_dir = Path("data")
    # quick deduplication by artifact hash
    existing = is_duplicate_by_sha256(artifact.sha256)
    if existing:
        return {
            "error": True,
            "reason": "DUPLICATE_ARTIFACT",
            "existing_document_id": existing,
        }

    raw_path = write_artifact(artifact, data_dir / "raw")

    extracted = extract(artifact.data, artifact.content_type)
    metadata = extract_metadata(artifact.data, artifact.content_type, extracted)
    normalized = normalize_text(extracted.text)

    document_id = str(uuid4())
    segments = segment_text(
        normalized,
        document_id,
        target_chars=config["processing"]["segment_target_chars"],
        overlap_chars=config["processing"]["segment_overlap_chars"],
    )

    source = SourceRecord(
        source_id=source_id_from_url(url),
        source_url=url,
        final_url=artifact.final_url,
        content_type=artifact.content_type,
        media_type=extracted.media_type,
        artifact_sha256=artifact.sha256,
        byte_size=len(artifact.data),
        extraction_status="extracted" if normalized else "failed",
    )

    extracted_record = {
        "document_id": document_id,
        "source_id": source.source_id,
        "source_url": source.source_url,
        "final_url": source.final_url,
        "artifact_sha256": source.artifact_sha256,
        "byte_size": source.byte_size,
        "content_type": source.content_type,
        "metadata": metadata,
        "extraction_method": extracted.method,
        "extraction_warnings": extracted.warnings,
        "text": normalized,
        "segments": [segment.model_dump(mode="json") for segment in segments],
    }

    # deduplication: near-duplicate detection on normalized text
    near = find_near_duplicate(normalized, threshold=0.9)
    if near:
        return {
            "error": True,
            "reason": "NEAR_DUPLICATE",
            "existing_document_id": near,
        }

    registry_path = data_dir / "registry" / f"{source.source_id}.json"
    extracted_path = data_dir / "extracted" / f"{document_id}.json"
    provenance_path = data_dir / "registry" / f"{source.source_id}.provenance.json"

    write_json(registry_path, source.model_dump(mode="json"))
    write_json(extracted_path, extracted_record)
    write_json(
        provenance_path,
        provenance_event(
            "capture_and_extract",
            entity_id=document_id,
            agent=f"arwen-policy-etl/{__version__}",
            input_ids=[artifact.sha256],
            attributes={
                "requested_url": url,
                "final_url": artifact.final_url,
                "capture_status": artifact.status_code,
                "etag": artifact.etag,
                "last_modified": artifact.last_modified,
                "extraction_method": extracted.method,
                "metadata": metadata,
            },
        ),
    )

    # register for deduplication lookup
    register_document(artifact.sha256, document_id, normalized)

    return {
        "source_id": source.source_id,
        "document_id": document_id,
        "raw_artifact": str(raw_path),
        "extracted_record": str(extracted_path),
        "segment_count": len(segments),
    }


def ingest_file(path: str) -> dict:
    source_path = Path(path)
    data = source_path.read_bytes()
    media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    extracted = extract(data, media_type)
    normalized = normalize_text(extracted.text)
    document_id = str(uuid4())
    segments = segment_text(document_id=document_id, text=normalized)

    return {
        "document_id": document_id,
        "media_type": media_type,
        "extraction_method": extracted.method,
        "text_length": len(normalized),
        "segment_count": len(segments),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="arwen-etl")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-config")

    ingest_url_parser = sub.add_parser("ingest-url")
    ingest_url_parser.add_argument("url")
    
    discover_parser = sub.add_parser("discover")
    discover_parser.add_argument("source_id", nargs="?", help="Optional source id to discover")

    ingest_file_parser = sub.add_parser("ingest-file")
    ingest_file_parser.add_argument("path")

    release_parser = sub.add_parser("build-release")
    release_parser.add_argument("--version", required=True)

    publish_parser = sub.add_parser("publish")
    publish_parser.add_argument("--version", required=True)
    publish_parser.add_argument(
        "--git-lfs",
        action="store_true",
        help="Prepare a git repository with Git LFS pointers for large files",
    )
    publish_parser.add_argument(
        "--push",
        action="store_true",
        help="When used with --git-lfs, attempt to push to the provided --repo-url",
    )
    publish_parser.add_argument(
        "--repo-url",
        help="Remote git URL to push to when using --git-lfs and --push",
    )
    publish_parser.add_argument(
        "--lfs-threshold",
        type=int,
        default=50 * 1024 * 1024,
        help="File size (bytes) threshold to track with Git LFS",
    )

    validate_release_parser = sub.add_parser("validate-release")
    validate_release_parser.add_argument("path")

    args = parser.parse_args()

    if args.command == "validate-config":
        load_pipeline_config()
        load_sources_config()
        print("Configuration valid.")
    elif args.command == "ingest-url":
        print(json.dumps(ingest_url(args.url), indent=2))
    elif args.command == "discover":
        registry = load_registry()
        targets = [s for s in registry.sources if s.enabled]
        if args.source_id:
            targets = [s for s in targets if s.id == args.source_id]

        output: dict = {}
        for s in targets:
            root = f"https://{s.domains[0]}"
            try:
                resp = httpx.get(root, timeout=10.0)
                resp.raise_for_status()
            except Exception as exc:
                print(f"Failed to fetch {root}: {exc}")
                continue

            discovered = discover_links_from_html(root, resp.text)
            out_path = Path("data") / "discovered" / f"{s.id}.json"
            write_json(out_path, [d.__dict__ for d in discovered])
            output[s.id] = len(discovered)

        print(json.dumps(output, indent=2))
    elif args.command == "pipeline":
        # simple pipeline: discover -> ingest for enabled registry entries
        registry = load_registry()
        targets = [s for s in registry.sources if s.enabled]
        output = {}
        for s in targets:
            root = f"https://{s.domains[0]}"
            try:
                resp = httpx.get(root, timeout=10.0)
                resp.raise_for_status()
            except Exception as exc:
                output[s.id] = {"error": str(exc)}
                continue

            discovered = discover_links_from_html(root, resp.text)
            count = 0
            for d in discovered:
                res = ingest_url(d.url)
                count += 1
                # if extraction low quality, enqueue for review
                if not res.get("error"):
                    doc = res.get("document_id")
                    # read extracted text to assess quality
                    extracted_path = Path(res.get("extracted_record"))
                    try:
                        rec = json.loads(extracted_path.read_text(encoding="utf-8"))
                        q = extraction_quality(rec.get("text", ""))
                        if q["score"] < 0.5:
                            enqueue_review(doc, "low_quality", {"score": q["score"]})
                    except Exception:
                        pass

            output[s.id] = {"discovered": len(discovered), "ingested": count}

        print(json.dumps(output, indent=2))
    elif args.command == "publish":
        # publish a release dir: first validate, then publish
        version = args.version
        release_dir = Path("data/releases") / version
        try:
            errors = validate_manifest(release_dir / "RELEASE_MANIFEST.json")
        except Exception as exc:
            print(f"Validation failed: {exc}")
            raise SystemExit(1) from exc

        if errors:
            for e in errors:
                print(e)
            raise SystemExit(1)

        token = os.environ.get("HF_TOKEN")
        repo_id = os.environ.get("HF_REPO", f"arwen-policy-etl/{version}")

        if args.git_lfs:
            # use git-lfs path
            try:
                result = publish_release_git_lfs(
                    release_dir,
                    repo_url=args.repo_url,
                    push=args.push,
                    lfs_threshold_bytes=args.lfs_threshold,
                )
            except Exception as exc:
                print(f"git-lfs publish failed: {exc}")
                raise SystemExit(1) from exc
        else:
            result = publish_release(release_dir, repo_id, token=token)

        print(json.dumps(result, indent=2))
    elif args.command == "ingest-file":
        print(json.dumps(ingest_file(args.path), indent=2))
    elif args.command == "build-release":
        path = build_release_manifest(Path("data/releases") / args.version, args.version)
        print(path)
    elif args.command == "validate-release":
        errors = validate_manifest(Path(args.path) / "RELEASE_MANIFEST.json")
        if errors:
            for error in errors:
                print(error)
            raise SystemExit(1)
        print("Release valid.")


if __name__ == "__main__":
    main()
