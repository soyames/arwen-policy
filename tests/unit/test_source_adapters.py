from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from arwen_etl import cli
from arwen_etl.registry import SourceDefinition, load_registry
from arwen_etl.sources.adapter import (
    GenericSourceAdapter,
    SourceDiscoveryBundle,
    SourceHealthRecord,
)
from arwen_etl.sources.generic import source_id_from_url
from arwen_etl.sources.health import build_source_health_index


def _fake_artifact() -> SimpleNamespace:
    html = (
        b"<html lang='en'><head>"
        b"<title>ICANN Draft Policy on Internet Governance and Domain Name System Oversight</title>"
        b"<meta property='article:published_time' content='2026-01-02T03:04:05Z' />"
        b"</head><body><a href='/policy'>Policy</a><p>This document outlines the multistakeholder approach.</p></body></html>"
    )
    return SimpleNamespace(
        source_id="abc123",
        final_url="https://example.org/home",
        status_code=200,
        content_type="text/html",
        data=html,
        sha256="a" * 64,
        etag=None,
        last_modified=None,
    )


def test_generic_source_adapter_discovers_links_and_records_health(monkeypatch):
    source = SourceDefinition(
        id="icann",
        name="ICANN",
        family="internet_governance",
        domains=["example.org"],
        discovery={"seed_urls": ["https://example.org/home"]},
    )

    monkeypatch.setattr(
        "arwen_etl.sources.adapter.load_pipeline_config",
        lambda: {
            "capture": {
                "timeout_seconds": 1,
                "max_download_mb": 1,
                "max_redirects": 1,
                "user_agent": "test",
                "respect_robots": True,
            }
        },
    )
    monkeypatch.setattr(
        "arwen_etl.sources.adapter.capture_url",
        lambda url, **kwargs: _fake_artifact(),
    )

    result = GenericSourceAdapter(source).discover()

    assert result.health.source_status == "reachable"
    assert result.health.title == "ICANN Draft Policy on Internet Governance and Domain Name System Oversight"
    assert result.health.document_identifier == "abc123"
    assert result.health.published_at is not None
    assert any(item.url == "https://example.org/policy" for item in result.discovered_urls)


def test_ingest_url_enriches_source_metadata(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    registry = load_registry(repo_root / "configs" / "sources.yaml")
    config = {
        "capture": {
            "allowed_schemes": ["https"],
            "timeout_seconds": 1,
            "max_download_mb": 1,
            "max_redirects": 1,
            "user_agent": "test",
            "respect_robots": True,
        },
        "processing": {
            "segment_target_chars": 1800,
            "segment_overlap_chars": 200,
        },
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_pipeline_config", lambda: config)
    monkeypatch.setattr(cli, "load_registry", lambda: registry)
    monkeypatch.setattr(cli, "capture_url", lambda url, **kwargs: _fake_artifact())
    monkeypatch.setattr(cli, "is_duplicate_by_sha256", lambda sha: None)
    monkeypatch.setattr(cli, "find_near_duplicate", lambda text, threshold=0.9: None)
    monkeypatch.setattr(cli, "register_document", lambda *args, **kwargs: None)

    result = cli.ingest_url("https://www.icann.org/example")
    assert result["source_id"] == source_id_from_url("https://www.icann.org/example")

    registry_file = tmp_path / "data" / "registry" / f"{result['source_id']}.json"
    payload = registry_file.read_text(encoding="utf-8")
    assert "\"source_family\": \"internet_governance\"" in payload
    assert "\"publisher\": \"ICANN\"" in payload
    assert "\"source_adapter\": \"generic-web-adapter\"" in payload


def test_build_source_health_index_summarizes_sources():
    bundle = SourceDiscoveryBundle(
        source_id="icann",
        source_name="ICANN",
        source_family="internet_governance",
        adapter="generic-web-adapter",
        health=SourceHealthRecord(
            source_id="icann",
            source_name="ICANN",
            source_family="internet_governance",
            adapter="generic-web-adapter",
            source_url="https://example.org/",
            source_status="reachable",
        ),
        discovered_urls=[],
    )
    index = build_source_health_index([bundle])
    assert index["source_count"] == 1
    assert index["status_counts"]["reachable"] == 1
    assert index["family_counts"]["internet_governance"] == 1


def test_generic_adapter_dedupes_discovered_urls(monkeypatch):
    source = SourceDefinition(
        id="icann",
        name="ICANN",
        family="internet_governance",
        domains=["example.org"],
        discovery={"seed_urls": ["https://example.org/home"]},
    )
    html = (
        b"<html><head><title>Dupes</title></head><body>"
        b"<a href='/policy'>A</a><a href='/policy'>B</a>"
        b"</body></html>"
    )
    artifact = SimpleNamespace(
        source_id="abc123",
        final_url="https://example.org/home",
        status_code=200,
        content_type="text/html",
        data=html,
        sha256="b" * 64,
        etag=None,
        last_modified=None,
    )
    monkeypatch.setattr(
        "arwen_etl.sources.adapter.load_pipeline_config",
        lambda: {
            "capture": {
                "timeout_seconds": 1,
                "max_download_mb": 1,
                "max_redirects": 1,
                "user_agent": "test",
                "respect_robots": True,
            }
        },
    )
    monkeypatch.setattr(
        "arwen_etl.sources.adapter.capture_url",
        lambda url, **kwargs: artifact,
    )

    result = GenericSourceAdapter(source).discover()
    policy_urls = [item.url for item in result.discovered_urls if item.url.endswith("/policy")]
    assert policy_urls.count("https://example.org/policy") == 1


def test_registry_discovery_seeds_present():
    registry = load_registry()
    icann = next(s for s in registry.sources if s.id == "icann")
    assert icann.discovery
    assert icann.discovery.get("seed_urls")
