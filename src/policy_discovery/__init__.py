from .base_adapter import BaseAdapter
from .icann_adapter import ICANNAdapter
from .igf_adapter import IGFAdapter
from .ietf_adapter import IETFAdapter
from .itu_adapter import ITUTAdapter
from .academic_adapter import AcademicAdapter
from .government_adapter import GovernmentAdapter

__all__ = [
    "BaseAdapter",
    "ICANNAdapter",
    "IGFAdapter",
    "IETFAdapter",
    "ITUTAdapter",
    "AcademicAdapter",
    "GovernmentAdapter",
]

ADAPTER_REGISTRY = {
    "icann": ICANNAdapter,
    "igf": IGFAdapter,
    "ietf": IETFAdapter,
    "itu": ITUTAdapter,
    "academic": AcademicAdapter,
    "government": GovernmentAdapter,
}

def get_adapter(source_id: str, config: dict = None) -> BaseAdapter:
    if source_id not in ADAPTER_REGISTRY:
        raise KeyError(f"No adapter registered for source_id: {source_id}")
    return ADAPTER_REGISTRY[source_id](source_id, config)

def discover_all(config: dict = None) -> dict:
    """
    Run discovery for all registered adapters (or a subset if config specifies).

    Args:
        config: Optional dict with keys:
            - "adapters": list of source_ids to run (default: all)
            - "max_pages": per-adapter page limit (overrides adapter default)

    Returns:
        Dict mapping source_id -> list of DiscoveredURL objects.
    """
    target_adapters = config.get("adapters", list(ADAPTER_REGISTRY.keys())) if config else list(ADAPTER_REGISTRY.keys())
    results = {}
    for sid in target_adapters:
        adapter = get_adapter(sid, config.get(sid, {}))
        results[sid] = adapter.discover_urls()
    return results

# ETL integration hook
def discover_and_process(config: dict = None) -> dict:
    """
    Run discovery and immediately process discovered URLs through ETL pipeline.

    Args:
        config: Optional dict with keys:
            - "adapters": list of source_ids to run (default: all)
            - "max_pages": per-adapter page limit (overrides adapter default)

    Returns:
        Dict mapping source_id -> list of ingested document IDs.
    """
    from ..arwen_etl.cli import ingest_url

    results = {}
    target_adapters = config.get("adapters", list(ADAPTER_REGISTRY.keys())) if config else list(ADAPTER_REGISTRY.keys())
    for sid in target_adapters:
        adapter = get_adapter(sid, config.get(sid, {}))
        urls = adapter.discover_urls()
        ingested_ids = []
        for url_obj in urls:
            # Pass the discovered URL and its provenance to ETL pipeline
            result = ingest_url(url_obj.url, provenance=url_obj.provenance if hasattr(url_obj, 'provenance') else None)
            if not result.get("error"):
                ingested_ids.append(result.get("document_id"))
        results[sid] = ingested_ids
    return results