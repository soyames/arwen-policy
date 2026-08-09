"""Benchmark schema for evaluating model configurations."""

from typing import Dict, Any

BENCHMARK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "provider": {"type": "string"},           # e.g., "qwen", "claude", "openrouter"
        "model_version": {"type": "string"},      # model identifier / revision
        "with_retrieval": {"type": "boolean"},
        "with_deliberation": {"type": "boolean"},
        "response": {"type": "string"},
        "score": {"type": "number"},
        "provenance": {"type": "object"},
        "latency_ms": {"type": "integer"},
    },
    "required": ["query", "provider", "model_version"],
}

def validate_benchmark_result(result: Dict[str, Any]) -> bool:
    """Simple validation against BENCHMARK_SCHEMA (not full JSON Schema)."""
    for field in BENCHMARK_SCHEMA["required"]:
        if field not in result:
            return False
    return True