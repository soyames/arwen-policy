import os
import time
from typing import Optional, List, Dict, Any

from . import ModelProvider


class QwenProvider(ModelProvider):
    """Qwen model provider using transformers or API.
    In CI/tests, this uses a lightweight mock.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("QWEN_MODEL", "qwen-14b")
        # In a real implementation you would load the model here
        # but we keep it lightweight for testing

    def generate(self, prompt: str, context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        start = time.time()
        # Mock generation: return a deterministic response for CI
        output_text = f"[QWEN-{self.model_id} MOCK] Response to: {prompt[:80]}..."
        latency_ms = int((time.time() - start) * 1000)

        return {
            "provider": "qwen",
            "model": self.model_id,
            "prompt": prompt,
            "context": context,
            "output": output_text,
            "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(output_text.split())},
            "latency_ms": latency_ms,
            "provenance": {
                "model_revision": "mock-rev",
                "timestamp": time.time(),
            },
            "error": None,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "qwen",
            "model_id": self.model_id,
            "revision": "mock-rev",
        }