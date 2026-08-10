"""Hugging Face Inference Provider for Arwen Policy Space.

Uses the official huggingface_hub InferenceClient with chat_completion,
which leverages HF Inference Providers (serverless, not the deprecated
api-inference endpoint). Designed for Spaces where Ollama and dedicated
endpoints are unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from . import ModelProvider
from .arwen_prompt import get_system_prompt

logger = logging.getLogger(__name__)

# Default model for HF Inference Providers — Qwen3-8B works with free tier.
# Configurable via MODEL_ID environment variable / Space Variable.
DEFAULT_MODEL = "Qwen/Qwen3-8B"


class HfInferenceProvider(ModelProvider):
    """Hugging Face serverless Inference Provider using InferenceClient.

    Requires HF_TOKEN in environment / Space Secrets.
    Model defaults to Qwen/Qwen3-8B, configurable via MODEL_ID env var.

    Uses InferenceClient.chat_completion() which:
    - Routes through HF Inference Providers (not deprecated api-inference)
    - Handles chat templating automatically
    - Supports serverless/free tier
    """

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL)
        self._hf_token = os.getenv("HF_TOKEN")
        self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, context: list[dict] | None = None) -> dict[str, Any]:
        start = time.time()

        if not self._hf_token:
            return self._error_result(
                prompt, start,
                "HF_TOKEN not set. Add it as a Space Secret in Settings.",
                "no_token",
            )

        # Lazy-init client (avoids import overhead when token is missing)
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(model=self.model_id, token=self._hf_token)
            except ImportError:
                return self._error_result(
                    prompt, start,
                    "huggingface_hub not installed. Required for inference.",
                    "missing_dependency",
                )
            except Exception as exc:
                return self._error_result(
                    prompt, start,
                    f"Failed to initialize InferenceClient: {exc}",
                    "client_init_failed",
                )

        return self._call_chat_completion(prompt, start)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "hf_inference_providers",
            "active_model": self.model_id,
            "backend": "hf_inference_providers" if self._hf_token else "unconfigured",
            "token_configured": bool(self._hf_token),
        }

    # ------------------------------------------------------------------
    # Chat completion via InferenceClient
    # ------------------------------------------------------------------

    def _call_chat_completion(self, prompt: str, start: float) -> dict[str, Any]:
        system_msg = get_system_prompt()
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
                top_p=0.9,
            )
        except Exception as exc:
            error_str = str(exc)
            code = "inference_failed"
            if "401" in error_str or "403" in error_str or "auth" in error_str.lower():
                code = "auth_error"
            elif "429" in error_str or "rate" in error_str.lower():
                code = "rate_limited"
            elif "503" in error_str or "loading" in error_str.lower():
                code = "model_loading"
            elif "timeout" in error_str.lower():
                code = "timeout"
            elif "not found" in error_str.lower() or "404" in error_str:
                code = "model_not_found"
            return self._error_result(prompt, start, error_str[:300], code)

        # Extract content from chat completion response
        try:
            output = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError):
            return self._error_result(
                prompt, start,
                "Unexpected response format from inference provider.",
                "bad_response",
            )

        return self._build_result(prompt, output, start)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(self, prompt: str, output: str, start: float) -> dict[str, Any]:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "provider": "hf_inference_providers",
            "active_model": self.model_id,
            "output": output,
            "usage": {
                "prompt_chars": len(prompt),
                "completion_chars": len(output),
            },
            "latency_ms": latency_ms,
            "provenance": {
                "backend": "hf_inference_providers",
                "model": self.model_id,
                "timestamp": time.time(),
            },
            "error": None,
        }

    def _error_result(
        self, prompt: str, start: float, message: str, error_code: str,
    ) -> dict[str, Any]:
        latency_ms = int((time.time() - start) * 1000)
        logger.warning("HF Inference error [%s]: %s", error_code, message[:120])
        return {
            "provider": "hf_inference_providers",
            "active_model": self.model_id,
            "output": None,
            "usage": {"prompt_chars": len(prompt), "completion_chars": 0},
            "latency_ms": latency_ms,
            "provenance": {
                "backend": "hf_inference_providers_error",
                "model": self.model_id,
                "timestamp": time.time(),
            },
            "error": {
                "code": error_code,
                "message": message,
            },
        }
