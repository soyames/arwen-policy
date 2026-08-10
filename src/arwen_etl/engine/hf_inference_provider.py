"""Hugging Face Inference API provider for Arwen Policy Space.

Uses the serverless HF Inference API with a configurable model.
Designed for Spaces where Ollama and dedicated endpoints are unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx

from . import ModelProvider
from .arwen_prompt import get_system_prompt

logger = logging.getLogger(__name__)

# Default model for HF Inference API — small enough for free tier.
# Configurable via MODEL_ID environment variable / Space Secret.
DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Fallback models if primary is unavailable
FALLBACK_MODELS = [
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

INFERENCE_API_URL = "https://api-inference.huggingface.co/models"


class HfInferenceProvider(ModelProvider):
    """Hugging Face serverless Inference API provider.

    Requires HF_TOKEN in environment/Space Secrets.
    Model defaults to Qwen/Qwen3-8B, configurable via MODEL_ID env var.
    """

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL)
        self._hf_token = os.getenv("HF_TOKEN")
        self._checked_availability: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, context: list[dict] | None = None) -> dict[str, Any]:
        start = time.time()

        if not self._hf_token:
            return self._error_result(
                prompt, context, start,
                "HF_TOKEN not set. Add it as a Space Secret.",
                "no_token",
            )

        return self._call_inference_api(prompt, context, start)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "provider": "hf_inference_api",
            "active_model": self.model_id,
            "backend": "hf_inference_api" if self._hf_token else "unconfigured",
            "token_configured": bool(self._hf_token),
        }

    # ------------------------------------------------------------------
    # HF Inference API call
    # ------------------------------------------------------------------

    def _call_inference_api(
        self, prompt: str, context: list[dict] | None, start: float,
    ) -> dict[str, Any]:
        api_url = f"{INFERENCE_API_URL}/{self.model_id}"
        headers = {
            "Authorization": f"Bearer {self._hf_token}",
            "Content-Type": "application/json",
        }

        system_msg = get_system_prompt()
        payload = {
            "inputs": (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        try:
            resp = httpx.post(api_url, json=payload, headers=headers, timeout=90)
        except httpx.TimeoutException:
            return self._error_result(
                prompt, context, start,
                f"Inference API timed out for model {self.model_id}.",
                "timeout",
            )
        except Exception as exc:
            return self._error_result(
                prompt, context, start,
                f"Inference API request failed: {exc}",
                "request_failed",
            )

        # Model loading (503) — retry once
        if resp.status_code == 503:
            wait = resp.json().get("estimated_time", 30)
            logger.info("Model %s loading (503), waiting %.0fs", self.model_id, wait)
            time.sleep(min(wait, 60))
            try:
                resp = httpx.post(api_url, json=payload, headers=headers, timeout=120)
            except Exception:
                return self._error_result(
                    prompt, context, start,
                    f"Model {self.model_id} still loading after retry.",
                    "model_loading",
                )

        if resp.status_code != 200:
            return self._error_result(
                prompt, context, start,
                f"Inference API error {resp.status_code}: {resp.text[:200]}",
                f"http_{resp.status_code}",
            )

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return self._error_result(
                prompt, context, start,
                "Inference API returned non-JSON response.",
                "bad_response",
            )

        output = data[0]["generated_text"] if isinstance(data, list) else str(data)

        return self._build_result(prompt, context, output, start, "hf_inference_api")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, prompt: str, context: list[dict] | None,
        output: str, start: float, backend: str,
    ) -> dict[str, Any]:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "provider": "hf_inference_api",
            "active_model": self.model_id,
            "output": output,
            "usage": {
                "prompt_chars": len(prompt),
                "completion_chars": len(output),
            },
            "latency_ms": latency_ms,
            "provenance": {
                "backend": backend,
                "model": self.model_id,
                "timestamp": time.time(),
            },
            "error": None,
        }

    def _error_result(
        self, prompt: str, context: list[dict] | None,
        start: float, message: str, error_code: str,
    ) -> dict[str, Any]:
        latency_ms = int((time.time() - start) * 1000)
        logger.warning("HF Inference error [%s]: %s", error_code, message)
        return {
            "provider": "hf_inference_api",
            "active_model": self.model_id,
            "output": None,
            "usage": {"prompt_chars": len(prompt), "completion_chars": 0},
            "latency_ms": latency_ms,
            "provenance": {
                "backend": "hf_inference_api_error",
                "model": self.model_id,
                "timestamp": time.time(),
            },
            "error": {
                "code": error_code,
                "message": message,
            },
        }
