"""Qwen3.6-27B model provider for Arwen Policy.

Production path: Hugging Face Inference API / Endpoint.
Development fallback: local Ollama (Qwen3 8.2B, NOT 27B).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from . import ModelProvider
from .arwen_prompt import get_system_prompt

logger = logging.getLogger(__name__)

# Verified from Qwen/Qwen3.6-27B config.json + model.safetensors.index.json:
#   model_type: qwen3_5_text
#   hidden_size: 5120, num_hidden_layers: 64
#   num_attention_heads: 24, num_key_value_heads: 4
#   intermediate_size: 17408, vocab_size: 248320
#   full_attention_interval: 4 (every 4th layer uses Gated Attention)
# LoRA target modules verified from weight map:
#   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
PRODUCTION_MODEL = "Qwen/Qwen3.6-27B"
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class QwenProvider(ModelProvider):
    """Qwen3.6-27B provider for Arwen Policy.

    Resolution order:
    1. HF_TOKEN + HF_INFERENCE_ENDPOINT (dedicated endpoint)
    2. HF_TOKEN + HF Inference API (serverless, production model)
    3. Local Ollama (development only — Qwen3 8.2B, NOT 27B)
    4. Deterministic fallback with explicit label
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("QWEN_MODEL", PRODUCTION_MODEL)
        self._ollama_available: bool | None = None
        self._backend: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        # 1. HF dedicated endpoint
        endpoint = os.getenv("HF_INFERENCE_ENDPOINT")
        hf_token = os.getenv("HF_TOKEN")
        if endpoint and hf_token:
            return self._generate_via_endpoint(endpoint, hf_token, prompt, context, start)

        # 2. HF serverless Inference API
        if hf_token:
            return self._generate_via_hf_api(hf_token, prompt, context, start)

        # 3. Local Ollama
        if self._check_ollama():
            return self._generate_ollama(prompt, context, start)

        # 4. Deterministic fallback
        return self._generate_fallback(prompt, context, start)

    def get_model_info(self) -> Dict[str, Any]:
        backend = self._resolve_backend()
        return {
            "provider": "qwen",
            "production_model": PRODUCTION_MODEL,
            "active_model": self.model_id,
            "backend": backend,
            "lora_target_modules": LORA_TARGET_MODULES,
            "architecture": {
                "model_type": "qwen3_5_text",
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "intermediate_size": 17408,
                "full_attention_interval": 4,
            },
        }

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import ollama
            ollama.list()
            self._ollama_available = True
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def _resolve_backend(self) -> str:
        if self._backend:
            return self._backend
        if os.getenv("HF_INFERENCE_ENDPOINT") and os.getenv("HF_TOKEN"):
            self._backend = "hf_endpoint"
        elif os.getenv("HF_TOKEN"):
            self._backend = "hf_inference_api"
        elif self._check_ollama():
            self._backend = "ollama_dev"
        else:
            self._backend = "fallback"
        return self._backend

    # ------------------------------------------------------------------
    # HF Inference Endpoint
    # ------------------------------------------------------------------

    def _generate_via_endpoint(
        self, endpoint: str, token: str, prompt: str,
        context: Optional[List[Dict]], start: float,
    ) -> Dict[str, Any]:
        import httpx

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.9,
                "do_sample": True,
            },
        }
        try:
            resp = httpx.post(endpoint, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            output = data[0]["generated_text"] if isinstance(data, list) else str(data)
        except Exception as exc:
            logger.warning("HF endpoint failed: %s", exc)
            return self._generate_ollama(prompt, context, start)

        return self._build_result(prompt, context, output, start, "hf_endpoint")

    # ------------------------------------------------------------------
    # HF Serverless Inference API
    # ------------------------------------------------------------------

    def _generate_via_hf_api(
        self, token: str, prompt: str,
        context: Optional[List[Dict]], start: float,
    ) -> Dict[str, Any]:
        import httpx

        api_url = f"https://api-inference.huggingface.co/models/{PRODUCTION_MODEL}"
        headers = {"Authorization": f"Bearer {token}"}
        system_msg = get_system_prompt()
        payload = {
            "inputs": f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                      f"<|im_start|>user\n{prompt}<|im_end|>\n"
                      f"<|im_start|>assistant\n",
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
            },
        }
        try:
            resp = httpx.post(api_url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 503:
                # Model is loading — return loading status
                logger.info("HF model loading (503). Estimated wait: %s",
                            resp.json().get("estimated_time", "unknown"))
                return self._build_result(
                    prompt, context,
                    "[Model loading on HF — retry in a moment]",
                    start, "hf_inference_api_loading",
                )
            resp.raise_for_status()
            data = resp.json()
            output = data[0]["generated_text"] if isinstance(data, list) else str(data)
        except Exception as exc:
            logger.warning("HF Inference API failed: %s", exc)
            if self._check_ollama():
                return self._generate_ollama(prompt, context, start)
            return self._generate_fallback(prompt, context, start)

        return self._build_result(prompt, context, output, start, "hf_inference_api")

    # ------------------------------------------------------------------
    # Local Ollama (development only — Qwen3 8.2B, NOT 27B)
    # ------------------------------------------------------------------

    def _generate_ollama(
        self, prompt: str, context: Optional[List[Dict]], start: float,
    ) -> Dict[str, Any]:
        import ollama

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = ollama.chat(model=self.model_id, messages=messages)
            output = resp.get("message", {}).get("content", "")
            return self._build_result(prompt, context, output, start, "ollama_dev")
        except Exception:
            return self._generate_fallback(prompt, context, start)

    # ------------------------------------------------------------------
    # Fallback (explicitly labeled)
    # ------------------------------------------------------------------

    def _generate_fallback(
        self, prompt: str, context: Optional[List[Dict]], start: float,
    ) -> Dict[str, Any]:
        logger.warning("All model backends unavailable — using deterministic fallback.")
        output = (
            '{"analysis": "FALLBACK: No model backend available. '
            'This is a deterministic placeholder, not a real LLM response. '
            'Set HF_TOKEN for HF Inference API or install Ollama for local dev.", '
            '"pro_argument": "", "pro_confidence": 0.0, '
            '"contra_argument": "", "contra_confidence": 0.0}'
        )
        return self._build_result(prompt, context, output, start, "fallback")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, prompt: str, context: Optional[List[Dict]],
        output: str, start: float, backend: str,
    ) -> Dict[str, Any]:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "provider": "qwen",
            "production_model": PRODUCTION_MODEL,
            "active_model": self.model_id,
            "prompt": prompt,
            "context": context,
            "output": output,
            "usage": {
                "prompt_chars": len(prompt),
                "completion_chars": len(output),
            },
            "latency_ms": latency_ms,
            "provenance": {
                "production_model": PRODUCTION_MODEL,
                "backend": backend,
                "lora_applied": False,
                "timestamp": time.time(),
            },
            "error": None if backend != "fallback" else "ALL_BACKENDS_UNAVAILABLE",
        }
