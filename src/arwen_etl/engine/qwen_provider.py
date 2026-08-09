"""Qwen model provider using local Ollama for real inference."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from . import ModelProvider


class QwenProvider(ModelProvider):
    """Real Qwen model provider backed by local Ollama.

    Falls back to a deterministic lightweight response when Ollama is unavailable
    (e.g. in CI), but logs a warning so the situation is observable.
    """

    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or os.getenv("QWEN_MODEL", "qwen3:latest")
        self._ollama_available: bool | None = None

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

    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        start = time.time()

        if self._check_ollama():
            return self._generate_ollama(prompt, context, start)
        return self._generate_fallback(prompt, context, start)

    def _generate_ollama(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]],
        start: float,
    ) -> Dict[str, Any]:
        import ollama

        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are an evidence-grounded policy analyst for the Arwen "
                    "Policy project. Answer using the provided evidence. "
                    "Preserve stakeholder disagreements. Disclose missing perspectives. "
                    "Never fabricate evidence or manufacture consensus. "
                    "When asked to analyse a claim, return a JSON object with: "
                    '{"pro_argument": "...", "pro_confidence": 0.0, '
                    '"contra_argument": "...", "contra_confidence": 0.0, '
                    '"analysis": "..."}.'
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = ollama.chat(model=self.model_id, messages=messages)
            output_text = response.get("message", {}).get("content", "")
        except Exception:
            return self._generate_fallback(prompt, context, start)

        latency_ms = int((time.time() - start) * 1000)

        return {
            "provider": "qwen",
            "model": self.model_id,
            "prompt": prompt,
            "context": context,
            "output": output_text,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output_text.split()),
            },
            "latency_ms": latency_ms,
            "provenance": {
                "model_revision": self.model_id,
                "backend": "ollama",
                "timestamp": time.time(),
            },
            "error": None,
        }

    def _generate_fallback(
        self,
        prompt: str,
        context: Optional[List[Dict[str, Any]]],
        start: float,
    ) -> Dict[str, Any]:
        """Deterministic fallback for environments without Ollama (e.g. CI)."""
        import logging

        logging.getLogger(__name__).warning(
            "Ollama not available — using deterministic fallback (not a real LLM). "
            "Install Ollama and pull '%s' for real inference.",
            self.model_id,
        )

        output_text = (
            f'{{"pro_argument": "The claim is supported by the evidence provided.", '
            f'"pro_confidence": 0.7, '
            f'"contra_argument": "Counterarguments should be considered; '
            f'additional stakeholder perspectives may alter the assessment.", '
            f'"contra_confidence": 0.5, '
            f'"analysis": "Deterministic fallback — Ollama not available."}}'
        )
        latency_ms = int((time.time() - start) * 1000)

        return {
            "provider": "qwen",
            "model": self.model_id,
            "prompt": prompt,
            "context": context,
            "output": output_text,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output_text.split()),
            },
            "latency_ms": latency_ms,
            "provenance": {
                "model_revision": "fallback",
                "backend": "deterministic",
                "timestamp": time.time(),
            },
            "error": "OLLAMA_UNAVAILABLE",
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "qwen",
            "model_id": self.model_id,
            "backend": "ollama" if self._check_ollama() else "fallback",
        }
