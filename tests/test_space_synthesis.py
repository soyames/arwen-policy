"""Tests for Space model synthesis — HF InferenceClient, config, error handling."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestHfInferenceProvider:
    """HfInferenceProvider configuration and error paths."""

    def test_no_token_reports_unconfigured(self):
        """Without HF_TOKEN, backend must report unconfigured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("MODEL_ID", None)
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            info = provider.get_model_info()
            assert info["token_configured"] is False
            assert info["backend"] == "unconfigured"

    def test_with_token_reports_configured(self):
        """With HF_TOKEN, backend must report configured."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            info = provider.get_model_info()
            assert info["token_configured"] is True

    def test_generate_without_token_returns_error(self):
        """generate() without HF_TOKEN returns structured error, never fake text."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("MODEL_ID", None)
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            result = provider.generate("test prompt")
            assert result["output"] is None
            assert result["error"] is not None
            assert result["error"]["code"] == "no_token"

    def test_default_model_is_qwen3_8b(self):
        """Default model must be Qwen/Qwen3-8B (not 27B)."""
        from arwen_etl.engine.hf_inference_provider import DEFAULT_MODEL
        assert "Qwen3-8B" in DEFAULT_MODEL

    def test_model_id_configurable(self):
        """MODEL_ID env var overrides default model."""
        with patch.dict(os.environ, {"MODEL_ID": "test/model"}, clear=False):
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            assert provider.model_id == "test/model"

    def test_successful_chat_completion(self):
        """Successful chat_completion() call returns output without error."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Policy analysis result"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with patch("huggingface_hub.InferenceClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat_completion.return_value = mock_response
                mock_client_cls.return_value = mock_client

                from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
                provider = HfInferenceProvider()
                result = provider.generate("test prompt")
                assert result["output"] == "Policy analysis result"
                assert result["error"] is None

    def test_inference_error_returns_structured_error(self):
        """InferenceClient errors return structured error, never fallback text."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with patch("huggingface_hub.InferenceClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat_completion.side_effect = RuntimeError("Model unavailable (503)")
                mock_client_cls.return_value = mock_client

                from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
                provider = HfInferenceProvider()
                result = provider.generate("test prompt")
                assert result["output"] is None
                assert result["error"] is not None
                assert "503" in result["error"]["message"]

    def test_auth_error_detected(self):
        """401/403 errors are classified as auth errors."""
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=False):
            with patch("huggingface_hub.InferenceClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.chat_completion.side_effect = RuntimeError("401 Unauthorized")
                mock_client_cls.return_value = mock_client

                from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
                provider = HfInferenceProvider()
                result = provider.generate("test")
                assert result["error"]["code"] == "auth_error"


class TestSpaceSynthesisFlow:
    """Space-level: retrieval works independently of model synthesis."""

    def test_retrieval_works_without_model(self):
        """Retrieval must succeed even when model provider is None."""
        from arwen_deliberation.council import DeliberationCouncil
        from arwen_engine.models import PolicyRequest
        from arwen_engine.pipeline import ArwenPolicyEngine
        from arwen_retrieval.models import CorpusRecord
        from arwen_retrieval.retriever import InMemoryRetriever
        from arwen_retrieval.service import RetrievalService

        records = [
            CorpusRecord(record_id="test-1", text="ICANN coordinates DNS.",
                         source_id="s1", document_id="d1", title="T1", url="http://x", topics=()),
        ]
        engine = ArwenPolicyEngine(
            retrieval=RetrievalService(InMemoryRetriever(records)),
            council=DeliberationCouncil(),
            model_provider=None,
        )
        answer = engine.analyze(PolicyRequest(question_id="q", question="What is DNS?", top_k=1))
        assert answer.status in ("success", "ready_for_model_synthesis")

    def test_engine_does_not_crash_without_model(self):
        """Engine must not crash when model_provider is None."""
        from arwen_deliberation.council import DeliberationCouncil
        from arwen_engine.pipeline import ArwenPolicyEngine
        from arwen_retrieval.retriever import InMemoryRetriever
        from arwen_retrieval.service import RetrievalService
        from arwen_retrieval.models import CorpusRecord
        records = [CorpusRecord(record_id="r1", text="Evidence.", source_id="s",
                                 document_id="d", title="T", url="http://x", topics=())]
        engine = ArwenPolicyEngine(
            retrieval=RetrievalService(InMemoryRetriever(records)),
            council=DeliberationCouncil(), model_provider=None,
        )
        from arwen_engine.models import PolicyRequest
        answer = engine.analyze(PolicyRequest(question_id="q", question="t", top_k=1))
        assert answer.evidence is not None


class TestNoFakeAnalysis:
    """Production must NEVER return deterministic fake analysis."""

    def test_hf_provider_never_returns_fallback(self):
        """HfInferenceProvider returns None output + error, never fake text."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            result = provider.generate("test")
            assert result["output"] is None
            assert "placeholder" not in str(result).lower()
            assert "FALLBACK" not in str(result).upper()
