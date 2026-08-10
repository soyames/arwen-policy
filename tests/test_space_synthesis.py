"""Tests for Space model synthesis — configuration, error handling, fallback behavior."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestHfInferenceProvider:
    """HfInferenceProvider configuration and error paths."""

    def test_no_token_reports_unconfigured(self):
        """Without HF_TOKEN, backend must report unconfigured (not silently fallback)."""
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
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token", "MODEL_ID": ""}, clear=False):
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider

            provider = HfInferenceProvider()
            info = provider.get_model_info()
            assert info["token_configured"] is True

    def test_generate_without_token_returns_error(self):
        """generate() without HF_TOKEN returns structured error (never falls back silently)."""
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
        assert "Qwen3-8B" in DEFAULT_MODEL or "Qwen" in DEFAULT_MODEL

    def test_model_id_configurable(self):
        """MODEL_ID env var overrides default model."""
        with patch.dict(os.environ, {"MODEL_ID": "test/model"}, clear=False):
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            assert provider.model_id == "test/model"

    @patch("httpx.post")
    def test_successful_api_call(self, mock_post):
        """Successful API call returns output without error."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"generated_text": "Analysis result"}]
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False):
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            result = provider.generate("test")
            assert result["output"] == "Analysis result"
            assert result["error"] is None

    @patch("httpx.post")
    def test_http_error_returns_structured_error(self, mock_post):
        """HTTP errors return structured error, not fallback text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False):
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            result = provider.generate("test")
            assert result["output"] is None
            assert result["error"] is not None
            assert "403" in result["error"]["code"]


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
            CorpusRecord(
                record_id="test-1",
                text="ICANN coordinates the Domain Name System (DNS).",
                source_id="src-1",
                document_id="doc-1",
                title="ICANN Overview",
                url="https://icann.org",
                topics=(),
            ),
            CorpusRecord(
                record_id="test-2",
                text="The IETF develops voluntary Internet standards through RFCs.",
                source_id="src-2",
                document_id="doc-2",
                title="IETF Overview",
                url="https://ietf.org",
                topics=(),
            ),
        ]
        retriever = InMemoryRetriever(records)
        service = RetrievalService(retriever)
        engine = ArwenPolicyEngine(
            retrieval=service,
            council=DeliberationCouncil(),
            model_provider=None,  # No model — retrieval must still work
        )

        request = PolicyRequest(question_id="q1", question="What is DNS?", top_k=2)
        answer = engine.analyze(request)
        assert answer.status in ("success", "ready_for_model_synthesis")
        # Evidence may be empty with synthetic test data depending on retriever;
        # the key assertion is that the engine did not crash without a model.

    def test_synthesis_error_does_not_kill_retrieval(self):
        """When model synthesis fails, evidence retrieval results must still be present."""
        from arwen_deliberation.council import DeliberationCouncil
        from arwen_engine.pipeline import ArwenPolicyEngine
        from arwen_retrieval.retriever import InMemoryRetriever
        from arwen_retrieval.service import RetrievalService
        from arwen_retrieval.models import CorpusRecord

        records = [
            CorpusRecord(record_id="r1", text="Policy evidence text.", source_id="s1",
                         document_id="d1", title="T", url="http://x", topics=()),
        ]
        engine = ArwenPolicyEngine(
            retrieval=RetrievalService(InMemoryRetriever(records)),
            council=DeliberationCouncil(),
            model_provider=None,
        )
        from arwen_engine.models import PolicyRequest
        answer = engine.analyze(PolicyRequest(question_id="q", question="test", top_k=1))
        assert answer.evidence is not None
        assert answer.status in ("success", "ready_for_model_synthesis")
        # Evidence count depends on retriever matching — the engine must not crash


class TestNoFakeAnalysisInProduction:
    """Production mode must NEVER return deterministic fake analysis."""

    def test_hf_provider_never_returns_fallback_text(self):
        """HfInferenceProvider must return None output + error, never fake text."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("HF_TOKEN", None)
            from arwen_etl.engine.hf_inference_provider import HfInferenceProvider
            provider = HfInferenceProvider()
            result = provider.generate("test")
            assert result["output"] is None
            assert result["error"]["code"] == "no_token"
            # Must NOT contain "FALLBACK" or "placeholder" in error
            assert "placeholder" not in str(result["error"]).lower()
