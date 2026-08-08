from arwen_deliberation.models import Perspective
from arwen_engine.models import PolicyRequest
from arwen_engine.pipeline import ArwenPolicyEngine
from arwen_retrieval.models import CorpusRecord
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService


def test_retrieval_deliberation_engine_chain():
    corpus = [
        CorpusRecord(
            record_id="doc-1-seg-1",
            text="Civil society supports transparent AI governance and public accountability.",
            source_id="source-1",
            document_id="doc-1",
            segment_id="seg-1",
            stakeholder_groups=("civil_society",),
            topics=("ai_governance",),
        ),
        CorpusRecord(
            record_id="doc-2-seg-1",
            text="Government supports risk-based AI governance with accountable oversight.",
            source_id="source-2",
            document_id="doc-2",
            segment_id="seg-1",
            stakeholder_groups=("government",),
            topics=("ai_governance",),
        ),
    ]
    engine = ArwenPolicyEngine(
        RetrievalService(InMemoryRetriever(corpus))
    )
    result = engine.analyze(
        PolicyRequest(
            question_id="integration-1",
            question="How should AI governance ensure accountability?",
            topics=("ai_governance",),
            stakeholder_groups=("government", "civil_society", "technical_community"),
        ),
        perspectives=[
            Perspective("government", "Risk-based oversight", evidence_record_ids=("doc-2-seg-1",)),
            Perspective(
                "civil_society",
                "Public accountability",
                evidence_record_ids=("doc-1-seg-1",),
            ),
        ],
    )
    assert len(result.evidence) == 2
    assert result.stakeholder_coverage["represented"] == ["civil_society", "government"]
    assert result.stakeholder_coverage["missing"] == ["technical_community"]
    assert result.deliberation["evidence_record_ids"] == ["doc-1-seg-1", "doc-2-seg-1"]
    assert result.status == "ready_for_model_synthesis"
