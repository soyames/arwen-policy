from arwen_deliberation.models import Perspective
from arwen_engine.models import PolicyRequest
from arwen_engine.pipeline import ArwenPolicyEngine
from arwen_retrieval.models import CorpusRecord
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService


def test_engine_carries_evidence_and_missing_groups():
    records = [
        CorpusRecord(
            "r1",
            "Internet governance requires transparent policy processes.",
            "s1",
            "d1",
            "seg1",
            stakeholder_groups=("civil_society",),
            topics=("internet_governance",),
        )
    ]
    engine = ArwenPolicyEngine(RetrievalService(InMemoryRetriever(records)))
    answer = engine.analyze(
        PolicyRequest(
            "q1",
            "transparent internet governance policy",
            topics=("internet_governance",),
            stakeholder_groups=("civil_society", "government"),
        ),
        [Perspective("civil_society", "Support transparency", evidence_record_ids=("r1",))],
    )
    assert answer.status == "ready_for_model_synthesis"
    assert answer.evidence[0]["record_id"] == "r1"
    assert answer.stakeholder_coverage["missing"] == ["government"]
    assert "never manufacture consensus" in answer.synthesis_prompt
    assert "human-rights-aware policy synthesis" in answer.synthesis_prompt
