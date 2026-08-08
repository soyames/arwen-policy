from arwen_retrieval.models import CorpusRecord, RetrievalQuery
from arwen_retrieval.retriever import InMemoryRetriever


def test_retrieval_ranks_matching_record_first():
    records = [
        CorpusRecord("a", "AI regulation and transparency", "s1", "d1", topics=("ai",)),
        CorpusRecord("b", "Domain name administration", "s2", "d2", topics=("dns",)),
    ]
    results = InMemoryRetriever(records).retrieve(RetrievalQuery("AI transparency", top_k=2))
    assert results[0].record.record_id == "a"
    assert results[0].score > 0


def test_retrieval_applies_topic_filter():
    records = [
        CorpusRecord("a", "AI regulation", "s1", "d1", topics=("ai",)),
        CorpusRecord("b", "AI regulation", "s2", "d2", topics=("dns",)),
    ]
    results = InMemoryRetriever(records).retrieve(
        RetrievalQuery("AI regulation", topics=("ai",), top_k=5)
    )
    assert [item.record.record_id for item in results] == ["a"]
