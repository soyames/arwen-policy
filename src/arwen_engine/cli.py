from __future__ import annotations

import argparse
import json

from arwen_retrieval.models import CorpusRecord
from arwen_retrieval.retriever import InMemoryRetriever
from arwen_retrieval.service import RetrievalService

from .models import PolicyRequest
from .pipeline import ArwenPolicyEngine


def main() -> None:
    parser = argparse.ArgumentParser(prog="arwen-policy")
    parser.add_argument("question")
    args = parser.parse_args()

    # The CLI is intentionally a deterministic smoke-test interface until the
    # production corpus loader and model adapter are configured.
    records = [
        CorpusRecord(
            record_id="smoke-1",
            source_id="smoke-source",
            document_id="smoke-document",
            segment_id="smoke-segment",
            text="Digital policy should remain transparent and multistakeholder.",
            stakeholder_groups=("civil_society",),
            topics=("digital_policy",),
        )
    ]
    engine = ArwenPolicyEngine(RetrievalService(InMemoryRetriever(records)))
    result = engine.analyze(PolicyRequest(question_id="cli", question=args.question))
    print(json.dumps(result.__dict__, indent=2, default=list))


if __name__ == "__main__":
    main()
