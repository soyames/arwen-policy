from arwen_benchmark.metrics import coverage_score, evidence_recall, reciprocal_rank
from arwen_benchmark.models import BenchmarkCase
from arwen_benchmark.runner import evaluate_case
from arwen_paper.manifest import experiment_manifest
from arwen_training.builder import build_instruction_example
from arwen_training.validation import validate_training_example


def test_benchmark_metrics():
    assert reciprocal_rank(["x", "b"], {"b"}) == 0.5
    assert evidence_recall(["a", "b"], {"b", "c"}) == 0.5
    assert coverage_score({"government"}, {"government", "users"}) == 0.5


def test_benchmark_runner():
    case = BenchmarkCase("c1", "q", ("b",), ("government",))
    result = evaluate_case(case, ["x", "b"], {"government"})
    assert result.reciprocal_rank == 0.5
    assert result.stakeholder_coverage == 1.0


def test_training_example_requires_provenance():
    example = build_instruction_example(
        question="q",
        evidence=[{"record_id": "r1"}],
        perspectives=[{"stakeholder_group": "government"}],
        answer="a",
    )
    assert validate_training_example(example) == []


def test_paper_manifest_is_reproducible_metadata():
    manifest = experiment_manifest(
        experiment_id="exp-1",
        config={"seed": 1},
        dataset_revisions={"corpus": "main"},
        code_revision="abc123",
    )
    assert manifest["code_revision"] == "abc123"
    assert manifest["dataset_revisions"]["corpus"] == "main"
