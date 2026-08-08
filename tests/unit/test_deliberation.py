from arwen_deliberation.council import DeliberationCouncil
from arwen_deliberation.models import Perspective, PolicyQuestion


def test_deliberation_reports_missing_groups():
    question = PolicyQuestion(
        "q1",
        "How should AI governance work?",
        required_stakeholder_groups=("government", "civil_society", "technical_community"),
    )
    perspectives = [
        Perspective("government", "Use risk-based rules", evidence_record_ids=("r1",)),
        Perspective("civil_society", "Use risk-based rules", evidence_record_ids=("r2",)),
    ]
    result = DeliberationCouncil().deliberate(question, perspectives)
    assert result.missing_groups == ("technical_community",)
    assert result.agreements == ("use risk-based rules",)


def test_deliberation_does_not_create_missing_perspective():
    question = PolicyQuestion("q1", "Question", required_stakeholder_groups=("users",))
    result = DeliberationCouncil().deliberate(question, [])
    assert result.perspectives == ()
    assert result.missing_groups == ("users",)
