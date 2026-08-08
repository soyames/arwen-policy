import pytest
from pydantic import ValidationError

from arwen_etl.registry import SourceDefinition


def test_valid_domains():
    s = SourceDefinition(id="x", name="n", family="f", domains=["example.com"])
    assert s.domains[0] == "example.com"


def test_invalid_domain_raises():
    with pytest.raises(ValidationError):
        SourceDefinition(id="x", name="n", family="f", domains=["bad/domain.com"])
