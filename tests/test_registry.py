from __future__ import annotations

from arwen_etl.registry import load_registry


def test_load_registry_and_find():
    reg = load_registry("configs/sources.yaml")
    assert reg.sources
    icann = reg.find_by_domain("www.icann.org")
    assert icann is not None
    assert icann.id == "icann"
