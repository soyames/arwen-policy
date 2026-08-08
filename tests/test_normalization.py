from arwen_etl.normalization import normalize_text


def test_normalize_text():
    assert normalize_text(" A\r\n\r\n B ") == "A\n\nB"
