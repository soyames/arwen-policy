import shutil
from pathlib import Path

from arwen_etl.deduplication import (
    canonicalize_url,
    content_identity,
    find_near_duplicate,
    is_duplicate_by_sha256,
    register_document,
)


def test_canonicalize_url():
    u = "HTTPS://Example.COM:443/path?utm_source=foo&a=1#frag"
    c = canonicalize_url(u)
    assert c.startswith("https://example.com/path")


def test_register_and_find_duplicate(tmp_path):
    # ensure clean data dir
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    text = "This is a unique document."
    sha = content_identity(text)
    assert is_duplicate_by_sha256(sha) is None

    register_document(sha, "doc-1", text, k=3)
    assert is_duplicate_by_sha256(sha) == "doc-1"


def test_near_duplicate_detection(tmp_path):
    # clean
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    text1 = "A quick brown fox jumps over the lazy dog"
    text2 = "A quick brown fox jumps over the lazy dog and runs away"
    register_document(content_identity(text1), "doc-a", text1, k=3)
    found = find_near_duplicate(text2, threshold=0.7, k=3)
    assert found == "doc-a"
