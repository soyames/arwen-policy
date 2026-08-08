import shutil
import threading
from pathlib import Path

from arwen_etl.deduplication import (
    content_identity,
    find_near_duplicate,
    is_duplicate_by_sha256,
    register_document,
)


def test_empty_and_short_texts():
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    empty = ""
    sha = content_identity(empty)
    register_document(sha, "doc-empty", empty, k=2)
    assert is_duplicate_by_sha256(sha) == "doc-empty"

    short = "Hi"
    sha2 = content_identity(short)
    register_document(sha2, "doc-short", short, k=2)
    assert is_duplicate_by_sha256(sha2) == "doc-short"


def test_threshold_edge_cases():
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    a = "abcdefg"
    b = "abcdefx"
    register_document(content_identity(a), "doc-a", a, k=2)
    # with small k, these may be detected as similar at low threshold
    found_low = find_near_duplicate(b, threshold=0.1, k=2)
    assert found_low == "doc-a"
    found_high = find_near_duplicate(b, threshold=0.99, k=2)
    assert found_high is None


def test_concurrent_registration():
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    texts = [f"doc content {i}" for i in range(20)]

    def worker(i):
        txt = texts[i]
        register_document(content_identity(txt), f"d{i}", txt, k=3)

    threads = []
    for i in range(len(texts)):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # verify all present
    for i in range(len(texts)):
        sha = content_identity(texts[i])
        assert is_duplicate_by_sha256(sha) == f"d{i}"
