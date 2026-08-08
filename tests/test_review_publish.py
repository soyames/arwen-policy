import shutil
from pathlib import Path

from arwen_etl.publish import publish_release
from arwen_etl.review import enqueue_review, list_queue, pop_review


def test_review_queue(tmp_path):
    # ensure clean data dir
    data_dir = Path("data")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    enqueue_review("doc-1", "low_quality", {"score": 0.2})
    q = list_queue()
    assert q and q[0]["document_id"] == "doc-1"
    popped = pop_review()
    assert popped["document_id"] == "doc-1"
    assert list_queue() == []


def test_publish_fallback(tmp_path):
    # create fake release dir
    release = tmp_path / "release"
    release.mkdir()
    (release / "file.txt").write_text("hello")

    res = publish_release(release, "test-repo")
    assert res["status"] in ("copied", "published")
    if res["status"] == "copied":
        p = Path(res["path"]) / "file.txt"
        assert p.exists()
