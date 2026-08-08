from arwen_etl import publish


class FakeApi:
    def __init__(self):
        self.calls = 0

    def create_repo(self, repo_id, token=None, repo_type=None):
        return None

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, token=None):
        # simulate failing twice then succeeding
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary failure")
        return {"status": "ok"}


def test_publish_retry(monkeypatch, tmp_path):
    # create fake release dir
    release = tmp_path / "release"
    release.mkdir()
    f = release / "bigfile.txt"
    f.write_bytes(b"x" * 1024)

    fake = FakeApi()

    monkeypatch.setattr(publish, "HfApi", lambda: fake)

    res = publish.publish_release(release, "test/repo", token="tok")
    assert res["status"] == "published"
    assert fake.calls >= 3
