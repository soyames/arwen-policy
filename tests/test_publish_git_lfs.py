import subprocess

from arwen_etl.publish import publish_release_git_lfs


def test_publish_git_lfs_push_retries(tmp_path, monkeypatch):
    # prepare a fake release dir with one small and one "large" file
    release = tmp_path / "release"
    release.mkdir()
    small = release / "small.txt"
    small.write_text("hello")

    # make a file that should be tracked by LFS (use small threshold in test)
    large = release / "big.bin"
    large.write_bytes(b"0" * 2048)

    # simulate subprocess.run behavior
    calls = []

    def fake_run(cmd, cwd=None, check=False, stdout=None, stderr=None):
        calls.append(list(cmd))
        # git-lfs and git --version checks succeed
        if cmd[0] in ("git-lfs", "git-lfs.exe") and "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[0] == "git" and "--version" in cmd:
            return subprocess.CompletedProcess(cmd, 0)

        # simulate 'git push' failing twice then succeeding
        if cmd[0] == "git" and "push" in cmd:
            fake_run.attempt += 1
            if fake_run.attempt < 3:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)

        # all other git/git-lfs commands succeed
        return subprocess.CompletedProcess(cmd, 0)

    fake_run.attempt = 0

    monkeypatch.setattr("arwen_etl.publish.subprocess.run", fake_run)

    # call helper with low threshold to mark big.bin for LFS
    result = publish_release_git_lfs(
        release,
        repo_url="https://example.com/repo.git",
        push=True,
        lfs_threshold_bytes=1024,
        push_attempts=4,
        push_backoff=0.01,
    )

    assert result.get("pushed") is True
    assert result.get("cleaned") is True

    # assert git lfs track was called for big.bin (relative path)
    tracked = any(call[:3] == ["git", "lfs", "track"] for call in calls)
    assert tracked
