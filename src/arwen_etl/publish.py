from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover - optional dependency
    HfApi = None


def _retry(fn, attempts: int = 3, backoff: float = 1.0):
    def wrapped(*args, **kwargs):
        last_exc = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                time.sleep(backoff * (2 ** i))
        raise last_exc

    return wrapped


def publish_release(release_dir: str | Path, repo_id: str, token: str | None = None) -> dict:
    """Publish release directory to a Hugging Face repo if available.

    Uses retries for HF uploads and falls back to copying to `data/published`.
    """
    root = Path(release_dir)
    if not root.exists():
        raise FileNotFoundError(release_dir)

    # If HF API available and token provided, attempt upload with retries.
    if HfApi and token:
        api = HfApi()
        try:
            api.create_repo(repo_id, token=token, repo_type="dataset")
        except Exception:
            pass

        upload = _retry(api.upload_file, attempts=4, backoff=1.0)

        for f in root.rglob("*"):
            if not f.is_file():
                continue

            path_in_repo = str(f.relative_to(root))
            try:
                # prefer streaming fileobj when available
                with f.open("rb") as fh:
                    upload(
                        path_or_fileobj=fh,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=token,
                    )
            except Exception:
                # best-effort: try with path string once more
                upload(
                    path_or_fileobj=str(f),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )

        return {"status": "published", "repo_id": repo_id}

    # optional: if caller asked for git-lfs based publish, this code path can be
    # invoked directly via `publish_release_git_lfs` below. The default fallback
    # remains a local copy into `data/published` so publishing works without git.

    # fallback: copy to data/published
    target = Path("data") / "published" / repo_id
    target.mkdir(parents=True, exist_ok=True)
    for f in root.rglob("*"):
        if f.is_file():
            rel = f.relative_to(root)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(f.read_bytes())

    return {"status": "copied", "path": str(target)}


def _check_executable(name: str) -> bool:
    try:
        cmd = [name, "--version"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def publish_release_git_lfs(
    release_dir: str | Path,
    repo_url: str | None = None,
    push: bool = False,
    lfs_threshold_bytes: int = 50 * 1024 * 1024,
    push_attempts: int = 3,
    push_backoff: float = 1.0,
    cleanup: bool = True,
) -> dict:
    """Prepare a git repository with Git LFS pointers for large files.

    - Copies the release into a temporary git repo directory.
    - Runs `git lfs track` for files larger than `lfs_threshold_bytes`.
    - Commits the tree. If `push` is True and `repo_url` provided, attempts a push.

    Note: This requires `git` and `git-lfs` available on PATH and appropriate
    credentials configured for pushing. This function is best-effort and will
    raise informative exceptions when commands fail.
    """
    root = Path(release_dir)
    if not root.exists():
        raise FileNotFoundError(release_dir)

    if not _check_executable("git"):
        raise RuntimeError("git is not available on PATH")
    if not _check_executable("git-lfs") and not _check_executable("git-lfs.exe"):
        raise RuntimeError("git-lfs is not available on PATH")

    tmpdir = Path(tempfile.mkdtemp(prefix="arwen_publish_"))
    try:
        # copy files into temp repo
        shutil.copytree(root, tmpdir / "release", dirs_exist_ok=True)
        repo_path = tmpdir / "release"

        def run_git(*args):
            subprocess.run(["git", *args], cwd=str(repo_path), check=True)

        run_git("init")
        # enable lfs locally
        subprocess.run(["git", "lfs", "install", "--local"], cwd=str(repo_path), check=True)

        # track large files
        patterns = set()
        for f in repo_path.rglob("*"):
            if not f.is_file():
                continue
            try:
                if f.stat().st_size >= lfs_threshold_bytes:
                    rel = str(f.relative_to(repo_path)).replace("\\", "/")
                    patterns.add(rel)
            except Exception:
                continue

        for p in sorted(patterns):
            # track specific path
            subprocess.run(["git", "lfs", "track", p], cwd=str(repo_path), check=True)

        run_git("add", "--all")
        run_git("commit", "-m", "Publish release (git-lfs prepared)")

        pushed = False
        if push:
            if not repo_url:
                raise ValueError("repo_url is required when push=True")

            # add remote if missing (may fail if exists; ignore)
            try:
                run_git("remote", "add", "origin", repo_url)
            except Exception:
                pass

            # ensure main branch
            run_git("branch", "-M", "main")

            # try pushing with retries and exponential backoff
            last_exc = None
            for attempt in range(push_attempts):
                try:
                    run_git("push", "-u", "origin", "main")
                    pushed = True
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 < push_attempts:
                        time.sleep(push_backoff * (2 ** attempt))
                        continue
                    raise last_exc from None

        result = {"status": "git_lfs_prepared", "path": str(repo_path), "pushed": pushed}

        # cleanup temporary repo if requested and push succeeded (or not pushing)
        if cleanup:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
                result["cleaned"] = True
            except Exception:
                result["cleaned"] = False

        return result
    except Exception:
        # cleanup on failure
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
