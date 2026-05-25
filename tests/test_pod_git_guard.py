import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "pod_git_guard.py"


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def commit_all(cwd: Path, message: str) -> None:
    git(cwd, "add", ".")
    git(cwd, "commit", "-m", message)


class TestPodGitGuard:
    def make_origin(self, tmp: Path) -> tuple[Path, Path]:
        source = tmp / "source"
        source.mkdir()
        git(source, "init", "-b", "main")
        git(source, "config", "user.name", "Pod Guard Test")
        git(source, "config", "user.email", "pod-guard@example.invalid")
        (source / "README.md").write_text("main\n")
        commit_all(source, "initial main")

        git(source, "checkout", "-b", "axel/hero")
        (source / "feature.txt").write_text("feature v1\n")
        commit_all(source, "initial feature")

        origin = tmp / "origin.git"
        git(tmp, "init", "--bare", "--initial-branch=main", str(origin))
        git(source, "remote", "add", "origin", str(origin))
        git(source, "push", "-u", "origin", "main", "axel/hero")
        return origin, source

    def clone_origin(self, tmp: Path, origin: Path, branch: str = "main") -> Path:
        pod = tmp / f"pod-{branch.replace('/', '-')}"
        git(tmp, "clone", "--branch", branch, str(origin), str(pod))
        git(pod, "config", "user.name", "Pod Guard Test")
        git(pod, "config", "user.email", "pod-guard@example.invalid")
        return pod

    def run_guard(
        self, pod: Path, expected_branch: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(GUARD),
                str(pod),
                expected_branch,
                "test pod",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_guard_checks_out_expected_branch_and_fast_forwards_to_origin(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            origin, source = self.make_origin(tmp)
            pod = self.clone_origin(tmp, origin, "main")

            (source / "feature.txt").write_text("feature v2\n")
            commit_all(source, "advance feature")
            git(source, "push", "origin", "axel/hero")

            result = self.run_guard(pod, "axel/hero")

            assert result.returncode == 0, result.stderr
            branch = git(pod, "branch", "--show-current").stdout.strip()
            head = git(pod, "rev-parse", "HEAD").stdout.strip()
            remote_head = git(
                pod, "rev-parse", "refs/remotes/origin/axel/hero"
            ).stdout.strip()
            status = git(pod, "status", "--porcelain=v1").stdout.strip()

        assert branch == "axel/hero"
        assert head == remote_head
        assert status == ""

    def test_guard_refuses_dirty_pod_checkout_before_switching_branch(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            origin, _source = self.make_origin(tmp)
            pod = self.clone_origin(tmp, origin, "main")
            (pod / "local.txt").write_text("not committed\n")

            result = self.run_guard(pod, "axel/hero")
            branch = git(pod, "branch", "--show-current").stdout.strip()

        assert result.returncode != 0
        assert "local git changes" in result.stderr
        assert "?? local.txt" in result.stderr
        assert branch == "main"

    def test_guard_refuses_clean_checkout_with_unpushed_local_commit(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            origin, _source = self.make_origin(tmp)
            pod = self.clone_origin(tmp, origin, "axel/hero")
            (pod / "feature.txt").write_text("local only\n")
            commit_all(pod, "local only")

            result = self.run_guard(pod, "axel/hero")

        assert result.returncode != 0
        assert "commits on axel/hero that are not on origin/axel/hero" in result.stderr

    def test_guard_does_not_switch_to_unpushed_expected_branch(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            origin, _source = self.make_origin(tmp)
            pod = self.clone_origin(tmp, origin, "axel/hero")
            (pod / "feature.txt").write_text("local only\n")
            commit_all(pod, "local only")
            git(pod, "checkout", "main")

            result = self.run_guard(pod, "axel/hero")
            branch = git(pod, "branch", "--show-current").stdout.strip()

        assert result.returncode != 0
        assert "commits on axel/hero that are not on origin/axel/hero" in result.stderr
        assert branch == "main"

    def test_guard_requires_expected_branch_from_local_worktree(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            origin, _source = self.make_origin(tmp)
            pod = self.clone_origin(tmp, origin, "main")

            result = self.run_guard(pod, "")

        assert result.returncode != 0
        assert "SWEHERO_POD_GIT_BRANCH is required" in result.stderr
