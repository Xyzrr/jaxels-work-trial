"""Tests for the pod-side Git checkout guard.

Training and SWE-bench eval runs can consume hours of GPU time and produce
metrics that reviewers compare across model checkpoints, datasets, and eval
stacks. Those results are only interpretable if the pod executed code that is
recoverable from Git. These tests exercise the guard with real temporary Git
repositories so the reproducibility contract is checked end to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "pod_git_guard.py"


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a real Git command in a fixture repository."""

    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def commit_all(cwd: Path, message: str) -> None:
    """Commit every file in the fixture repository."""

    git(cwd, "add", ".")
    git(cwd, "commit", "-m", message)


class TestPodGitGuard:
    """Verify the guard rejects pod states that would make runs unreproducible."""

    def make_origin(self, tmp: Path) -> tuple[Path, Path]:
        """Create a bare origin with main and axel/hero branches.

        The source checkout stands in for the workstation branch that launches a
        training/eval job. The bare origin stands in for GitHub, which is the
        only code source the pod can safely reproduce.
        """

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
        """Clone the origin into a pod-like checkout."""

        pod = tmp / f"pod-{branch.replace('/', '-')}"
        git(tmp, "clone", "--branch", branch, str(origin), str(pod))
        git(pod, "config", "user.name", "Pod Guard Test")
        git(pod, "config", "user.email", "pod-guard@example.invalid")
        return pod

    def run_guard(
        self, pod: Path, expected_branch: str
    ) -> subprocess.CompletedProcess[str]:
        """Run the guard as a subprocess so CLI errors match pod launch behavior."""

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

    def test_guard_checks_out_expected_branch_and_fast_forwards_to_origin(
        self, tmp_path: Path
    ) -> None:
        origin, source = self.make_origin(tmp_path)
        pod = self.clone_origin(tmp_path, origin, "main")

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

        # Successful launch preparation means the pod is exactly on the branch
        # selected by the workstation and at the same commit as origin. The
        # clean status is what lets training/eval metadata name a commit without
        # hiding local pod-only code changes.
        assert branch == "axel/hero"
        assert head == remote_head
        assert status == ""

    def test_guard_refuses_dirty_pod_checkout_before_switching_branch(
        self, tmp_path: Path
    ) -> None:
        origin, _source = self.make_origin(tmp_path)
        pod = self.clone_origin(tmp_path, origin, "main")
        (pod / "local.txt").write_text("not committed\n")

        result = self.run_guard(pod, "axel/hero")
        branch = git(pod, "branch", "--show-current").stdout.strip()

        # Dirty files in the pod might be ad-hoc debugging edits, generated code,
        # or local launcher changes. Any of those could affect a model run while
        # being absent from the commit recorded in the run metadata.
        assert result.returncode != 0
        assert "local git changes" in result.stderr
        assert "?? local.txt" in result.stderr
        assert branch == "main"

    def test_guard_refuses_clean_checkout_with_unpushed_local_commit(
        self, tmp_path: Path
    ) -> None:
        origin, _source = self.make_origin(tmp_path)
        pod = self.clone_origin(tmp_path, origin, "axel/hero")
        (pod / "feature.txt").write_text("local only\n")
        commit_all(pod, "local only")

        result = self.run_guard(pod, "axel/hero")

        # A clean Git status is not enough: a local commit that has not reached
        # origin is still unrecoverable by reviewers and future reruns.
        assert result.returncode != 0
        assert "commits on axel/hero that are not on origin/axel/hero" in result.stderr

    def test_guard_does_not_switch_to_unpushed_expected_branch(
        self, tmp_path: Path
    ) -> None:
        origin, _source = self.make_origin(tmp_path)
        pod = self.clone_origin(tmp_path, origin, "axel/hero")
        (pod / "feature.txt").write_text("local only\n")
        commit_all(pod, "local only")
        git(pod, "checkout", "main")

        result = self.run_guard(pod, "axel/hero")
        branch = git(pod, "branch", "--show-current").stdout.strip()

        # The guard checks the target branch for unpushed commits before
        # switching to it. That prevents the pod from silently launching with a
        # local-only branch tip just because the current branch was clean.
        assert result.returncode != 0
        assert "commits on axel/hero that are not on origin/axel/hero" in result.stderr
        assert branch == "main"

    def test_guard_requires_expected_branch_from_local_worktree(
        self, tmp_path: Path
    ) -> None:
        origin, _source = self.make_origin(tmp_path)
        pod = self.clone_origin(tmp_path, origin, "main")

        result = self.run_guard(pod, "")

        # The expected branch is the handshake from the workstation launcher to
        # the pod. Without it, the pod cannot know which reviewed branch should
        # define the training/eval run.
        assert result.returncode != 0
        assert "SWEHERO_POD_GIT_BRANCH is required" in result.stderr
