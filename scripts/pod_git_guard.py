#!/usr/bin/env -S uv run python
"""Keep pod-side training/eval launches pinned to a pushed git branch.

Long ML runs are expensive and hard to interpret when the executing code cannot
be tied back to a commit. The workstation meta-wrapper pushes a clean branch and
sets ``SWEHERO_POD_GIT_BRANCH`` before entering the pod; this guard enforces the
other half of that contract inside the pod checkout.

The goal is not general-purpose git convenience. The guard is deliberately
conservative: it refuses dirty files, refuses unpushed pod commits, fetches the
expected branch from origin, and fast-forwards the pod checkout before a new
training/eval job starts. That makes run metadata and later debugging point at
code reviewers can actually retrieve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class PodGitGuardError(RuntimeError):
    """Raised when the pod checkout cannot be made reproducible."""


def _git(
    repo_dir: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run git inside the pod checkout and capture diagnostics for callers."""

    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        text=True,
        capture_output=True,
        check=check,
    )


def _git_stdout(repo_dir: Path, *args: str) -> str:
    return _git(repo_dir, *args).stdout.strip()


def pod_git_status(repo_dir: Path) -> str:
    """Return porcelain status so callers can report exact dirty files."""

    return _git(repo_dir, "status", "--porcelain=v1").stdout


def require_clean_pod_git_status(repo_dir: Path, label: str) -> None:
    """Fail if local pod files would make the launched run unreproducible."""

    status = pod_git_status(repo_dir)
    if status:
        indented = "".join(f"  {line}\n" for line in status.splitlines())
        raise PodGitGuardError(
            f"{label} has local git changes; clean it before launching:\n{indented}".rstrip()
        )


def _check_ref_format(branch: str) -> bool:
    """Use git's own branch-name validator before interpolating a refspec."""

    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _merge_base_is_ancestor(repo_dir: Path, ancestor: str, descendant: str) -> bool:
    """Return whether ``ancestor`` is already contained in ``descendant``."""

    result = _git(
        repo_dir,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    return result.returncode == 0


def require_pod_git_checkout(
    repo_dir: str | Path,
    expected_branch: str,
    label: str = "pod execution directory",
) -> None:
    """Make ``repo_dir`` exactly match ``origin/<expected_branch>``.

    Training and SWE-bench eval artifacts record git state as part of their
    provenance. This function makes that provenance meaningful by ensuring the
    pod executes the same branch the workstation selected and pushed.
    """

    repo_path = Path(repo_dir)
    if not expected_branch:
        raise PodGitGuardError(
            "SWEHERO_POD_GIT_BRANCH is required so the pod can match the current "
            "local worktree branch"
        )
    if not _check_ref_format(expected_branch):
        raise PodGitGuardError(f"invalid SWEHERO_POD_GIT_BRANCH: {expected_branch}")
    if (
        _git(repo_path, "rev-parse", "--is-inside-work-tree", check=False).returncode
        != 0
    ):
        raise PodGitGuardError(f"{label} is not a git worktree: {repo_path}")

    # Require the repository root rather than an arbitrary subdirectory so git
    # status, checkout, and merge checks cover every file that can affect the
    # launched training/eval process.
    top_level = Path(_git_stdout(repo_path, "rev-parse", "--show-toplevel")).resolve()
    physical_repo_dir = repo_path.resolve()
    if top_level != physical_repo_dir:
        raise PodGitGuardError(f"{label} must be the git top-level: {repo_path}")

    # Check cleanliness before switching branches. This avoids hiding local pod
    # edits by changing branches or letting them accidentally alter a run.
    require_clean_pod_git_status(repo_path, label)
    if _git(repo_path, "remote", "get-url", "origin", check=False).returncode != 0:
        raise PodGitGuardError(f"{label} does not have an origin remote")

    remote_ref = f"refs/remotes/origin/{expected_branch}"
    # Fetch only the selected branch into its remote-tracking ref. The leading
    # plus keeps the pod view aligned with origin even if the branch was
    # force-updated before the launch.
    fetch = _git(
        repo_path,
        "fetch",
        "--prune",
        "origin",
        f"+refs/heads/{expected_branch}:{remote_ref}",
        check=False,
    )
    if fetch.returncode != 0:
        raise PodGitGuardError(f"could not fetch origin/{expected_branch} for {label}")
    if (
        _git(
            repo_path, "show-ref", "--verify", "--quiet", remote_ref, check=False
        ).returncode
        != 0
    ):
        raise PodGitGuardError(f"origin/{expected_branch} does not exist for {label}")

    if (
        _git(
            repo_path,
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{expected_branch}",
            check=False,
        ).returncode
        == 0
    ):
        # If the branch already exists in the pod, do not switch to it when it
        # contains unpushed commits. Those commits would make the job differ
        # from origin and from the workstation's pushed branch.
        if not _merge_base_is_ancestor(repo_path, expected_branch, remote_ref):
            raise PodGitGuardError(
                f"{label} has commits on {expected_branch} that are not on "
                f"origin/{expected_branch}; push or reset them before launching"
            )
        _git(repo_path, "checkout", "--quiet", expected_branch)
    else:
        # A fresh pod clone may not have the selected branch locally yet. Create
        # a tracking branch from the just-fetched remote ref instead of checking
        # out a detached commit, so later diagnostics can report the branch name.
        _git(
            repo_path,
            "checkout",
            "--quiet",
            "-b",
            expected_branch,
            "--track",
            remote_ref,
        )

    current_branch = _git_stdout(repo_path, "branch", "--show-current")
    if current_branch != expected_branch:
        raise PodGitGuardError(
            f"{label} is on branch '{current_branch}', expected '{expected_branch}'"
        )

    # At this point HEAD should be either equal to or behind origin. Anything
    # ahead of origin is unreproducible pod-only code and must be rejected.
    if not _merge_base_is_ancestor(repo_path, "HEAD", remote_ref):
        raise PodGitGuardError(
            f"{label} has commits that are not on origin/{expected_branch}; "
            "push or reset them before launching"
        )

    # Fast-forward only. This updates stale pod checkouts without creating merge
    # commits or resolving conflicts inside the pod.
    _git(repo_path, "merge", "--ff-only", "--quiet", remote_ref)

    head = _git_stdout(repo_path, "rev-parse", "HEAD")
    remote_head = _git_stdout(repo_path, "rev-parse", f"{remote_ref}^{{commit}}")
    if head != remote_head:
        raise PodGitGuardError(
            f"{label} is not at origin/{expected_branch} after fast-forward: "
            f"{head} != {remote_head}"
        )

    # Check one more time because checkout/merge can update tracked files. A
    # launch should start from an exact remote commit with no local file drift.
    require_clean_pod_git_status(repo_path, label)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_dir")
    parser.add_argument("expected_branch")
    parser.add_argument("label", nargs="?", default="pod execution directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_pod_git_checkout(args.repo_dir, args.expected_branch, args.label)
    except PodGitGuardError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
