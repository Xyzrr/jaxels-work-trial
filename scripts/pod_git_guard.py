#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class PodGitGuardError(RuntimeError):
    pass


def _git(
    repo_dir: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        text=True,
        capture_output=True,
        check=check,
    )


def _git_stdout(repo_dir: Path, *args: str) -> str:
    return _git(repo_dir, *args).stdout.strip()


def pod_git_status(repo_dir: Path) -> str:
    return _git(repo_dir, "status", "--porcelain=v1").stdout


def require_clean_pod_git_status(repo_dir: Path, label: str) -> None:
    status = pod_git_status(repo_dir)
    if status:
        indented = "".join(f"  {line}\n" for line in status.splitlines())
        raise PodGitGuardError(
            f"{label} has local git changes; clean it before launching:\n{indented}".rstrip()
        )


def _check_ref_format(branch: str) -> bool:
    result = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _merge_base_is_ancestor(repo_dir: Path, ancestor: str, descendant: str) -> bool:
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

    top_level = Path(_git_stdout(repo_path, "rev-parse", "--show-toplevel")).resolve()
    physical_repo_dir = repo_path.resolve()
    if top_level != physical_repo_dir:
        raise PodGitGuardError(f"{label} must be the git top-level: {repo_path}")

    require_clean_pod_git_status(repo_path, label)
    if _git(repo_path, "remote", "get-url", "origin", check=False).returncode != 0:
        raise PodGitGuardError(f"{label} does not have an origin remote")

    remote_ref = f"refs/remotes/origin/{expected_branch}"
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
        if not _merge_base_is_ancestor(repo_path, expected_branch, remote_ref):
            raise PodGitGuardError(
                f"{label} has commits on {expected_branch} that are not on "
                f"origin/{expected_branch}; push or reset them before launching"
            )
        _git(repo_path, "checkout", "--quiet", expected_branch)
    else:
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

    if not _merge_base_is_ancestor(repo_path, "HEAD", remote_ref):
        raise PodGitGuardError(
            f"{label} has commits that are not on origin/{expected_branch}; "
            "push or reset them before launching"
        )
    _git(repo_path, "merge", "--ff-only", "--quiet", remote_ref)

    head = _git_stdout(repo_path, "rev-parse", "HEAD")
    remote_head = _git_stdout(repo_path, "rev-parse", f"{remote_ref}^{{commit}}")
    if head != remote_head:
        raise PodGitGuardError(
            f"{label} is not at origin/{expected_branch} after fast-forward: "
            f"{head} != {remote_head}"
        )
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
