"""Shared pod-startup guards for training, eval, and image-prebuild launchers.

The lower-level launchers all run expensive ML-adjacent workloads on the same
GPU pod. Before they start a real job, they need two common checks:

* the process is actually running in the pod runtime with required binaries; and
* the pod checkout matches the clean branch that the workstation wrapper pushed.

This module keeps those checks centralized so training and eval wrappers do not
drift into subtly different reproducibility contracts.
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from scripts import pod_git_guard
from scripts.pod_utils import die, env_flag, is_falsey, is_truthy


def should_enforce_pod_git_guard(repo_dir: str | Path) -> bool:
    """Return whether this repo path should be pinned to the pushed branch."""

    value = env_flag(os.environ.get("SWEHERO_POD_GIT_ENFORCE"))
    if is_truthy(value):
        return True
    if is_falsey(value):
        return False
    if value not in {"auto", "AUTO"}:
        die(
            "SWEHERO_POD_GIT_ENFORCE must be auto, 1, or 0; got:\n"
            f"  {os.environ.get('SWEHERO_POD_GIT_ENFORCE')}",
            exit_code=2,
        )
    # Auto mode protects the canonical pod checkout but stays out of local unit
    # tests and temporary scratch directories. The env name is legacy, but the
    # contract now applies to all midtraining pod workloads.
    return str(repo_dir) == os.environ.get(
        "SWEHERO_POD_GIT_ROOT",
        "/workspace/jaxels-work-trial",
    )


def prepare_pod_checkout(
    repo_dir: str | Path,
    label: str = "pod execution directory",
) -> None:
    """Run the git guard before starting a new pod-side job.

    Reconnect-only paths can skip this by returning before they launch new work.
    That avoids mutating an already-running tmux session while still enforcing
    clean, pushed code for fresh training/eval/prebuild jobs.
    """

    repo_path = Path(repo_dir)
    if not should_enforce_pod_git_guard(repo_path):
        return
    if not repo_path.is_dir():
        die(f"workspace not found: {repo_path}")
    if shutil.which("git") is None:
        die("git not found; recreate the pod with manifests/midtraining-hostpath.yaml")
    try:
        pod_git_guard.require_pod_git_checkout(
            repo_path,
            os.environ.get("SWEHERO_POD_GIT_BRANCH", ""),
            label,
        )
    except pod_git_guard.PodGitGuardError as exc:
        die(str(exc))


def require_pod_runtime(workspace_root: str | Path, *binaries: str) -> None:
    """Fail early when a pod-only launcher is run in the wrong environment."""

    workspace_path = Path(workspace_root)
    if platform.system() == "Darwin":
        die("this launcher is pod-only; run it from the Kubernetes GPU pod")
    if not Path("/workspace").is_dir():
        die("expected /workspace hostPath; run from the GPU pod")
    if not workspace_path.is_dir():
        die(f"workspace not found: {workspace_path}")

    for binary in binaries:
        if shutil.which(binary) is None:
            # Missing tools here usually mean the Kubernetes manifest or pod
            # startup bootstrap did not run, not that the experiment preset is
            # wrong. Fail before vLLM, OpenHands, Docker, or TorchTitan produce
            # less obvious downstream errors.
            die(
                f"{binary} not found; recreate the pod with "
                "manifests/midtraining-hostpath.yaml"
            )
