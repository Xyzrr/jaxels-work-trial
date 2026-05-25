from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from scripts import pod_git_guard
from scripts.pod_utils import die, env_flag, is_falsey, is_truthy


def should_enforce_pod_git_guard(repo_dir: str | Path) -> bool:
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
    return str(repo_dir) == os.environ.get(
        "SWEHERO_POD_GIT_ROOT",
        "/workspace/jaxels-work-trial",
    )


def prepare_pod_checkout(
    repo_dir: str | Path,
    label: str = "pod execution directory",
) -> None:
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
    workspace_path = Path(workspace_root)
    if platform.system() == "Darwin":
        die("this launcher is pod-only; run it from the Kubernetes GPU pod")
    if not Path("/workspace").is_dir():
        die("expected /workspace hostPath; run from the GPU pod")
    if not workspace_path.is_dir():
        die(f"workspace not found: {workspace_path}")

    for binary in binaries:
        if shutil.which(binary) is None:
            die(
                f"{binary} not found; recreate the pod with "
                "manifests/midtraining-hostpath.yaml"
            )
