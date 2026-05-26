#!/usr/bin/env -S uv run python
"""Workstation entrypoint for launching midtraining workloads on the GPU pod.

This script is intentionally a pod transport wrapper, not an ML experiment
launcher. Training/eval decisions such as model checkpoint, dataset, context
length, optimizer, OpenHands stack, and grading behavior belong in preset files
or in the pod-side workload arguments. This wrapper only makes sure the
workstation state is reproducible in the pod and then execs the selected
pod-side script.

That separation matters for readability: a future experiment can reuse the
same Kubernetes handoff without inheriting hidden SWE-Hero, Qwen, vLLM, or
OpenHands defaults from this file.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pod_utils import command_output, die, exec_process, repo_root, run

ROOT_DIR = repo_root()
DEFAULT_KUBECONFIG = ROOT_DIR / "tmp" / "pod-creds" / "kubeconfig.yaml"

# Forward only secrets and runtime plumbing. Experiment settings are deliberately
# excluded: model IDs, datasets, context modes, sampling parameters, optimizer
# settings, and eval stack choices should travel through explicit preset/CLI
# arguments so the launched run can be reproduced from command history.
FORWARDED_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "LLM_API_KEY",
    "TORCHTITAN_POD_VENV",
    "TORCHTITAN_POD_SETUP_SCRIPT",
    "SWEHERO_POD_TMUX_SESSION",
    "SWEHERO_POD_SUPERVISOR",
    "SWEHERO_POD_TMUX_ATTACH",
    "SWEHERO_POD_TMUX_LOG_DIR",
    "SWEHERO_POD_TMUX_ENV_DIR",
    "VLLM_VENV",
    "VLLM_REQUIREMENTS_PATH",
    "VLLM_FORCE_RESTART",
    "VLLM_VISIBLE_DEVICES",
    "VLLM_GPU",
    "VLLM_NCCL_CUMEM_ENABLE",
    "EVAL_VENV",
    "OPENHANDS_EVAL_POETRY_VERSION",
    "OPENHANDS_EVAL_TMUX_SESSION",
    "OPENHANDS_EVAL_ATTACH",
    "OPENHANDS_EVAL_TMUX_LOG_DIR",
    "REQUIRED_GPU_COUNT",
    "UV_TOOL_DIR",
    "UV_CACHE_DIR",
    "UV_PYTHON_INSTALL_DIR",
)


USAGE = """\
Usage: scripts/run_midtraining_pod.py [launcher options] WORKLOAD [args...]

Run a midtraining workload on the canonical Kubernetes GPU pod.

Workloads:
  train                 scripts/run_qwen_swehero_torchtitan_pod.py
  eval                  scripts/run_openhands_swebench_eval_pod.py
  prebuild              scripts/prebuild_openhands_swebench_images_pod.py
  scripts/path.py       Any repository-relative pod script

Launcher options:
  --kubeconfig PATH     Default: tmp/pod-creds/kubeconfig.yaml
  --namespace NAME      Default: midtraining
  --pod-name NAME       Default: midtraining-dev
  --workspace-root PATH Default: /workspace/jaxels-work-trial
  --branch NAME         Default: current local branch
  --no-push             Do not push the branch before kubectl exec.
  --no-tty              Do not allocate an interactive TTY for kubectl exec.
  -h, --help            Show this help.

The launcher pushes the selected branch by default, enters the GPU pod with
kubectl exec, sets SWEHERO_POD_GIT_BRANCH for the pod-side git guard, and then
starts the selected workload from the pod checkout.
"""


def selected_workload_script(workload: str) -> str:
    """Map human workload aliases to the concrete pod-side script."""

    match workload:
        case "train" | "training" | "torchtitan":
            return "scripts/run_qwen_swehero_torchtitan_pod.py"
        case "eval" | "openhands-eval" | "swebench-eval":
            return "scripts/run_openhands_swebench_eval_pod.py"
        case "prebuild" | "image-prebuild" | "openhands-prebuild":
            return "scripts/prebuild_openhands_swebench_images_pod.py"
        case _:
            return workload


def require_clean_local_checkout() -> None:
    """Refuse launches that would not match the pushed branch inside the pod."""

    status = command_output(["git", "-C", str(ROOT_DIR), "status", "--porcelain=v1"])
    if status:
        print(
            "error: local checkout has uncommitted changes; commit or stash them "
            "before launching a pod workload.\n\n"
            "The pod runs the pushed branch from origin, so dirty local files would "
            "not be\nvisible to the job:",
            file=sys.stderr,
        )
        for line in status.splitlines():
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)


def parse_launcher(argv: list[str]) -> tuple[dict[str, object], str, list[str]]:
    """Parse only meta-launcher options and leave workload args untouched."""

    options: dict[str, object] = {
        "kubeconfig": str(DEFAULT_KUBECONFIG),
        "namespace": "midtraining",
        "pod_name": "midtraining-dev",
        "workspace_root": "/workspace/jaxels-work-trial",
        "branch": "",
        "push_branch": True,
        "allocate_tty": "auto",
    }
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            index += 1
            break
        if arg in {
            "--kubeconfig",
            "--namespace",
            "--pod-name",
            "--workspace-root",
            "--branch",
        }:
            if index + 1 >= len(argv):
                die(f"{arg} requires a value")
            key = arg[2:].replace("-", "_")
            options[key] = argv[index + 1]
            index += 2
            continue
        if arg == "--no-push":
            # Useful for local/test invocations where the caller is explicitly
            # responsible for making the pod checkout match --branch.
            options["push_branch"] = False
            index += 1
            continue
        if arg == "--no-tty":
            options["allocate_tty"] = "0"
            index += 1
            continue
        if arg in {"-h", "--help"}:
            print(USAGE, end="")
            raise SystemExit(0)
        if arg.startswith("-"):
            die(f"unknown launcher option: {arg}", exit_code=1)
        break

    remaining = argv[index:]
    if not remaining:
        print(USAGE, end="", file=sys.stderr)
        raise SystemExit(2)
    workload = remaining[0]
    # Everything after the workload is passed verbatim to the pod script. That
    # is where experiment presets and one-off overrides belong.
    return options, workload, remaining[1:]


def main(argv: list[str] | None = None) -> int:
    options, workload, workload_args = parse_launcher(
        sys.argv[1:] if argv is None else argv
    )
    workload_script = selected_workload_script(workload)
    if workload_script.startswith("/"):
        die(f"workload script must be repository-relative, got: {workload_script}")
    if not (ROOT_DIR / workload_script).is_file():
        die(f"workload script not found: {workload_script}")

    branch = str(options["branch"])
    push_branch = bool(options["push_branch"])
    if (not branch or push_branch) and shutil.which("git") is None:
        die("git not found")
    if not branch:
        branch = command_output(
            ["git", "-C", str(ROOT_DIR), "branch", "--show-current"]
        )
    if not branch:
        die("could not determine current branch; pass --branch explicitly")

    if shutil.which("kubectl") is None:
        die("kubectl not found")
    kubeconfig = Path(str(options["kubeconfig"]))
    if not kubeconfig.is_file():
        die(f"kubeconfig not found: {kubeconfig}")

    if push_branch:
        # Pod workloads run from the branch checked out under /workspace, not
        # from the workstation filesystem. Push only from a clean checkout so
        # training/eval metadata can point at a commit that actually contains
        # the code being executed.
        require_clean_local_checkout()
        run(["git", "-C", str(ROOT_DIR), "push", "-u", "origin", branch])

    kubectl_tty_args: list[str] = []
    allocate_tty = str(options["allocate_tty"])
    if allocate_tty == "auto":
        if sys.stdin.isatty() and sys.stdout.isatty():
            kubectl_tty_args = ["-it"]
    elif allocate_tty == "1":
        kubectl_tty_args = ["-it"]

    workspace_root = str(options["workspace_root"])

    # These three variables describe the pod checkout contract. The legacy
    # SWEHERO_POD_GIT_BRANCH name remains because pod-side guards already use
    # it, but it applies to all current midtraining workloads.
    env_args = [
        f"SWEHERO_POD_GIT_BRANCH={branch}",
        f"MIDTRAINING_POD_WORKSPACE_ROOT={workspace_root}",
        f"WORKSPACE_ROOT={workspace_root}",
    ]
    env_args.extend(
        f"{env_name}={os.environ[env_name]}"
        for env_name in FORWARDED_ENV_NAMES
        if env_name in os.environ
    )

    # Use bash -lc with "$1"/shift so workspace_root and workload paths are
    # passed as arguments rather than interpolated into a shell string. The
    # final command is the pod-side script plus workload_args exactly as the user
    # supplied them.
    exec_process(
        [
            "kubectl",
            "--kubeconfig",
            str(kubeconfig),
            "exec",
            *kubectl_tty_args,
            "-n",
            str(options["namespace"]),
            str(options["pod_name"]),
            "--",
            "bash",
            "-lc",
            'cd "$1" && shift && exec "$@"',
            "bash",
            workspace_root,
            "env",
            *env_args,
            f"{workspace_root}/{workload_script}",
            *workload_args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
