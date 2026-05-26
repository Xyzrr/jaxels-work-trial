"""Tests for the workstation-to-pod midtraining launcher.

The launcher is deliberately a transport layer: it syncs the local Git branch
into the Kubernetes GPU pod and then runs a chosen pod-side script. It should
not decide which model, dataset, context length, grader, or optimizer an ML
experiment uses. Those choices belong in preset files and workload arguments so
reviewers can reproduce a training or eval run from the command line alone.
"""

from __future__ import annotations

import os
import py_compile
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_midtraining_pod.py"
COMMON = REPO_ROOT / "scripts" / "pod_startup_common.py"


def write_executable(path: Path, content: str) -> None:
    """Write a fake command that records how the launcher calls external tools."""

    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(0o755)


class TestRunMidtrainingPodLauncher:
    """Exercise the workstation wrapper without touching a real cluster."""

    def test_python_syntax_is_valid(self) -> None:
        py_compile.compile(str(SCRIPT), doraise=True)
        py_compile.compile(str(COMMON), doraise=True)

    def test_eval_workload_pushes_branch_and_execs_pod_wrapper(
        self, tmp_path: Path
    ) -> None:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        kubeconfig = tmp_path / "kubeconfig.yaml"
        kubeconfig.write_text("apiVersion: v1\n")
        git_log = tmp_path / "git.log"
        kubectl_log = tmp_path / "kubectl.log"
        write_executable(
            fake_bin / "git",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            while [[ "${1:-}" == "-C" ]]; do
              shift 2
            done
            case "${1:-}" in
              branch)
                [[ "${2:-}" == "--show-current" ]] || exit 99
                echo "axel/test"
                ;;
              status)
                exit 0
                ;;
              push)
                printf 'git' >> "$FAKE_GIT_LOG"
                printf ' %q' "$@" >> "$FAKE_GIT_LOG"
                printf '\\n' >> "$FAKE_GIT_LOG"
                ;;
              *)
                echo "unexpected git invocation: $*" >&2
                exit 99
                ;;
            esac
            """,
        )
        write_executable(
            fake_bin / "kubectl",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            printf 'kubectl' >> "$FAKE_KUBECTL_LOG"
            printf ' %q' "$@" >> "$FAKE_KUBECTL_LOG"
            printf '\\n' >> "$FAKE_KUBECTL_LOG"
            """,
        )

        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GIT_LOG": str(git_log),
            "FAKE_KUBECTL_LOG": str(kubectl_log),
        }
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--kubeconfig",
                str(kubeconfig),
                "--namespace",
                "training",
                "--pod-name",
                "gpu-dev",
                "--workspace-root",
                "/workspace/custom-repo",
                "eval",
                "--eval-limit",
                "1",
                "--no-attach",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        git_invocation = git_log.read_text()
        kubectl_invocation = shlex.split(kubectl_log.read_text())

        # The pod does not see dirty workstation files. Pushing the branch before
        # kubectl exec makes evaluation results attributable to a real commit,
        # which is essential when comparing model checkpoints or grader changes.
        assert git_invocation == "git push -u origin axel/test\n"
        assert "--kubeconfig" in kubectl_invocation
        assert str(kubeconfig) in kubectl_invocation
        assert "exec" in kubectl_invocation
        assert "-n" in kubectl_invocation
        assert "training" in kubectl_invocation
        assert "gpu-dev" in kubectl_invocation
        assert "SWEHERO_POD_GIT_BRANCH=axel/test" in kubectl_invocation
        assert (
            "MIDTRAINING_POD_WORKSPACE_ROOT=/workspace/custom-repo"
            in kubectl_invocation
        )
        assert (
            "/workspace/custom-repo/scripts/run_openhands_swebench_eval_pod.py"
            in kubectl_invocation
        )
        # Workload arguments remain verbatim after "eval". The launcher should
        # not reinterpret eval-limit, presets, model IDs, sampling choices, or
        # any other ML/eval setting because that would create hidden defaults.
        assert "--eval-limit" in kubectl_invocation
        assert "1" in kubectl_invocation
        assert "--no-attach" in kubectl_invocation

    def test_no_push_with_explicit_branch_does_not_require_git(
        self, tmp_path: Path
    ) -> None:
        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        kubeconfig = tmp_path / "kubeconfig.yaml"
        kubeconfig.write_text("apiVersion: v1\n")
        kubectl_log = tmp_path / "kubectl.log"
        write_executable(
            fake_bin / "kubectl",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%q ' "$@" > "$FAKE_KUBECTL_LOG"
            """,
        )
        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_KUBECTL_LOG": str(kubectl_log),
        }

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--kubeconfig",
                str(kubeconfig),
                "--branch",
                "axel/manual",
                "--no-push",
                "prebuild",
                "--eval-limit",
                "2",
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        kubectl_invocation = shlex.split(kubectl_log.read_text())

        # --no-push is an explicit escape hatch for tests or already-synced pod
        # branches. It still passes the branch to the pod-side git guard so the
        # running workload can verify which code revision it should use.
        assert "SWEHERO_POD_GIT_BRANCH=axel/manual" in kubectl_invocation
        assert (
            "/workspace/jaxels-work-trial/scripts/prebuild_openhands_swebench_images_pod.py"
            in kubectl_invocation
        )
        # Image prebuild is eval infrastructure, not a model-training decision:
        # the limit flag is forwarded to the prebuild script instead of being
        # interpreted by the generic pod launcher.
        assert "--eval-limit" in kubectl_invocation
        assert "2" in kubectl_invocation
