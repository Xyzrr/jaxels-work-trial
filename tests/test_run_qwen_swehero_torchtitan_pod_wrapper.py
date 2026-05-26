"""Tests for the pod-local Qwen SWE-HERO TorchTitan wrapper.

The wrapper is runtime plumbing around the real training entrypoint. It should
make long GPU training runs reconnectable, repair the pinned TorchTitan CUDA
venv, and then forward every model/data/training argument unchanged to
``qwen_swehero_train.py``. These tests make that boundary explicit so future
changes do not hide ML experiment choices in the pod supervisor layer.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "run_qwen_swehero_torchtitan_pod.py"


def write_executable(path: Path, content: str) -> None:
    """Write a tiny executable shim used to observe wrapper behavior."""

    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(0o755)


class TestQwenSweHeroPodWrapper:
    """Exercise the wrapper without starting real tmux or TorchTitan training."""

    def make_fake_runtime(self, tmp: Path) -> dict[str, str]:
        """Create fake pod tools and return the environment that exposes them.

        The actual wrapper launches a pinned TorchTitan runtime because CUDA,
        PyTorch, TorchTitan, and TorchAO versions are tightly coupled for model
        training. The tests replace those tools with shims that log invocations,
        so we can verify the wrapper's process control without importing the
        training stack or touching GPUs.
        """

        fake_bin = tmp / "fake-bin"
        fake_bin.mkdir()
        venv_bin = tmp / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        runtime_log = tmp / "runtime.log"
        tmux_log = tmp / "tmux.log"
        setup_log = tmp / "setup.log"
        python_template = tmp / "python-template"
        torchrun_template = tmp / "torchrun-template"

        python_shim = f"""
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${{1:-}}" == "-" ]]; then
              exec {shlex.quote(sys.executable)} "$@"
            fi
            printf 'python' >> "$FAKE_RUNTIME_LOG"
            printf ' %q' "$@" >> "$FAKE_RUNTIME_LOG"
            printf '\\n' >> "$FAKE_RUNTIME_LOG"
            """
        torchrun_shim = """
            #!/usr/bin/env bash
            exit 0
            """
        # The Python shim delegates stdin programs back to the real test Python.
        # The wrapper uses that path when it needs Python only to parse @arg files
        # before the TorchTitan venv exists; normal training invocations are
        # logged instead of executed.
        write_executable(venv_bin / "python", python_shim)
        write_executable(venv_bin / "torchrun", torchrun_shim)
        write_executable(python_template, python_shim)
        write_executable(torchrun_template, torchrun_shim)
        write_executable(
            tmp / "setup.sh",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            printf 'setup' >> "$FAKE_SETUP_LOG"
            printf ' %q' "$@" >> "$FAKE_SETUP_LOG"
            printf '\\n' >> "$FAKE_SETUP_LOG"
            if [[ "${FAKE_SETUP_INSTALL_VENV:-0}" == "1" ]]; then
              mkdir -p "$TORCHTITAN_POD_VENV/bin"
              cp "$FAKE_PYTHON_TEMPLATE" "$TORCHTITAN_POD_VENV/bin/python"
              cp "$FAKE_TORCHRUN_TEMPLATE" "$TORCHTITAN_POD_VENV/bin/torchrun"
            fi
            """,
        )
        write_executable(
            fake_bin / "tmux",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            printf 'tmux' >> "$FAKE_TMUX_LOG"
            printf ' %q' "$@" >> "$FAKE_TMUX_LOG"
            printf '\\n' >> "$FAKE_TMUX_LOG"
            if [[ "${1:-}" == "has-session" ]]; then
              if [[ "${FAKE_TMUX_HAS_SESSION:-0}" == "1" ]]; then
                exit 0
              fi
              exit 1
            fi
            exit 0
            """,
        )

        return {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "TORCHTITAN_POD_VENV": str(tmp / "venv"),
            "TORCHTITAN_POD_SETUP_SCRIPT": str(tmp / "setup.sh"),
            "SWEHERO_POD_GIT_ENFORCE": "0",
            "SWEHERO_POD_TMUX_LOG_DIR": str(tmp / "runlogs"),
            "SWEHERO_POD_TMUX_ENV_DIR": str(tmp / "tmux-env"),
            "FAKE_RUNTIME_LOG": str(runtime_log),
            "FAKE_TMUX_LOG": str(tmux_log),
            "FAKE_SETUP_LOG": str(setup_log),
            "FAKE_PYTHON_TEMPLATE": str(python_template),
            "FAKE_TORCHRUN_TEMPLATE": str(torchrun_template),
        }

    def run_wrapper(
        self,
        tmp: Path,
        args: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
        remove_venv: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        env = self.make_fake_runtime(tmp)
        env.update(env_overrides or {})
        if remove_venv:
            shutil.rmtree(tmp / "venv")
        return subprocess.run(
            [sys.executable, str(WRAPPER), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_forced_supervisor_starts_tmux_session_without_running_verifier(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/qwen-smoke", "--dry-run"],
                env_overrides={
                    "SWEHERO_POD_SUPERVISOR": "1",
                    "SWEHERO_POD_TMUX_ATTACH": "1",
                },
            )

            assert result.returncode == 0, result.stderr
            tmux_log = (tmp / "tmux.log").read_text()
            setup_log_exists = (tmp / "setup.log").exists()
            runtime_log_exists = (tmp / "runtime.log").exists()

        # Starting tmux should not run the verifier/training process in the parent
        # wrapper. The parent only creates a supervised child session so the GPU
        # job can survive a dropped pod connection.
        assert "has-session -t swehero-qwen-smoke" in tmux_log
        assert "new-session -d -s swehero-qwen-smoke" in tmux_log
        assert "pipe-pane -o -t swehero-qwen-smoke:0.0" in tmux_log
        assert "attach-session -t swehero-qwen-smoke" in tmux_log
        assert "Started supervised SWE-HERO session" in result.stdout
        assert not setup_log_exists
        assert not runtime_log_exists

    def test_existing_supervised_session_reconnects_without_starting_new_job(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir=/workspace/runs/qwen-prod"],
                env_overrides={
                    "SWEHERO_POD_SUPERVISOR": "1",
                    "SWEHERO_POD_TMUX_ATTACH": "1",
                    "FAKE_TMUX_HAS_SESSION": "1",
                },
            )

            assert result.returncode == 0, result.stderr
            tmux_log = (tmp / "tmux.log").read_text()
            setup_log_exists = (tmp / "setup.log").exists()
            runtime_log_exists = (tmp / "runtime.log").exists()

        # Reconnecting to an existing session must be side-effect free. A running
        # distributed training job already owns its model state, checkpoints, and
        # optimizer state; starting a new job here would corrupt that lifecycle.
        assert "has-session -t swehero-qwen-prod" in tmux_log
        assert "new-session" not in tmux_log
        assert "attach-session -t swehero-qwen-prod" in tmux_log
        assert "Found existing supervised SWE-HERO session" in result.stdout
        assert "Attaching now." in result.stdout
        assert not setup_log_exists
        assert not runtime_log_exists

    def test_forced_supervisor_without_tty_starts_detached_and_returns(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/detached", "--dry-run"],
                env_overrides={"SWEHERO_POD_SUPERVISOR": "1"},
            )

            assert result.returncode == 0, result.stderr
            tmux_log = (tmp / "tmux.log").read_text()
            setup_log_exists = (tmp / "setup.log").exists()
            runtime_log_exists = (tmp / "runtime.log").exists()

        # In non-interactive contexts the wrapper should still start the tmux
        # supervisor but return immediately, leaving the training session detached
        # for later inspection.
        assert "new-session -d -s swehero-detached" in tmux_log
        assert "attach-session" not in tmux_log
        assert "No interactive terminal is available" in result.stdout
        assert not setup_log_exists
        assert not runtime_log_exists

    def test_supervisor_session_name_uses_out_dir_from_argument_file(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            arg_file = tmp / "launch.args"
            arg_file.write_text(
                "\n".join(
                    [
                        "# production command shape",
                        "--out-dir /workspace/runs/from-arg-file",
                        "--dry-run",
                    ]
                )
            )
            result = self.run_wrapper(
                tmp,
                [f"@{arg_file}"],
                env_overrides={
                    "SWEHERO_POD_SUPERVISOR": "1",
                    "SWEHERO_POD_TMUX_ATTACH": "1",
                },
            )

            assert result.returncode == 0, result.stderr
            tmux_log = (tmp / "tmux.log").read_text()

        # @arg files are the project convention for reproducible experiment
        # presets. The supervisor expands them only far enough to derive the
        # output directory and a stable session name, not to reinterpret training
        # hyperparameters.
        assert "has-session -t swehero-from-arg-file" in tmux_log
        assert "new-session -d -s swehero-from-arg-file" in tmux_log

    def test_supervisor_can_start_before_pod_venv_exists(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/fresh-pod", "--dry-run"],
                env_overrides={
                    "SWEHERO_POD_SUPERVISOR": "1",
                    "SWEHERO_POD_TMUX_ATTACH": "0",
                },
                remove_venv=True,
            )

            assert result.returncode == 0, result.stderr
            tmux_log = (tmp / "tmux.log").read_text()

        # A fresh pod may not have the pinned TorchTitan venv yet. The supervisor
        # still has to start because the tmux child is the process that repairs
        # the CUDA/PyTorch runtime before invoking training.
        assert "new-session -d -s swehero-fresh-pod" in tmux_log
        assert "Canonical TorchTitan venv is missing" not in result.stderr

    def test_supervisor_child_runs_existing_launcher_path_directly(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/direct-child", "--dry-run"],
                env_overrides={
                    "SWEHERO_POD_SUPERVISOR": "1",
                    "SWEHERO_POD_SUPERVISOR_CHILD": "1",
                },
            )

            assert result.returncode == 0, result.stderr
            setup_log = (tmp / "setup.log").read_text()
            runtime_log = (tmp / "runtime.log").read_text()
            tmux_log_exists = (tmp / "tmux.log").exists()

        # Once inside the tmux child, the wrapper must run the setup script and
        # then call the training script directly. This is where the pinned runtime
        # is repaired before the ML recipe is interpreted.
        assert "--venv" in setup_log
        assert "--verify-only" not in setup_log
        assert "scripts/qwen_swehero_train.py" in runtime_log
        # Arguments are forwarded intact so model, dataset, context length, and
        # optimizer choices remain owned by qwen_swehero_train.py and its presets.
        assert "--out-dir /workspace/runs/direct-child --dry-run" in runtime_log
        assert not tmux_log_exists

    def test_direct_launch_repairs_missing_venv_before_training_entrypoint(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/fresh-direct", "--dry-run"],
                env_overrides={"FAKE_SETUP_INSTALL_VENV": "1"},
                remove_venv=True,
            )

            assert result.returncode == 0, result.stderr
            setup_log = (tmp / "setup.log").read_text()
            runtime_log = (tmp / "runtime.log").read_text()

        # Direct non-supervised launches also repair the venv first. Otherwise a
        # stale CUDA/PyTorch/TorchTitan stack could fail or alter training behavior
        # before the explicit Qwen/SWE-HERO recipe is even reached.
        assert "--venv" in setup_log
        assert "--verify-only" not in setup_log
        assert "scripts/qwen_swehero_train.py" in runtime_log
        assert "--out-dir /workspace/runs/fresh-direct --dry-run" in runtime_log

    def test_default_noninteractive_launch_keeps_existing_direct_behavior(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/noninteractive", "--dry-run"],
            )

            assert result.returncode == 0, result.stderr
            setup_log = (tmp / "setup.log").read_text()
            runtime_log = (tmp / "runtime.log").read_text()
            tmux_log_exists = (tmp / "tmux.log").exists()

        # Non-interactive automation keeps the original direct behavior unless
        # supervision is explicitly requested. That matters for CI and wrapper
        # tests that expect one process to exec the training entrypoint.
        assert "--venv" in setup_log
        assert "--verify-only" not in setup_log
        assert "scripts/qwen_swehero_train.py" in runtime_log
        assert "--out-dir /workspace/runs/noninteractive --dry-run" in runtime_log
        assert not tmux_log_exists
