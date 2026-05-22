import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "scripts" / "run_qwen_swehero_torchtitan_pod.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).lstrip())
    path.chmod(0o755)


class QwenSweHeroPodWrapperTests(unittest.TestCase):
    def make_fake_runtime(self, tmp: Path) -> dict[str, str]:
        fake_bin = tmp / "fake-bin"
        fake_bin.mkdir()
        venv_bin = tmp / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        runtime_log = tmp / "runtime.log"
        tmux_log = tmp / "tmux.log"
        setup_log = tmp / "setup.log"

        write_executable(
            venv_bin / "python",
            f"""
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${{1:-}}" == "-" ]]; then
              exec {shlex.quote(sys.executable)} "$@"
            fi
            printf 'python' >> "$FAKE_RUNTIME_LOG"
            printf ' %q' "$@" >> "$FAKE_RUNTIME_LOG"
            printf '\\n' >> "$FAKE_RUNTIME_LOG"
            """,
        )
        write_executable(
            venv_bin / "torchrun",
            """
            #!/usr/bin/env bash
            exit 0
            """,
        )
        write_executable(
            tmp / "setup.sh",
            """
            #!/usr/bin/env bash
            set -euo pipefail
            printf 'setup' >> "$FAKE_SETUP_LOG"
            printf ' %q' "$@" >> "$FAKE_SETUP_LOG"
            printf '\\n' >> "$FAKE_SETUP_LOG"
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
            "SWEHERO_POD_TMUX_LOG_DIR": str(tmp / "runlogs"),
            "SWEHERO_POD_TMUX_ENV_DIR": str(tmp / "tmux-env"),
            "FAKE_RUNTIME_LOG": str(runtime_log),
            "FAKE_TMUX_LOG": str(tmux_log),
            "FAKE_SETUP_LOG": str(setup_log),
        }

    def run_wrapper(
        self,
        tmp: Path,
        args: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = self.make_fake_runtime(tmp)
        env.update(env_overrides or {})
        return subprocess.run(
            [str(WRAPPER), *args],
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

            self.assertEqual(result.returncode, 0, result.stderr)
            tmux_log = (tmp / "tmux.log").read_text()
            setup_log_exists = (tmp / "setup.log").exists()
            runtime_log_exists = (tmp / "runtime.log").exists()

        self.assertIn("has-session -t swehero-qwen-smoke", tmux_log)
        self.assertIn("new-session -d -s swehero-qwen-smoke", tmux_log)
        self.assertIn("pipe-pane -o -t swehero-qwen-smoke:0.0", tmux_log)
        self.assertIn("attach-session -t swehero-qwen-smoke", tmux_log)
        self.assertIn("Started supervised SWE-HERO session", result.stdout)
        self.assertFalse(setup_log_exists)
        self.assertFalse(runtime_log_exists)

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

            self.assertEqual(result.returncode, 0, result.stderr)
            tmux_log = (tmp / "tmux.log").read_text()
            setup_log_exists = (tmp / "setup.log").exists()
            runtime_log_exists = (tmp / "runtime.log").exists()

        self.assertIn("has-session -t swehero-qwen-prod", tmux_log)
        self.assertNotIn("new-session", tmux_log)
        self.assertIn("attach-session -t swehero-qwen-prod", tmux_log)
        self.assertIn("Found existing supervised SWE-HERO session", result.stdout)
        self.assertIn("Attaching now.", result.stdout)
        self.assertFalse(setup_log_exists)
        self.assertFalse(runtime_log_exists)

    def test_forced_supervisor_without_tty_starts_detached_and_returns(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/detached", "--dry-run"],
                env_overrides={"SWEHERO_POD_SUPERVISOR": "1"},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            tmux_log = (tmp / "tmux.log").read_text()
            setup_log_exists = (tmp / "setup.log").exists()
            runtime_log_exists = (tmp / "runtime.log").exists()

        self.assertIn("new-session -d -s swehero-detached", tmux_log)
        self.assertNotIn("attach-session", tmux_log)
        self.assertIn("No interactive terminal is available", result.stdout)
        self.assertFalse(setup_log_exists)
        self.assertFalse(runtime_log_exists)

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

            self.assertEqual(result.returncode, 0, result.stderr)
            tmux_log = (tmp / "tmux.log").read_text()

        self.assertIn("has-session -t swehero-from-arg-file", tmux_log)
        self.assertIn("new-session -d -s swehero-from-arg-file", tmux_log)

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

            self.assertEqual(result.returncode, 0, result.stderr)
            setup_log = (tmp / "setup.log").read_text()
            runtime_log = (tmp / "runtime.log").read_text()
            tmux_log_exists = (tmp / "tmux.log").exists()

        self.assertIn("--verify-only --venv", setup_log)
        self.assertIn("scripts/qwen_swehero_train.py", runtime_log)
        self.assertIn("--out-dir /workspace/runs/direct-child --dry-run", runtime_log)
        self.assertFalse(tmux_log_exists)

    def test_default_noninteractive_launch_keeps_existing_direct_behavior(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.run_wrapper(
                tmp,
                ["--out-dir", "/workspace/runs/noninteractive", "--dry-run"],
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            setup_log = (tmp / "setup.log").read_text()
            runtime_log = (tmp / "runtime.log").read_text()
            tmux_log_exists = (tmp / "tmux.log").exists()

        self.assertIn("--verify-only --venv", setup_log)
        self.assertIn("scripts/qwen_swehero_train.py", runtime_log)
        self.assertIn("--out-dir /workspace/runs/noninteractive --dry-run", runtime_log)
        self.assertFalse(tmux_log_exists)


if __name__ == "__main__":
    unittest.main()
