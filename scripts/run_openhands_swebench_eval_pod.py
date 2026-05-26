#!/usr/bin/env python3
"""Pod-side launcher for OpenHands SWE-bench evals served by local vLLM.

This file is the glue between three systems that are easy to confuse:

* OpenHands drives the agent loop and asks an LLM to edit a repository.
* vLLM serves the model on the GPU pod through an OpenAI-compatible API.
* SWE-bench grades the generated patch in Docker after the agent finishes.

Most of the choices below are not generic Kubernetes plumbing; they encode ML
serving constraints for long-context coding models. Comments intentionally spell
out why we choose specific context lengths, GPU layouts, and concurrency limits
so engineers who do not work with model serving can still reason about changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import (
    openhands_eval_launcher_defaults,
    openhands_eval_worker_selection,
    pod_startup_common,
)
from scripts import (
    openhands_swebench_eval as eval_script,
)
from scripts.pod_utils import die, exec_process, repo_root, shell_join, shell_quote

ROOT_DIR = repo_root()
OPENHANDS_EVAL_UV_VERSION = "0.11.16"
UV_X86_64_UNKNOWN_LINUX_GNU_SHA256 = (
    "74947fe2c03315cf07e82ab3acc703eddef01aba4d5232a98e4c6825ec116131"
)
# Qwen2.5-Coder is trained with a 32k-token native position window. Asking vLLM
# to serve 128k without an explicit long-context strategy would make the model
# extrapolate beyond its native positional embeddings in an undefined way.
QWEN_NATIVE_CONTEXT_LENGTH = 32768


USAGE = """\
Usage: scripts/run_openhands_swebench_eval_pod.py [options]

Launch the canonical OpenHands SWE-bench Verified pass@1 eval from the GPU pod.
For workstation launches, use: scripts/run_midtraining_pod.py eval [options]

Options:
  --config PATH           Argparse preset file. Defaults to the 7B 128k preset.
  --eval-limit N          Run N instances. Omit for the full Verified split.
  --eval-ids IDS          Comma-separated SWE-bench instance IDs to run.
  --preflight-only        Check vLLM tool calls and Docker, then exit.
  --skip-swebench-eval    Generate patches without running SWE-bench grading.
  --output-dir PATH       Eval output directory. Defaults to a timestamped pod path.
  --run-id NAME           Timestamp/name component used by the default output dir.
  --foreground            Run in this shell instead of supervising with tmux.
  --attach                Attach to the tmux session after launch.
  --no-attach             Do not attach to the tmux session after launch.
  -h, --help              Show this help.

Environment overrides:
  WORKSPACE_ROOT          Default: /workspace/jaxels-work-trial
  LLM_API_KEY             Default: local-llm, or dummy-key for the SWE-Lego
                          eval stack to match its vendored OpenHands/vLLM
                          scripts.
  VLLM_VENV               Default: /workspace/venvs/openhands-vllm
  VLLM_REQUIREMENTS_PATH  Default: requirements/openhands-vllm.txt
  VLLM_FORCE_RESTART      Set to 1 to replace an already-running vLLM server.
  VLLM_NCCL_CUMEM_ENABLE  Default: auto. The launcher sets NCCL_CUMEM_ENABLE=1
                          for multi-GPU vLLM servers to avoid tiny /dev/shm
                          failures on the pod.
  EVAL_VENV               Default: /workspace/venvs/openhands-eval-pod-py312
  OPENHANDS_EVAL_POETRY_VERSION
                          Default: 2.1.3
  REQUIRED_GPU_COUNT      Default: 8
  SWEHERO_POD_GIT_BRANCH  Required for new pod-side launches. Set by
                          scripts/run_midtraining_pod.py from the selected
                          local branch; the pod startup guard fast-forwards it
                          from origin.
  OPENHANDS_EVAL_TMUX_SESSION
  OPENHANDS_EVAL_ATTACH   Default: 1 for interactive shells, otherwise 0
"""


def run(
    args: list[str], *, check: bool = True, **kwargs
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, check=check, **kwargs)


def run_output(args: list[str]) -> str:
    return run(args, capture_output=True).stdout.strip()


def env_bool(value: str) -> bool:
    return value in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


class EvalLauncher:
    """Coordinate pod-local model serving, OpenHands inference, and grading."""

    def __init__(self) -> None:
        uv_version_env = os.environ.get("UV_VERSION")
        if uv_version_env and uv_version_env != OPENHANDS_EVAL_UV_VERSION:
            die(
                "UV_VERSION override is not supported; expected uv "
                f"{OPENHANDS_EVAL_UV_VERSION}, got {uv_version_env}"
            )
        self.workspace_root = Path(
            os.environ.get("WORKSPACE_ROOT", "/workspace/jaxels-work-trial")
        )
        # The default preset is the paper-aligned SWE-HERO eval recipe: Qwen
        # 2.5 Coder 7B, OpenHands, SWE-bench Verified, and a 128k YaRN context
        # window. Other eval stacks still come from explicit preset files.
        self.config_preset = (
            ROOT_DIR
            / "configs/eval/openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args"
        )
        # vLLM requires an API key even for a local server. The value is not a
        # secret for these pod-local evals; it is only a bearer token shared by
        # OpenHands, the router, and the model server.
        self.llm_api_key_explicit = "LLM_API_KEY" in os.environ
        self.llm_api_key = os.environ.get("LLM_API_KEY", "local-llm")
        # Keep model-serving dependencies separate from OpenHands dependencies.
        # vLLM pins CUDA-adjacent packages tightly, while OpenHands and
        # SWE-bench bring their own Python dependency constraints.
        self.vllm_venv = Path(
            os.environ.get("VLLM_VENV", "/workspace/venvs/openhands-vllm")
        )
        self.vllm_requirements_path = Path(
            os.environ.get(
                "VLLM_REQUIREMENTS_PATH",
                str(ROOT_DIR / "requirements" / "openhands-vllm.txt"),
            )
        )
        self.vllm_python_version = os.environ.get("VLLM_PYTHON_VERSION", "3.12")
        self.vllm_visible_devices = os.environ.get(
            "VLLM_VISIBLE_DEVICES", os.environ.get("VLLM_GPU", "")
        )
        # Restart is opt-in because loading a coding model into GPU memory can
        # take minutes. A context signature below detects when reuse is safe.
        self.vllm_force_restart = os.environ.get("VLLM_FORCE_RESTART", "0")
        self.vllm_nccl_cumem_enable = os.environ.get("VLLM_NCCL_CUMEM_ENABLE", "auto")
        self.vllm_tmux_session = os.environ.get(
            "VLLM_TMUX_SESSION", "openhands-vllm-7b"
        )
        self.vllm_tmux_session_prefix = os.environ.get(
            "VLLM_TMUX_SESSION_PREFIX", "openhands-vllm-7b-gpu"
        )
        self.vllm_router_tmux_session = os.environ.get(
            "VLLM_ROUTER_TMUX_SESSION", "openhands-vllm-router"
        )
        self.eval_venv = Path(
            os.environ.get("EVAL_VENV", "/workspace/venvs/openhands-eval-pod-py312")
        )
        self.openhands_eval_python_version = os.environ.get(
            "OPENHANDS_EVAL_PYTHON_VERSION", "3.12"
        )
        self.openhands_eval_poetry_version = os.environ.get(
            "OPENHANDS_EVAL_POETRY_VERSION", "2.1.3"
        )
        self.docker_tmux_session = os.environ.get(
            "DOCKER_TMUX_SESSION", "openhands-dockerd"
        )
        self.required_gpu_count = int(os.environ.get("REQUIRED_GPU_COUNT", "8"))
        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self.output_dir_explicit = False
        self.output_dir = Path(
            f"/workspace/eval-runs/openhands-swebench-verified-pass1/{self.run_id}"
        )
        self.tmux_session = os.environ.get(
            "OPENHANDS_EVAL_TMUX_SESSION", f"openhands-swebench-eval-{self.run_id}"
        )
        self.tmux_log_dir = Path(
            os.environ.get("OPENHANDS_EVAL_TMUX_LOG_DIR", "/workspace/runlogs")
        )
        self.tmux_log_path = self.tmux_log_dir / f"{self.tmux_session}.log"
        self.uv_tool_dir = Path(os.environ.get("UV_TOOL_DIR", "/workspace/uv"))
        self.uv_cache_dir = Path(os.environ.get("UV_CACHE_DIR", "/workspace/.cache/uv"))
        self.uv_python_install_dir = Path(
            os.environ.get("UV_PYTHON_INSTALL_DIR", "/workspace/python")
        )
        self.bootstrap_python = os.environ.get("PYTHON", "")
        self.eval_python_ready = False
        self.vllm_python_ready = False
        self.eval_limit = ""
        self.eval_ids = ""
        self.preflight_only = False
        self.skip_swebench_eval = False
        self.foreground = False
        self.attach = os.environ.get(
            "OPENHANDS_EVAL_ATTACH", "1" if sys.stdout.isatty() else "0"
        )

    def parse(self, argv: list[str]) -> None:
        index = 0
        while index < len(argv):
            arg = argv[index]
            if arg in {
                "--config",
                "--eval-limit",
                "--eval-ids",
                "--output-dir",
                "--run-id",
            }:
                if index + 1 >= len(argv):
                    die(f"{arg} requires a value")
                value = argv[index + 1]
                if arg == "--config":
                    self.config_preset = Path(value)
                elif arg == "--eval-limit":
                    self.eval_limit = value
                elif arg == "--eval-ids":
                    self.eval_ids = value
                elif arg == "--output-dir":
                    self.output_dir = Path(value)
                    self.output_dir_explicit = True
                elif arg == "--run-id":
                    self.run_id = value
                    if not self.output_dir_explicit:
                        self.output_dir = Path(
                            f"/workspace/eval-runs/openhands-swebench-verified-pass1/{self.run_id}"
                        )
                    self.tmux_session = os.environ.get(
                        "OPENHANDS_EVAL_TMUX_SESSION",
                        f"openhands-swebench-eval-{self.run_id}",
                    )
                    self.tmux_log_path = self.tmux_log_dir / f"{self.tmux_session}.log"
                index += 2
            elif arg == "--preflight-only":
                self.preflight_only = True
                index += 1
            elif arg == "--skip-swebench-eval":
                self.skip_swebench_eval = True
                index += 1
            elif arg == "--foreground":
                self.foreground = True
                index += 1
            elif arg == "--attach":
                self.attach = "1"
                index += 1
            elif arg == "--no-attach":
                self.attach = "0"
                index += 1
            elif arg in {"-h", "--help"}:
                print(USAGE, end="")
                raise SystemExit(0)
            else:
                die(f"unknown option: {arg}")
        if self.eval_limit and self.eval_ids:
            die("--eval-limit and --eval-ids are mutually exclusive")

    def resolve_config_preset_path(self) -> Path:
        if self.config_preset.is_absolute():
            return self.config_preset
        return ROOT_DIR / self.config_preset

    def resolve_eval_config(self, config_path: Path) -> dict[str, object]:
        """Parse the preset once and project eval settings into launcher state."""

        if not config_path.is_file():
            raise SystemExit(f"eval config preset not found: {config_path}")
        # The eval script owns preset semantics. Reusing its parser here avoids a
        # second source of truth for ML knobs such as context mode, sampling, and
        # vLLM parallelism.
        args = eval_script.parse_args(
            [
                f"@{config_path}",
                "--dry-run",
                "--output-dir",
                str(self.output_dir),
            ]
        )
        context_spec = eval_script.context_mode_spec(args.context_mode)
        # The pod launcher must know whether vLLM needs permission to exceed the
        # model's native context window. For Qwen2.5-Coder, that is only valid
        # when the preset also supplies the matching YaRN rope scaling config.
        return {
            "CONFIG_PRESET_PATH": config_path,
            "EVAL_STACK": args.eval_stack,
            "MODEL_ID": args.model_id,
            "SERVED_MODEL_NAME": args.served_model_name,
            "LITELLM_MODEL": args.litellm_model or "",
            "CONTEXT_MODE": args.context_mode,
            "MAX_INPUT_TOKENS": args.max_input_tokens,
            "CONTEXT_ALLOW_LONG_MAX_MODEL_LEN": "1"
            if context_spec.allow_long_max_model_len
            else "0",
            "OPENHANDS_REPO": args.openhands_repo,
            "OPENHANDS_REF": args.openhands_ref,
            "OPENHANDS_DIR": eval_script.effective_openhands_dir(args),
            "OPENHANDS_POETRY_VERSION_FROM_CONFIG": args.openhands_poetry_version,
            "SWE_LEGO_REPO": args.swe_lego_repo,
            "SWE_LEGO_REF": args.swe_lego_ref,
            "SWE_LEGO_DIR": args.swe_lego_dir,
            "SWE_LEGO_SWEBENCH_DIR": eval_script.effective_swebench_dir(args) or "",
            "DOCKER_SMOKE_IMAGE": args.docker_smoke_image,
            "VLLM_HOST": args.vllm_host,
            "VLLM_PORT": args.vllm_port,
            "VLLM_MAX_MODEL_LEN": args.vllm_max_model_len,
            "VLLM_MAX_NUM_SEQS": args.vllm_max_num_seqs or "",
            "VLLM_ROPE_SCALING": args.vllm_rope_scaling or "",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"
            if context_spec.allow_long_max_model_len
            and args.vllm_max_model_len > QWEN_NATIVE_CONTEXT_LENGTH
            else "0",
            "VLLM_ENFORCE_EAGER": "1" if args.vllm_enforce_eager else "0",
            "VLLM_TENSOR_PARALLEL_SIZE": args.vllm_tensor_parallel_size,
            "VLLM_PIPELINE_PARALLEL_SIZE": args.vllm_pipeline_parallel_size,
            "VLLM_SERVER_COUNT": args.vllm_server_count,
            "VLLM_AGENT_TASKS_PER_SERVER": args.vllm_agent_tasks_per_server,
            "VLLM_USE_ROUTER": "1" if args.vllm_use_router else "0",
            "VLLM_ROUTER_PORT": args.vllm_router_port,
            "VLLM_GPU_MEMORY_UTILIZATION": args.vllm_gpu_memory_utilization,
            "VLLM_DTYPE": args.vllm_dtype,
            "VLLM_ENABLE_AUTO_TOOL_CHOICE": "1"
            if args.vllm_enable_auto_tool_choice
            else "0",
            "VLLM_TOOL_CALL_PARSER": args.vllm_tool_call_parser,
            "VLLM_DISTRIBUTED_EXECUTOR_BACKEND": args.vllm_distributed_executor_backend
            or "",
            "CONFIG_NUM_WORKERS": args.num_workers,
            "SWEBENCH_CACHE_LEVEL": args.swebench_cache_level,
            "SWEBENCH_TIMEOUT": args.swebench_timeout,
            "SWEBENCH_MAX_WORKERS": args.swebench_max_workers,
        }

    def supervised_env_args(self) -> list[str]:
        return [
            f"SWEHERO_POD_GIT_BRANCH={os.environ.get('SWEHERO_POD_GIT_BRANCH', '')}",
            f"LLM_API_KEY={self.llm_api_key}",
            f"VLLM_VENV={self.vllm_venv}",
            f"VLLM_REQUIREMENTS_PATH={self.vllm_requirements_path}",
            f"VLLM_PYTHON_VERSION={self.vllm_python_version}",
            f"VLLM_VISIBLE_DEVICES={self.vllm_visible_devices}",
            f"VLLM_FORCE_RESTART={self.vllm_force_restart}",
            f"VLLM_NCCL_CUMEM_ENABLE={self.vllm_nccl_cumem_enable}",
            f"VLLM_TMUX_SESSION={self.vllm_tmux_session}",
            f"VLLM_TMUX_SESSION_PREFIX={self.vllm_tmux_session_prefix}",
            f"VLLM_ROUTER_TMUX_SESSION={self.vllm_router_tmux_session}",
            f"EVAL_VENV={self.eval_venv}",
            f"OPENHANDS_EVAL_PYTHON_VERSION={self.openhands_eval_python_version}",
            f"OPENHANDS_EVAL_POETRY_VERSION={self.openhands_eval_poetry_version}",
            f"DOCKER_TMUX_SESSION={self.docker_tmux_session}",
            f"REQUIRED_GPU_COUNT={self.required_gpu_count}",
            f"OPENHANDS_EVAL_TMUX_LOG_DIR={self.tmux_log_dir}",
            f"UV_TOOL_DIR={self.uv_tool_dir}",
            f"UV_CACHE_DIR={self.uv_cache_dir}",
            f"UV_PYTHON_INSTALL_DIR={self.uv_python_install_dir}",
        ]

    def launch_tmux_if_needed(self, config_path: Path) -> None:
        if self.foreground:
            return
        if shutil.which("tmux") is None:
            die("tmux is required for supervised pod launches")
        self.tmux_log_dir.mkdir(parents=True, exist_ok=True)
        if (
            run(
                ["tmux", "has-session", "-t", self.tmux_session], check=False
            ).returncode
            == 0
        ):
            print(f"tmux session already exists: {self.tmux_session}")
        else:
            pod_startup_common.prepare_pod_checkout(
                self.workspace_root,
                "OpenHands eval pod execution directory",
            )
            script_path = Path(__file__).resolve()
            command = (
                f"cd {shell_quote(self.workspace_root)} && env "
                f"{shell_join(self.supervised_env_args())} "
                f"{shell_quote(script_path)} --foreground"
            )
            if self.eval_limit:
                command += f" --eval-limit {shell_quote(self.eval_limit)}"
            if self.eval_ids:
                command += f" --eval-ids {shell_quote(self.eval_ids)}"
            if self.preflight_only:
                command += " --preflight-only"
            if self.skip_swebench_eval:
                command += " --skip-swebench-eval"
            command += f" --config {shell_quote(config_path)}"
            command += f" --output-dir {shell_quote(self.output_dir)}"
            run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    self.tmux_session,
                    f"set -euo pipefail; {command} 2>&1 | tee -a {shell_quote(self.tmux_log_path)}",
                ]
            )
            print(f"launched tmux session: {self.tmux_session}")
        print(f"log: {self.tmux_log_path}")
        if self.attach == "1":
            exec_process(["tmux", "attach-session", "-t", self.tmux_session])
        raise SystemExit(0)

    def uv_version_matches(self, uv_bin: Path) -> bool:
        if not uv_bin.exists():
            return False
        try:
            actual = run_output([str(uv_bin), "--version"])
        except Exception:
            return False
        return actual.startswith(f"uv {OPENHANDS_EVAL_UV_VERSION}")

    def require_uv_version(self, uv_bin: Path) -> None:
        actual = run_output([str(uv_bin), "--version"])
        if not actual.startswith(f"uv {OPENHANDS_EVAL_UV_VERSION}"):
            die(
                f"Wrong uv binary at {uv_bin}: expected uv "
                f"{OPENHANDS_EVAL_UV_VERSION}, found: {actual}"
            )
        print(actual, file=sys.stderr)

    def ensure_uv(self) -> Path:
        uv_bin_env = os.environ.get("UV_BIN")
        if uv_bin_env:
            uv_bin = Path(uv_bin_env)
            if not uv_bin.exists() or not os.access(uv_bin, os.X_OK):
                die(f"UV_BIN is not executable: {uv_bin}")
            self.require_uv_version(uv_bin)
            return uv_bin

        managed_dir = self.uv_tool_dir / f"uv-{OPENHANDS_EVAL_UV_VERSION}"
        uv_bin = managed_dir / "uv"
        if uv_bin.exists():
            if self.uv_version_matches(uv_bin):
                self.require_uv_version(uv_bin)
                return uv_bin
            print(
                f"Removing wrong uv binary from pinned tool directory: {uv_bin}",
                file=sys.stderr,
            )
            shutil.rmtree(managed_dir)

        system_uv = shutil.which("uv")
        if system_uv and self.uv_version_matches(Path(system_uv)):
            self.require_uv_version(Path(system_uv))
            return Path(system_uv)

        if os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
            die(
                f"uv {OPENHANDS_EVAL_UV_VERSION} is required. Install it or set UV_BIN=/path/to/uv."
            )
        bootstrap_python = self.bootstrap_python or shutil.which("python3")
        if not bootstrap_python:
            die("pinned uv is missing and no bootstrap Python is available")

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            archive = tmp / "uv.tar.gz"
            url = (
                "https://github.com/astral-sh/uv/releases/download/"
                f"{OPENHANDS_EVAL_UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
            )
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                archive.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            actual_sha256 = (
                __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
            )
            if actual_sha256 != UV_X86_64_UNKNOWN_LINUX_GNU_SHA256:
                raise SystemExit(
                    "uv archive checksum mismatch: expected "
                    f"{UV_X86_64_UNKNOWN_LINUX_GNU_SHA256}, found {actual_sha256}"
                )
            with tarfile.open(archive) as tar:
                tar.extractall(tmp)
            managed_dir.mkdir(parents=True, exist_ok=True)
            extracted = tmp / "uv-x86_64-unknown-linux-gnu"
            shutil.copy2(extracted / "uv", managed_dir / "uv")
            shutil.copy2(extracted / "uvx", managed_dir / "uvx")
            (managed_dir / "uv").chmod(0o755)
            (managed_dir / "uvx").chmod(0o755)
        self.require_uv_version(uv_bin)
        return uv_bin

    def venv_python_matches(self, venv_path: Path, expected_version: str) -> bool:
        python = venv_path / "bin" / "python"
        if not python.exists():
            return False
        code = (
            "import sys; expected=tuple(int(p) for p in sys.argv[1].split('.')); "
            "raise SystemExit(0 if sys.version_info[:len(expected)] == expected else 1)"
        )
        return (
            run([str(python), "-c", code, expected_version], check=False).returncode
            == 0
        )

    def ensure_python_venv(
        self, venv_path: Path, python_version: str, uv_bin: Path
    ) -> None:
        if self.venv_python_matches(venv_path, python_version):
            return
        shutil.rmtree(venv_path, ignore_errors=True)
        venv_path.parent.mkdir(parents=True, exist_ok=True)
        self.uv_python_install_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["UV_PYTHON_DOWNLOADS"] = "automatic"
        run(
            [
                str(uv_bin),
                "python",
                "install",
                python_version,
                "--install-dir",
                str(self.uv_python_install_dir),
                "--no-bin",
            ],
            env=env,
        )
        env["UV_PYTHON_INSTALL_DIR"] = str(self.uv_python_install_dir)
        run(
            [
                str(uv_bin),
                "venv",
                "--no-project",
                "--python",
                python_version,
                "--seed",
                str(venv_path),
            ],
            env=env,
        )

    def ensure_eval_python(self, uv_bin: Path) -> None:
        """Create the Python runtime used for OpenHands and SWE-bench grading."""

        if self.eval_python_ready:
            return
        self.ensure_python_venv(
            self.eval_venv, self.openhands_eval_python_version, uv_bin
        )
        run(
            [
                str(uv_bin),
                "pip",
                "install",
                "--python",
                str(self.eval_venv / "bin" / "python"),
                f"poetry=={self.openhands_eval_poetry_version}",
            ]
        )
        run(
            [
                str(uv_bin),
                "pip",
                "check",
                "--python",
                str(self.eval_venv / "bin" / "python"),
            ]
        )
        code = (
            "import json, subprocess, sys, time; "
            "from importlib.metadata import version; from pathlib import Path; "
            "venv=Path(sys.argv[1]); expected=sys.argv[2]; uv=Path(sys.argv[3]); "
            "actual=version('poetry'); "
            "assert actual == expected, f'poetry version mismatch: expected {expected}, found {actual}'; "
            "record={'created_at_unix': time.time(), 'python': sys.version, 'venv': str(venv), "
            "'uv': subprocess.check_output([str(uv), '--version'], text=True).strip(), 'poetry': actual}; "
            "(venv/'openhands-eval-runtime.json').write_text(json.dumps(record, indent=2))"
        )
        run(
            [
                str(self.eval_venv / "bin" / "python"),
                "-c",
                code,
                str(self.eval_venv),
                self.openhands_eval_poetry_version,
                str(uv_bin),
            ]
        )
        self.eval_python_ready = True

    def ensure_vllm_python(self, uv_bin: Path) -> None:
        """Create the Python runtime used only for the vLLM model server."""

        if self.vllm_python_ready:
            return
        if not self.vllm_requirements_path.is_file():
            die(f"vLLM requirements file not found: {self.vllm_requirements_path}")
        self.ensure_python_venv(self.vllm_venv, self.vllm_python_version, uv_bin)
        resolved_requirements = self.vllm_venv / "openhands-vllm-resolved.txt"
        # Compile then sync rather than installing the input file directly. That
        # records the full transitive vLLM stack, which matters for GPU serving
        # because CUDA, transformers, and vLLM versions must remain compatible.
        run(
            [
                str(uv_bin),
                "pip",
                "compile",
                "--python",
                str(self.vllm_venv / "bin" / "python"),
                "--output-file",
                str(resolved_requirements),
                str(self.vllm_requirements_path),
            ]
        )
        run(
            [
                str(uv_bin),
                "pip",
                "sync",
                "--python",
                str(self.vllm_venv / "bin" / "python"),
                str(resolved_requirements),
            ]
        )
        run(
            [
                str(uv_bin),
                "pip",
                "check",
                "--python",
                str(self.vllm_venv / "bin" / "python"),
            ]
        )
        expected_vllm = None
        for line in self.vllm_requirements_path.read_text().splitlines():
            match = re.match(r"^vllm==(.+)$", line.strip())
            if match:
                expected_vllm = match.group(1)
                break
        if expected_vllm is None:
            die(f"{self.vllm_requirements_path} must pin vllm with vllm==...")
        # vLLM is the model-serving engine, so silently drifting this version can
        # change context-length behavior, scheduler behavior, or tool-call JSON.
        actual_vllm = run(
            [
                str(self.vllm_venv / "bin" / "python"),
                "-c",
                "from importlib.metadata import version; print(version('vllm'))",
            ],
            capture_output=True,
        ).stdout.strip()
        if actual_vllm != expected_vllm:
            die(f"vllm version mismatch: expected {expected_vllm}, found {actual_vllm}")
        if not (self.vllm_venv / "bin" / "vllm").exists():
            die(f"vLLM CLI missing: {self.vllm_venv / 'bin' / 'vllm'}")
        self.vllm_python_ready = True

    def poetry_install_openhands_dependencies(self) -> None:
        poetry_env = dict(os.environ)
        poetry_env["PATH"] = f"{self.eval_venv / 'bin'}:{poetry_env.get('PATH', '')}"
        poetry_env["POETRY_VIRTUALENVS_PATH"] = "/workspace/venvs/poetry-pod"
        poetry_env["POETRY_CACHE_DIR"] = "/workspace/.cache/poetry-pod"
        run(
            [
                "poetry",
                "-C",
                str(self.openhands_dir),
                "env",
                "use",
                str(self.eval_venv / "bin" / "python"),
            ],
            env=poetry_env,
        )
        if (
            run(
                ["poetry", "-C", str(self.openhands_dir), "sync", "--help"],
                env=poetry_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        ):
            run(
                [
                    "poetry",
                    "-C",
                    str(self.openhands_dir),
                    "sync",
                    "--with",
                    "evaluation,test",
                    "--no-root",
                ],
                env=poetry_env,
            )
        else:
            run(
                [
                    "poetry",
                    "-C",
                    str(self.openhands_dir),
                    "install",
                    "--sync",
                    "--with",
                    "evaluation,test",
                    "--no-root",
                ],
                env=poetry_env,
            )

    def ensure_docker(self) -> None:
        if (
            run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            != 0
        ):
            run(
                ["tmux", "kill-session", "-t", self.docker_tmux_session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    self.docker_tmux_session,
                    f"dockerd --host=unix:///var/run/docker.sock > /workspace/runlogs/{self.docker_tmux_session}.log 2>&1",
                ]
            )
        for _ in range(90):
            if (
                run(
                    ["docker", "info"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            ):
                break
            time.sleep(1)
        if (
            run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            != 0
        ):
            die(
                f"Docker daemon did not become ready; see /workspace/runlogs/{self.docker_tmux_session}.log"
            )
        run(
            ["docker", "run", "--rm", self.docker_smoke_image],
            stdout=subprocess.DEVNULL,
        )
        run(["docker", "buildx", "version"], stdout=subprocess.DEVNULL)

    def ensure_openhands_checkout(self) -> None:
        """Check out the exact eval harness requested by the preset."""

        if self.eval_stack == "swe-lego":
            # SWE-Lego vendors the OpenHands and SWE-bench revisions it expects.
            # Using upstream OpenHands here would mix eval harness assumptions
            # with a checkpoint that was published for the vendored stack.
            swe_lego_dir = Path(self.swe_lego_dir)
            if not (swe_lego_dir / ".git").is_dir():
                swe_lego_dir.parent.mkdir(parents=True, exist_ok=True)
                run(["git", "clone", self.swe_lego_repo, str(swe_lego_dir)])
            if run(
                ["git", "-C", str(swe_lego_dir), "status", "--porcelain"],
                capture_output=True,
            ).stdout:
                die(f"{swe_lego_dir} has local changes; clean it before launching eval")
            run(
                [
                    "git",
                    "-C",
                    str(swe_lego_dir),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    self.swe_lego_ref,
                ]
            )
            run(
                [
                    "git",
                    "-C",
                    str(swe_lego_dir),
                    "checkout",
                    "--detach",
                    self.swe_lego_ref,
                ]
            )
            if not self.openhands_dir.is_dir():
                die(f"SWE-Lego OpenHands directory missing: {self.openhands_dir}")
            if not Path(self.swe_lego_swebench_dir).is_dir():
                die(
                    f"SWE-Lego SWE-bench directory missing: {self.swe_lego_swebench_dir}"
                )
            return

        if not (self.openhands_dir / ".git").is_dir():
            self.openhands_dir.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "git",
                    "clone",
                    "--branch",
                    self.openhands_ref,
                    "--depth",
                    "1",
                    self.openhands_repo,
                    str(self.openhands_dir),
                ]
            )
        if run(
            ["git", "-C", str(self.openhands_dir), "status", "--porcelain"],
            capture_output=True,
        ).stdout:
            die(
                f"{self.openhands_dir} has local changes; clean it before launching eval"
            )
        run(
            [
                "git",
                "-C",
                str(self.openhands_dir),
                "fetch",
                "--tags",
                "--depth",
                "1",
                "origin",
                self.openhands_ref,
            ]
        )
        run(
            [
                "git",
                "-C",
                str(self.openhands_dir),
                "checkout",
                "--detach",
                self.openhands_ref,
            ]
        )

    def ensure_openhands_dependencies(self, uv_bin: Path) -> None:
        """Install the selected OpenHands/SWE-bench stack into the eval venv."""

        self.ensure_openhands_checkout()
        self.poetry_install_openhands_dependencies()
        if self.eval_stack == "swe-lego":
            # SWE-Lego pins a vendored SWE-bench package. Installing it editable
            # makes Python imports resolve to the same grader code the stack was
            # released with, instead of the upstream package in this repo.
            run(
                [
                    str(uv_bin),
                    "pip",
                    "install",
                    "--python",
                    str(self.eval_venv / "bin" / "python"),
                    "-e",
                    self.swe_lego_swebench_dir,
                ]
            )
            run(
                [
                    str(uv_bin),
                    "pip",
                    "check",
                    "--python",
                    str(self.eval_venv / "bin" / "python"),
                ]
            )

    def pod_ip(self) -> str:
        return run(["hostname", "-I"], capture_output=True).stdout.split()[0]

    def vllm_session_name(self, gpu: int) -> str:
        return f"{self.vllm_tmux_session_prefix}-{gpu}"

    def effective_vllm_nccl_cumem_enable(self) -> str:
        if self.vllm_nccl_cumem_enable == "auto":
            if self.vllm_parallel_gpu_count > 1:
                # NCCL coordinates GPU-to-GPU communication for tensor/pipeline
                # parallel serving. The pod's /dev/shm is small, and enabling
                # CUDA memory-backed NCCL avoids fragile shared-memory failures.
                return "1"
            return ""
        return self.vllm_nccl_cumem_enable

    def vllm_context_signature(self, gpu: int, port: int) -> str:
        """Describe the model-serving contract that makes endpoint reuse safe."""

        # A live /models response only proves that something is serving. For eval
        # reproducibility, the endpoint must also match the preset's model,
        # context window, rope scaling, precision, GPU layout, and tool parser.
        return "\n".join(
            [
                f"CONTEXT_MODE={self.context_mode}",
                f"CONFIG_PRESET_PATH={self.config_preset_path}",
                f"MODEL_ID={self.model_id}",
                f"SERVED_MODEL_NAME={self.served_model_name}",
                f"MAX_INPUT_TOKENS={self.max_input_tokens}",
                f"VLLM_MAX_MODEL_LEN={self.vllm_max_model_len}",
                f"VLLM_MAX_NUM_SEQS={self.vllm_max_num_seqs}",
                f"VLLM_ROPE_SCALING={self.vllm_rope_scaling}",
                f"VLLM_DTYPE={self.vllm_dtype}",
                f"VLLM_GPU_MEMORY_UTILIZATION={self.vllm_gpu_memory_utilization}",
                f"VLLM_TENSOR_PARALLEL_SIZE={self.vllm_tensor_parallel_size}",
                f"VLLM_PIPELINE_PARALLEL_SIZE={self.vllm_pipeline_parallel_size}",
                f"VLLM_DISTRIBUTED_EXECUTOR_BACKEND={self.vllm_distributed_executor_backend}",
                f"VLLM_ENFORCE_EAGER={self.vllm_enforce_eager}",
                f"VLLM_ENABLE_AUTO_TOOL_CHOICE={self.vllm_enable_auto_tool_choice}",
                f"VLLM_TOOL_CALL_PARSER={self.vllm_tool_call_parser}",
                f"NCCL_CUMEM_ENABLE={self.effective_vllm_nccl_cumem_enable()}",
                f"GPU={gpu}",
                f"PORT={port}",
                "",
            ]
        )

    def kill_process_pattern(self, pattern: str) -> None:
        if (
            run(
                ["pgrep", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            != 0
        ):
            return
        run(
            ["pkill", "-TERM", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            if (
                run(
                    ["pgrep", "-f", pattern],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                != 0
            ):
                return
            time.sleep(1)
        run(
            ["pkill", "-KILL", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def cleanup_vllm_runtime(self) -> None:
        """Remove stale model-serving processes and IPC files before restart."""

        self.kill_process_pattern(str(self.vllm_venv / "bin" / "vllm"))
        self.kill_process_pattern(
            f"{self.vllm_venv / 'bin' / 'python'} -c from multiprocessing"
        )
        for pattern in ("/dev/shm/psm_*", "/dev/shm/sem.mp-*", "/dev/shm/nccl-*"):
            # Multi-GPU vLLM can leave multiprocessing and NCCL shared-memory
            # handles behind after a hard kill. Removing them prevents the next
            # server from inheriting stale inter-process communication state.
            for path in Path("/").glob(pattern.removeprefix("/")):
                path.unlink(missing_ok=True)

    def ensure_vllm_server(
        self, uv_bin: Path, ip: str, gpu: int, port: int, session: str
    ) -> None:
        """Start or reuse one vLLM OpenAI-compatible endpoint."""

        base_url = f"http://{ip}:{port}/v1"
        context_path = self.tmux_log_dir / f"{session}.context"
        expected_context = self.vllm_context_signature(gpu, port)
        if env_bool(self.vllm_force_restart):
            run(
                ["tmux", "kill-session", "-t", session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        curl_ok = (
            run(
                [
                    "curl",
                    "-fsS",
                    "-H",
                    f"Authorization: Bearer {self.llm_api_key}",
                    f"{base_url}/models",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        if curl_ok:
            if (
                not env_bool(self.vllm_force_restart)
                and context_path.is_file()
                and context_path.read_text() == expected_context
            ):
                # Reuse is safe only when the endpoint and the recorded serving
                # contract match. Otherwise a previous eval could silently serve
                # the wrong model length, precision, or rope scaling.
                return
            print(
                f"restarting vLLM endpoint on {base_url} for context mode {self.context_mode}"
            )
            run(
                ["tmux", "kill-session", "-t", session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                if (
                    run(
                        [
                            "curl",
                            "-fsS",
                            "-H",
                            f"Authorization: Bearer {self.llm_api_key}",
                            f"{base_url}/models",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    ).returncode
                    != 0
                ):
                    break
                time.sleep(1)
            if (
                run(
                    [
                        "curl",
                        "-fsS",
                        "-H",
                        f"Authorization: Bearer {self.llm_api_key}",
                        f"{base_url}/models",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            ):
                die(
                    f"non-matching vLLM endpoint is still serving on {base_url}; stop it or set a different VLLM_PORT"
                )

        self.ensure_vllm_python(uv_bin)
        if not (self.vllm_venv / "bin" / "vllm").exists():
            die(f"missing vLLM binary: {self.vllm_venv / 'bin' / 'vllm'}")
        run(
            ["tmux", "kill-session", "-t", session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        env_parts: list[str] = []
        if self.vllm_visible_devices and self.vllm_visible_devices != "all":
            # A manual override is useful for single-server debugging. It is
            # disallowed for multi-replica runs later because mapping several
            # server processes onto one custom device list is ambiguous.
            env_parts.append(f"CUDA_VISIBLE_DEVICES={self.vllm_visible_devices}")
        elif self.vllm_server_count == 1 and self.vllm_parallel_gpu_count > 1:
            # A single large model server can shard one model across multiple
            # GPUs. CUDA_VISIBLE_DEVICES must expose the whole shard group to the
            # one vLLM process.
            env_parts.append(
                f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(self.vllm_parallel_gpu_count))}"
            )
        else:
            # The standard SWE-HERO eval path runs one vLLM replica per GPU.
            # The router then spreads OpenHands requests across replicas.
            env_parts.append(f"CUDA_VISIBLE_DEVICES={gpu}")
        if env_bool(self.vllm_allow_long_max_model_len):
            # vLLM protects users from accidentally exceeding a model's native
            # context length. We enable the override only after preset validation
            # proved that the corresponding long-context mode is intentional.
            env_parts.append("VLLM_ALLOW_LONG_MAX_MODEL_LEN=1")
        nccl_cumem_enable = self.effective_vllm_nccl_cumem_enable()
        if nccl_cumem_enable:
            env_parts.append(f"NCCL_CUMEM_ENABLE={nccl_cumem_enable}")
        command = [
            str(self.vllm_venv / "bin" / "vllm"),
            "serve",
            self.model_id,
            "--host",
            self.vllm_host,
            "--port",
            str(port),
            "--api-key",
            self.llm_api_key,
            "--served-model-name",
            self.served_model_name,
            "--max-model-len",
            str(self.vllm_max_model_len),
        ]
        if self.vllm_rope_scaling:
            # RoPE is the position-encoding mechanism that lets the model know
            # token order. YaRN scaling stretches that mechanism from Qwen's
            # native 32k window to the paper's 128k evaluation window.
            command.extend(["--rope-scaling", self.vllm_rope_scaling])
        command.extend(
            [
                "--tensor-parallel-size",
                str(self.vllm_tensor_parallel_size),
                "--pipeline-parallel-size",
                str(self.vllm_pipeline_parallel_size),
            ]
        )
        if self.vllm_max_num_seqs:
            # This caps how many prompts vLLM batches together. Higher values
            # improve throughput but consume more KV-cache memory per server.
            command.extend(["--max-num-seqs", str(self.vllm_max_num_seqs)])
        command.extend(
            [
                "--gpu-memory-utilization",
                str(self.vllm_gpu_memory_utilization),
                "--dtype",
                self.vllm_dtype,
            ]
        )
        if env_bool(self.vllm_enable_auto_tool_choice):
            # OpenHands uses tool calls to drive repository edits. Native
            # tool-call parsing catches malformed model output before it becomes
            # an agent action.
            command.append("--enable-auto-tool-choice")
            if self.vllm_tool_call_parser:
                command.extend(["--tool-call-parser", self.vllm_tool_call_parser])
        if self.vllm_distributed_executor_backend:
            # Multi-GPU serving needs a worker launcher. The default "mp"
            # backend keeps all workers inside this pod instead of relying on a
            # separate distributed cluster.
            command.extend(
                [
                    "--distributed-executor-backend",
                    self.vllm_distributed_executor_backend,
                ]
            )
        if env_bool(self.vllm_enforce_eager):
            # Eager execution trades some graph-compiler optimization for more
            # predictable startup/debug behavior in this prototype eval stack.
            command.append("--enforce-eager")
        tmux_command = (
            f"cd {shell_quote(self.workspace_root)} && "
            f"{shell_join(env_parts)} {shell_join(command)} > /workspace/runlogs/{session}.log 2>&1"
        )
        run(["tmux", "new-session", "-d", "-s", session, tmux_command])
        for _ in range(240):
            if (
                run(
                    [
                        "curl",
                        "-fsS",
                        "-H",
                        f"Authorization: Bearer {self.llm_api_key}",
                        f"{base_url}/models",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            ):
                context_path.write_text(expected_context)
                return
            time.sleep(2)
        die(f"vLLM did not become ready; see /workspace/runlogs/{session}.log")

    def ensure_vllm_router(self, ip: str, backend_args: list[str]) -> None:
        """Start or reuse the local router that load-balances vLLM replicas."""

        router_url = f"http://{ip}:{self.vllm_router_port}/v1"
        context_path = self.tmux_log_dir / f"{self.vllm_router_tmux_session}.context"
        expected_context = "\n".join(
            [
                f"CONTEXT_MODE={self.context_mode}",
                f"CONFIG_PRESET_PATH={self.config_preset_path}",
                f"VLLM_ROUTER_PORT={self.vllm_router_port}",
                f"VLLM_AGENT_TASKS_PER_SERVER={self.vllm_agent_tasks_per_server}",
                "LLM_API_KEY_SET=1",
                *[f"BACKEND={arg}" for arg in backend_args],
                "",
            ]
        )
        if env_bool(self.vllm_force_restart):
            run(
                ["tmux", "kill-session", "-t", self.vllm_router_tmux_session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if (
            run(
                [
                    "curl",
                    "-fsS",
                    "-H",
                    f"Authorization: Bearer {self.llm_api_key}",
                    f"{router_url}/models",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        ):
            if (
                not env_bool(self.vllm_force_restart)
                and context_path.is_file()
                and context_path.read_text() == expected_context
            ):
                # The router has no model weights, but it still encodes eval
                # concurrency. Reusing a router with the wrong backend list can
                # overload one GPU or send requests to a stale model server.
                return
            print(
                f"restarting vLLM router on {router_url} for context mode {self.context_mode}"
            )
            run(
                ["tmux", "kill-session", "-t", self.vllm_router_tmux_session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        run(
            ["tmux", "kill-session", "-t", self.vllm_router_tmux_session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        router_command = [
            str(self.eval_venv / "bin" / "python"),
            "scripts/openai_vllm_router.py",
            "--listen-host",
            "0.0.0.0",
            "--listen-port",
            str(self.vllm_router_port),
            "--api-key",
            self.llm_api_key,
            "--per-backend-concurrency",
            str(self.vllm_agent_tasks_per_server),
            *backend_args,
        ]
        command = f"cd {shell_quote(self.workspace_root)} && {shell_join(router_command)} > /workspace/runlogs/{self.vllm_router_tmux_session}.log 2>&1"
        run(["tmux", "new-session", "-d", "-s", self.vllm_router_tmux_session, command])
        for _ in range(90):
            if (
                run(
                    [
                        "curl",
                        "-fsS",
                        "-H",
                        f"Authorization: Bearer {self.llm_api_key}",
                        f"{router_url}/models",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            ):
                context_path.write_text(expected_context)
                return
            time.sleep(1)
        die(
            f"vLLM router did not become ready; see /workspace/runlogs/{self.vllm_router_tmux_session}.log"
        )

    def ensure_vllm_stack(self, uv_bin: Path, ip: str) -> None:
        """Ensure all configured vLLM replicas and the optional router exist."""

        if env_bool(self.vllm_force_restart):
            run(
                ["tmux", "kill-session", "-t", self.vllm_tmux_session],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if self.vllm_server_count == 1 and self.vllm_parallel_gpu_count > 1:
                for stale_gpu in range(self.visible_gpu_count):
                    run(
                        [
                            "tmux",
                            "kill-session",
                            "-t",
                            self.vllm_session_name(stale_gpu),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            if not env_bool(self.vllm_use_router):
                run(
                    ["tmux", "kill-session", "-t", self.vllm_router_tmux_session],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self.cleanup_vllm_runtime()

        backend_args: list[str] = []
        for gpu in range(self.vllm_server_count):
            # In the common replica layout, port N maps to GPU N. For a single
            # multi-GPU server, the loop runs once and that server receives the
            # whole tensor/pipeline-parallel device group.
            port = self.vllm_port + gpu
            session = self.vllm_session_name(gpu)
            self.ensure_vllm_server(uv_bin, ip, gpu, port, session)
            backend_args.extend(["--backend", f"http://{ip}:{port}/v1"])
        if env_bool(self.vllm_use_router):
            self.ensure_vllm_router(ip, backend_args)

    def assign_config(self, values: dict[str, object]) -> None:
        """Copy parsed preset values onto the launcher instance."""

        for key, value in values.items():
            setattr(self, key.lower(), value)
        self.config_preset_path = Path(values["CONFIG_PRESET_PATH"])
        self.openhands_dir = Path(str(values["OPENHANDS_DIR"]))
        if values.get("OPENHANDS_POETRY_VERSION_FROM_CONFIG"):
            self.openhands_eval_poetry_version = str(
                values["OPENHANDS_POETRY_VERSION_FROM_CONFIG"]
            )

    def run_foreground(self, uv_bin: Path) -> None:
        """Run the full pod-side eval flow in the current process."""

        pod_startup_common.require_pod_runtime(
            self.workspace_root, "nvidia-smi", "docker", "curl", "git"
        )
        pod_startup_common.prepare_pod_checkout(
            self.workspace_root,
            "OpenHands eval pod execution directory",
        )
        os.environ["UV_CACHE_DIR"] = str(self.uv_cache_dir)
        os.environ["UV_PYTHON_INSTALL_DIR"] = str(self.uv_python_install_dir)
        visible_gpu_output = run(
            ["nvidia-smi", "--list-gpus"], capture_output=True
        ).stdout
        self.visible_gpu_count = len(
            [line for line in visible_gpu_output.splitlines() if line.strip()]
        )
        if self.visible_gpu_count < self.required_gpu_count:
            die(
                f"expected at least {self.required_gpu_count} visible GPUs, found {self.visible_gpu_count}"
            )
        # Tensor parallelism splits individual model layers across GPUs.
        # Pipeline parallelism splits groups of layers across GPUs. Multiplying
        # them gives the number of GPUs one vLLM process needs to serve a single
        # model replica.
        self.vllm_parallel_gpu_count = int(self.vllm_tensor_parallel_size) * int(
            self.vllm_pipeline_parallel_size
        )
        if int(self.vllm_server_count) == 1:
            # SWE-Lego's published contract uses one multi-GPU vLLM server for a
            # larger long-context model, so the parallel group may span GPUs.
            if self.vllm_parallel_gpu_count > self.visible_gpu_count:
                die(
                    f"vLLM parallel size exceeds visible GPUs: {self.vllm_parallel_gpu_count} > {self.visible_gpu_count}"
                )
        else:
            # The current OpenHands/SWE-HERO path scales throughput by running
            # independent one-GPU replicas. That keeps each replica simple and
            # lets the router apply a fixed per-GPU request budget.
            if int(self.vllm_server_count) > self.visible_gpu_count:
                die(
                    f"vLLM server count exceeds visible GPUs: {self.vllm_server_count} > {self.visible_gpu_count}"
                )
            if (
                int(self.vllm_tensor_parallel_size) != 1
                or int(self.vllm_pipeline_parallel_size) != 1
            ):
                die(
                    "multi-replica pod eval uses one vLLM process per GPU; keep VLLM_TENSOR_PARALLEL_SIZE=1 and VLLM_PIPELINE_PARALLEL_SIZE=1 when VLLM_SERVER_COUNT>1"
                )
        if not env_bool(str(self.vllm_use_router)) and int(self.vllm_server_count) != 1:
            die("direct vLLM base URL without router requires VLLM_SERVER_COUNT=1")
        if (
            self.vllm_visible_devices
            and self.vllm_visible_devices != "all"
            and int(self.vllm_server_count) != 1
        ):
            die(
                "VLLM_VISIBLE_DEVICES/VLLM_GPU override is only supported with VLLM_SERVER_COUNT=1"
            )

        os.chdir(self.workspace_root)
        self.tmux_log_dir.mkdir(parents=True, exist_ok=True)
        Path("/workspace/runlogs").mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_docker()
        self.ensure_eval_python(uv_bin)
        pod_ip = os.environ.get("POD_IP", self.pod_ip())
        self.ensure_vllm_stack(uv_bin, pod_ip)

        total_agent_workers = int(self.vllm_server_count) * int(
            self.vllm_agent_tasks_per_server
        )
        # OpenHands worker count is tied to serving capacity. Too many workers
        # create long queues and can exhaust vLLM KV-cache memory; too few leave
        # expensive GPUs idle.
        if env_bool(str(self.vllm_use_router)):
            llm_base_url = f"http://{pod_ip}:{self.vllm_router_port}/v1"
        else:
            llm_base_url = f"http://{pod_ip}:{self.vllm_port}/v1"
        eval_num_workers = (
            openhands_eval_worker_selection.select_openhands_eval_num_workers(
                str(self.eval_stack),
                self.eval_limit,
                self.eval_ids,
                int(self.config_num_workers),
                total_agent_workers,
            )
        )
        eval_args = [
            f"@{self.config_preset_path}",
            "--base-url",
            llm_base_url,
            "--output-dir",
            str(self.output_dir),
            "--openhands-dir",
            str(self.openhands_dir),
            "--openhands-ref",
            str(self.openhands_ref),
            "--num-workers",
            str(eval_num_workers),
        ]
        if self.eval_limit:
            eval_args.extend(["--eval-limit", self.eval_limit])
        if self.eval_ids:
            eval_args.extend(["--eval-ids", self.eval_ids])
        if self.preflight_only:
            eval_args.append("--preflight-only")
        else:
            # Skip this in preflight mode because that mode only verifies that
            # vLLM can emit a structured tool call; it does not run OpenHands.
            self.ensure_openhands_dependencies(uv_bin)
        if self.skip_swebench_eval:
            eval_args.append("--skip-swebench-eval")

        env = dict(os.environ)
        env["PATH"] = f"{self.eval_venv / 'bin'}:{env.get('PATH', '')}"
        env["LLM_API_KEY"] = self.llm_api_key
        env["POETRY_VIRTUALENVS_PATH"] = "/workspace/venvs/poetry-pod"
        env["POETRY_CACHE_DIR"] = "/workspace/.cache/poetry-pod"
        exec_process(
            [
                str(self.eval_venv / "bin" / "python"),
                "scripts/openhands_swebench_eval.py",
                *eval_args,
            ],
            env=env,
        )


def main(argv: list[str] | None = None) -> int:
    launcher = EvalLauncher()
    launcher.parse(sys.argv[1:] if argv is None else argv)
    config_preset_path = launcher.resolve_config_preset_path()
    config_values = launcher.resolve_eval_config(config_preset_path)
    launcher.assign_config(config_values)
    launcher.llm_api_key = (
        openhands_eval_launcher_defaults.select_openhands_llm_api_key(
            str(launcher.eval_stack),
            launcher.llm_api_key_explicit,
            launcher.llm_api_key,
        )
    )
    launcher.launch_tmux_if_needed(config_preset_path)
    uv_bin = launcher.ensure_uv()
    launcher.run_foreground(uv_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
