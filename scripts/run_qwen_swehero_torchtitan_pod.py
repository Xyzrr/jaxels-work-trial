#!/usr/bin/env python3
"""Pod-local launcher for the Qwen SWE-Hero TorchTitan training entrypoint.

This wrapper is deliberately about runtime plumbing, not ML experiment design.
The actual model, dataset, context length, optimizer, checkpointing, and other
training choices live in preset/CLI arguments consumed by
``scripts/qwen_swehero_train.py``. Keeping this file thin prevents a future
non-SWE-Hero experiment from inheriting hidden Qwen or SWE-Hero assumptions just
because it also runs through the TorchTitan pod runtime.

The main ML-adjacent decision made here is runtime isolation: TorchTitan training
uses a pinned CUDA/PyTorch/TorchTitan environment that differs from the pod's
generic Python environment. This launcher creates or repairs that environment
before handing control to the training script, so distributed startup failures
are less likely to be caused by accidental package drift.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import pod_startup_common
from scripts.pod_utils import (
    die,
    exec_process,
    repo_root,
    shell_quote,
    write_shell_exports,
)

ROOT_DIR = repo_root()
SELF_PATH = ROOT_DIR / "scripts" / Path(__file__).name

# The venv path is runtime plumbing, not an experiment setting. The default name
# remains SWE-Hero-flavored for compatibility with existing pods, but callers
# should choose model/data behavior through preset arguments forwarded to the
# training entrypoint below.
VENV_PATH = Path(
    os.environ.get("TORCHTITAN_POD_VENV", "/workspace/venvs/torchtitan-swehero-cu128")
)

# Keep setup outside the training script so every launch, including dry runs and
# resumed runs, starts from the same pinned TorchTitan runtime contract.
SETUP_SCRIPT = Path(
    os.environ.get(
        "TORCHTITAN_POD_SETUP_SCRIPT",
        str(ROOT_DIR / "scripts" / "setup_torchtitan_pod_venv.py"),
    )
)

# Used only to derive a reconnectable tmux session name when the caller did not
# pass --out-dir. The canonical training preset still owns the real output path.
DEFAULT_OUT_DIR = "/workspace/qwen25-coder7b-swehero-torchtitan"


def python_for_arg_parsing() -> str:
    """Find a Python interpreter before the TorchTitan venv necessarily exists.

    The tmux supervisor needs to inspect ``@preset`` files and ``--out-dir``
    before it decides which session to create or reattach. On a fresh pod, the
    TorchTitan venv may not exist yet, so this helper falls back to the base
    Python installed by the pod manifest.
    """

    venv_python = VENV_PATH / "bin" / "python"
    if venv_python.exists() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    for candidate in ("python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    print(
        "python3 is required to parse launcher argument files before the pod venv exists.\n"
        "The canonical pod manifest installs python3 at container startup.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def expanded_args(args: list[str]) -> list[str]:
    """Expand argparse ``@file`` arguments just far enough for wrapper metadata.

    The training script also parses these presets later. This wrapper expands
    them only to find runtime metadata such as ``--out-dir`` for tmux naming; it
    deliberately does not interpret model, dataset, or optimizer flags here.
    """

    tokens: list[str] = []

    def expand_arg(arg: str) -> None:
        if not arg.startswith("@"):
            tokens.append(arg)
            return
        raw_path = Path(arg[1:])
        candidates = [raw_path]
        if not raw_path.is_absolute():
            candidates.append(ROOT_DIR / raw_path)
        for candidate in candidates:
            if candidate.is_file():
                for line in candidate.read_text().splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    for token in shlex.split(stripped):
                        expand_arg(token)
                return
        tokens.append(arg)

    for value in args:
        expand_arg(value)
    return tokens


def resolved_out_dir(args: list[str]) -> str:
    """Resolve the launch output directory without owning training semantics."""

    # Keep the explicit bootstrap Python check from the shell wrapper.
    python_for_arg_parsing()
    out_dir = DEFAULT_OUT_DIR
    tokens = expanded_args(args)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--out-dir" and index + 1 < len(tokens):
            out_dir = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--out-dir="):
            out_dir = token.split("=", 1)[1]
        index += 1
    return out_dir


def tmux_session_name(args: list[str]) -> str:
    """Derive a stable, shell-safe tmux session name for this launch."""

    if os.environ.get("SWEHERO_POD_TMUX_SESSION"):
        return os.environ["SWEHERO_POD_TMUX_SESSION"]
    base = Path(resolved_out_dir(args).rstrip("/") or "default").name
    if base in {"", ".", "/"}:
        base = "default"
    sanitized = "".join(ch if ch.isalnum() or ch in "_-" else "-" for ch in base)[:48]
    sanitized = sanitized.strip("-") or "default"
    return f"swehero-{sanitized}"


def should_use_tmux_supervisor() -> bool:
    """Decide whether this launch should run inside a reconnectable tmux session."""

    if os.environ.get("SWEHERO_POD_SUPERVISOR_CHILD") == "1":
        return False
    if os.environ.get("TMUX"):
        return False
    value = os.environ.get("SWEHERO_POD_SUPERVISOR", "auto")
    if value in {"1", "true", "TRUE", "yes", "YES", "on", "ON", "tmux", "TMUX"}:
        return True
    if value in {
        "0",
        "false",
        "FALSE",
        "no",
        "NO",
        "off",
        "OFF",
        "direct",
        "DIRECT",
        "none",
        "NONE",
    }:
        return False
    if value in {"auto", "AUTO", ""}:
        return sys.stdin.isatty() and sys.stdout.isatty()
    die(
        f"SWEHERO_POD_SUPERVISOR must be auto, tmux/1, or direct/0; got:\n  {value}",
        exit_code=2,
    )


def should_attach_tmux_client() -> bool:
    """Decide whether to attach after creating or finding a tmux session."""

    value = os.environ.get("SWEHERO_POD_TMUX_ATTACH", "auto")
    if value in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}:
        return True
    if value in {"0", "false", "FALSE", "no", "NO", "off", "OFF"}:
        return False
    if value in {"auto", "AUTO", ""}:
        return sys.stdin.isatty() and sys.stdout.isatty()
    die(f"SWEHERO_POD_TMUX_ATTACH must be auto, 1, or 0; got:\n  {value}", exit_code=2)


def attach_or_create_tmux_session(args: list[str]) -> None:
    """Attach to an existing supervised run or start a detached supervised run.

    Long distributed training jobs should survive a dropped ``kubectl exec`` or
    laptop network connection. tmux is the pod-local supervisor used here; it is
    intentionally outside the ML stack and does not change any training
    arguments.
    """

    if shutil.which("tmux") is None:
        print(
            "tmux is required for reconnectable supervised pod launches.\n\n"
            "The canonical pod manifest installs tmux at container startup. If this is an\n"
            "older running pod, recreate it from manifests/midtraining-hostpath.yaml or\n"
            "install tmux in the container before launching. To intentionally bypass this\n"
            "supervisor for non-interactive automation, set SWEHERO_POD_SUPERVISOR=0.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    session_name = tmux_session_name(args)
    log_dir = Path(os.environ.get("SWEHERO_POD_TMUX_LOG_DIR", "/workspace/runlogs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = log_dir / f"{session_name}.tmux.log"
    env_dir = Path(
        os.environ.get("SWEHERO_POD_TMUX_ENV_DIR", os.environ.get("TMPDIR", "/tmp"))
    )
    env_dir.mkdir(parents=True, exist_ok=True)

    if (
        subprocess.run(
            ["tmux", "has-session", "-t", session_name], check=False
        ).returncode
        == 0
    ):
        print(
            "Found existing supervised SWE-HERO session:\n"
            f"  {session_name}\n"
            "Transcript:\n"
            f"  {transcript_path}"
        )
        if should_attach_tmux_client():
            print("Attaching now.")
            exec_process(["tmux", "attach-session", "-t", session_name])
        print(
            "No interactive terminal is available, so the existing session was left running.\n"
            "Attach later with:\n"
            f"  tmux attach-session -t {session_name}"
        )
        raise SystemExit(0)

    # Starting a new training process is the point where the pod checkout must
    # match the branch and cleanliness rules. Reattaching above intentionally
    # skips this guard so reconnects do not mutate an already-running job.
    pod_startup_common.prepare_pod_checkout(
        ROOT_DIR, "TorchTitan pod execution directory"
    )
    fd, env_file_name = tempfile.mkstemp(prefix=f"{session_name}.env.", dir=env_dir)
    os.close(fd)
    env_file = Path(env_file_name)
    write_shell_exports(env_file, dict(os.environ))

    # tmux receives one shell command, so preserve the caller's environment in a
    # temporary export file and quote every forwarded argument. The child process
    # sets SWEHERO_POD_SUPERVISOR_CHILD=1 to avoid recursively spawning tmux.
    command = (
        f"source {shell_quote(env_file)}; "
        f"rm -f {shell_quote(env_file)}; "
        "tmux set-option -w remain-on-exit on >/dev/null 2>&1 || true; "
        "tmux set-option history-limit 200000 >/dev/null 2>&1 || true; "
        f"cd {shell_quote(ROOT_DIR)} && "
        "export SWEHERO_POD_SUPERVISOR_CHILD=1; "
        f"exec {shell_quote(SELF_PATH)}"
    )
    for arg in args:
        command += f" {shell_quote(arg)}"

    created = subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            f"exec bash -lc {shell_quote(command)}",
        ],
        check=False,
    )
    if created.returncode != 0:
        env_file.unlink(missing_ok=True)
        raise SystemExit(created.returncode)
    subprocess.run(
        [
            "tmux",
            "pipe-pane",
            "-o",
            "-t",
            f"{session_name}:0.0",
            f"cat >> {shell_quote(transcript_path)}",
        ],
        check=True,
    )

    print(
        "Started supervised SWE-HERO session:\n"
        f"  {session_name}\n\n"
        "If the pod connection drops, the job keeps running in tmux. Reconnect with the\n"
        "same launcher command, or directly with:\n"
        f"  tmux attach-session -t {session_name}\n\n"
        "Transcript:\n"
        f"  {transcript_path}"
    )
    if should_attach_tmux_client():
        exec_process(["tmux", "attach-session", "-t", session_name])
    print(
        "No interactive terminal is available, so the session was started detached.\n"
        "Attach later with:\n"
        f"  tmux attach-session -t {session_name}"
    )
    raise SystemExit(0)


def main(argv: list[str] | None = None) -> int:
    """Prepare the pod runtime and exec the Qwen SWE-Hero training script."""

    args = sys.argv[1:] if argv is None else argv
    if should_use_tmux_supervisor():
        attach_or_create_tmux_session(args)

    # Direct execution reaches this point either because supervision is disabled
    # or because this process is already the tmux child. The checkout guard runs
    # immediately before the actual training launch.
    pod_startup_common.prepare_pod_checkout(
        ROOT_DIR, "TorchTitan pod execution directory"
    )

    # Synchronize the pinned TorchTitan runtime before invoking any training
    # code. This matters for ML correctness because PyTorch, CUDA, TorchAO, and
    # TorchTitan internals are tightly coupled; a stale dependency can produce
    # failures or numerically different behavior before the model configuration
    # itself is even considered.
    setup = subprocess.run([str(SETUP_SCRIPT), "--venv", str(VENV_PATH)], check=False)
    if setup.returncode != 0:
        return setup.returncode
    env = dict(os.environ)
    env["PATH"] = f"{VENV_PATH / 'bin'}:{env.get('PATH', '')}"

    # From here onward, the lower-level training entrypoint owns the ML recipe.
    # This wrapper forwards args unchanged so preset ordering and CLI overrides
    # behave exactly as if qwen_swehero_train.py had been run directly inside the
    # canonical venv.
    exec_process(
        [
            str(VENV_PATH / "bin" / "python"),
            str(ROOT_DIR / "scripts" / "qwen_swehero_train.py"),
            *args,
        ],
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
