"""Shared helpers for pod-side launchers.

This module intentionally stays small and boring. Training, eval, and image
prebuild launchers all need the same subprocess, shell-quoting, env-flag, and
process-handoff behavior, and duplicating those details would make the pod
workflows drift in subtle ways.

The functions here do not encode experiment settings such as model, dataset,
context length, batch size, or optimizer choices. They are runtime plumbing used
to make those higher-level ML workflows launch consistently.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

TRUE_VALUES = {"1", "true", "TRUE", "yes", "YES", "on", "ON", "tmux", "TMUX"}
FALSE_VALUES = {
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
}


def repo_root() -> Path:
    """Return the repository root for scripts executed from arbitrary cwd."""

    return Path(__file__).resolve().parents[1]


def die(message: str, *, exit_code: int = 1) -> NoReturn:
    """Print a consistent launcher error and terminate."""

    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def require_command(binary: str) -> None:
    """Fail early when a pod/bootstrap dependency is missing."""

    if shutil.which(binary) is None:
        die(f"{binary} not found")


def run(
    args: list[str] | tuple[str, ...],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with script-friendly defaults.

    Callers pass structured argument lists, not shell strings. That keeps
    user-provided paths, branch names, and preset values from being reinterpreted
    by a shell unless a launcher explicitly chooses to build a shell command.
    """

    return subprocess.run(
        [str(arg) for arg in args],
        cwd=None if cwd is None else str(cwd),
        env=env,
        check=check,
        capture_output=capture_output,
        text=text,
    )


def command_output(
    args: list[str] | tuple[str, ...], *, cwd: str | Path | None = None
) -> str:
    """Return trimmed stdout for commands that are expected to print one value."""

    return run(args, cwd=cwd, capture_output=True).stdout.strip()


def shell_quote(value: str | Path) -> str:
    """Quote one value for the few places that must compose shell text."""

    return shlex.quote(str(value))


def shell_join(args: list[str] | tuple[str, ...]) -> str:
    """Render a structured argv as shell-safe text for tmux/bash command lines."""

    return shlex.join([str(arg) for arg in args])


def env_flag(value: str | None, *, default: str = "auto") -> str:
    """Normalize optional environment flags while preserving explicit values."""

    if value is None or value == "":
        return default
    return value


def is_truthy(value: str | None) -> bool:
    """Return whether a launcher env flag explicitly means enabled."""

    return value in TRUE_VALUES


def is_falsey(value: str | None) -> bool:
    """Return whether a launcher env flag explicitly means disabled."""

    return value in FALSE_VALUES


def write_shell_exports(path: Path, env: dict[str, str]) -> None:
    """Write shell exports for passing runtime plumbing through tmux.

    The generated file may contain secrets or pod credentials, so it is written
    owner-readable only. Invalid shell variable names are skipped instead of
    producing a file that fails when sourced.
    """

    with path.open("w") as handle:
        for key, value in sorted(env.items()):
            if not key.replace("_", "A").isalnum() or key[0].isdigit():
                continue
            handle.write(f"export {key}={shell_quote(value)}\n")
    path.chmod(0o600)


def executable_script_path(path: Path) -> str:
    """Return a script path in argv form.

    Kept as a named helper so callers read as "execute this script" even though
    the current implementation is simply `str(path)`.
    """

    return str(path)


def exec_process(
    args: list[str] | tuple[str, ...], *, env: dict[str, str] | None = None
) -> NoReturn:
    """Replace the current process with another command.

    Pod launchers use this for tmux attach and final training/eval handoff so
    signal handling and exit status belong to the real long-running process
    rather than to an extra Python wrapper.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    if env is None:
        os.execvp(str(args[0]), [str(arg) for arg in args])
    os.execvpe(str(args[0]), [str(arg) for arg in args], env)
    raise AssertionError("exec returned unexpectedly")
