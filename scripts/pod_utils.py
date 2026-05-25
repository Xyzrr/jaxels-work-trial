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
    return Path(__file__).resolve().parents[1]


def die(message: str, *, exit_code: int = 1) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def require_command(binary: str) -> None:
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
    return run(args, cwd=cwd, capture_output=True).stdout.strip()


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def shell_join(args: list[str] | tuple[str, ...]) -> str:
    return shlex.join([str(arg) for arg in args])


def env_flag(value: str | None, *, default: str = "auto") -> str:
    if value is None or value == "":
        return default
    return value


def is_truthy(value: str | None) -> bool:
    return value in TRUE_VALUES


def is_falsey(value: str | None) -> bool:
    return value in FALSE_VALUES


def write_shell_exports(path: Path, env: dict[str, str]) -> None:
    with path.open("w") as handle:
        for key, value in sorted(env.items()):
            if not key.replace("_", "A").isalnum() or key[0].isdigit():
                continue
            handle.write(f"export {key}={shell_quote(value)}\n")
    path.chmod(0o600)


def executable_script_path(path: Path) -> str:
    return str(path)


def exec_process(
    args: list[str] | tuple[str, ...], *, env: dict[str, str] | None = None
) -> NoReturn:
    sys.stdout.flush()
    sys.stderr.flush()
    if env is None:
        os.execvp(str(args[0]), [str(arg) for arg in args])
    os.execvpe(str(args[0]), [str(arg) for arg in args], env)
    raise AssertionError("exec returned unexpectedly")
