#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationProcess:
    name: str
    command: tuple[str, ...]


@dataclass
class ValidationResult:
    name: str
    command: tuple[str, ...]
    returncode: int
    output: str


def validation_processes() -> tuple[ValidationProcess, ...]:
    return (
        ValidationProcess("pytest", ("pytest",)),
        ValidationProcess("ruff check", ("ruff", "check", ".")),
        ValidationProcess("ruff format", ("ruff", "format", "--check", ".")),
    )


def run_process(process: ValidationProcess, results: list[ValidationResult]) -> None:
    completed = subprocess.run(
        process.command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    results.append(
        ValidationResult(
            name=process.name,
            command=process.command,
            returncode=completed.returncode,
            output=completed.stdout,
        )
    )


def print_grouped_result(result: ValidationResult) -> None:
    command = " ".join(result.command)
    print(f"===== {result.name}: {command} =====")
    if result.output:
        print(result.output, end="" if result.output.endswith("\n") else "\n")
    else:
        print("(no output)")
    print(f"===== {result.name}: exit {result.returncode} =====")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project validation with grouped parallel output.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print validation subprocesses without running them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    processes = validation_processes()
    if args.list:
        for process in processes:
            print(f"{process.name}: {' '.join(process.command)}")
        return 0

    results: list[ValidationResult] = []
    threads = [
        threading.Thread(target=run_process, args=(process, results), daemon=False)
        for process in processes
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    by_name = {result.name: result for result in results}
    failed = False
    for process in processes:
        result = by_name[process.name]
        print_grouped_result(result)
        failed = failed or result.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
