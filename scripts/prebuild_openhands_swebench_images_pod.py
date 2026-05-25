#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import openhands_swebench_eval as eval_script
from scripts import pod_startup_common
from scripts.pod_utils import die, exec_process, repo_root, shell_quote

ROOT_DIR = repo_root()


USAGE = """\
Usage: scripts/prebuild_openhands_swebench_images_pod.py [options]

Prebuild OpenHands runtime Docker images for SWE-bench tasks on the GPU pod.
For workstation launches, use: scripts/run_midtraining_pod.py prebuild [options]

Options:
  --config PATH           Argparse eval preset. Defaults to the 7B 128k preset.
  --eval-limit N          Prebuild N instances. Omit for the full split.
  --parallel-builds N     Concurrent Docker runtime image builds. Default: 4.
  --tmux-session NAME     Default: openhands-swebench-image-prebuild
  --replace-session       Kill and restart an existing tmux session.
  --foreground            Run in this shell. The normal entrypoint uses tmux.
  --attach                Attach to the tmux session after launch.
  --no-attach             Do not attach to the tmux session after launch.
  --help                  Show this help.

Environment:
  SWEHERO_POD_GIT_BRANCH  Required when launching a new tmux job. Set by
                          scripts/run_midtraining_pod.py from the selected
                          local branch.
  EVAL_VENV               Default: /workspace/venvs/openhands-eval-pod-py312
  OPENHANDS_EVAL_POETRY_VERSION
                          Default: 2.1.3, or the preset's
                          --openhands-poetry-version when set.
"""


PREBUILD_IMAGES_CODE = r"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
from datasets import load_dataset
from docker.errors import ImageNotFound
from evaluation.benchmarks.swe_bench.run_infer import (
    get_instance_docker_image,
    set_dataset_type,
)
from openhands.runtime.builder import DockerRuntimeBuilder
from openhands.runtime.utils.runtime_build import (
    build_runtime_image,
    get_hash_for_lock_files,
    get_hash_for_source_files,
    get_runtime_image_repo,
)
from openhands import __version__ as openhands_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--parallel-builds", type=int, default=1)
    args = parser.parse_args()
    if args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive when provided")
    if args.parallel_builds <= 0:
        raise ValueError("--parallel-builds must be positive")
    return args


def local_image_exists(client: docker.DockerClient, image_name: str) -> bool:
    try:
        client.images.get(image_name)
        return True
    except ImageNotFound:
        return False


def runtime_target_image(base_image: str, source_hash: str) -> str:
    lock_tag = f"oh_v{openhands_version}_{get_hash_for_lock_files(base_image, False)}"
    return f"{get_runtime_image_repo()}:{lock_tag}_{source_hash}"


def build_runtime_job(job: dict[str, str]) -> str:
    client = docker.from_env()
    try:
        if local_image_exists(client, job["target_image"]):
            return "skipped"

        builder = DockerRuntimeBuilder(client)
        image_name = build_runtime_image(
            job["base_image"],
            builder,
            platform="linux/amd64",
            enable_browser=False,
        )
        if image_name != job["target_image"]:
            raise RuntimeError(
                f"unexpected image tag: expected {job['target_image']}, got {image_name}"
            )
        if not local_image_exists(client, job["target_image"]):
            raise RuntimeError(
                f"image build finished but target is missing: {job['target_image']}"
            )
        return "built"
    finally:
        client.close()


def main() -> None:
    args = parse_args()
    set_dataset_type(args.dataset)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.eval_limit is not None:
        dataset = dataset.select(range(min(args.eval_limit, len(dataset))))

    source_hash = get_hash_for_source_files()
    jobs: list[dict[str, str]] = []
    seen_targets: set[str] = set()
    for row in dataset:
        instance_id = row["instance_id"]
        base_image = get_instance_docker_image(
            instance_id,
            swebench_official_image=True,
        )
        target_image = runtime_target_image(base_image, source_hash)
        if target_image in seen_targets:
            continue
        seen_targets.add(target_image)
        jobs.append(
            {
                "instance_id": instance_id,
                "base_image": base_image,
                "target_image": target_image,
            }
        )

    built = 0
    skipped = 0
    started_at = time.time()
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "instances": len(dataset),
                "unique_runtime_images": len(jobs),
                "parallel_builds": args.parallel_builds,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    probe_client = docker.from_env()
    build_jobs: list[dict[str, str]] = []
    try:
        for index, job in enumerate(jobs, start=1):
            target_image = job["target_image"]
            if local_image_exists(probe_client, target_image):
                skipped += 1
                print(
                    f"[{index}/{len(jobs)}] skip {target_image} ({job['instance_id']})",
                    flush=True,
                )
                continue

            queued_job = dict(job)
            queued_job["index"] = str(index)
            build_jobs.append(queued_job)
            print(
                f"[{index}/{len(jobs)}] queue build {target_image} from {job['base_image']} ({job['instance_id']})",
                flush=True,
            )
    finally:
        probe_client.close()

    if build_jobs:
        active_workers = min(args.parallel_builds, len(build_jobs))
        print(
            json.dumps(
                {
                    "scheduled_builds": len(build_jobs),
                    "active_workers": active_workers,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            future_to_job = {
                executor.submit(build_runtime_job, job): job for job in build_jobs
            }
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                target_image = job["target_image"]
                try:
                    status = future.result()
                except Exception as exc:
                    for pending in future_to_job:
                        pending.cancel()
                    raise RuntimeError(
                        f"failed to build {target_image} ({job['instance_id']})"
                    ) from exc

                if status == "skipped":
                    skipped += 1
                    print(
                        f"[{job['index']}/{len(jobs)}] skip {target_image} ({job['instance_id']})",
                        flush=True,
                    )
                else:
                    built += 1
                    print(
                        f"[{job['index']}/{len(jobs)}] built {target_image} ({job['instance_id']})",
                        flush=True,
                    )

    print(
        json.dumps(
            {
                "built": built,
                "skipped": skipped,
                "unique_runtime_images": len(jobs),
                "elapsed_seconds": round(time.time() - started_at, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
"""


def run(
    args: list[str], *, check: bool = True, **kwargs
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, check=check, **kwargs)


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_eval_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise SystemExit(f"eval config preset not found: {config_path}")
    args = eval_script.parse_args(
        [
            f"@{config_path}",
            "--dry-run",
            "--output-dir",
            "/tmp/openhands-image-prebuild-scaffold",
        ]
    )
    return {
        "CONFIG_PRESET_PATH": config_path,
        "EVAL_STACK": args.eval_stack,
        "DATASET": args.dataset,
        "SPLIT": args.split,
        "RUNTIME": args.runtime,
        "OPENHANDS_REPO": args.openhands_repo,
        "OPENHANDS_REF": args.openhands_ref,
        "OPENHANDS_DIR": eval_script.effective_openhands_dir(args),
        "OPENHANDS_POETRY_VERSION_FROM_CONFIG": args.openhands_poetry_version,
        "SWE_LEGO_REPO": args.swe_lego_repo,
        "SWE_LEGO_REF": args.swe_lego_ref,
        "SWE_LEGO_DIR": args.swe_lego_dir,
        "SWE_LEGO_SWEBENCH_DIR": eval_script.effective_swebench_dir(args) or "",
        "DOCKER_SMOKE_IMAGE": args.docker_smoke_image,
    }


def optional_int(raw: str) -> int | None:
    return None if raw == "" else int(raw)


def now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def comparable_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": context["schema_version"],
        "kind": context["kind"],
        "tmux_session": context["tmux_session"],
        "workspace_root": context["workspace_root"],
        "script_path": context["script_path"],
        "requested": context["requested"],
        "resolved_eval_config": context["resolved_eval_config"],
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(flatten(child, child_prefix))
    return flattened


class PrebuildLauncher:
    def __init__(self) -> None:
        self.config_preset = (
            ROOT_DIR
            / "configs/eval/openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args"
        )
        self.eval_limit = ""
        self.parallel_builds = 4
        self.tmux_session = "openhands-swebench-image-prebuild"
        self.tmux_log_dir = Path("/workspace/runlogs")
        self.foreground = False
        self.foreground_worker = False
        self.replace_session = False
        self.eval_venv = Path(
            os.environ.get("EVAL_VENV", "/workspace/venvs/openhands-eval-pod-py312")
        )
        self.openhands_eval_poetry_version = os.environ.get(
            "OPENHANDS_EVAL_POETRY_VERSION", "2.1.3"
        )
        self.attach = "1" if sys.stdout.isatty() else "0"
        self.foreground_worker_process: subprocess.Popen[str] | None = None

    def parse(self, argv: list[str]) -> None:
        index = 0
        while index < len(argv):
            arg = argv[index]
            if arg in {
                "--config",
                "--eval-limit",
                "--parallel-builds",
                "--tmux-session",
            }:
                if index + 1 >= len(argv):
                    die(f"{arg} requires a value")
                value = argv[index + 1]
                if arg == "--config":
                    self.config_preset = Path(value)
                elif arg == "--eval-limit":
                    if not value.isdigit() or int(value) <= 0:
                        die("--eval-limit must be positive")
                    self.eval_limit = value
                elif arg == "--parallel-builds":
                    if not value.isdigit() or int(value) <= 0:
                        die("--parallel-builds must be positive")
                    self.parallel_builds = int(value)
                elif arg == "--tmux-session":
                    self.tmux_session = value
                index += 2
            elif arg == "--replace-session":
                self.replace_session = True
                index += 1
            elif arg == "--foreground":
                self.foreground = True
                index += 1
            elif arg == "--foreground-worker":
                self.foreground = True
                self.foreground_worker = True
                index += 1
            elif arg == "--attach":
                self.attach = "1"
                index += 1
            elif arg == "--no-attach":
                self.attach = "0"
                index += 1
            elif arg == "--help":
                print(USAGE, end="")
                raise SystemExit(0)
            else:
                die(f"unknown option: {arg}")
        if self.foreground and self.replace_session:
            die("--replace-session only applies to tmux-supervised launches")

    def assign_config(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self, key.lower(), value)
        self.config_preset_path = Path(values["CONFIG_PRESET_PATH"])
        self.openhands_dir = Path(str(values["OPENHANDS_DIR"]))
        if values.get("OPENHANDS_POETRY_VERSION_FROM_CONFIG"):
            self.openhands_eval_poetry_version = str(
                values["OPENHANDS_POETRY_VERSION_FROM_CONFIG"]
            )
        if self.runtime != "docker":
            die("prebuild only applies to --runtime docker configs")
        self.tmux_log_path = self.tmux_log_dir / f"{self.tmux_session}.log"
        self.tmux_context_path = self.tmux_log_dir / f"{self.tmux_session}.context.json"

    def build_context(
        self, script_path: Path, foreground_command: str
    ) -> dict[str, Any]:
        git_branch = run(
            ["git", "-C", str(ROOT_DIR), "branch", "--show-current"],
            capture_output=True,
            check=False,
        ).stdout.strip()
        git_commit = run(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
        ).stdout.strip()
        return {
            "schema_version": 1,
            "kind": "openhands_swebench_image_prebuild",
            "tmux_session": self.tmux_session,
            "created_at_utc": now_utc(),
            "workspace_root": str(ROOT_DIR),
            "script_path": str(script_path),
            "foreground_command": foreground_command,
            "requested": {
                "config": str(self.config_preset_path),
                "eval_limit": optional_int(self.eval_limit),
                "parallel_builds": self.parallel_builds,
                "eval_venv": str(self.eval_venv),
                "openhands_eval_poetry_version": self.openhands_eval_poetry_version,
            },
            "resolved_eval_config": {
                "eval_stack": self.eval_stack,
                "dataset": self.dataset,
                "split": self.split,
                "runtime": self.runtime,
                "openhands_repo": self.openhands_repo,
                "openhands_ref": self.openhands_ref,
                "openhands_dir": str(self.openhands_dir),
                "swe_lego_repo": self.swe_lego_repo,
                "swe_lego_ref": self.swe_lego_ref,
                "swe_lego_dir": self.swe_lego_dir,
                "swe_lego_swebench_dir": self.swe_lego_swebench_dir,
                "docker_smoke_image": self.docker_smoke_image,
            },
            "git": {"branch": git_branch or None, "commit": git_commit or None},
        }

    def write_tmux_context(self, script_path: Path, foreground_command: str) -> None:
        context = self.build_context(script_path, foreground_command)
        self.tmux_context_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            prefix=f"{self.tmux_context_path.name}.",
            suffix=".tmp",
            dir=self.tmux_context_path.parent,
            delete=False,
        ) as handle:
            json.dump(context, handle, indent=2, sort_keys=True)
            handle.write("\n")
            tmp_name = handle.name
        Path(tmp_name).replace(self.tmux_context_path)

    def compare_tmux_context(self, script_path: Path, foreground_command: str) -> None:
        requested = comparable_context(
            self.build_context(script_path, foreground_command)
        )
        if not self.tmux_context_path.is_file():
            print(
                f"error: tmux session already exists but launch context is missing: {self.tmux_context_path}",
                file=sys.stderr,
            )
            print(
                "Use --replace-session to restart it with the requested context, "
                "or --tmux-session NAME to launch a separate prebuild.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            existing = comparable_context(
                json.loads(self.tmux_context_path.read_text())
            )
        except Exception as exc:
            print(
                f"error: tmux session launch context is unreadable or unsupported: {self.tmux_context_path}",
                file=sys.stderr,
            )
            print(f"reason: {exc}", file=sys.stderr)
            print(
                "Use --replace-session to restart it with a fresh context, "
                "or --tmux-session NAME to launch a separate prebuild.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        if existing == requested:
            return
        existing_flat = flatten(existing)
        requested_flat = flatten(requested)
        print(
            f"error: tmux session already exists with different launch context: {self.tmux_session}",
            file=sys.stderr,
        )
        print(f"context: {self.tmux_context_path}\n", file=sys.stderr)
        for path in sorted(set(existing_flat) | set(requested_flat)):
            existing_value = existing_flat.get(path, "<missing>")
            requested_value = requested_flat.get(path, "<missing>")
            if existing_value == requested_value:
                continue
            print(f"{path}:", file=sys.stderr)
            print(
                f"  existing:  {json.dumps(existing_value, sort_keys=True)}",
                file=sys.stderr,
            )
            print(
                f"  requested: {json.dumps(requested_value, sort_keys=True)}",
                file=sys.stderr,
            )
        print(
            "\nUse --replace-session to restart it with the requested context, "
            "or --tmux-session NAME to launch a separate prebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    def foreground_prebuild_pids(self) -> list[int]:
        result = run(["ps", "-eo", "pid=,args="], capture_output=True, check=False)
        pids: list[int] = []
        current_pid = os.getpid()
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            raw_pid, _, args = stripped.partition(" ")
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid == current_pid:
                continue
            tokens = args.split()
            if "prebuild_openhands_swebench_images_pod.py" not in args:
                continue
            if "--foreground" not in tokens and "--foreground-worker" not in tokens:
                continue
            try:
                session_index = tokens.index("--tmux-session")
            except ValueError:
                continue
            if (
                session_index + 1 < len(tokens)
                and tokens[session_index + 1] == self.tmux_session
            ):
                pids.append(pid)
        return pids

    def ensure_no_foreground_prebuild(self) -> None:
        pids = self.foreground_prebuild_pids()
        if not pids:
            return
        print(
            f"error: foreground prebuild process already exists for tmux session: {self.tmux_session}",
            file=sys.stderr,
        )
        for pid in pids:
            print(f"  pid: {pid}", file=sys.stderr)
        print(
            "Use --replace-session to terminate it before launching a new prebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    def kill_foreground_prebuilds(self) -> None:
        pids = self.foreground_prebuild_pids()
        if not pids:
            return
        pgid_output = run(
            ["ps", "-o", "pgid=", "-p", ",".join(str(pid) for pid in pids)],
            capture_output=True,
            check=False,
        ).stdout
        pgids = sorted(
            {int(raw.strip()) for raw in pgid_output.splitlines() if raw.strip()}
        )
        if not pgids:
            return
        print(
            f"terminating foreground prebuild process groups for {self.tmux_session}: {' '.join(map(str, pgids))}"
        )
        for pgid in pgids:
            run(["kill", "-TERM", f"-{pgid}"], check=False)
        time.sleep(5)
        if not self.foreground_prebuild_pids():
            return
        for pgid in pgids:
            run(["kill", "-KILL", f"-{pgid}"], check=False)

    def launch_tmux_if_needed(self) -> None:
        if self.foreground:
            return
        pod_startup_common.require_pod_runtime(ROOT_DIR, "tmux")
        self.tmux_log_dir.mkdir(parents=True, exist_ok=True)
        script_path = Path(__file__).resolve()
        command = (
            f"cd {shell_quote(ROOT_DIR)} && exec env "
            f"SWEHERO_POD_GIT_BRANCH={shell_quote(os.environ.get('SWEHERO_POD_GIT_BRANCH', ''))} "
            f"EVAL_VENV={shell_quote(self.eval_venv)} "
            f"OPENHANDS_EVAL_POETRY_VERSION={shell_quote(self.openhands_eval_poetry_version)} "
            f"{shell_quote(script_path)} --foreground --config {shell_quote(self.config_preset_path)} "
            f"--tmux-session {shell_quote(self.tmux_session)} --parallel-builds {shell_quote(str(self.parallel_builds))}"
        )
        if self.eval_limit:
            command += f" --eval-limit {shell_quote(self.eval_limit)}"
        launch_session = True
        if (
            run(
                ["tmux", "has-session", "-t", self.tmux_session], check=False
            ).returncode
            == 0
        ):
            if self.replace_session:
                print(f"replacing tmux session: {self.tmux_session}")
            else:
                self.compare_tmux_context(script_path, command)
                print(f"tmux session already exists: {self.tmux_session}")
                launch_session = False
        if launch_session:
            pod_startup_common.prepare_pod_checkout(
                ROOT_DIR,
                "OpenHands image prebuild pod execution directory",
            )
            if self.replace_session:
                run(["tmux", "kill-session", "-t", self.tmux_session], check=False)
                self.kill_foreground_prebuilds()
            else:
                self.ensure_no_foreground_prebuild()
            self.write_tmux_context(script_path, command)
            tmux_command = f"set -euo pipefail; exec > >(tee -a {shell_quote(self.tmux_log_path)}) 2>&1; {command}"
            if (
                run(
                    [
                        "tmux",
                        "new-session",
                        "-d",
                        "-s",
                        self.tmux_session,
                        tmux_command,
                    ],
                    check=False,
                ).returncode
                != 0
            ):
                self.tmux_context_path.unlink(missing_ok=True)
                die(f"failed to launch tmux session: {self.tmux_session}")
            run(
                [
                    "tmux",
                    "set-option",
                    "-t",
                    self.tmux_session,
                    "@swehero_launch_context",
                    str(self.tmux_context_path),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"launched tmux session: {self.tmux_session}")
        print(f"context: {self.tmux_context_path}")
        print(f"log: {self.tmux_log_path}")
        if self.attach == "1":
            exec_process(["tmux", "attach-session", "-t", self.tmux_session])
        raise SystemExit(0)

    def terminate_foreground_worker(self, *_: object) -> None:
        process = self.foreground_worker_process
        if process is None or process.poll() is not None:
            return
        print(
            f"foreground prebuild received termination; terminating child process group: {process.pid}",
            file=sys.stderr,
        )
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(
                "foreground prebuild child process group did not exit after TERM; "
                f"sending KILL: {process.pid}",
                file=sys.stderr,
            )
            os.killpg(process.pid, signal.SIGKILL)

    def run_supervised_foreground_worker(self) -> None:
        if shutil.which("setsid") is None:
            die(
                "setsid not found; recreate the pod with manifests/midtraining-hostpath.yaml"
            )
        script_path = Path(__file__).resolve()
        worker_command = [
            "env",
            f"SWEHERO_POD_GIT_BRANCH={os.environ.get('SWEHERO_POD_GIT_BRANCH', '')}",
            f"EVAL_VENV={self.eval_venv}",
            f"OPENHANDS_EVAL_POETRY_VERSION={self.openhands_eval_poetry_version}",
            str(script_path),
            "--foreground-worker",
            "--config",
            str(self.config_preset_path),
            "--tmux-session",
            self.tmux_session,
            "--parallel-builds",
            str(self.parallel_builds),
        ]
        if self.eval_limit:
            worker_command.extend(["--eval-limit", self.eval_limit])
        self.foreground_worker_process = subprocess.Popen(
            ["setsid", *worker_command], text=True
        )
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.terminate_foreground_worker)
        try:
            raise SystemExit(self.foreground_worker_process.wait())
        finally:
            self.terminate_foreground_worker()

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
            run(["tmux", "kill-session", "-t", "openhands-dockerd"], check=False)
            run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    "openhands-dockerd",
                    "dockerd --host=unix:///var/run/docker.sock > /workspace/runlogs/openhands-dockerd.log 2>&1",
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
                "Docker daemon did not become ready; see /workspace/runlogs/openhands-dockerd.log"
            )
        run(
            ["docker", "run", "--rm", self.docker_smoke_image],
            stdout=subprocess.DEVNULL,
        )
        run(["docker", "buildx", "version"], stdout=subprocess.DEVNULL)

    def ensure_openhands_checkout(self) -> None:
        if self.eval_stack == "swe-lego":
            swe_lego_dir = Path(self.swe_lego_dir)
            if not (swe_lego_dir / ".git").is_dir():
                swe_lego_dir.parent.mkdir(parents=True, exist_ok=True)
                run(["git", "clone", self.swe_lego_repo, str(swe_lego_dir)])
            if run(
                ["git", "-C", str(swe_lego_dir), "status", "--porcelain"],
                capture_output=True,
            ).stdout:
                die(
                    f"{swe_lego_dir} has local changes; clean it before prebuilding images"
                )
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
                f"{self.openhands_dir} has local changes; clean it before prebuilding images"
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

    def venv_python_matches(self) -> bool:
        python = self.eval_venv / "bin" / "python"
        if not python.exists():
            return False
        code = (
            "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        )
        return run([str(python), "-c", code], check=False).returncode == 0

    def ensure_eval_python(self) -> None:
        uv_bin = Path("/workspace/uv/uv-0.11.16/uv")
        if not uv_bin.exists():
            die("uv 0.11.16 not found at /workspace/uv/uv-0.11.16/uv")
        env = dict(os.environ)
        env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/workspace/.cache/uv")
        env["UV_PYTHON_INSTALL_DIR"] = env.get(
            "UV_PYTHON_INSTALL_DIR", "/workspace/python"
        )
        if not self.venv_python_matches():
            shutil.rmtree(self.eval_venv, ignore_errors=True)
            self.eval_venv.parent.mkdir(parents=True, exist_ok=True)
            Path(env["UV_PYTHON_INSTALL_DIR"]).mkdir(parents=True, exist_ok=True)
            run(
                [
                    str(uv_bin),
                    "python",
                    "install",
                    "3.12",
                    "--install-dir",
                    env["UV_PYTHON_INSTALL_DIR"],
                    "--no-bin",
                ],
                env={**env, "UV_PYTHON_DOWNLOADS": "automatic"},
            )
            run(
                [
                    str(uv_bin),
                    "venv",
                    "--no-project",
                    "--python",
                    "3.12",
                    "--seed",
                    str(self.eval_venv),
                ],
                env=env,
            )
        run(
            [
                str(uv_bin),
                "pip",
                "install",
                "--python",
                str(self.eval_venv / "bin" / "python"),
                f"poetry=={self.openhands_eval_poetry_version}",
            ],
            env=env,
        )

    def ensure_openhands_dependencies(self) -> None:
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

    def prebuild_images(self) -> None:
        args = [
            "--dataset",
            self.dataset,
            "--split",
            self.split,
            "--parallel-builds",
            str(self.parallel_builds),
        ]
        if self.eval_limit:
            args.extend(["--eval-limit", self.eval_limit])
        poetry_env = dict(os.environ)
        poetry_env["PATH"] = f"{self.eval_venv / 'bin'}:{poetry_env.get('PATH', '')}"
        poetry_env["POETRY_VIRTUALENVS_PATH"] = "/workspace/venvs/poetry-pod"
        poetry_env["POETRY_CACHE_DIR"] = "/workspace/.cache/poetry-pod"
        run(
            ["poetry", "-C", str(self.openhands_dir), "run", "python", "-", *args],
            input=PREBUILD_IMAGES_CODE,
            env=poetry_env,
        )

    def run_foreground(self) -> None:
        pod_startup_common.require_pod_runtime(
            ROOT_DIR, "docker", "git", "python3", "tmux"
        )
        if not self.foreground_worker:
            self.run_supervised_foreground_worker()
        self.tmux_log_dir.mkdir(parents=True, exist_ok=True)
        Path("/workspace/runlogs").mkdir(parents=True, exist_ok=True)
        pod_startup_common.prepare_pod_checkout(
            ROOT_DIR,
            "OpenHands image prebuild pod execution directory",
        )
        self.ensure_docker()
        self.ensure_eval_python()
        self.ensure_openhands_checkout()
        self.ensure_openhands_dependencies()
        self.prebuild_images()


def main(argv: list[str] | None = None) -> int:
    launcher = PrebuildLauncher()
    launcher.parse(sys.argv[1:] if argv is None else argv)
    config_path = resolve_path(launcher.config_preset)
    launcher.assign_config(resolve_eval_config(config_path))
    launcher.launch_tmux_if_needed()
    launcher.run_foreground()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
