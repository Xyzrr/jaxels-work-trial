#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/pod_git_guard.sh
source "$ROOT_DIR/scripts/pod_git_guard.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/prebuild_openhands_swebench_images_pod.sh [options]

Prebuild OpenHands runtime Docker images for SWE-bench tasks on the GPU pod.

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
  SWEHERO_POD_GIT_BRANCH  Required when launching a new tmux job.
  EVAL_VENV               Default: /workspace/venvs/openhands-eval-pod-py312
  OPENHANDS_EVAL_POETRY_VERSION
                          Default: 2.1.3, or the preset's
                          --openhands-poetry-version when set.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

quote_args() {
  (($#)) || return 0
  printf "%q " "$@"
}

ensure_pod_git_checkout() {
  command -v git >/dev/null 2>&1 || die "git not found; recreate the pod with manifests/midtraining-hostpath.yaml"
  swehero_require_pod_git_checkout \
    "$ROOT_DIR" \
    "${SWEHERO_POD_GIT_BRANCH:-}" \
    "OpenHands image prebuild pod execution directory"
}

resolve_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf "%s\n" "$raw"
  else
    printf "%s\n" "$ROOT_DIR/$raw"
  fi
}

resolve_eval_config() {
  local config_path="$1"
  local bootstrap_python="python3"
  command -v "$bootstrap_python" >/dev/null 2>&1 || \
    die "python3 is required to read eval presets"
  "$bootstrap_python" - "$ROOT_DIR" "$config_path" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
if not config_path.is_file():
    raise SystemExit(f"eval config preset not found: {config_path}")

sys.path.insert(0, str(root))
from scripts import openhands_swebench_eval as eval_script

args = eval_script.parse_args(
    [
        f"@{config_path}",
        "--dry-run",
        "--output-dir",
        "/tmp/openhands-image-prebuild-scaffold",
    ]
)
values = {
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
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

tmux_launch_context() {
  local mode="$1"
  local context_path="$2"
  local script_path="$3"
  local foreground_command="$4"
  local git_branch
  local git_commit
  git_branch="$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || true)"
  git_commit="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"

  python3 - \
    "$mode" \
    "$context_path" \
    "$ROOT_DIR" \
    "$script_path" \
    "$TMUX_SESSION" \
    "$CONFIG_PRESET_PATH" \
    "${EVAL_LIMIT:-}" \
    "$PARALLEL_BUILDS" \
    "$EVAL_VENV" \
    "$DATASET" \
    "$SPLIT" \
    "$RUNTIME" \
    "$EVAL_STACK" \
    "$OPENHANDS_REPO" \
    "$OPENHANDS_REF" \
    "$OPENHANDS_DIR" \
    "$OPENHANDS_EVAL_POETRY_VERSION" \
    "$SWE_LEGO_REPO" \
    "$SWE_LEGO_REF" \
    "$SWE_LEGO_DIR" \
    "$SWE_LEGO_SWEBENCH_DIR" \
    "$DOCKER_SMOKE_IMAGE" \
    "$git_branch" \
    "$git_commit" \
    "$foreground_command" <<'PY'
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


(
    mode,
    context_path_raw,
    workspace_root,
    script_path,
    tmux_session,
    config_path,
    eval_limit,
    parallel_builds,
    eval_venv,
    dataset,
    split,
    runtime,
    eval_stack,
    openhands_repo,
    openhands_ref,
    openhands_dir,
    openhands_eval_poetry_version,
    swe_lego_repo,
    swe_lego_ref,
    swe_lego_dir,
    swe_lego_swebench_dir,
    docker_smoke_image,
    git_branch,
    git_commit,
    foreground_command,
) = sys.argv[1:]

context_path = Path(context_path_raw)


def optional_int(raw: str) -> int | None:
    return None if raw == "" else int(raw)


def now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_context() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "openhands_swebench_image_prebuild",
        "tmux_session": tmux_session,
        "created_at_utc": now_utc(),
        "workspace_root": workspace_root,
        "script_path": script_path,
        "foreground_command": foreground_command,
        "requested": {
            "config": config_path,
            "eval_limit": optional_int(eval_limit),
            "parallel_builds": int(parallel_builds),
            "eval_venv": eval_venv,
            "openhands_eval_poetry_version": openhands_eval_poetry_version,
        },
        "resolved_eval_config": {
            "eval_stack": eval_stack,
            "dataset": dataset,
            "split": split,
            "runtime": runtime,
            "openhands_repo": openhands_repo,
            "openhands_ref": openhands_ref,
            "openhands_dir": openhands_dir,
            "swe_lego_repo": swe_lego_repo,
            "swe_lego_ref": swe_lego_ref,
            "swe_lego_dir": swe_lego_dir,
            "swe_lego_swebench_dir": swe_lego_swebench_dir,
            "docker_smoke_image": docker_smoke_image,
        },
        "git": {
            "branch": git_branch or None,
            "commit": git_commit or None,
        },
    }


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


def format_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def write_context(context: dict[str, Any]) -> None:
    context_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{context_path.name}.",
        suffix=".tmp",
        dir=str(context_path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(context, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, context_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


requested_context = build_context()
if mode == "write":
    write_context(requested_context)
elif mode == "compare":
    if not context_path.is_file():
        print(
            f"error: tmux session already exists but launch context is missing: {context_path}",
            file=sys.stderr,
        )
        print(
            "Use --replace-session to restart it with the requested context, "
            "or --tmux-session NAME to launch a separate prebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        existing_context = json.loads(context_path.read_text())
        existing = comparable_context(existing_context)
    except Exception as exc:
        print(
            f"error: tmux session launch context is unreadable or unsupported: {context_path}",
            file=sys.stderr,
        )
        print(f"reason: {exc}", file=sys.stderr)
        print(
            "Use --replace-session to restart it with a fresh context, "
            "or --tmux-session NAME to launch a separate prebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    requested = comparable_context(requested_context)
    if existing != requested:
        existing_flat = flatten(existing)
        requested_flat = flatten(requested)
        paths = sorted(set(existing_flat) | set(requested_flat))
        print(
            f"error: tmux session already exists with different launch context: {tmux_session}",
            file=sys.stderr,
        )
        print(f"context: {context_path}", file=sys.stderr)
        print("", file=sys.stderr)
        for path in paths:
            existing_value = existing_flat.get(path, "<missing>")
            requested_value = requested_flat.get(path, "<missing>")
            if existing_value == requested_value:
                continue
            print(f"{path}:", file=sys.stderr)
            print(f"  existing:  {format_value(existing_value)}", file=sys.stderr)
            print(f"  requested: {format_value(requested_value)}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Use --replace-session to restart it with the requested context, "
            "or --tmux-session NAME to launch a separate prebuild.",
            file=sys.stderr,
        )
        raise SystemExit(1)
else:
    raise SystemExit(f"unknown context mode: {mode}")
PY
}

write_tmux_context() {
  tmux_launch_context write "$@"
}

compare_tmux_context() {
  tmux_launch_context compare "$@"
}

foreground_prebuild_pids() {
  python3 - "$TMUX_SESSION" <<'PY'
from __future__ import annotations

import os
import subprocess
import sys

session = sys.argv[1]
current_pid = os.getpid()
try:
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
except subprocess.CalledProcessError:
    raise SystemExit(0)

for line in output.splitlines():
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
    if "prebuild_openhands_swebench_images_pod.sh" not in args:
        continue
    if "--foreground" not in tokens:
        continue
    try:
        session_index = tokens.index("--tmux-session")
    except ValueError:
        continue
    if session_index + 1 >= len(tokens) or tokens[session_index + 1] != session:
        continue
    print(pid)
PY
}

ensure_no_foreground_prebuild() {
  local pids
  pids="$(foreground_prebuild_pids)"
  if [[ -n "$pids" ]]; then
    echo "error: foreground prebuild process already exists for tmux session: $TMUX_SESSION" >&2
    printf "%s\n" "$pids" | sed "s/^/  pid: /" >&2
    echo "Use --replace-session to terminate it before launching a new prebuild." >&2
    return 1
  fi
}

kill_foreground_prebuilds() {
  local pids
  pids="$(foreground_prebuild_pids)"
  [[ -n "$pids" ]] || return 0

  local pgids
  pgids="$(ps -o pgid= -p $pids | tr -d " " | sort -u)"
  [[ -n "$pgids" ]] || return 0

  echo "terminating foreground prebuild process groups for $TMUX_SESSION: $pgids"
  local pgid
  for pgid in $pgids; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  sleep 5
  pids="$(foreground_prebuild_pids)"
  [[ -n "$pids" ]] || return 0
  pgids="$(ps -o pgid= -p $pids | tr -d " " | sort -u)"
  for pgid in $pgids; do
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
}

ensure_docker() {
  if ! docker info >/dev/null 2>&1; then
    tmux kill-session -t openhands-dockerd 2>/dev/null || true
    tmux new-session -d -s openhands-dockerd \
      "dockerd --host=unix:///var/run/docker.sock > /workspace/runlogs/openhands-dockerd.log 2>&1"
  fi

  for _ in $(seq 1 90); do
    docker info >/dev/null 2>&1 && break
    sleep 1
  done
  docker info >/dev/null 2>&1 || die "Docker daemon did not become ready; see /workspace/runlogs/openhands-dockerd.log"
  docker run --rm "$DOCKER_SMOKE_IMAGE" >/dev/null
  docker buildx version >/dev/null
}

ensure_openhands_checkout() {
  if [[ "$EVAL_STACK" == "swe-lego" ]]; then
    if [[ ! -d "$SWE_LEGO_DIR/.git" ]]; then
      mkdir -p "$(dirname "$SWE_LEGO_DIR")"
      git clone "$SWE_LEGO_REPO" "$SWE_LEGO_DIR"
    fi
    if [[ -n "$(git -C "$SWE_LEGO_DIR" status --porcelain)" ]]; then
      die "$SWE_LEGO_DIR has local changes; clean it before prebuilding images"
    fi
    git -C "$SWE_LEGO_DIR" fetch --depth 1 origin "$SWE_LEGO_REF"
    git -C "$SWE_LEGO_DIR" checkout --detach "$SWE_LEGO_REF"
    [[ -d "$OPENHANDS_DIR" ]] || die "SWE-Lego OpenHands directory missing: $OPENHANDS_DIR"
    [[ -d "$SWE_LEGO_SWEBENCH_DIR" ]] || die "SWE-Lego SWE-bench directory missing: $SWE_LEGO_SWEBENCH_DIR"
    return
  fi

  if [[ ! -d "$OPENHANDS_DIR/.git" ]]; then
    mkdir -p "$(dirname "$OPENHANDS_DIR")"
    git clone --branch "$OPENHANDS_REF" --depth 1 "$OPENHANDS_REPO" "$OPENHANDS_DIR"
  fi
  if [[ -n "$(git -C "$OPENHANDS_DIR" status --porcelain)" ]]; then
    die "$OPENHANDS_DIR has local changes; clean it before prebuilding images"
  fi
  git -C "$OPENHANDS_DIR" fetch --tags --depth 1 origin "$OPENHANDS_REF"
  git -C "$OPENHANDS_DIR" checkout --detach "$OPENHANDS_REF"
}

venv_python_matches() {
  local venv_path="$1"
  [[ -x "$venv_path/bin/python" ]] || return 1
  "$venv_path/bin/python" - <<'PY'
from __future__ import annotations

import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
}

ensure_eval_python() {
  local uv_bin="/workspace/uv/uv-0.11.16/uv"
  [[ -x "$uv_bin" ]] || die "uv 0.11.16 not found at $uv_bin"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/python}"

  if ! venv_python_matches "$EVAL_VENV"; then
    rm -rf "$EVAL_VENV"
    mkdir -p "$(dirname "$EVAL_VENV")" "$UV_PYTHON_INSTALL_DIR"
    UV_PYTHON_DOWNLOADS=automatic "$uv_bin" python install 3.12 \
      --install-dir "$UV_PYTHON_INSTALL_DIR" \
      --no-bin
    UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
      "$uv_bin" venv --no-project --python 3.12 --seed "$EVAL_VENV"
  fi
  "$uv_bin" pip install --python "$EVAL_VENV/bin/python" "poetry==${OPENHANDS_EVAL_POETRY_VERSION}"
}

ensure_openhands_dependencies() {
  local poetry_env=(
    PATH="$EVAL_VENV/bin:$PATH"
    POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-pod
    POETRY_CACHE_DIR=/workspace/.cache/poetry-pod
  )
  env "${poetry_env[@]}" poetry -C "$OPENHANDS_DIR" env use "$EVAL_VENV/bin/python"
  if env "${poetry_env[@]}" poetry -C "$OPENHANDS_DIR" sync --help >/dev/null 2>&1; then
    env "${poetry_env[@]}" poetry -C "$OPENHANDS_DIR" sync --with evaluation,test --no-root
  else
    env "${poetry_env[@]}" poetry -C "$OPENHANDS_DIR" install --sync --with evaluation,test --no-root
  fi
}

prebuild_images() {
  local args=(
    --dataset "$DATASET"
    --split "$SPLIT"
    --parallel-builds "$PARALLEL_BUILDS"
  )
  if [[ -n "$EVAL_LIMIT" ]]; then
    args+=(--eval-limit "$EVAL_LIMIT")
  fi

  PATH="$EVAL_VENV/bin:$PATH" \
    POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-pod \
    POETRY_CACHE_DIR=/workspace/.cache/poetry-pod \
    poetry -C "$OPENHANDS_DIR" run python - "${args[@]}" <<'PY'
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
PY
}

CONFIG_PRESET="$ROOT_DIR/configs/eval/openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args"
EVAL_LIMIT=""
PARALLEL_BUILDS=4
TMUX_SESSION="openhands-swebench-image-prebuild"
TMUX_LOG_DIR="/workspace/runlogs"
FOREGROUND=0
REPLACE_SESSION=0
EVAL_VENV="${EVAL_VENV:-/workspace/venvs/openhands-eval-pod-py312}"
OPENHANDS_EVAL_POETRY_VERSION="${OPENHANDS_EVAL_POETRY_VERSION:-2.1.3}"
if [[ -t 1 ]]; then
  ATTACH=1
else
  ATTACH=0
fi

while (($#)); do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || die "--config requires a value"
      CONFIG_PRESET="$2"
      shift 2
      ;;
    --eval-limit)
      [[ $# -ge 2 ]] || die "--eval-limit requires a value"
      EVAL_LIMIT="$2"
      [[ "$EVAL_LIMIT" =~ ^[0-9]+$ && "$EVAL_LIMIT" -gt 0 ]] || die "--eval-limit must be positive"
      shift 2
      ;;
    --parallel-builds)
      [[ $# -ge 2 ]] || die "--parallel-builds requires a value"
      PARALLEL_BUILDS="$2"
      [[ "$PARALLEL_BUILDS" =~ ^[0-9]+$ && "$PARALLEL_BUILDS" -gt 0 ]] || die "--parallel-builds must be positive"
      shift 2
      ;;
    --tmux-session)
      [[ $# -ge 2 ]] || die "--tmux-session requires a value"
      TMUX_SESSION="$2"
      shift 2
      ;;
    --replace-session)
      REPLACE_SESSION=1
      shift
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --attach)
      ATTACH=1
      shift
      ;;
    --no-attach)
      ATTACH=0
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if [[ "$FOREGROUND" == "1" && "$REPLACE_SESSION" == "1" ]]; then
  die "--replace-session only applies to tmux-supervised launches"
fi

CONFIG_PRESET_PATH="$(resolve_path "$CONFIG_PRESET")"
eval "$(resolve_eval_config "$CONFIG_PRESET_PATH")"
if [[ -n "${OPENHANDS_POETRY_VERSION_FROM_CONFIG:-}" ]]; then
  OPENHANDS_EVAL_POETRY_VERSION="$OPENHANDS_POETRY_VERSION_FROM_CONFIG"
fi
[[ "$RUNTIME" == "docker" ]] || die "prebuild only applies to --runtime docker configs"
TMUX_LOG_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.log"
TMUX_CONTEXT_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.context.json"

if [[ "$FOREGROUND" != "1" ]]; then
  [[ "$(uname -s)" != "Darwin" ]] || die "this launcher is pod-only; run it from the Kubernetes GPU pod"
  [[ -d /workspace ]] || die "expected /workspace hostPath; run from the GPU pod"
  command -v tmux >/dev/null 2>&1 || die "tmux is required for pod prebuilds"
  mkdir -p "$TMUX_LOG_DIR"
  script_path="$(realpath "$0")"
  command="cd $(quote_args "$ROOT_DIR") && SWEHERO_POD_GIT_BRANCH=$(quote_args "${SWEHERO_POD_GIT_BRANCH:-}") EVAL_VENV=$(quote_args "$EVAL_VENV") OPENHANDS_EVAL_POETRY_VERSION=$(quote_args "$OPENHANDS_EVAL_POETRY_VERSION") $(quote_args "$script_path") --foreground --config $(quote_args "$CONFIG_PRESET_PATH") --tmux-session $(quote_args "$TMUX_SESSION") --parallel-builds $(quote_args "$PARALLEL_BUILDS")"
  if [[ -n "$EVAL_LIMIT" ]]; then
    command+=" --eval-limit $(quote_args "$EVAL_LIMIT")"
  fi
  LAUNCH_SESSION=1
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    if [[ "$REPLACE_SESSION" == "1" ]]; then
      echo "replacing tmux session: $TMUX_SESSION"
    else
      compare_tmux_context "$TMUX_CONTEXT_PATH" "$script_path" "$command"
      echo "tmux session already exists: $TMUX_SESSION"
      LAUNCH_SESSION=0
    fi
  fi
  if [[ "$LAUNCH_SESSION" == "1" ]]; then
    ensure_pod_git_checkout
    eval "$(resolve_eval_config "$CONFIG_PRESET_PATH")"
    if [[ -n "${OPENHANDS_POETRY_VERSION_FROM_CONFIG:-}" ]]; then
      OPENHANDS_EVAL_POETRY_VERSION="$OPENHANDS_POETRY_VERSION_FROM_CONFIG"
    fi
    [[ "$RUNTIME" == "docker" ]] || die "prebuild only applies to --runtime docker configs"
    if [[ "$REPLACE_SESSION" == "1" ]]; then
      tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
      kill_foreground_prebuilds
    else
      ensure_no_foreground_prebuild
    fi
    write_tmux_context "$TMUX_CONTEXT_PATH" "$script_path" "$command"
    if ! tmux new-session -d -s "$TMUX_SESSION" "set -euo pipefail; $command 2>&1 | tee -a $(quote_args "$TMUX_LOG_PATH")"; then
      rm -f "$TMUX_CONTEXT_PATH"
      die "failed to launch tmux session: $TMUX_SESSION"
    fi
    tmux set-option -t "$TMUX_SESSION" @swehero_launch_context "$TMUX_CONTEXT_PATH" >/dev/null 2>&1 || true
    echo "launched tmux session: $TMUX_SESSION"
  fi
  echo "context: $TMUX_CONTEXT_PATH"
  echo "log: $TMUX_LOG_PATH"
  if [[ "$ATTACH" == "1" ]]; then
    exec tmux attach-session -t "$TMUX_SESSION"
  fi
  exit 0
fi

[[ "$(uname -s)" != "Darwin" ]] || die "this launcher is pod-only; run it from the Kubernetes GPU pod"
[[ -d /workspace ]] || die "expected /workspace hostPath; run from the GPU pod"
command -v docker >/dev/null 2>&1 || die "docker not found; recreate the pod with manifests/midtraining-hostpath.yaml"
command -v git >/dev/null 2>&1 || die "git not found; recreate the pod with manifests/midtraining-hostpath.yaml"
command -v python3 >/dev/null 2>&1 || die "python3 not found; recreate the pod with manifests/midtraining-hostpath.yaml"
command -v tmux >/dev/null 2>&1 || die "tmux not found; recreate the pod with manifests/midtraining-hostpath.yaml"

mkdir -p "$TMUX_LOG_DIR" /workspace/runlogs
ensure_pod_git_checkout
ensure_docker
ensure_eval_python
ensure_openhands_checkout
ensure_openhands_dependencies
prebuild_images
