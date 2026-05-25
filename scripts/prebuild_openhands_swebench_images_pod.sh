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
    "DATASET": args.dataset,
    "SPLIT": args.split,
    "RUNTIME": args.runtime,
    "OPENHANDS_REPO": args.openhands_repo,
    "OPENHANDS_REF": args.openhands_ref,
    "OPENHANDS_DIR": args.openhands_dir,
    "DOCKER_SMOKE_IMAGE": args.docker_smoke_image,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
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
  "$uv_bin" pip install --python "$EVAL_VENV/bin/python" "poetry==2.1.3"
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
from openhands.version import get_version


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
    lock_tag = f"oh_v{get_version()}_{get_hash_for_lock_files(base_image, False)}"
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
[[ "$RUNTIME" == "docker" ]] || die "prebuild only applies to --runtime docker configs"
TMUX_LOG_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.log"

if [[ "$FOREGROUND" != "1" ]]; then
  [[ "$(uname -s)" != "Darwin" ]] || die "this launcher is pod-only; run it from the Kubernetes GPU pod"
  [[ -d /workspace ]] || die "expected /workspace hostPath; run from the GPU pod"
  command -v tmux >/dev/null 2>&1 || die "tmux is required for pod prebuilds"
  mkdir -p "$TMUX_LOG_DIR"
  LAUNCH_SESSION=1
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    if [[ "$REPLACE_SESSION" == "1" ]]; then
      echo "replacing tmux session: $TMUX_SESSION"
    else
      echo "tmux session already exists: $TMUX_SESSION"
      LAUNCH_SESSION=0
    fi
  fi
  if [[ "$LAUNCH_SESSION" == "1" ]]; then
    ensure_pod_git_checkout
    if [[ "$REPLACE_SESSION" == "1" ]]; then
      tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    fi
    script_path="$(realpath "$0")"
    command="cd $(quote_args "$ROOT_DIR") && SWEHERO_POD_GIT_BRANCH=$(quote_args "$SWEHERO_POD_GIT_BRANCH") EVAL_VENV=$(quote_args "$EVAL_VENV") $(quote_args "$script_path") --foreground --config $(quote_args "$CONFIG_PRESET_PATH") --tmux-session $(quote_args "$TMUX_SESSION") --parallel-builds $(quote_args "$PARALLEL_BUILDS")"
    if [[ -n "$EVAL_LIMIT" ]]; then
      command+=" --eval-limit $(quote_args "$EVAL_LIMIT")"
    fi
    tmux new-session -d -s "$TMUX_SESSION" "set -euo pipefail; $command 2>&1 | tee -a $(quote_args "$TMUX_LOG_PATH")"
    echo "launched tmux session: $TMUX_SESSION"
  fi
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
