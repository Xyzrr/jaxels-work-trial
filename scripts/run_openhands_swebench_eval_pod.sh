#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck source=scripts/pod_startup_common.sh
source "$ROOT_DIR/scripts/pod_startup_common.sh"
# shellcheck source=scripts/openhands_eval_launcher_defaults.sh
source "$ROOT_DIR/scripts/openhands_eval_launcher_defaults.sh"
# shellcheck source=scripts/openhands_eval_worker_selection.sh
source "$ROOT_DIR/scripts/openhands_eval_worker_selection.sh"

usage() {
  cat <<'USAGE'
Usage: scripts/run_openhands_swebench_eval_pod.sh [options]

Launch the canonical OpenHands SWE-bench Verified pass@1 eval from the GPU pod.
For workstation launches, use: scripts/run_midtraining_pod.sh eval [options]

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
                          scripts/run_midtraining_pod.sh from the selected
                          local branch; the pod startup guard fast-forwards it
                          from origin.
  OPENHANDS_EVAL_TMUX_SESSION
  OPENHANDS_EVAL_ATTACH   Default: 1 for interactive shells, otherwise 0
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

readonly OPENHANDS_EVAL_UV_VERSION="0.11.16"
readonly UV_X86_64_UNKNOWN_LINUX_GNU_SHA256="74947fe2c03315cf07e82ab3acc703eddef01aba4d5232a98e4c6825ec116131"
if [[ -n "${UV_VERSION:-}" && "$UV_VERSION" != "$OPENHANDS_EVAL_UV_VERSION" ]]; then
  die "UV_VERSION override is not supported; expected uv ${OPENHANDS_EVAL_UV_VERSION}, got ${UV_VERSION}"
fi
readonly UV_VERSION="$OPENHANDS_EVAL_UV_VERSION"

quote_args() {
  (($#)) || return 0
  printf "%q " "$@"
}

supervised_env_args() {
  quote_args \
    "SWEHERO_POD_GIT_BRANCH=${SWEHERO_POD_GIT_BRANCH:-}" \
    "LLM_API_KEY=$LLM_API_KEY" \
    "VLLM_VENV=$VLLM_VENV" \
    "VLLM_REQUIREMENTS_PATH=$VLLM_REQUIREMENTS_PATH" \
    "VLLM_PYTHON_VERSION=$VLLM_PYTHON_VERSION" \
    "VLLM_VISIBLE_DEVICES=$VLLM_VISIBLE_DEVICES" \
    "VLLM_FORCE_RESTART=$VLLM_FORCE_RESTART" \
    "VLLM_NCCL_CUMEM_ENABLE=$VLLM_NCCL_CUMEM_ENABLE" \
    "VLLM_TMUX_SESSION=$VLLM_TMUX_SESSION" \
    "VLLM_TMUX_SESSION_PREFIX=$VLLM_TMUX_SESSION_PREFIX" \
    "VLLM_ROUTER_TMUX_SESSION=$VLLM_ROUTER_TMUX_SESSION" \
    "EVAL_VENV=$EVAL_VENV" \
    "OPENHANDS_EVAL_PYTHON_VERSION=$OPENHANDS_EVAL_PYTHON_VERSION" \
    "OPENHANDS_EVAL_POETRY_VERSION=$OPENHANDS_EVAL_POETRY_VERSION" \
    "DOCKER_TMUX_SESSION=$DOCKER_TMUX_SESSION" \
    "REQUIRED_GPU_COUNT=$REQUIRED_GPU_COUNT" \
    "OPENHANDS_EVAL_TMUX_LOG_DIR=$TMUX_LOG_DIR" \
    "UV_TOOL_DIR=$UV_TOOL_DIR" \
    "UV_CACHE_DIR=$UV_CACHE_DIR" \
    "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
}

readonly QWEN_NATIVE_CONTEXT_LENGTH=32768
readonly PAPER_CONTEXT_LENGTH=131072
readonly PAPER_YARN_ROPE_SCALING='{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/jaxels-work-trial}"
CONFIG_PRESET="$ROOT_DIR/configs/eval/openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args"
if [[ -v LLM_API_KEY ]]; then
  LLM_API_KEY_EXPLICIT=1
else
  LLM_API_KEY_EXPLICIT=0
fi
LLM_API_KEY="${LLM_API_KEY:-local-llm}"
VLLM_VENV="${VLLM_VENV:-/workspace/venvs/openhands-vllm}"
VLLM_REQUIREMENTS_PATH="${VLLM_REQUIREMENTS_PATH:-$ROOT_DIR/requirements/openhands-vllm.txt}"
VLLM_PYTHON_VERSION="${VLLM_PYTHON_VERSION:-3.12}"
VLLM_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES:-${VLLM_GPU:-}}"
VLLM_FORCE_RESTART="${VLLM_FORCE_RESTART:-0}"
VLLM_NCCL_CUMEM_ENABLE="${VLLM_NCCL_CUMEM_ENABLE:-auto}"
VLLM_TMUX_SESSION="${VLLM_TMUX_SESSION:-openhands-vllm-7b}"
VLLM_TMUX_SESSION_PREFIX="${VLLM_TMUX_SESSION_PREFIX:-openhands-vllm-7b-gpu}"
VLLM_ROUTER_TMUX_SESSION="${VLLM_ROUTER_TMUX_SESSION:-openhands-vllm-router}"
EVAL_VENV="${EVAL_VENV:-/workspace/venvs/openhands-eval-pod-py312}"
OPENHANDS_EVAL_PYTHON_VERSION="${OPENHANDS_EVAL_PYTHON_VERSION:-3.12}"
# OpenHands 0.62.0 locks Poetry to 2.1.3; keep the launcher tool version in
# step with that checkout instead of floating to the newest Poetry release.
OPENHANDS_EVAL_POETRY_VERSION="${OPENHANDS_EVAL_POETRY_VERSION:-2.1.3}"
DOCKER_TMUX_SESSION="${DOCKER_TMUX_SESSION:-openhands-dockerd}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-8}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR_EXPLICIT=0
OUTPUT_DIR="/workspace/eval-runs/openhands-swebench-verified-pass1/${RUN_ID}"
TMUX_SESSION="${OPENHANDS_EVAL_TMUX_SESSION:-openhands-swebench-eval-${RUN_ID}}"
TMUX_LOG_DIR="${OPENHANDS_EVAL_TMUX_LOG_DIR:-/workspace/runlogs}"
TMUX_LOG_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.log"
UV_TOOL_DIR="${UV_TOOL_DIR:-/workspace/uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/python}"
BOOTSTRAP_PYTHON="${PYTHON:-}"
EVAL_PYTHON_READY=0
VLLM_PYTHON_READY=0

EVAL_LIMIT=""
EVAL_IDS=""
PREFLIGHT_ONLY=0
SKIP_SWEBENCH_EVAL=0
FOREGROUND=0
if [[ -t 1 ]]; then
  ATTACH="${OPENHANDS_EVAL_ATTACH:-1}"
else
  ATTACH="${OPENHANDS_EVAL_ATTACH:-0}"
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
      shift 2
      ;;
    --eval-ids)
      [[ $# -ge 2 ]] || die "--eval-ids requires a value"
      EVAL_IDS="$2"
      shift 2
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --skip-swebench-eval)
      SKIP_SWEBENCH_EVAL=1
      shift
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      OUTPUT_DIR="$2"
      OUTPUT_DIR_EXPLICIT=1
      shift 2
      ;;
    --run-id)
      [[ $# -ge 2 ]] || die "--run-id requires a value"
      RUN_ID="$2"
      if [[ "$OUTPUT_DIR_EXPLICIT" != "1" ]]; then
        OUTPUT_DIR="/workspace/eval-runs/openhands-swebench-verified-pass1/${RUN_ID}"
      fi
      TMUX_SESSION="${OPENHANDS_EVAL_TMUX_SESSION:-openhands-swebench-eval-${RUN_ID}}"
      TMUX_LOG_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.log"
      shift 2
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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if [[ -n "$EVAL_LIMIT" && -n "$EVAL_IDS" ]]; then
  die "--eval-limit and --eval-ids are mutually exclusive"
fi

resolve_config_preset_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf "%s\n" "$raw"
  else
    printf "%s\n" "$ROOT_DIR/$raw"
  fi
}

resolve_eval_config() {
  local config_path="$1"
  local bootstrap_python="${BOOTSTRAP_PYTHON:-python3}"
  command -v "$bootstrap_python" >/dev/null 2>&1 || \
    die "python3 is required to read eval presets before venv setup"
  "$bootstrap_python" - "$ROOT_DIR" "$config_path" "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import shlex
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
output_dir = sys.argv[3]
if not config_path.is_file():
    raise SystemExit(f"eval config preset not found: {config_path}")
sys.path.insert(0, str(root))
from scripts import openhands_swebench_eval as eval_script

args = eval_script.parse_args(
    [
        f"@{config_path}",
        "--dry-run",
        "--output-dir",
        output_dir,
    ]
)

values = {
    "CONFIG_PRESET_PATH": str(config_path),
    "EVAL_STACK": args.eval_stack,
    "MODEL_ID": args.model_id,
    "SERVED_MODEL_NAME": args.served_model_name,
    "LITELLM_MODEL": args.litellm_model or "",
    "CONTEXT_MODE": args.context_mode,
    "MAX_INPUT_TOKENS": args.max_input_tokens,
    "CONTEXT_ALLOW_LONG_MAX_MODEL_LEN": (
        "1" if eval_script.context_mode_spec(args.context_mode).allow_long_max_model_len else "0"
    ),
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
    if eval_script.context_mode_spec(args.context_mode).allow_long_max_model_len
    and args.vllm_max_model_len > eval_script.QWEN_NATIVE_CONTEXT_LENGTH
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
    "VLLM_DISTRIBUTED_EXECUTOR_BACKEND": (
        args.vllm_distributed_executor_backend or ""
    ),
    "CONFIG_NUM_WORKERS": args.num_workers,
    "SWEBENCH_CACHE_LEVEL": args.swebench_cache_level,
    "SWEBENCH_TIMEOUT": args.swebench_timeout,
    "SWEBENCH_MAX_WORKERS": args.swebench_max_workers,
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

CONFIG_PRESET_PATH="$(resolve_config_preset_path "$CONFIG_PRESET")"
eval "$(resolve_eval_config "$CONFIG_PRESET_PATH")"
LLM_API_KEY="$(
  select_openhands_llm_api_key \
    "$EVAL_STACK" \
    "$LLM_API_KEY_EXPLICIT" \
    "$LLM_API_KEY"
)"
if [[ -n "$OPENHANDS_POETRY_VERSION_FROM_CONFIG" ]]; then
  OPENHANDS_EVAL_POETRY_VERSION="$OPENHANDS_POETRY_VERSION_FROM_CONFIG"
fi

if [[ "$FOREGROUND" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || die "tmux is required for supervised pod launches"
  mkdir -p "$TMUX_LOG_DIR"
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session already exists: $TMUX_SESSION"
  else
    midtraining_prepare_pod_checkout \
      "$WORKSPACE_ROOT" \
      "OpenHands eval pod execution directory"
    script_path="$(realpath "$0")"
    command="cd $(quote_args "$WORKSPACE_ROOT") && env $(supervised_env_args)$(quote_args "$script_path") --foreground"
    if [[ -n "$EVAL_LIMIT" ]]; then
      command+=" --eval-limit $(quote_args "$EVAL_LIMIT")"
    fi
    if [[ -n "$EVAL_IDS" ]]; then
      command+=" --eval-ids $(quote_args "$EVAL_IDS")"
    fi
    if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
      command+=" --preflight-only"
    fi
    if [[ "$SKIP_SWEBENCH_EVAL" == "1" ]]; then
      command+=" --skip-swebench-eval"
    fi
    command+=" --config $(quote_args "$CONFIG_PRESET_PATH")"
    command+=" --output-dir $(quote_args "$OUTPUT_DIR")"
    tmux new-session -d -s "$TMUX_SESSION" "set -euo pipefail; $command 2>&1 | tee -a $(quote_args "$TMUX_LOG_PATH")"
    echo "launched tmux session: $TMUX_SESSION"
  fi
  echo "log: $TMUX_LOG_PATH"
  if [[ "$ATTACH" == "1" ]]; then
    exec tmux attach-session -t "$TMUX_SESSION"
  fi
  exit 0
fi

uv_version_matches() {
  local uv_bin="$1"
  local actual
  actual="$("$uv_bin" --version 2>/dev/null || true)"
  [[ "$actual" == "uv $UV_VERSION"* ]]
}

require_uv_version() {
  local uv_bin="$1"
  local actual
  actual="$("$uv_bin" --version)"
  if [[ "$actual" != "uv $UV_VERSION"* ]]; then
    echo "Wrong uv binary at $uv_bin: expected uv $UV_VERSION, found: $actual" >&2
    return 1
  fi
  echo "$actual" >&2
}

ensure_uv() {
  local uv_bin="${UV_BIN:-}"
  if [[ -n "$uv_bin" ]]; then
    [[ -x "$uv_bin" ]] || die "UV_BIN is not executable: $uv_bin"
    require_uv_version "$uv_bin"
    printf "%s\n" "$uv_bin"
    return
  fi

  local managed_dir="$UV_TOOL_DIR/uv-$UV_VERSION"
  uv_bin="$managed_dir/uv"
  if [[ -x "$uv_bin" ]]; then
    if uv_version_matches "$uv_bin"; then
      require_uv_version "$uv_bin"
      printf "%s\n" "$uv_bin"
      return
    fi
    echo "Removing wrong uv binary from pinned tool directory: $uv_bin" >&2
    rm -rf "$managed_dir"
  fi

  if command -v uv >/dev/null 2>&1; then
    local system_uv
    system_uv="$(command -v uv)"
    if uv_version_matches "$system_uv"; then
      require_uv_version "$system_uv"
      printf "%s\n" "$system_uv"
      return
    fi
  fi

  [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
    die "uv $UV_VERSION is required. Install it or set UV_BIN=/path/to/uv."

  local bootstrap_python="${BOOTSTRAP_PYTHON:-python3}"
  command -v "$bootstrap_python" >/dev/null 2>&1 || \
    die "pinned uv is missing and no bootstrap Python is available"

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  mkdir -p "$managed_dir"
  "$bootstrap_python" - "$UV_VERSION" "$UV_X86_64_UNKNOWN_LINUX_GNU_SHA256" "$tmp_dir/uv.tar.gz" <<'PY'
from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request

version, expected_sha256, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
url = (
    "https://github.com/astral-sh/uv/releases/download/"
    f"{version}/uv-x86_64-unknown-linux-gnu.tar.gz"
)
with urllib.request.urlopen(url, timeout=120) as response, open(output_path, "wb") as out:
    shutil.copyfileobj(response, out)
actual_sha256 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"uv archive checksum mismatch: expected {expected_sha256}, found {actual_sha256}"
    )
PY
  tar -xzf "$tmp_dir/uv.tar.gz" -C "$tmp_dir"
  cp "$tmp_dir/uv-x86_64-unknown-linux-gnu/uv" "$managed_dir/uv"
  cp "$tmp_dir/uv-x86_64-unknown-linux-gnu/uvx" "$managed_dir/uvx"
  chmod 0755 "$managed_dir/uv" "$managed_dir/uvx"
  rm -rf "$tmp_dir"
  require_uv_version "$uv_bin"
  printf "%s\n" "$uv_bin"
}

venv_python_matches() {
  local venv_path="$1"
  local expected_version="$2"
  [[ -x "$venv_path/bin/python" ]] || return 1
  "$venv_path/bin/python" - "$expected_version" <<'PY'
from __future__ import annotations

import sys

expected = tuple(int(part) for part in sys.argv[1].split("."))
actual = sys.version_info[: len(expected)]
raise SystemExit(0 if actual == expected else 1)
PY
}

ensure_python_venv() {
  local venv_path="$1"
  local python_version="$2"

  if ! venv_python_matches "$venv_path" "$python_version"; then
    rm -rf "$venv_path"
    mkdir -p "$(dirname "$venv_path")" "$UV_PYTHON_INSTALL_DIR"
    UV_PYTHON_DOWNLOADS=automatic "$PINNED_UV_BIN" python install "$python_version" \
      --install-dir "$UV_PYTHON_INSTALL_DIR" \
      --no-bin
    UV_PYTHON_INSTALL_DIR="$UV_PYTHON_INSTALL_DIR" \
      "$PINNED_UV_BIN" venv --no-project --python "$python_version" --seed "$venv_path"
  fi
}

ensure_eval_python() {
  if [[ "$EVAL_PYTHON_READY" == "1" ]]; then
    return
  fi
  ensure_python_venv "$EVAL_VENV" "$OPENHANDS_EVAL_PYTHON_VERSION"
  "$PINNED_UV_BIN" pip install \
    --python "$EVAL_VENV/bin/python" \
    "poetry==${OPENHANDS_EVAL_POETRY_VERSION}"
  "$PINNED_UV_BIN" pip check --python "$EVAL_VENV/bin/python"
  "$EVAL_VENV/bin/python" - "$EVAL_VENV" "$OPENHANDS_EVAL_POETRY_VERSION" "$PINNED_UV_BIN" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

venv = Path(sys.argv[1])
expected_poetry = sys.argv[2]
uv_bin = Path(sys.argv[3])
actual_poetry = version("poetry")
if actual_poetry != expected_poetry:
    raise SystemExit(
        f"poetry version mismatch: expected {expected_poetry}, found {actual_poetry}"
    )
record = {
    "created_at_unix": time.time(),
    "python": sys.version,
    "venv": str(venv),
    "uv": subprocess.check_output([str(uv_bin), "--version"], text=True).strip(),
    "poetry": actual_poetry,
}
(venv / "openhands-eval-runtime.json").write_text(json.dumps(record, indent=2))
PY
  EVAL_PYTHON_READY=1
}

ensure_vllm_python() {
  if [[ "$VLLM_PYTHON_READY" == "1" ]]; then
    return
  fi
  [[ -f "$VLLM_REQUIREMENTS_PATH" ]] || die "vLLM requirements file not found: $VLLM_REQUIREMENTS_PATH"
  ensure_python_venv "$VLLM_VENV" "$VLLM_PYTHON_VERSION"
  local resolved_requirements="$VLLM_VENV/openhands-vllm-resolved.txt"
  "$PINNED_UV_BIN" pip compile \
    --python "$VLLM_VENV/bin/python" \
    --output-file "$resolved_requirements" \
    "$VLLM_REQUIREMENTS_PATH"
  "$PINNED_UV_BIN" pip sync --python "$VLLM_VENV/bin/python" "$resolved_requirements"
  "$PINNED_UV_BIN" pip check --python "$VLLM_VENV/bin/python"
  "$VLLM_VENV/bin/python" - "$VLLM_VENV" "$VLLM_REQUIREMENTS_PATH" "$PINNED_UV_BIN" <<'PY'
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

venv = Path(sys.argv[1])
requirements = Path(sys.argv[2])
uv_bin = Path(sys.argv[3])
expected_vllm = None
for line in requirements.read_text().splitlines():
    match = re.match(r"^vllm==(.+)$", line.strip())
    if match:
        expected_vllm = match.group(1)
        break
if expected_vllm is None:
    raise SystemExit(f"{requirements} must pin vllm with vllm==...")
actual_vllm = version("vllm")
if actual_vllm != expected_vllm:
    raise SystemExit(
        f"vllm version mismatch: expected {expected_vllm}, found {actual_vllm}"
    )
vllm_bin = venv / "bin" / "vllm"
if not vllm_bin.exists():
    raise SystemExit(f"vLLM CLI missing: {vllm_bin}")
record = {
    "created_at_unix": time.time(),
    "python": sys.version,
    "venv": str(venv),
    "requirements": str(requirements),
    "uv": subprocess.check_output([str(uv_bin), "--version"], text=True).strip(),
    "packages": {
        package: version(package)
        for package in ("vllm", "torch", "transformers", "tokenizers")
    },
}
(venv / "openhands-vllm-runtime.json").write_text(json.dumps(record, indent=2))
PY
  VLLM_PYTHON_READY=1
}

poetry_install_openhands_dependencies() {
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

midtraining_require_pod_runtime "$WORKSPACE_ROOT" nvidia-smi docker curl git
midtraining_prepare_pod_checkout \
  "$WORKSPACE_ROOT" \
  "OpenHands eval pod execution directory"
export UV_CACHE_DIR
export UV_PYTHON_INSTALL_DIR
PINNED_UV_BIN="$(ensure_uv)"
VISIBLE_GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
(( VISIBLE_GPU_COUNT >= REQUIRED_GPU_COUNT )) || die "expected at least ${REQUIRED_GPU_COUNT} visible GPUs, found ${VISIBLE_GPU_COUNT}"
VLLM_PARALLEL_GPU_COUNT=$((VLLM_TENSOR_PARALLEL_SIZE * VLLM_PIPELINE_PARALLEL_SIZE))
if [[ "$VLLM_SERVER_COUNT" -eq 1 ]]; then
  (( VLLM_PARALLEL_GPU_COUNT <= VISIBLE_GPU_COUNT )) || die "vLLM parallel size exceeds visible GPUs: ${VLLM_PARALLEL_GPU_COUNT} > ${VISIBLE_GPU_COUNT}"
else
  (( VLLM_SERVER_COUNT <= VISIBLE_GPU_COUNT )) || die "vLLM server count exceeds visible GPUs: ${VLLM_SERVER_COUNT} > ${VISIBLE_GPU_COUNT}"
  (( VLLM_TENSOR_PARALLEL_SIZE == 1 && VLLM_PIPELINE_PARALLEL_SIZE == 1 )) || die "multi-replica pod eval uses one vLLM process per GPU; keep VLLM_TENSOR_PARALLEL_SIZE=1 and VLLM_PIPELINE_PARALLEL_SIZE=1 when VLLM_SERVER_COUNT>1"
fi
if [[ "$VLLM_USE_ROUTER" != "1" && "$VLLM_USE_ROUTER" != "true" && "$VLLM_SERVER_COUNT" -ne 1 ]]; then
  die "direct vLLM base URL without router requires VLLM_SERVER_COUNT=1"
fi
if [[ -n "$VLLM_VISIBLE_DEVICES" && "$VLLM_VISIBLE_DEVICES" != "all" && "$VLLM_SERVER_COUNT" -ne 1 ]]; then
  die "VLLM_VISIBLE_DEVICES/VLLM_GPU override is only supported with VLLM_SERVER_COUNT=1"
fi

cd "$WORKSPACE_ROOT"
mkdir -p "$TMUX_LOG_DIR" /workspace/runlogs "$OUTPUT_DIR"

ensure_docker() {
  if ! docker info >/dev/null 2>&1; then
    tmux kill-session -t "$DOCKER_TMUX_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$DOCKER_TMUX_SESSION" \
      "dockerd --host=unix:///var/run/docker.sock > /workspace/runlogs/${DOCKER_TMUX_SESSION}.log 2>&1"
  fi

  for _ in $(seq 1 90); do
    docker info >/dev/null 2>&1 && break
    sleep 1
  done
  docker info >/dev/null 2>&1 || die "Docker daemon did not become ready; see /workspace/runlogs/${DOCKER_TMUX_SESSION}.log"
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
      die "$SWE_LEGO_DIR has local changes; clean it before launching eval"
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
    die "$OPENHANDS_DIR has local changes; clean it before launching eval"
  fi
  git -C "$OPENHANDS_DIR" fetch --tags --depth 1 origin "$OPENHANDS_REF"
  git -C "$OPENHANDS_DIR" checkout --detach "$OPENHANDS_REF"
}

ensure_openhands_dependencies() {
  ensure_openhands_checkout
  poetry_install_openhands_dependencies
  if [[ "$EVAL_STACK" == "swe-lego" ]]; then
    "$PINNED_UV_BIN" pip install \
      --python "$EVAL_VENV/bin/python" \
      -e "$SWE_LEGO_SWEBENCH_DIR"
    "$PINNED_UV_BIN" pip check --python "$EVAL_VENV/bin/python"
  fi
}

pod_ip() {
  hostname -I | awk '{print $1}'
}

vllm_session_name() {
  local gpu="$1"
  printf "%s-%s" "$VLLM_TMUX_SESSION_PREFIX" "$gpu"
}

vllm_context_signature() {
  local gpu="$1"
  local port="$2"
  cat <<EOF
CONTEXT_MODE=$CONTEXT_MODE
CONFIG_PRESET_PATH=$CONFIG_PRESET_PATH
MODEL_ID=$MODEL_ID
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
MAX_INPUT_TOKENS=$MAX_INPUT_TOKENS
VLLM_MAX_MODEL_LEN=$VLLM_MAX_MODEL_LEN
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
VLLM_ROPE_SCALING=$VLLM_ROPE_SCALING
VLLM_DTYPE=$VLLM_DTYPE
VLLM_GPU_MEMORY_UTILIZATION=$VLLM_GPU_MEMORY_UTILIZATION
VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE
VLLM_PIPELINE_PARALLEL_SIZE=$VLLM_PIPELINE_PARALLEL_SIZE
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=$VLLM_DISTRIBUTED_EXECUTOR_BACKEND
VLLM_ENFORCE_EAGER=$VLLM_ENFORCE_EAGER
VLLM_ENABLE_AUTO_TOOL_CHOICE=$VLLM_ENABLE_AUTO_TOOL_CHOICE
VLLM_TOOL_CALL_PARSER=$VLLM_TOOL_CALL_PARSER
NCCL_CUMEM_ENABLE=$(effective_vllm_nccl_cumem_enable)
GPU=$gpu
PORT=$port
EOF
}

effective_vllm_nccl_cumem_enable() {
  if [[ "$VLLM_NCCL_CUMEM_ENABLE" == "auto" ]]; then
    if [[ "$VLLM_PARALLEL_GPU_COUNT" -gt 1 ]]; then
      printf "1\n"
    fi
    return
  fi
  printf "%s\n" "$VLLM_NCCL_CUMEM_ENABLE"
}

kill_process_pattern() {
  local pattern="$1"
  if ! pgrep -f "$pattern" >/dev/null 2>&1; then
    return
  fi

  pkill -TERM -f "$pattern" 2>/dev/null || true
  for _ in $(seq 1 15); do
    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  pkill -KILL -f "$pattern" 2>/dev/null || true
}

cleanup_vllm_runtime() {
  kill_process_pattern "$VLLM_VENV/bin/vllm"
  kill_process_pattern "$VLLM_VENV/bin/python -c from multiprocessing"
  rm -f /dev/shm/psm_* /dev/shm/sem.mp-* /dev/shm/nccl-* 2>/dev/null || true
}

ensure_vllm_server() {
  local ip="$1"
  local gpu="$2"
  local port="$3"
  local session="$4"
  local base_url="http://${ip}:${port}/v1"
  local context_path="${TMUX_LOG_DIR}/${session}.context"
  local expected_context
  expected_context="$(vllm_context_signature "$gpu" "$port")"
  if [[ "$VLLM_FORCE_RESTART" == "1" || "$VLLM_FORCE_RESTART" == "true" ]]; then
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
  if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1; then
    if [[ "$VLLM_FORCE_RESTART" != "1" && "$VLLM_FORCE_RESTART" != "true" ]] \
      && [[ -f "$context_path" ]] \
      && [[ "$(cat "$context_path")" == "$expected_context" ]]; then
      return
    fi
    echo "restarting vLLM endpoint on ${base_url} for context mode ${CONTEXT_MODE}"
    tmux kill-session -t "$session" 2>/dev/null || true
    for _ in $(seq 1 30); do
      curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1 || break
      sleep 1
    done
    if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1; then
      die "non-matching vLLM endpoint is still serving on ${base_url}; stop it or set a different VLLM_PORT"
    fi
  fi

  ensure_vllm_python
  [[ -x "$VLLM_VENV/bin/vllm" ]] || die "missing vLLM binary: $VLLM_VENV/bin/vllm"
  tmux kill-session -t "$session" 2>/dev/null || true
  local vllm_eager_arg=()
  local vllm_distributed_arg=()
  local cuda_visible_arg=()
  local vllm_long_context_env=()
  local vllm_nccl_env=()
  local vllm_rope_arg=()
  local vllm_max_num_seqs_arg=()
  local vllm_tool_args=()
  if [[ "$VLLM_ENFORCE_EAGER" == "1" || "$VLLM_ENFORCE_EAGER" == "true" ]]; then
    vllm_eager_arg=(--enforce-eager)
  fi
  if [[ -n "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    vllm_distributed_arg=(--distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
  fi
  if [[ -n "$VLLM_VISIBLE_DEVICES" && "$VLLM_VISIBLE_DEVICES" != "all" ]]; then
    cuda_visible_arg=(CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE_DEVICES")
  elif [[ "$VLLM_SERVER_COUNT" -eq 1 && "$VLLM_PARALLEL_GPU_COUNT" -gt 1 ]]; then
    local visible_devices=""
    local i
    for i in $(seq 0 $((VLLM_PARALLEL_GPU_COUNT - 1))); do
      if [[ -n "$visible_devices" ]]; then
        visible_devices+=","
      fi
      visible_devices+="$i"
    done
    cuda_visible_arg=(CUDA_VISIBLE_DEVICES="$visible_devices")
  else
    cuda_visible_arg=(CUDA_VISIBLE_DEVICES="$gpu")
  fi
  if [[ "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" == "1" || "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" == "true" ]]; then
    vllm_long_context_env=(VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
  fi
  local nccl_cumem_enable
  nccl_cumem_enable="$(effective_vllm_nccl_cumem_enable)"
  if [[ -n "$nccl_cumem_enable" ]]; then
    vllm_nccl_env=(NCCL_CUMEM_ENABLE="$nccl_cumem_enable")
  fi
  if [[ -n "$VLLM_ROPE_SCALING" ]]; then
    vllm_rope_arg=(--rope-scaling "$VLLM_ROPE_SCALING")
  fi
  if [[ -n "$VLLM_MAX_NUM_SEQS" ]]; then
    vllm_max_num_seqs_arg=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
  fi
  if [[ "$VLLM_ENABLE_AUTO_TOOL_CHOICE" == "1" || "$VLLM_ENABLE_AUTO_TOOL_CHOICE" == "true" ]]; then
    vllm_tool_args=(--enable-auto-tool-choice)
    if [[ -n "$VLLM_TOOL_CALL_PARSER" ]]; then
      vllm_tool_args+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
    fi
  fi
  tmux new-session -d -s "$session" \
    "cd $(quote_args "$WORKSPACE_ROOT") && $(quote_args "${cuda_visible_arg[@]}") $(quote_args "${vllm_long_context_env[@]}") $(quote_args "${vllm_nccl_env[@]}") $(quote_args "$VLLM_VENV/bin/vllm") serve $(quote_args "$MODEL_ID") --host $(quote_args "$VLLM_HOST") --port $(quote_args "$port") --api-key $(quote_args "$LLM_API_KEY") --served-model-name $(quote_args "$SERVED_MODEL_NAME") --max-model-len $(quote_args "$VLLM_MAX_MODEL_LEN") $(quote_args "${vllm_rope_arg[@]}") --tensor-parallel-size $(quote_args "$VLLM_TENSOR_PARALLEL_SIZE") --pipeline-parallel-size $(quote_args "$VLLM_PIPELINE_PARALLEL_SIZE") $(quote_args "${vllm_max_num_seqs_arg[@]}") --gpu-memory-utilization $(quote_args "$VLLM_GPU_MEMORY_UTILIZATION") --dtype $(quote_args "$VLLM_DTYPE") $(quote_args "${vllm_tool_args[@]}") $(quote_args "${vllm_distributed_arg[@]}") $(quote_args "${vllm_eager_arg[@]}") > /workspace/runlogs/${session}.log 2>&1"

  for _ in $(seq 1 240); do
    if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1; then
      printf "%s\n" "$expected_context" > "$context_path"
      return
    fi
    sleep 2
  done
  die "vLLM did not become ready; see /workspace/runlogs/${session}.log"
}

ensure_vllm_router() {
  local ip="$1"
  local router_url="http://${ip}:${VLLM_ROUTER_PORT}/v1"
  shift
  local backend_args=("$@")
  local context_path="${TMUX_LOG_DIR}/${VLLM_ROUTER_TMUX_SESSION}.context"
  local expected_context
  expected_context="$(
    {
      printf "CONTEXT_MODE=%s\n" "$CONTEXT_MODE"
      printf "CONFIG_PRESET_PATH=%s\n" "$CONFIG_PRESET_PATH"
      printf "VLLM_ROUTER_PORT=%s\n" "$VLLM_ROUTER_PORT"
      printf "VLLM_AGENT_TASKS_PER_SERVER=%s\n" "$VLLM_AGENT_TASKS_PER_SERVER"
      printf "LLM_API_KEY_SET=%s\n" "1"
      printf "BACKEND=%s\n" "${backend_args[@]}"
    }
  )"
  if [[ "$VLLM_FORCE_RESTART" == "1" || "$VLLM_FORCE_RESTART" == "true" ]]; then
    tmux kill-session -t "$VLLM_ROUTER_TMUX_SESSION" 2>/dev/null || true
  fi
  if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${router_url}/models" >/dev/null 2>&1; then
    if [[ "$VLLM_FORCE_RESTART" != "1" && "$VLLM_FORCE_RESTART" != "true" ]] \
      && [[ -f "$context_path" ]] \
      && [[ "$(cat "$context_path")" == "$expected_context" ]]; then
      return
    fi
    echo "restarting vLLM router on ${router_url} for context mode ${CONTEXT_MODE}"
    tmux kill-session -t "$VLLM_ROUTER_TMUX_SESSION" 2>/dev/null || true
    for _ in $(seq 1 30); do
      curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${router_url}/models" >/dev/null 2>&1 || break
      sleep 1
    done
    if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${router_url}/models" >/dev/null 2>&1; then
      die "non-matching vLLM router is still serving on ${router_url}; stop it or set a different VLLM_ROUTER_PORT"
    fi
  fi
  tmux kill-session -t "$VLLM_ROUTER_TMUX_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$VLLM_ROUTER_TMUX_SESSION" \
    "cd $(quote_args "$WORKSPACE_ROOT") && $(quote_args "$EVAL_VENV/bin/python") scripts/openai_vllm_router.py --listen-host 0.0.0.0 --listen-port $(quote_args "$VLLM_ROUTER_PORT") --api-key $(quote_args "$LLM_API_KEY") --per-backend-concurrency $(quote_args "$VLLM_AGENT_TASKS_PER_SERVER") $(quote_args "${backend_args[@]}") > /workspace/runlogs/${VLLM_ROUTER_TMUX_SESSION}.log 2>&1"

  for _ in $(seq 1 90); do
    if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${router_url}/models" >/dev/null 2>&1; then
      printf "%s\n" "$expected_context" > "$context_path"
      return
    fi
    sleep 1
  done
  die "vLLM router did not become ready; see /workspace/runlogs/${VLLM_ROUTER_TMUX_SESSION}.log"
}

ensure_vllm_stack() {
  local ip="$1"
  if [[ "$VLLM_FORCE_RESTART" == "1" || "$VLLM_FORCE_RESTART" == "true" ]]; then
    tmux kill-session -t "$VLLM_TMUX_SESSION" 2>/dev/null || true
    if [[ "$VLLM_SERVER_COUNT" -eq 1 && "$VLLM_PARALLEL_GPU_COUNT" -gt 1 ]]; then
      local stale_gpu
      for stale_gpu in $(seq 0 $((VISIBLE_GPU_COUNT - 1))); do
        tmux kill-session -t "$(vllm_session_name "$stale_gpu")" 2>/dev/null || true
      done
    fi
    if [[ "$VLLM_USE_ROUTER" != "1" && "$VLLM_USE_ROUTER" != "true" ]]; then
      tmux kill-session -t "$VLLM_ROUTER_TMUX_SESSION" 2>/dev/null || true
    fi
    cleanup_vllm_runtime
  fi

  local backend_args=()
  local gpu port session
  for gpu in $(seq 0 $((VLLM_SERVER_COUNT - 1))); do
    port=$((VLLM_PORT + gpu))
    session="$(vllm_session_name "$gpu")"
    ensure_vllm_server "$ip" "$gpu" "$port" "$session"
    backend_args+=(--backend "http://${ip}:${port}/v1")
  done
  if [[ "$VLLM_USE_ROUTER" == "1" || "$VLLM_USE_ROUTER" == "true" ]]; then
    ensure_vllm_router "$ip" "${backend_args[@]}"
  fi
}

ensure_docker
ensure_eval_python
POD_IP="${POD_IP:-$(pod_ip)}"
ensure_vllm_stack "$POD_IP"

TOTAL_AGENT_WORKERS=$((VLLM_SERVER_COUNT * VLLM_AGENT_TASKS_PER_SERVER))
if [[ "$VLLM_USE_ROUTER" == "1" || "$VLLM_USE_ROUTER" == "true" ]]; then
  LLM_BASE_URL="http://${POD_IP}:${VLLM_ROUTER_PORT}/v1"
else
  LLM_BASE_URL="http://${POD_IP}:${VLLM_PORT}/v1"
fi
EVAL_NUM_WORKERS="$(
  select_openhands_eval_num_workers \
    "$EVAL_STACK" \
    "$EVAL_LIMIT" \
    "$EVAL_IDS" \
    "$CONFIG_NUM_WORKERS" \
    "$TOTAL_AGENT_WORKERS"
)"

eval_args=(
  "@$CONFIG_PRESET_PATH"
  --base-url "$LLM_BASE_URL"
  --output-dir "$OUTPUT_DIR"
  --openhands-dir "$OPENHANDS_DIR"
  --openhands-ref "$OPENHANDS_REF"
  --num-workers "$EVAL_NUM_WORKERS"
)

if [[ -n "$EVAL_LIMIT" ]]; then
  eval_args+=(--eval-limit "$EVAL_LIMIT")
fi
if [[ -n "$EVAL_IDS" ]]; then
  eval_args+=(--eval-ids "$EVAL_IDS")
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  eval_args+=(--preflight-only)
else
  ensure_openhands_dependencies
fi
if [[ "$SKIP_SWEBENCH_EVAL" == "1" ]]; then
  eval_args+=(--skip-swebench-eval)
fi

PATH="$EVAL_VENV/bin:$PATH" \
  LLM_API_KEY="$LLM_API_KEY" \
  POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-pod \
  POETRY_CACHE_DIR=/workspace/.cache/poetry-pod \
  "$EVAL_VENV/bin/python" scripts/openhands_swebench_eval.py "${eval_args[@]}"
