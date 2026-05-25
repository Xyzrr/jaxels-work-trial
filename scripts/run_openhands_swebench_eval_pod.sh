#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_openhands_swebench_eval_pod.sh [options]

Launch the canonical OpenHands SWE-bench Verified pass@1 eval from the GPU pod.

Options:
  --smoke                 Run one SWE-bench Verified instance.
  --eval-limit N          Run N instances. Omit for the full Verified split.
  --full                  Clear any eval limit and run the full Verified split.
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
  MODEL_ID                Default: /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct
  SERVED_MODEL_NAME       Default: Qwen/Qwen2.5-Coder-7B-Instruct
  LITELLM_MODEL           Default: openai/Qwen/Qwen2.5-Coder-7B-Instruct
  LLM_API_KEY             Default: local-llm
  VLLM_VENV               Default: /workspace/venvs/openhands-vllm
  VLLM_TENSOR_PARALLEL_SIZE
                          Default: 1
  VLLM_PIPELINE_PARALLEL_SIZE
                          Default: 1
  VLLM_SERVER_COUNT       Default: 8
  VLLM_AGENT_TASKS_PER_SERVER
                          Default: 24
  VLLM_ROUTER_PORT        Default: 8090
  VLLM_DTYPE              Default: bfloat16
  VLLM_FORCE_RESTART      Set to 1 to replace an already-running vLLM server.
  EVAL_VENV               Default: /workspace/venvs/openhands-eval-pod-py312
  OPENHANDS_DIR           Default: /workspace/eval-runs/OpenHands
  OPENHANDS_REF           Default: 0.62.0
  MAX_OUTPUT_TOKENS       Default: 4096. Set to none only for ablations.
  REQUIRED_GPU_COUNT      Default: 8
  OPENHANDS_EVAL_TMUX_SESSION
  OPENHANDS_EVAL_ATTACH   Default: 1 for interactive shells, otherwise 0
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

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/jaxels-work-trial}"
MODEL_ID="${MODEL_ID:-/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-Coder-7B-Instruct}"
LITELLM_MODEL="${LITELLM_MODEL:-openai/Qwen/Qwen2.5-Coder-7B-Instruct}"
LLM_API_KEY="${LLM_API_KEY:-local-llm}"
VLLM_VENV="${VLLM_VENV:-/workspace/venvs/openhands-vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_VISIBLE_DEVICES="${VLLM_VISIBLE_DEVICES:-${VLLM_GPU:-}}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
VLLM_SERVER_COUNT="${VLLM_SERVER_COUNT:-8}"
VLLM_AGENT_TASKS_PER_SERVER="${VLLM_AGENT_TASKS_PER_SERVER:-24}"
VLLM_ROUTER_PORT="${VLLM_ROUTER_PORT:-8090}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_FORCE_RESTART="${VLLM_FORCE_RESTART:-0}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
VLLM_TMUX_SESSION="${VLLM_TMUX_SESSION:-openhands-vllm-7b}"
VLLM_TMUX_SESSION_PREFIX="${VLLM_TMUX_SESSION_PREFIX:-openhands-vllm-7b-gpu}"
VLLM_ROUTER_TMUX_SESSION="${VLLM_ROUTER_TMUX_SESSION:-openhands-vllm-router}"
EVAL_VENV="${EVAL_VENV:-/workspace/venvs/openhands-eval-pod-py312}"
OPENHANDS_DIR="${OPENHANDS_DIR:-/workspace/eval-runs/OpenHands}"
OPENHANDS_REF="${OPENHANDS_REF:-0.62.0}"
OPENHANDS_REPO="${OPENHANDS_REPO:-https://github.com/OpenHands/OpenHands.git}"
DOCKER_TMUX_SESSION="${DOCKER_TMUX_SESSION:-openhands-dockerd}"
DOCKER_SMOKE_IMAGE="${DOCKER_SMOKE_IMAGE:-hello-world:latest}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-8}"
RUN_ID="${OPENHANDS_EVAL_RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR_EXPLICIT=1
else
  OUTPUT_DIR_EXPLICIT=0
fi
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/eval-runs/openhands-swebench-verified-pass1/${RUN_ID}}"
TMUX_SESSION="${OPENHANDS_EVAL_TMUX_SESSION:-openhands-swebench-eval-${RUN_ID}}"
TMUX_LOG_DIR="${OPENHANDS_EVAL_TMUX_LOG_DIR:-/workspace/runlogs}"
TMUX_LOG_PATH="${TMUX_LOG_DIR}/${TMUX_SESSION}.log"

EVAL_LIMIT="${EVAL_LIMIT:-}"
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
    --smoke)
      EVAL_LIMIT=1
      shift
      ;;
    --eval-limit)
      [[ $# -ge 2 ]] || die "--eval-limit requires a value"
      EVAL_LIMIT="$2"
      shift 2
      ;;
    --full)
      EVAL_LIMIT=""
      shift
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

if [[ "$FOREGROUND" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || die "tmux is required for supervised pod launches"
  mkdir -p "$TMUX_LOG_DIR"
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "tmux session already exists: $TMUX_SESSION"
  else
    script_path="$(realpath "$0")"
    command="cd $(quote_args "$WORKSPACE_ROOT") && $(quote_args "$script_path") --foreground"
    if [[ -n "$EVAL_LIMIT" ]]; then
      command+=" --eval-limit $(quote_args "$EVAL_LIMIT")"
    fi
    if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
      command+=" --preflight-only"
    fi
    if [[ "$SKIP_SWEBENCH_EVAL" == "1" ]]; then
      command+=" --skip-swebench-eval"
    fi
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

if [[ "$(uname -s)" == "Darwin" ]]; then
  die "this launcher is pod-only; run it from the Kubernetes GPU pod"
fi
[[ -d /workspace ]] || die "expected /workspace hostPath; run from the GPU pod"
[[ -d "$WORKSPACE_ROOT" ]] || die "workspace not found: $WORKSPACE_ROOT"
[[ -x /workspace/uv/uv-0.11.16/uv ]] || die "missing pinned uv at /workspace/uv/uv-0.11.16/uv"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; run from the GPU pod"
command -v docker >/dev/null 2>&1 || die "docker not found; recreate the pod with manifests/midtraining-hostpath.yaml"
command -v curl >/dev/null 2>&1 || die "curl not found; recreate the pod with manifests/midtraining-hostpath.yaml"
command -v git >/dev/null 2>&1 || die "git not found; recreate the pod with manifests/midtraining-hostpath.yaml"
VISIBLE_GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
(( VISIBLE_GPU_COUNT >= REQUIRED_GPU_COUNT )) || die "expected at least ${REQUIRED_GPU_COUNT} visible GPUs, found ${VISIBLE_GPU_COUNT}"
(( VLLM_SERVER_COUNT <= VISIBLE_GPU_COUNT )) || die "vLLM server count exceeds visible GPUs: ${VLLM_SERVER_COUNT} > ${VISIBLE_GPU_COUNT}"
(( VLLM_TENSOR_PARALLEL_SIZE == 1 && VLLM_PIPELINE_PARALLEL_SIZE == 1 )) || die "canonical pod eval uses one vLLM per GPU; keep VLLM_TENSOR_PARALLEL_SIZE=1 and VLLM_PIPELINE_PARALLEL_SIZE=1"
if [[ -n "$VLLM_VISIBLE_DEVICES" && "$VLLM_VISIBLE_DEVICES" != "all" && "$VLLM_SERVER_COUNT" -ne 1 ]]; then
  die "VLLM_VISIBLE_DEVICES/VLLM_GPU override is only supported with VLLM_SERVER_COUNT=1"
fi

cd "$WORKSPACE_ROOT"
mkdir -p /workspace/runlogs "$OUTPUT_DIR"

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

ensure_eval_python() {
  if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
    /workspace/uv/uv-0.11.16/uv venv "$EVAL_VENV" --python 3.12
  fi
  /workspace/uv/uv-0.11.16/uv pip install \
    --python "$EVAL_VENV/bin/python" \
    pip poetry
}

ensure_openhands_checkout() {
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
  PATH="$EVAL_VENV/bin:$PATH" \
    POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-pod \
    POETRY_CACHE_DIR=/workspace/.cache/poetry-pod \
    poetry -C "$OPENHANDS_DIR" env use "$EVAL_VENV/bin/python"
  PATH="$EVAL_VENV/bin:$PATH" \
    POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-pod \
    POETRY_CACHE_DIR=/workspace/.cache/poetry-pod \
    poetry -C "$OPENHANDS_DIR" install --with evaluation,test --no-root
}

pod_ip() {
  hostname -I | awk '{print $1}'
}

vllm_session_name() {
  local gpu="$1"
  printf "%s-%s" "$VLLM_TMUX_SESSION_PREFIX" "$gpu"
}

ensure_vllm_server() {
  local ip="$1"
  local gpu="$2"
  local port="$3"
  local session="$4"
  local base_url="http://${ip}:${port}/v1"
  if [[ "$VLLM_FORCE_RESTART" == "1" || "$VLLM_FORCE_RESTART" == "true" ]]; then
    tmux kill-session -t "$session" 2>/dev/null || true
  fi
  if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1; then
    return
  fi

  [[ -x "$VLLM_VENV/bin/vllm" ]] || die "missing vLLM binary: $VLLM_VENV/bin/vllm"
  tmux kill-session -t "$session" 2>/dev/null || true
  local vllm_eager_arg=()
  local vllm_distributed_arg=()
  local cuda_visible_arg=()
  if [[ "$VLLM_ENFORCE_EAGER" == "1" || "$VLLM_ENFORCE_EAGER" == "true" ]]; then
    vllm_eager_arg=(--enforce-eager)
  fi
  if [[ -n "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    vllm_distributed_arg=(--distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
  fi
  if [[ -n "$VLLM_VISIBLE_DEVICES" && "$VLLM_VISIBLE_DEVICES" != "all" ]]; then
    cuda_visible_arg=(CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE_DEVICES")
  else
    cuda_visible_arg=(CUDA_VISIBLE_DEVICES="$gpu")
  fi
  tmux new-session -d -s "$session" \
    "cd $(quote_args "$WORKSPACE_ROOT") && $(quote_args "${cuda_visible_arg[@]}") VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 $(quote_args "$VLLM_VENV/bin/vllm") serve $(quote_args "$MODEL_ID") --host 0.0.0.0 --port $(quote_args "$port") --api-key $(quote_args "$LLM_API_KEY") --served-model-name $(quote_args "$SERVED_MODEL_NAME") --max-model-len 131072 --tensor-parallel-size 1 --pipeline-parallel-size 1 --gpu-memory-utilization $(quote_args "$VLLM_GPU_MEMORY_UTILIZATION") --dtype $(quote_args "$VLLM_DTYPE") --enable-auto-tool-choice --tool-call-parser hermes $(quote_args "${vllm_distributed_arg[@]}") $(quote_args "${vllm_eager_arg[@]}") > /workspace/runlogs/${session}.log 2>&1"

  for _ in $(seq 1 240); do
    curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${base_url}/models" >/dev/null 2>&1 && return
    sleep 2
  done
  die "vLLM did not become ready; see /workspace/runlogs/${session}.log"
}

ensure_vllm_router() {
  local ip="$1"
  local router_url="http://${ip}:${VLLM_ROUTER_PORT}/v1"
  shift
  local backend_args=("$@")
  if [[ "$VLLM_FORCE_RESTART" == "1" || "$VLLM_FORCE_RESTART" == "true" ]]; then
    tmux kill-session -t "$VLLM_ROUTER_TMUX_SESSION" 2>/dev/null || true
  fi
  if curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${router_url}/models" >/dev/null 2>&1; then
    return
  fi
  tmux kill-session -t "$VLLM_ROUTER_TMUX_SESSION" 2>/dev/null || true
  tmux new-session -d -s "$VLLM_ROUTER_TMUX_SESSION" \
    "cd $(quote_args "$WORKSPACE_ROOT") && $(quote_args "$EVAL_VENV/bin/python") scripts/openai_vllm_router.py --listen-host 0.0.0.0 --listen-port $(quote_args "$VLLM_ROUTER_PORT") --api-key $(quote_args "$LLM_API_KEY") --per-backend-concurrency $(quote_args "$VLLM_AGENT_TASKS_PER_SERVER") $(quote_args "${backend_args[@]}") > /workspace/runlogs/${VLLM_ROUTER_TMUX_SESSION}.log 2>&1"

  for _ in $(seq 1 90); do
    curl -fsS -H "Authorization: Bearer ${LLM_API_KEY}" "${router_url}/models" >/dev/null 2>&1 && return
    sleep 1
  done
  die "vLLM router did not become ready; see /workspace/runlogs/${VLLM_ROUTER_TMUX_SESSION}.log"
}

ensure_vllm_stack() {
  local ip="$1"
  if [[ "$VLLM_FORCE_RESTART" == "1" || "$VLLM_FORCE_RESTART" == "true" ]]; then
    tmux kill-session -t "$VLLM_TMUX_SESSION" 2>/dev/null || true
  fi

  local backend_args=()
  local gpu port session
  for gpu in $(seq 0 $((VLLM_SERVER_COUNT - 1))); do
    port=$((VLLM_PORT + gpu))
    session="$(vllm_session_name "$gpu")"
    ensure_vllm_server "$ip" "$gpu" "$port" "$session"
    backend_args+=(--backend "http://${ip}:${port}/v1")
  done
  ensure_vllm_router "$ip" "${backend_args[@]}"
}

ensure_docker
ensure_eval_python
POD_IP="${POD_IP:-$(pod_ip)}"
ensure_vllm_stack "$POD_IP"

TOTAL_AGENT_WORKERS=$((VLLM_SERVER_COUNT * VLLM_AGENT_TASKS_PER_SERVER))
if [[ -n "${NUM_WORKERS:-}" ]]; then
  EVAL_NUM_WORKERS="$NUM_WORKERS"
elif [[ -n "$EVAL_LIMIT" && "$EVAL_LIMIT" -gt 0 && "$EVAL_LIMIT" -lt "$TOTAL_AGENT_WORKERS" ]]; then
  EVAL_NUM_WORKERS="$EVAL_LIMIT"
else
  EVAL_NUM_WORKERS="$TOTAL_AGENT_WORKERS"
fi

eval_args=(
  --model-id "$MODEL_ID"
  --served-model-name "$SERVED_MODEL_NAME"
  --litellm-model "$LITELLM_MODEL"
  --base-url "http://${POD_IP}:${VLLM_ROUTER_PORT}/v1"
  --api-key "$LLM_API_KEY"
  --output-dir "$OUTPUT_DIR"
  --openhands-dir "$OPENHANDS_DIR"
  --openhands-ref "$OPENHANDS_REF"
  --num-workers "$EVAL_NUM_WORKERS"
  --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
  --vllm-pipeline-parallel-size "$VLLM_PIPELINE_PARALLEL_SIZE"
  --vllm-server-count "$VLLM_SERVER_COUNT"
  --vllm-agent-tasks-per-server "$VLLM_AGENT_TASKS_PER_SERVER"
  --vllm-router-port "$VLLM_ROUTER_PORT"
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm-dtype "$VLLM_DTYPE"
  --vllm-distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND"
)
if [[ "$VLLM_ENFORCE_EAGER" == "1" || "$VLLM_ENFORCE_EAGER" == "true" ]]; then
  eval_args+=(--vllm-enforce-eager)
else
  eval_args+=(--no-vllm-enforce-eager)
fi

if [[ -n "$EVAL_LIMIT" ]]; then
  eval_args+=(--eval-limit "$EVAL_LIMIT")
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
  POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-pod \
  POETRY_CACHE_DIR=/workspace/.cache/poetry-pod \
  "$EVAL_VENV/bin/python" scripts/openhands_swebench_eval.py "${eval_args[@]}"
