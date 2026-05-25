#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_KUBECONFIG="$ROOT_DIR/tmp/pod-creds/kubeconfig.yaml"
KUBECONFIG_PATH="$DEFAULT_KUBECONFIG"
NAMESPACE="midtraining"
POD_NAME="midtraining-dev"
WORKSPACE_ROOT="/workspace/jaxels-work-trial"
BRANCH=""
PUSH_BRANCH=1
ALLOCATE_TTY="auto"
FORWARDED_ENV_NAMES=(
  HF_TOKEN
  HUGGING_FACE_HUB_TOKEN
  WANDB_API_KEY
  LLM_API_KEY
  TORCHTITAN_POD_VENV
  TORCHTITAN_POD_SETUP_SCRIPT
  SWEHERO_POD_TMUX_SESSION
  SWEHERO_POD_SUPERVISOR
  SWEHERO_POD_TMUX_ATTACH
  SWEHERO_POD_TMUX_LOG_DIR
  SWEHERO_POD_TMUX_ENV_DIR
  VLLM_VENV
  VLLM_REQUIREMENTS_PATH
  VLLM_FORCE_RESTART
  VLLM_VISIBLE_DEVICES
  VLLM_GPU
  VLLM_NCCL_CUMEM_ENABLE
  EVAL_VENV
  OPENHANDS_EVAL_POETRY_VERSION
  OPENHANDS_EVAL_TMUX_SESSION
  OPENHANDS_EVAL_ATTACH
  OPENHANDS_EVAL_TMUX_LOG_DIR
  REQUIRED_GPU_COUNT
  UV_TOOL_DIR
  UV_CACHE_DIR
  UV_PYTHON_INSTALL_DIR
)

usage() {
  cat <<'USAGE'
Usage: scripts/run_midtraining_pod.sh [launcher options] WORKLOAD [args...]

Run a midtraining workload on the canonical Kubernetes GPU pod.

Workloads:
  train                 scripts/run_qwen_swehero_torchtitan_pod.sh
  eval                  scripts/run_openhands_swebench_eval_pod.sh
  prebuild              scripts/prebuild_openhands_swebench_images_pod.sh
  scripts/path.sh       Any repository-relative pod script

Launcher options:
  --kubeconfig PATH     Default: tmp/pod-creds/kubeconfig.yaml
  --namespace NAME      Default: midtraining
  --pod-name NAME       Default: midtraining-dev
  --workspace-root PATH Default: /workspace/jaxels-work-trial
  --branch NAME         Default: current local branch
  --no-push             Do not push the branch before kubectl exec.
  --no-tty              Do not allocate an interactive TTY for kubectl exec.
  -h, --help            Show this help.

The launcher pushes the selected branch by default, enters the GPU pod with
kubectl exec, sets SWEHERO_POD_GIT_BRANCH for the pod-side git guard, and then
starts the selected workload from the pod checkout.
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

selected_workload_script() {
  local workload="$1"
  case "$workload" in
    train|training|torchtitan)
      printf "scripts/run_qwen_swehero_torchtitan_pod.sh\n"
      ;;
    eval|openhands-eval|swebench-eval)
      printf "scripts/run_openhands_swebench_eval_pod.sh\n"
      ;;
    prebuild|image-prebuild|openhands-prebuild)
      printf "scripts/prebuild_openhands_swebench_images_pod.sh\n"
      ;;
    *)
      printf "%s\n" "$workload"
      ;;
  esac
}

require_clean_local_checkout() {
  local status
  status="$(git -C "$ROOT_DIR" status --porcelain=v1)"
  if [[ -n "$status" ]]; then
    cat >&2 <<EOF
error: local checkout has uncommitted changes; commit or stash them before launching a pod workload.

The pod runs the pushed branch from origin, so dirty local files would not be
visible to the job:
EOF
    printf "%s\n" "$status" | sed "s/^/  /" >&2
    exit 1
  fi
}

while (($#)); do
  case "$1" in
    --kubeconfig)
      [[ $# -ge 2 ]] || die "--kubeconfig requires a value"
      KUBECONFIG_PATH="$2"
      shift 2
      ;;
    --namespace)
      [[ $# -ge 2 ]] || die "--namespace requires a value"
      NAMESPACE="$2"
      shift 2
      ;;
    --pod-name)
      [[ $# -ge 2 ]] || die "--pod-name requires a value"
      POD_NAME="$2"
      shift 2
      ;;
    --workspace-root)
      [[ $# -ge 2 ]] || die "--workspace-root requires a value"
      WORKSPACE_ROOT="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || die "--branch requires a value"
      BRANCH="$2"
      shift 2
      ;;
    --no-push)
      PUSH_BRANCH=0
      shift
      ;;
    --no-tty)
      ALLOCATE_TTY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "unknown launcher option: $1"
      ;;
    *)
      break
      ;;
  esac
done

(($#)) || {
  usage >&2
  exit 2
}

WORKLOAD="$1"
shift
WORKLOAD_SCRIPT="$(selected_workload_script "$WORKLOAD")"
if [[ "$WORKLOAD_SCRIPT" = /* ]]; then
  die "workload script must be repository-relative, got: $WORKLOAD_SCRIPT"
fi
[[ -f "$ROOT_DIR/$WORKLOAD_SCRIPT" ]] || die "workload script not found: $WORKLOAD_SCRIPT"

if [[ -z "$BRANCH" || "$PUSH_BRANCH" == "1" ]]; then
  command -v git >/dev/null 2>&1 || die "git not found"
fi
if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git -C "$ROOT_DIR" branch --show-current)"
fi
[[ -n "$BRANCH" ]] || die "could not determine current branch; pass --branch explicitly"

command -v kubectl >/dev/null 2>&1 || die "kubectl not found"
[[ -f "$KUBECONFIG_PATH" ]] || die "kubeconfig not found: $KUBECONFIG_PATH"

if [[ "$PUSH_BRANCH" == "1" ]]; then
  require_clean_local_checkout
  git -C "$ROOT_DIR" push -u origin "$BRANCH"
fi

kubectl_tty_args=()
if [[ "$ALLOCATE_TTY" == "auto" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    kubectl_tty_args=(-it)
  fi
elif [[ "$ALLOCATE_TTY" == "1" ]]; then
  kubectl_tty_args=(-it)
else
  kubectl_tty_args=()
fi

env_args=(
  "SWEHERO_POD_GIT_BRANCH=$BRANCH"
  "MIDTRAINING_POD_WORKSPACE_ROOT=$WORKSPACE_ROOT"
  "WORKSPACE_ROOT=$WORKSPACE_ROOT"
)
for env_name in "${FORWARDED_ENV_NAMES[@]}"; do
  if [[ "${!env_name+x}" == "x" ]]; then
    env_args+=("$env_name=${!env_name}")
  fi
done

exec kubectl --kubeconfig "$KUBECONFIG_PATH" \
  exec "${kubectl_tty_args[@]}" -n "$NAMESPACE" "$POD_NAME" -- \
  bash -lc 'cd "$1" && shift && exec "$@"' \
  bash \
  "$WORKSPACE_ROOT" \
  env \
  "${env_args[@]}" \
  "$WORKSPACE_ROOT/$WORKLOAD_SCRIPT" \
  "$@"
