#!/usr/bin/env bash
# Shared pod-side startup checks for launchers that must run repository code as
# pushed to origin. Source this file from pod workload wrappers.

midtraining_pod_startup_error() {
  echo "error: $*" >&2
}

midtraining_should_enforce_pod_git_guard() {
  local repo_dir="$1"
  case "${SWEHERO_POD_GIT_ENFORCE:-auto}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
    auto|AUTO|"")
      [[ "$repo_dir" == "${SWEHERO_POD_GIT_ROOT:-/workspace/jaxels-work-trial}" ]]
      return
      ;;
    *)
      cat >&2 <<EOF
SWEHERO_POD_GIT_ENFORCE must be auto, 1, or 0; got:
  $SWEHERO_POD_GIT_ENFORCE
EOF
      return 2
      ;;
  esac
}

midtraining_prepare_pod_checkout() {
  local repo_dir="$1"
  local label="${2:-pod execution directory}"
  local enforce_status

  if midtraining_should_enforce_pod_git_guard "$repo_dir"; then
    enforce_status=0
  else
    enforce_status=$?
  fi
  if [[ "$enforce_status" == "1" ]]; then
    return 0
  fi
  if [[ "$enforce_status" != "0" ]]; then
    return "$enforce_status"
  fi

  [[ -d "$repo_dir" ]] || {
    midtraining_pod_startup_error "workspace not found: $repo_dir"
    return 1
  }
  command -v git >/dev/null 2>&1 || {
    midtraining_pod_startup_error "git not found; recreate the pod with manifests/midtraining-hostpath.yaml"
    return 1
  }

  # shellcheck source=scripts/pod_git_guard.sh
  source "$repo_dir/scripts/pod_git_guard.sh"
  swehero_require_pod_git_checkout \
    "$repo_dir" \
    "${SWEHERO_POD_GIT_BRANCH:-}" \
    "$label"
}

midtraining_require_pod_runtime() {
  local workspace_root="$1"
  shift

  if [[ "$(uname -s)" == "Darwin" ]]; then
    midtraining_pod_startup_error "this launcher is pod-only; run it from the Kubernetes GPU pod"
    return 1
  fi
  [[ -d /workspace ]] || {
    midtraining_pod_startup_error "expected /workspace hostPath; run from the GPU pod"
    return 1
  }
  [[ -d "$workspace_root" ]] || {
    midtraining_pod_startup_error "workspace not found: $workspace_root"
    return 1
  }

  local binary
  for binary in "$@"; do
    command -v "$binary" >/dev/null 2>&1 || {
      midtraining_pod_startup_error "$binary not found; recreate the pod with manifests/midtraining-hostpath.yaml"
      return 1
    }
  done
}
