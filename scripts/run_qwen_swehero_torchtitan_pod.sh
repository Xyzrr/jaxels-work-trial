#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_PATH="$ROOT_DIR/scripts/$(basename -- "${BASH_SOURCE[0]}")"
VENV_PATH="${TORCHTITAN_POD_VENV:-/workspace/venvs/torchtitan-swehero-cu128}"
SETUP_SCRIPT="${TORCHTITAN_POD_SETUP_SCRIPT:-$ROOT_DIR/scripts/setup_torchtitan_pod_venv.sh}"
DEFAULT_OUT_DIR="/workspace/qwen25-coder7b-swehero-torchtitan"

# shellcheck source=scripts/pod_git_guard.sh
source "$ROOT_DIR/scripts/pod_git_guard.sh"

shell_quote() {
  printf "%q" "$1"
}

should_enforce_pod_git_guard() {
  case "${SWEHERO_POD_GIT_ENFORCE:-auto}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
    auto|AUTO|"")
      [[ "$ROOT_DIR" == "${SWEHERO_POD_GIT_ROOT:-/workspace/jaxels-work-trial}" ]]
      return
      ;;
    *)
      cat >&2 <<EOF
SWEHERO_POD_GIT_ENFORCE must be auto, 1, or 0; got:
  $SWEHERO_POD_GIT_ENFORCE
EOF
      exit 2
      ;;
  esac
}

ensure_pod_git_checkout() {
  should_enforce_pod_git_guard || return 0
  swehero_require_pod_git_checkout \
    "$ROOT_DIR" \
    "${SWEHERO_POD_GIT_BRANCH:-}" \
    "TorchTitan pod execution directory"
}

python_for_arg_parsing() {
  if [[ -x "$VENV_PATH/bin/python" ]]; then
    printf "%s\n" "$VENV_PATH/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  cat >&2 <<'EOF'
python3 is required to parse launcher argument files before the pod venv exists.
The canonical pod manifest installs python3 at container startup.
EOF
  exit 1
}

resolved_out_dir() {
  "$(python_for_arg_parsing)" - "$ROOT_DIR" "$DEFAULT_OUT_DIR" "$@" <<'PY'
import os
import shlex
import sys
from pathlib import Path

root = Path(sys.argv[1])
default_out_dir = sys.argv[2]
tokens: list[str] = []


def expand_arg(arg: str) -> None:
    if not arg.startswith("@"):
        tokens.append(arg)
        return

    raw_path = Path(arg[1:])
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(root / raw_path)
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open() as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    for token in shlex.split(stripped):
                        expand_arg(token)
            return
    tokens.append(arg)


for value in sys.argv[3:]:
    expand_arg(value)

out_dir = default_out_dir
index = 0
while index < len(tokens):
    token = tokens[index]
    if token == "--out-dir" and index + 1 < len(tokens):
        out_dir = tokens[index + 1]
        index += 2
        continue
    if token.startswith("--out-dir="):
        out_dir = token.split("=", 1)[1]
    index += 1

print(out_dir)
PY
}

tmux_session_name() {
  if [[ -n "${SWEHERO_POD_TMUX_SESSION:-}" ]]; then
    printf "%s\n" "$SWEHERO_POD_TMUX_SESSION"
    return
  fi

  local out_dir
  out_dir="$(resolved_out_dir "$@")"
  local base
  base="$(basename -- "${out_dir%/}")"
  if [[ -z "$base" || "$base" == "." || "$base" == "/" ]]; then
    base="default"
  fi
  base="$(printf "%s" "$base" | tr -c "A-Za-z0-9_-" "-" | cut -c1-48)"
  base="${base%-}"
  base="${base#-}"
  if [[ -z "$base" ]]; then
    base="default"
  fi
  printf "swehero-%s\n" "$base"
}

should_use_tmux_supervisor() {
  if [[ "${SWEHERO_POD_SUPERVISOR_CHILD:-0}" == "1" ]]; then
    return 1
  fi
  if [[ -n "${TMUX:-}" ]]; then
    return 1
  fi

  case "${SWEHERO_POD_SUPERVISOR:-auto}" in
    1|true|TRUE|yes|YES|on|ON|tmux|TMUX)
      return 0
      ;;
    0|false|FALSE|no|NO|off|OFF|direct|DIRECT|none|NONE)
      return 1
      ;;
    auto|AUTO|"")
      [[ -t 0 && -t 1 ]]
      return
      ;;
    *)
      cat >&2 <<EOF
SWEHERO_POD_SUPERVISOR must be auto, tmux/1, or direct/0; got:
  $SWEHERO_POD_SUPERVISOR
EOF
      exit 2
      ;;
  esac
}

should_attach_tmux_client() {
  case "${SWEHERO_POD_TMUX_ATTACH:-auto}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    0|false|FALSE|no|NO|off|OFF)
      return 1
      ;;
    auto|AUTO|"")
      [[ -t 0 && -t 1 ]]
      return
      ;;
    *)
      cat >&2 <<EOF
SWEHERO_POD_TMUX_ATTACH must be auto, 1, or 0; got:
  $SWEHERO_POD_TMUX_ATTACH
EOF
      exit 2
      ;;
  esac
}

attach_or_create_tmux_session() {
  if ! command -v tmux >/dev/null 2>&1; then
    cat >&2 <<'EOF'
tmux is required for reconnectable supervised pod launches.

The canonical pod manifest installs tmux at container startup. If this is an
older running pod, recreate it from manifests/midtraining-hostpath.yaml or
install tmux in the container before launching. To intentionally bypass this
supervisor for non-interactive automation, set SWEHERO_POD_SUPERVISOR=0.
EOF
    exit 1
  fi

  local session_name
  session_name="$(tmux_session_name "$@")"
  local log_dir="${SWEHERO_POD_TMUX_LOG_DIR:-/workspace/runlogs}"
  mkdir -p "$log_dir"
  local transcript_path="$log_dir/$session_name.tmux.log"
  local env_dir="${SWEHERO_POD_TMUX_ENV_DIR:-${TMPDIR:-/tmp}}"
  mkdir -p "$env_dir"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    cat <<EOF
Found existing supervised SWE-HERO session:
  $session_name
Transcript:
  $transcript_path
EOF
    if should_attach_tmux_client; then
      echo "Attaching now."
      exec tmux attach-session -t "$session_name"
    fi
    cat <<EOF
No interactive terminal is available, so the existing session was left running.
Attach later with:
  tmux attach-session -t $session_name
EOF
    exit 0
  fi

  ensure_pod_git_checkout

  local env_file
  env_file="$(mktemp "$env_dir/$session_name.env.XXXXXX")"
  chmod 600 "$env_file"
  export -p >"$env_file"

  local command
  command="source $(shell_quote "$env_file"); "
  command+="rm -f $(shell_quote "$env_file"); "
  command+="tmux set-option -w remain-on-exit on >/dev/null 2>&1 || true; "
  command+="tmux set-option history-limit 200000 >/dev/null 2>&1 || true; "
  command+="cd $(shell_quote "$ROOT_DIR") && "
  command+="export SWEHERO_POD_SUPERVISOR_CHILD=1; "
  command+="exec $(shell_quote "$SELF_PATH")"
  local arg
  for arg in "$@"; do
    command+=" $(shell_quote "$arg")"
  done

  if ! tmux new-session -d -s "$session_name" "exec bash -lc $(shell_quote "$command")"; then
    rm -f "$env_file"
    exit 1
  fi
  tmux pipe-pane -o -t "$session_name:0.0" "cat >> $(shell_quote "$transcript_path")"

  cat <<EOF
Started supervised SWE-HERO session:
  $session_name

If the pod connection drops, the job keeps running in tmux. Reconnect with the
same launcher command, or directly with:
  tmux attach-session -t $session_name

Transcript:
  $transcript_path
EOF
  if should_attach_tmux_client; then
    exec tmux attach-session -t "$session_name"
  fi
  cat <<EOF
No interactive terminal is available, so the session was started detached.
Attach later with:
  tmux attach-session -t $session_name
EOF
  exit 0
}

if should_use_tmux_supervisor "$@"; then
  attach_or_create_tmux_session "$@"
fi

ensure_pod_git_checkout

"$SETUP_SCRIPT" --venv "$VENV_PATH"

export PATH="$VENV_PATH/bin:$PATH"

exec "$VENV_PATH/bin/python" "$ROOT_DIR/scripts/qwen_swehero_train.py" "$@"
