#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${TORCHTITAN_POD_VENV:-/workspace/venvs/torchtitan-swehero-cu128}"
SETUP_SCRIPT="$ROOT_DIR/scripts/setup_torchtitan_pod_venv.sh"

if [[ ! -x "$VENV_PATH/bin/python" ]]; then
  cat >&2 <<EOF
Canonical TorchTitan venv is missing:
  $VENV_PATH

Create it first:
  $SETUP_SCRIPT --recreate
EOF
  exit 1
fi

"$SETUP_SCRIPT" --verify-only --venv "$VENV_PATH" >/dev/null

export PATH="$VENV_PATH/bin:$PATH"
export TORCHRUN_BIN="${TORCHRUN_BIN:-$VENV_PATH/bin/torchrun}"

exec "$VENV_PATH/bin/python" "$ROOT_DIR/scripts/qwen_swehero_train.py" "$@"
