#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_torchtitan_pod_venv.sh [--recreate] [--verify-only] [--venv PATH]

Build or verify the canonical uv-managed GPU pod venv for the vendored TorchTitan
SWE-HERO trainer. This is the only supported pod runtime for
scripts/run_qwen_swehero_torchtitan_pod.sh.

Defaults:
  --venv /workspace/venvs/torchtitan-swehero-cu128

Options:
  --recreate      Delete the venv before rebuilding it.
  --verify-only   Do not install anything; only verify the existing venv.
  --venv PATH     Override the venv path.
  -h, --help      Show this help.
EOF
}

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${TORCHTITAN_POD_VENV:-/workspace/venvs/torchtitan-swehero-cu128}"
REQUIREMENTS_PATH="${TORCHTITAN_POD_REQUIREMENTS:-$ROOT_DIR/requirements/torchtitan-pod-cu128.txt}"
LOCK_PATH="${TORCHTITAN_POD_LOCK:-$ROOT_DIR/requirements/torchtitan-pod-cu128.lock}"
PYTHON_BIN="${PYTHON:-python3}"
UV_VERSION="${UV_VERSION:-0.10.9}"
UV_TOOL_DIR="${UV_TOOL_DIR:-/workspace/uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
RECREATE=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
  echo "Requirements file not found: $REQUIREMENTS_PATH" >&2
  exit 1
fi
INSTALL_REQUIREMENTS_PATH="$REQUIREMENTS_PATH"
if [[ -f "$LOCK_PATH" ]]; then
  INSTALL_REQUIREMENTS_PATH="$LOCK_PATH"
fi

ensure_uv() {
  local uv_bin="${UV_BIN:-}"
  if [[ -n "$uv_bin" && -x "$uv_bin" ]]; then
    "$uv_bin" --version >&2
    printf '%s\n' "$uv_bin"
    return
  fi

  local managed_dir="$UV_TOOL_DIR/uv-$UV_VERSION"
  uv_bin="$managed_dir/uv"
  if [[ -x "$uv_bin" ]]; then
    "$uv_bin" --version >&2
    printf '%s\n' "$uv_bin"
    return
  fi

  if command -v uv >/dev/null 2>&1; then
    local system_uv
    system_uv="$(command -v uv)"
    if "$system_uv" --version | grep -Fq "uv $UV_VERSION"; then
      "$system_uv" --version >&2
      printf '%s\n' "$system_uv"
      return
    fi
  fi

  if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "uv $UV_VERSION is required. Install it or set UV_BIN=/path/to/uv." >&2
    exit 1
  fi
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  mkdir -p "$managed_dir"
  "$PYTHON_BIN" - "$UV_VERSION" "$tmp_dir/uv.tar.gz" <<'PY'
from __future__ import annotations

import shutil
import sys
import urllib.request

version, output_path = sys.argv[1], sys.argv[2]
url = (
    "https://github.com/astral-sh/uv/releases/download/"
    f"{version}/uv-x86_64-unknown-linux-gnu.tar.gz"
)
with urllib.request.urlopen(url, timeout=120) as response, open(output_path, "wb") as out:
    shutil.copyfileobj(response, out)
PY
  tar -xzf "$tmp_dir/uv.tar.gz" -C "$tmp_dir"
  cp "$tmp_dir/uv-x86_64-unknown-linux-gnu/uv" "$managed_dir/uv"
  cp "$tmp_dir/uv-x86_64-unknown-linux-gnu/uvx" "$managed_dir/uvx"
  chmod 0755 "$managed_dir/uv" "$managed_dir/uvx"
  rm -rf "$tmp_dir"
  "$uv_bin" --version >&2
  printf '%s\n' "$uv_bin"
}

UV_BIN="$(ensure_uv)"
export UV_CACHE_DIR
export UV_INDEX_STRATEGY="${UV_INDEX_STRATEGY:-unsafe-best-match}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
  if [[ "$RECREATE" -eq 1 && -e "$VENV_PATH" ]]; then
    rm -rf "$VENV_PATH"
  fi

  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    mkdir -p "$(dirname "$VENV_PATH")"
    "$UV_BIN" venv --no-project --python "$PYTHON_BIN" --seed "$VENV_PATH"
  fi

  "$UV_BIN" pip sync --python "$VENV_PATH/bin/python" "$INSTALL_REQUIREMENTS_PATH"
  "$UV_BIN" pip install --python "$VENV_PATH/bin/python" --no-deps -e "$ROOT_DIR/torchtitan"
fi

"$UV_BIN" pip check --python "$VENV_PATH/bin/python"

"$VENV_PATH/bin/python" - "$ROOT_DIR" "$VENV_PATH" "$INSTALL_REQUIREMENTS_PATH" <<'PY'
from __future__ import annotations

import importlib
import json
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path

root = Path(sys.argv[1])
venv = Path(sys.argv[2])
requirements = Path(sys.argv[3])

required: dict[str, str] = {}
for line in requirements.read_text().splitlines():
    match = re.match(r"^(torch|torchao|torchdata)==(.+)$", line.strip())
    if match:
        required[match.group(1)] = match.group(2)

for package, expected in required.items():
    actual = version(package)
    if actual != expected:
        raise SystemExit(
            f"{package} version mismatch: expected {expected}, found {actual}"
        )

import torch

if torch.__version__ != required["torch"]:
    raise SystemExit(
        f"torch.__version__ mismatch: expected {required['torch']}, found {torch.__version__}"
    )
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected CUDA 12.8 torch wheel, found {torch.version.cuda}")

from torch.distributed.fsdp import DataParallelMeshDims, MixedPrecisionPolicy, fully_shard
from torchao.float8 import Float8LinearConfig

Float8LinearConfig.from_recipe_name("rowwise")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false in the pod runtime")
cuda_value = torch.ones(1, device="cuda").item()
if cuda_value != 1.0:
    raise SystemExit("CUDA smoke tensor returned the wrong value")

for module in (
    "datasets",
    "einops",
    "fsspec",
    "huggingface_hub",
    "safetensors",
    "tensorboard",
    "tokenizers",
    "transformers",
    "tyro",
    "wandb",
):
    importlib.import_module(module)

import torchtitan
import torchtitan.distributed.full_dtensor
import torchtitan.models.llama3.parallelize

record = {
    "created_at_unix": time.time(),
    "python": sys.version,
    "venv": str(venv),
    "requirements": str(requirements),
    "repo_root": str(root),
    "torch_cuda": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "critical_imports": {
        "DataParallelMeshDims": repr(DataParallelMeshDims),
        "MixedPrecisionPolicy": repr(MixedPrecisionPolicy),
        "fully_shard": repr(fully_shard),
        "Float8LinearConfig": repr(Float8LinearConfig),
    },
    "packages": {
        package: version(package)
        for package in (
            "torch",
            "torchao",
            "torchdata",
            "datasets",
            "tokenizers",
            "transformers",
            "torchtitan",
        )
    },
}
(venv / "torchtitan-swehero-runtime.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
PY

cat <<EOF
Canonical TorchTitan SWE-HERO venv is ready:
  $VENV_PATH

Run training through:
  $ROOT_DIR/scripts/run_qwen_swehero_torchtitan_pod.sh [qwen_swehero_train.py args...]
EOF
