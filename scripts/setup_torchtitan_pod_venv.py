#!/usr/bin/env python3
"""Create or verify the pinned TorchTitan training runtime on the GPU pod.

This script intentionally does not run through the project `uv` environment.
It is the bootstrap layer that creates the separate pod venv used for
distributed TorchTitan training. That venv pins CUDA, PyTorch, TorchAO, and the
vendored TorchTitan checkout together so a training job does not accidentally
inherit packages from the workstation or from a previous pod image.

The important ML/runtime contract is:

* PyTorch must be the CUDA 12.8 build expected by the GPU pod;
* TorchTitan must import with the FSDP and mixed-precision APIs this project
  uses for full-model training; and
* TorchAO float8 support must be present because the training launcher can ask
  TorchTitan to use FP8 linear layers where TorchTitan marks that path safe.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# The pod training runtime is deliberately pinned separately from the local
# project runtime. TorchTitan and PyTorch internals are tightly coupled; allowing
# a newer uv/Python resolver path to drift here can change CUDA wheels or
# dependency versions before any training code runs.
TORCHTITAN_POD_UV_VERSION = "0.11.16"
TORCHTITAN_POD_PYTHON_VERSION = "3.10.12"
UV_X86_64_UNKNOWN_LINUX_GNU_SHA256 = (
    "74947fe2c03315cf07e82ab3acc703eddef01aba4d5232a98e4c6825ec116131"
)


USAGE = """\
Usage: scripts/setup_torchtitan_pod_venv.py [--recreate] [--verify-only] [--venv PATH]

Build or verify the canonical uv-managed GPU pod venv for the vendored TorchTitan
SWE-HERO trainer. This is the only supported pod runtime for
scripts/run_qwen_swehero_torchtitan_pod.py.

Defaults:
  --venv /workspace/venvs/torchtitan-swehero-cu128

Options:
  --recreate      Delete the venv before rebuilding it.
  --verify-only   Do not install anything; only verify the existing venv.
  --venv PATH     Override the venv path.
  -h, --help      Show this help.
"""


def parse_args(argv: list[str]) -> tuple[Path, bool, bool]:
    venv_path = Path(
        os.environ.get(
            "TORCHTITAN_POD_VENV", "/workspace/venvs/torchtitan-swehero-cu128"
        )
    )
    recreate = False
    verify_only = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--recreate":
            recreate = True
            index += 1
        elif arg == "--verify-only":
            verify_only = True
            index += 1
        elif arg == "--venv":
            if index + 1 >= len(argv):
                print("--venv requires a value", file=sys.stderr)
                raise SystemExit(2)
            venv_path = Path(argv[index + 1])
            index += 2
        elif arg in {"-h", "--help"}:
            print(USAGE, end="")
            raise SystemExit(0)
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            print(USAGE, end="", file=sys.stderr)
            raise SystemExit(2)
    return venv_path, recreate, verify_only


def run(
    args: list[str], *, check: bool = True, **kwargs
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, check=check, **kwargs)


def uv_version_matches(uv_bin: Path, expected: str) -> bool:
    try:
        actual = run([str(uv_bin), "--version"], capture_output=True).stdout
    except Exception:
        return False
    return actual.startswith(f"uv {expected}")


def require_uv_version(uv_bin: Path, expected: str) -> None:
    actual = run([str(uv_bin), "--version"], capture_output=True).stdout.strip()
    if not actual.startswith(f"uv {expected}"):
        print(
            f"Wrong uv binary at {uv_bin}: expected uv {expected}, found: {actual}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(actual, file=sys.stderr)


def download_linux_uv(
    version_value: str, expected_sha256: str, destination_dir: Path
) -> None:
    """Install the exact Linux uv binary needed before any venv exists."""

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        archive = tmp / "uv.tar.gz"
        url = (
            "https://github.com/astral-sh/uv/releases/download/"
            f"{version_value}/uv-x86_64-unknown-linux-gnu.tar.gz"
        )
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            archive.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        actual_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            # This bootstrap path runs before the project dependency lock can
            # protect us. The checksum keeps the pod from executing a corrupted
            # or unexpected uv binary while preparing the training runtime.
            raise SystemExit(
                f"uv archive checksum mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
        with tarfile.open(archive) as tar:
            tar.extractall(tmp)
        extracted = tmp / "uv-x86_64-unknown-linux-gnu"
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted / "uv", destination_dir / "uv")
        shutil.copy2(extracted / "uvx", destination_dir / "uvx")
        (destination_dir / "uv").chmod(0o755)
        (destination_dir / "uvx").chmod(0o755)


def ensure_uv() -> Path:
    """Return a uv binary that exactly matches the TorchTitan pod contract."""

    uv_version_env = os.environ.get("UV_VERSION")
    if uv_version_env and uv_version_env != TORCHTITAN_POD_UV_VERSION:
        print(
            "UV_VERSION override is not supported for this training runtime.\n"
            f"Expected uv {TORCHTITAN_POD_UV_VERSION}, but UV_VERSION={uv_version_env} was set.\n"
            "Unset UV_VERSION and rerun this script.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    uv_bin_env = os.environ.get("UV_BIN")
    if uv_bin_env:
        # UV_BIN is a runtime-plumbing escape hatch for repairing a pod. It is
        # still version-checked because resolver drift can change the CUDA/PyTorch
        # dependency set that the later sync installs.
        uv_bin = Path(uv_bin_env)
        if not uv_bin.exists() or not os.access(uv_bin, os.X_OK):
            print(f"UV_BIN is not executable: {uv_bin}", file=sys.stderr)
            raise SystemExit(1)
        require_uv_version(uv_bin, TORCHTITAN_POD_UV_VERSION)
        return uv_bin

    uv_tool_dir = Path(os.environ.get("UV_TOOL_DIR", "/workspace/uv"))
    managed_dir = uv_tool_dir / f"uv-{TORCHTITAN_POD_UV_VERSION}"
    uv_bin = managed_dir / "uv"
    if uv_bin.exists():
        if uv_version_matches(uv_bin, TORCHTITAN_POD_UV_VERSION):
            require_uv_version(uv_bin, TORCHTITAN_POD_UV_VERSION)
            return uv_bin
        # A wrong binary in the managed location is more dangerous than a
        # missing binary because it looks canonical to future runs. Remove it so
        # the verified download path can recreate the expected tool.
        print(
            f"Removing wrong uv binary from pinned tool directory: {uv_bin}",
            file=sys.stderr,
        )
        shutil.rmtree(managed_dir)

    system_uv = shutil.which("uv")
    if system_uv and uv_version_matches(Path(system_uv), TORCHTITAN_POD_UV_VERSION):
        require_uv_version(Path(system_uv), TORCHTITAN_POD_UV_VERSION)
        return Path(system_uv)

    if os.uname().sysname != "Linux" or os.uname().machine != "x86_64":
        print(
            f"uv {TORCHTITAN_POD_UV_VERSION} is required. Install it or set UV_BIN=/path/to/uv.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    bootstrap_python = os.environ.get("PYTHON") or shutil.which("python3")
    if not bootstrap_python:
        print(
            "Pinned uv binary is missing and no bootstrap Python is available.\n"
            "Expected uv at:\n"
            f"  {uv_bin}\n\n"
            "Either restore the pinned uv binary under /workspace or set PYTHON to a working\n"
            "interpreter for one-time uv bootstrapping.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    download_linux_uv(
        TORCHTITAN_POD_UV_VERSION,
        UV_X86_64_UNKNOWN_LINUX_GNU_SHA256,
        managed_dir,
    )
    require_uv_version(uv_bin, TORCHTITAN_POD_UV_VERSION)
    return uv_bin


def ensure_python(uv_bin: Path) -> str:
    """Return the CPython interpreter used to build the TorchTitan venv."""

    python_bin = os.environ.get("PYTHON")
    if python_bin:
        # PYTHON is only used for bootstrap/repair. We do not infer experiment
        # settings from it; the actual training packages still come from the
        # pinned requirements file below.
        if not Path(python_bin).exists() and shutil.which(python_bin) is None:
            print(f"PYTHON is not executable or on PATH: {python_bin}", file=sys.stderr)
            raise SystemExit(1)
        return python_bin

    python_install_dir = Path(
        os.environ.get("UV_PYTHON_INSTALL_DIR", "/workspace/python")
    )
    python_install_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["UV_PYTHON_DOWNLOADS"] = "automatic"
    # Install CPython into /workspace so the venv can be recreated on fresh pods
    # without depending on the container image's system Python version.
    run(
        [
            str(uv_bin),
            "python",
            "install",
            TORCHTITAN_POD_PYTHON_VERSION,
            "--install-dir",
            str(python_install_dir),
            "--no-bin",
        ],
        env=env,
    )
    env["UV_PYTHON_INSTALL_DIR"] = str(python_install_dir)
    return run(
        [
            str(uv_bin),
            "python",
            "find",
            TORCHTITAN_POD_PYTHON_VERSION,
            "--managed-python",
            "--resolve-links",
        ],
        env=env,
        capture_output=True,
    ).stdout.strip()


def verify_runtime(
    root: Path,
    venv: Path,
    requirements: Path,
    uv_bin: Path,
    expected_uv_version: str,
) -> None:
    """Prove the venv can run the training stack before a job starts."""

    actual_uv_version = run(
        [str(uv_bin), "--version"], capture_output=True
    ).stdout.strip()
    if not actual_uv_version.startswith(f"uv {expected_uv_version}"):
        raise SystemExit(
            f"uv version mismatch: expected uv {expected_uv_version}, found {actual_uv_version}"
        )

    required: dict[str, str] = {}
    for line in requirements.read_text().splitlines():
        match = re.match(r"^(torch|torchao|torchdata)==(.+)$", line.strip())
        if match:
            required[match.group(1)] = match.group(2)

    for package, expected in required.items():
        # These three packages form the critical training runtime:
        # torch provides CUDA kernels and distributed primitives, torchao provides
        # optional FP8/quantization helpers, and torchdata is used by TorchTitan's
        # input pipeline.
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
        # The wheel CUDA version must match the pod contract. A CPU wheel or a
        # different CUDA build can import successfully but fail later inside
        # distributed training or NCCL setup.
        raise SystemExit(f"expected CUDA 12.8 torch wheel, found {torch.version.cuda}")

    from torch.distributed.fsdp import (
        DataParallelMeshDims,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torchao.float8 import Float8LinearConfig

    # These imports are not decorative. The training launcher asks TorchTitan for
    # FSDP-style sharding, mixed precision, and optional FP8 linear layers. If
    # any symbol moved or vanished, fail here instead of during a multi-GPU job.
    Float8LinearConfig.from_recipe_name("rowwise")
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false in the pod runtime")
    # Run one CUDA tensor operation to prove the selected wheel can allocate on
    # the GPU, not just import its Python package.
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
        # These packages support dataset loading, tokenization, checkpoint file
        # formats, config parsing, TensorBoard/W&B logging, and other training
        # plumbing. Importing them here catches incomplete venv repairs.
        importlib.import_module(module)

    # Persist a venv-local manifest so each training run can record exactly what
    # runtime was verified without re-running this script during provenance
    # collection.
    record = {
        "created_at_unix": time.time(),
        "python": sys.version,
        "venv": str(venv),
        "requirements": str(requirements),
        "repo_root": str(root),
        "uv": actual_uv_version,
        "uv_bin": str(uv_bin),
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


def main(argv: list[str] | None = None) -> int:
    venv_path, recreate, verify_only = parse_args(
        sys.argv[1:] if argv is None else argv
    )
    # The requirements input is split into a human-maintained requirement file
    # and an optional lock generated for the pod. Install from the lock when it
    # exists, but still use the input file as the source for critical version
    # checks in verify_runtime.
    requirements_path = Path(
        os.environ.get(
            "TORCHTITAN_POD_REQUIREMENTS",
            str(ROOT_DIR / "requirements" / "torchtitan-pod-cu128.txt"),
        )
    )
    lock_path = Path(
        os.environ.get(
            "TORCHTITAN_POD_LOCK",
            str(ROOT_DIR / "requirements" / "torchtitan-pod-cu128.lock"),
        )
    )
    if not requirements_path.is_file():
        print(f"Requirements file not found: {requirements_path}", file=sys.stderr)
        return 1
    install_requirements_path = lock_path if lock_path.is_file() else requirements_path

    uv_bin = ensure_uv()
    python_bin = ensure_python(uv_bin)
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = env.get("UV_CACHE_DIR", "/workspace/.cache/uv")
    # PyTorch CUDA wheels are often available from multiple indexes. The
    # unsafe-best-match strategy is intentional here because the lock/requirements
    # pin the exact wheel set and the pod runtime must be able to select those
    # CUDA-specific artifacts.
    env["UV_INDEX_STRATEGY"] = env.get("UV_INDEX_STRATEGY", "unsafe-best-match")
    env["UV_LINK_MODE"] = env.get("UV_LINK_MODE", "hardlink")
    # After ensure_python has installed the managed interpreter, do not let uv
    # download a different Python during venv creation or dependency sync.
    env["UV_PYTHON_DOWNLOADS"] = env.get("UV_PYTHON_DOWNLOADS", "never")

    if not verify_only:
        if recreate and venv_path.exists():
            shutil.rmtree(venv_path)
        if not (venv_path / "bin" / "python").exists():
            venv_path.parent.mkdir(parents=True, exist_ok=True)
            # --no-project prevents uv from mixing this pod training venv with
            # the repository's local development pyproject/lock. TorchTitan's
            # CUDA runtime is intentionally isolated.
            run(
                [
                    str(uv_bin),
                    "venv",
                    "--no-project",
                    "--python",
                    python_bin,
                    "--seed",
                    str(venv_path),
                ],
                env=env,
            )
        # Sync enforces the complete runtime set rather than incrementally
        # installing packages. That removes stale CUDA or TorchTitan-adjacent
        # packages from previous pod experiments.
        run(
            [
                str(uv_bin),
                "pip",
                "sync",
                "--python",
                str(venv_path / "bin" / "python"),
                str(install_requirements_path),
            ],
            env=env,
        )
        # TorchTitan is vendored in this repo and intentionally outside the
        # project uv environment. Install it editable without dependencies so the
        # pinned pod requirements remain the single source for package versions.
        run(
            [
                str(uv_bin),
                "pip",
                "install",
                "--python",
                str(venv_path / "bin" / "python"),
                "--no-deps",
                "-e",
                str(ROOT_DIR / "torchtitan"),
            ],
            env=env,
        )

    # pip check catches dependency metadata conflicts; verify_runtime catches the
    # ML-specific contract that pip metadata alone cannot express.
    run(
        [str(uv_bin), "pip", "check", "--python", str(venv_path / "bin" / "python")],
        env=env,
    )
    verify_code = (
        "from scripts.setup_torchtitan_pod_venv import verify_runtime; "
        "from pathlib import Path; import sys; "
        "verify_runtime(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), "
        "Path(sys.argv[4]), sys.argv[5])"
    )
    run(
        [
            str(venv_path / "bin" / "python"),
            "-c",
            verify_code,
            str(ROOT_DIR),
            str(venv_path),
            str(install_requirements_path),
            str(uv_bin),
            TORCHTITAN_POD_UV_VERSION,
        ],
        env={**env, "PYTHONPATH": str(ROOT_DIR)},
    )

    print(
        "Canonical TorchTitan SWE-HERO venv is ready:\n"
        f"  {venv_path}\n\n"
        "Run training through:\n"
        f"  {ROOT_DIR}/scripts/run_qwen_swehero_torchtitan_pod.py [qwen_swehero_train.py args...]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
