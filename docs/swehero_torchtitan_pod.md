# SWE-HERO TorchTitan Pod Runtime

This repository vendors TorchTitan source directly under `torchtitan/`. That
source expects a PyTorch nightly or a PyTorch source build, not the pod's
pre-existing `/workspace/venv`.

The canonical pod runtime is:

```bash
cd /workspace/jaxels-work-trial
scripts/setup_torchtitan_pod_venv.sh --recreate
scripts/run_qwen_swehero_torchtitan_pod.sh \
  --out-dir /workspace/qwen25-coder7b-swehero-torchtitan \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct
```

Do not launch this job with bare `python`, bare `torchrun`, or
`/workspace/venv`. The run wrapper verifies the canonical venv first, prepends
that venv to `PATH`, and points `TORCHRUN_BIN` at the venv's `torchrun`.

## Locked Runtime

The setup script bootstraps exactly `uv 0.11.16` under
`/workspace/uv/uv-0.11.16` when that exact version is not already installed.
The version is pinned inside `scripts/setup_torchtitan_pod_venv.sh`; do not set
`UV_VERSION` or use an unversioned installer URL. If `UV_BIN` is provided, the
script verifies it reports `uv 0.11.16` before using it. The downloaded Linux
x86_64 archive is also checked against its pinned SHA256.

After `uv` is established, the setup script creates the venv with `uv venv`,
syncs dependencies with `uv pip sync`, installs vendored TorchTitan with
`uv pip install --no-deps -e torchtitan`, and verifies the result with
`uv pip check`.

The fully resolved Linux pod lock is
`requirements/torchtitan-pod-cu128.lock`. The setup script installs from that
lock when it is present, so transitive dependencies are not floating between
pod runs.

The human-readable root requirement file,
`requirements/torchtitan-pod-cu128.txt`, records the critical Torch stack pins:

```text
torch==2.12.0.dev20260408+cu128
torchao==0.18.0.dev20260407+cu128
torchdata==0.12.0.dev20260408+cpu
```

The rest of the Python dependencies are installed from the vendored
TorchTitan requirement file, `torchtitan/.ci/docker/requirements.txt`, plus the
Hugging Face packages used by our SWE-HERO materialization scripts. Those
resolved transitive versions are frozen in the lock file.

This uses CUDA 12.8 wheels because the current 8xH100 pod driver is
`570.195.03`. The newer CUDA 13.0 nightly index is not the canonical runtime
for this pod unless the driver and the pinned requirement file are updated
together and revalidated.

## Built-In Verification

`scripts/setup_torchtitan_pod_venv.sh` fails unless the venv can:

- import `DataParallelMeshDims` directly from `torch.distributed.fsdp`;
- import TorchAO float8 support and construct the `rowwise` recipe;
- run a CUDA smoke tensor on the H100;
- import the vendored TorchTitan modules that require the exported FSDP API;
- pass `uv pip check`.

It also writes a venv-local metadata file at
`$TORCHTITAN_POD_VENV/torchtitan-swehero-runtime.json` with the exact package
versions, `uv` binary path/version, and critical imports used for the run.

`scripts/qwen_swehero_train.py` also validates the active runtime before
launching training, so dependency mismatches fail before data prep or
distributed startup.

## HF Logits Parity

Before a real run, verify that the TorchTitan model definition and
`Qwen25StateDictAdapter` load the same initial model as the Hugging Face
reference:

```bash
cd /workspace/jaxels-work-trial
$TORCHTITAN_POD_VENV/bin/python scripts/qwen_swehero_logits_parity.py \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --reference-model-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --reference-context paper-yarn-128k \
  --json-out /workspace/qwen25-coder7b-swehero-parity.json
```

`paper-yarn-128k` is the default because the SWE-HERO paper fine-tunes
Qwen2.5-Coder-Instruct with YaRN extending the native 32k context to 128k.
The script also supports `--reference-context standard-hf` for checking against
the unmodified HF config, but that is not the training recipe used here.

To make the training launcher run this preflight check before data prep and
TorchTitan startup, add:

```bash
--verify-hf-logits-parity
```

## Smoke Run Command

For the public dataset subset smoke test on the pod:

```bash
cd /workspace/jaxels-work-trial
scripts/run_qwen_swehero_torchtitan_pod.sh \
  --out-dir /workspace/qwen25-coder7b-swehero-soak32k \
  --skip-data-prep \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --max-length 32768 \
  --buckets 32768 \
  --bucket-cp 32768:2 \
  --nproc-per-node 8 \
  --num-train-epochs 100 \
  --enable-fp8 \
  --fp8-recipe rowwise \
  --checkpoint-interval 1000 \
  --checkpoint-async-mode disabled \
  --metrics-log-freq 1 \
  --log-rank 0
```

The production run should use the same wrapper with the final filtered dataset
and the full bucket plan.
