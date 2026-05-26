# TorchTitan Pod Runtime

## Overview

This is the GPU-pod runbook for TorchTitan training. It still uses the
Qwen2.5-Coder-7B direct-to-hero SWE-Hero run as the canonical example, but new
training experiments should be preset-driven rather than SWE-Hero hardcoded.

Use this document when launching, reproducing, or debugging pod training. For
shared ML vocabulary and project-wide configuration rules, see
[`../AGENTS.md`](../AGENTS.md). For the local Python/uv boundary, see
[`python_uv_project.md`](python_uv_project.md).

Do not modify `torchtitan/` unless explicitly asked. It is vendored source and
expects the locked pod runtime below, not the pod's pre-existing
`/workspace/venv`.

## Quick Commands

Canonical production-shaped launch:

```bash
scripts/run_midtraining_pod.py train \
  @configs/training/qwen25-coder-7b-direct-to-hero.args \
  --out-dir /workspace/qwen25-coder7b-swehero-torchtitan \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct
```

Paper-aligned production launch:

```bash
scripts/run_midtraining_pod.py train \
  @configs/training/qwen25-coder-7b-direct-to-hero.args \
  --production-mode \
  --enable-wandb
```

Recreate the TorchTitan pod venv intentionally:

```bash
scripts/setup_torchtitan_pod_venv.py --recreate
```

List or attach to supervised pod sessions:

```bash
tmux ls
tmux attach-session -t swehero-qwen25-coder7b-swehero-torchtitan
```

## Runtime Contract

Run `scripts/run_midtraining_pod.py train` from the workstation checkout. The
meta-wrapper pushes the current clean branch, enters `midtraining-dev` with
`tmp/pod-creds/kubeconfig.yaml`, sets the legacy `SWEHERO_POD_GIT_BRANCH`
runtime variable inside the pod, and starts the lower-level TorchTitan wrapper
from `/workspace/jaxels-work-trial`.

The default experiment preset is:

```text
configs/training/qwen25-coder-7b-direct-to-hero.args
```

It contains the paper-aligned model, dataset, context, optimizer, bucket,
context-parallelism, checkpoint, and launch settings that used to be implicit
launcher defaults. New experiments should copy a preset, edit the copy, and
pass that preset explicitly with `@configs/training/...`.

The direct-to-hero preset pins the base checkpoint to:

```text
Qwen/Qwen2.5-Coder-7B-Instruct@c03e6d358207e414f1eca0bb1891e29f1db0e242
```

That revision is passed to Hugging Face asset downloads, recorded in the data
manifest and run spec, and checked during preflight.

CLI flags after the preset are valid one-off overrides; normal argparse order
means later values win. Keep secrets and pod/runtime plumbing as environment
only: `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `WANDB_API_KEY`,
`TORCHTITAN_POD_VENV`, `TORCHTITAN_POD_SUPERVISOR`, and tmux controls are not
experiment settings and do not belong in presets.

`SWEHERO_POD_GIT_BRANCH` is a pod-side legacy compatibility name. New shared
controls should use neutral names.

## Supervised Sessions

When the pod wrapper runs from an interactive pod terminal, it creates or
attaches to a `tmux` session named from `--out-dir`. Training continues inside
that pod-local session if `kubectl exec` disconnects. Rerunning the same
wrapper command reconnects instead of starting a duplicate job.

The wrapper records the outer launcher transcript under:

```text
/workspace/runlogs/<session>.tmux.log
```

Useful controls:

```bash
# Override the derived session name for a launch.
SWEHERO_POD_TMUX_SESSION=swehero-7b-prod \
  scripts/run_midtraining_pod.py train \
    @configs/training/qwen25-coder-7b-direct-to-hero.args \
    --production-mode --enable-wandb

# Force a supervised detached launch from a non-interactive exec.
SWEHERO_POD_SUPERVISOR=1 SWEHERO_POD_TMUX_ATTACH=0 \
  scripts/run_midtraining_pod.py train \
    @configs/training/qwen25-coder-7b-direct-to-hero.args \
    --production-mode --enable-wandb

# Bypass tmux intentionally for non-interactive automation.
SWEHERO_POD_SUPERVISOR=0 \
  scripts/run_midtraining_pod.py train \
    @configs/training/qwen25-coder-7b-direct-to-hero.args \
    --dry-run
```

## Pod And Git Access

The host and container workspace root are both `/workspace`. The pod manifest
uses `hostPath.path: /workspace` with `type: Directory`, so the GPU node must
prepare `/workspace` as a real directory or mountpoint before the pod is
created. Do not rely on a host symlink.

The project remote currently uses GitHub SSH. To let the pod run the same Git
operations as the workstation, create or refresh the Kubernetes Secret mounted
by `manifests/midtraining-hostpath.yaml`:

```bash
mkdir -p tmp/pod-creds
git_name="$(git config --global --get user.name)"
git_email="$(git config --global --get user.email)"
printf '[user]\n\tname = %s\n\temail = %s\n' "$git_name" "$git_email" \
  > tmp/pod-creds/gitconfig
ssh-keygen -F github.com -f "$HOME/.ssh/known_hosts" \
  | sed '/^#/d' > tmp/pod-creds/github_known_hosts
test -s tmp/pod-creds/github_known_hosts \
  || ssh-keyscan github.com > tmp/pod-creds/github_known_hosts

KUBECONFIG=tmp/pod-creds/kubeconfig.yaml \
  kubectl create secret generic midtraining-git-ssh -n midtraining \
    --from-file=id_ed25519="$HOME/.ssh/id_ed25519" \
    --from-file=id_ed25519.pub="$HOME/.ssh/id_ed25519.pub" \
    --from-file=known_hosts=tmp/pod-creds/github_known_hosts \
    --from-file=gitconfig=tmp/pod-creds/gitconfig \
    --dry-run=client -o yaml \
  | KUBECONFIG=tmp/pod-creds/kubeconfig.yaml kubectl apply -f -
```

This keeps private key material in a cluster Secret and ignored local files,
not git. Use a repo-scoped deploy key if the pod should only access this repo.
Running pods do not gain new volume mounts, so recreate `midtraining-dev` after
creating the Secret when durable Git access is needed across pod restarts.

Verify pod Git access:

```bash
KUBECONFIG=tmp/pod-creds/kubeconfig.yaml \
  kubectl exec -n midtraining midtraining-dev -- \
    bash -lc 'ssh -T git@github.com || true; git -C /workspace/jaxels-work-trial pull --ff-only'
```

For new launches, `scripts/run_midtraining_pod.py` refuses to push if the local
checkout has uncommitted changes, then pushes the selected branch and enters
the pod. The pod-side startup guard refuses to launch unless
`/workspace/jaxels-work-trial` is clean, checked out to that branch, and
fast-forwardable to `origin/<branch>`.

Production launches require workspace root `/workspace/jaxels-work-trial`.
Override `--workspace-root` only for non-production local tests.

## Locked Runtime

The CUDA base image does not include the required Python runtime. The pod
entrypoint uses pinned `uv 0.11.16` to install CPython `3.10.12` under
`/workspace/python`; the setup script then creates the canonical venv under:

```text
/workspace/venvs/torchtitan-swehero-cu128
```

Do not launch training with bare `python`, bare `torchrun`, or
`/workspace/venv`. The wrapper creates or repairs the canonical venv, prepends
it to `PATH`, and launches the training entrypoint with that venv's Python so
`torchrun` resolves from the same runtime.

The setup script bootstraps exactly `uv 0.11.16` under:

```text
/workspace/uv/uv-0.11.16
```

Do not set `UV_VERSION` or use an unversioned installer URL. If `UV_BIN` is
provided, the setup verifies that it reports `uv 0.11.16`. The downloaded
Linux x86_64 archive is checked against its pinned SHA256.

After `uv` exists, setup creates the venv with `uv venv`, syncs dependencies
with `uv pip sync`, installs vendored TorchTitan with
`uv pip install --no-deps -e torchtitan`, and verifies with `uv pip check`.

The resolved Linux pod lock is:

```text
requirements/torchtitan-pod-cu128.lock
```

The human-readable root requirement file records the critical Torch stack:

```text
torch==2.12.0.dev20260408+cu128
torchao==0.18.0.dev20260407+cu128
torchdata==0.12.0.dev20260408+cpu
```

These pins are part of the ML runtime. `torch` supplies tensor math, FSDP,
compiler paths, and CUDA kernels. `torchao` supplies the FP8 recipe.
`torchdata` supplies data-loading primitives expected by vendored TorchTitan.

The rest of the Python dependencies come from
`torchtitan/.ci/docker/requirements.txt` plus Hugging Face packages used by the
SWE-Hero materialization scripts. Resolved transitive versions are frozen in
the lock file.

This runtime uses CUDA 12.8 wheels because the current 8xH100 pod driver is
`570.195.03`. Do not switch to CUDA 13.0 nightlies unless the driver and pinned
requirement file are updated together and revalidated.

## Built-In Verification

`scripts/setup_torchtitan_pod_venv.py` fails unless the venv can:

- import `DataParallelMeshDims` directly from `torch.distributed.fsdp`;
- import TorchAO float8 support and construct the `rowwise` recipe;
- run a CUDA smoke tensor on the H100;
- import vendored TorchTitan modules that require the exported FSDP API;
- pass `uv pip check`.

It writes:

```text
$TORCHTITAN_POD_VENV/torchtitan-swehero-runtime.json
```

That file records exact package versions, `uv` binary path/version, and
critical imports. `scripts/qwen_swehero_train.py` also validates the active
runtime before data prep or distributed startup.

## Training Dataset

The canonical training artifact is:

```text
datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/
```

It is a context-capped one-rollout public approximation, not the exact paper
manifest. Provenance and row-count caveats live in
[`../notes/swe-hero-dataset-discrepancy.md`](../notes/swe-hero-dataset-discrepancy.md).

On the pod, the trainer defaults to:

```text
/workspace/datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout
```

If that directory is missing, rebuild it out of band before a production
launch. The raw source revision is:

```text
nvidia/SWE-Hero-openhands-trajectories@5b2ed21270ad773a50163e2999c510f0cbb92cfa
```

Build and refresh the local artifact:

```bash
cd /workspace/jaxels-work-trial
$TORCHTITAN_POD_VENV/bin/python scripts/prepare_swehero_historical_one_rollout.py \
  --output-dir /workspace/datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout \
  --overwrite
$TORCHTITAN_POD_VENV/bin/python scripts/refresh_swehero_context_capped_one_rollout.py \
  --dataset-path /workspace/datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout \
  --output-dir /workspace/datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout \
  --tokenizer-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --max-shifted-context 131072 \
  --overwrite
```

The 2026-05-22 refresh starts from 12,633 raw selected rows, replaces 23
over-context rows, excludes 16 tasks with no fitting accepted rollout, and
keeps 12,617 rows. The final max shifted input length is 130,126.

The refreshed artifact writes `metadata.json`, `selection_manifest.jsonl`, and
`context_filter_report.json`. Do not point production at the raw uncapped
artifact; it still contains rows that exceed the 128k training context.

For quick real-data GPU smoke tests, use `--num-examples 64` or another cap.
The default `--num-examples 0` materializes all usable examples.

## HF Logits Parity

Before a real run, verify that TorchTitan and `Qwen25StateDictAdapter` load the
same initial model as Hugging Face:

```bash
cd /workspace/jaxels-work-trial
$TORCHTITAN_POD_VENV/bin/python scripts/qwen_swehero_logits_parity.py \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --reference-model-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --reference-context paper-yarn-128k \
  --json-out /workspace/qwen25-coder7b-swehero-parity.json
```

Matching logits proves the weights, tokenizer/config assumptions, and 128k
YaRN position encoding align before a multi-hour training job. Use
`--reference-context standard-hf` only for the unmodified HF config, not the
paper-aligned training recipe.

To make the launcher run this preflight before data prep and TorchTitan
startup, add:

```bash
--verify-hf-logits-parity
```

## Smoke Runs

Synthetic all-bucket/CP launch-path smoke:

```bash
scripts/run_midtraining_pod.py train \
  @configs/training/qwen25-coder-7b-direct-to-hero.args \
  --out-dir /workspace/qwen25-coder7b-swehero-all-bucket-cp-smoke \
  --overwrite-output \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --smoke-synthetic-buckets \
  --smoke-synthetic-examples-per-bucket 1 \
  --max-length 8192 \
  --buckets 1024,2048,4096,8192 \
  --bucket-cp 1024:1,2048:2,4096:4,8192:8 \
  --nproc-per-node 8 \
  --global-batch-size 8 \
  --num-train-epochs 8 \
  --checkpoint-interval 1000 \
  --checkpoint-async-mode disabled \
  --metrics-log-freq 1 \
  --log-rank 0 \
  --no-compile \
  --no-enable-fp8
```

This mode materializes tiny synthetic tokenized records only. It exercises
configured buckets and CP degrees, but it is not a training dataset.

For a small real-data smoke, replace the synthetic flags with
`--num-examples 64` and use the target bucket plan. That path is closer to
production data but does not guarantee every configured bucket is non-empty.

Lifecycle smoke:

```bash
cd /workspace/jaxels-work-trial
$TORCHTITAN_POD_VENV/bin/python scripts/qwen_swehero_gpu_lifecycle_smoke.py \
  --out-dir /workspace/qwen25-coder7b-swehero-lifecycle-smoke \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --nproc-per-node 8
```

The lifecycle wrapper runs the trainer with a tiny synthetic bucket, requires
the step-1 DCP validation report, checks final DCP checkpoint plus Hugging Face
export, invokes the same immutable run spec with `--resume`, and validates the
completed run again.

Final acceptance smoke:

```bash
cd /workspace/jaxels-work-trial
$TORCHTITAN_POD_VENV/bin/python scripts/qwen_swehero_gpu_lifecycle_smoke.py \
  --out-dir /workspace/qwen25-coder7b-swehero-final-acceptance-smoke \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --dataset-path /workspace/datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout-final-acceptance-subset \
  --production-acceptance-smoke \
  --bucket 32768 \
  --num-examples 1 \
  --max-streamed-examples 1 \
  --nproc-per-node 8
```

That artifact must be a real one-row Parquet subset built from the cached
one-rollout dataset, not synthetic JSONL, and should include `metadata.json`
and `selection_manifest.jsonl`.

The final acceptance path keeps production Git provenance, canonical workspace,
real dataset, first-step checkpoint validation, final checkpoint/export
validation, resume contract, and durable W&B requirements enabled. It records
the subset, shortened bucket, and one-step cap as acceptance-only deviations.

## Production Gates

The production run should use the canonical wrapper with `--production-mode`
and no `--num-examples`, so it tokenizes the full cached one-rollout dataset
and uses the full bucket plan.

Production mode rejects:

- dry runs;
- synthetic buckets;
- subset caps;
- step caps;
- shortened context;
- alternate bucket curricula;
- bucket stages whose example count is smaller than data-parallel degree;
- non-canonical workspace roots;
- missing `git`;
- dirty repository state;
- missing W&B metrics or non-durable W&B modes.

Include `--enable-wandb` and do not set `--wandb-mode offline` or
`--wandb-mode disabled` for the paper-aligned run.

Before `torchrun`, the launcher checks output disk, free GPU memory, CPU
memory, and write throughput. Defaults:

```bash
--min-free-disk-gb 100 \
--min-free-gpu-memory-gb 60 \
--min-free-cpu-memory-gb 32 \
--min-write-throughput-mb-s 50 \
--write-throughput-probe-mb 64
```

## Output And Logs

Every launcher invocation acquires a sidecar lock:

```text
<out-dir>.launch.lock
```

The lock lives outside `--out-dir`, so `--overwrite-output` cannot delete a
lock held by another process. If the process is killed with `SIGKILL` or the
pod dies, remove the sidecar only after confirming no matching launcher is
still running.

Each bucket-stage attempt writes torchrun stdout and stderr under:

```text
$OUT_DIR/torchrun_logs/
```

Exact per-attempt paths are recorded in `stage_status.json`.

Training-affecting TorchTitan environment controls should be launcher
arguments, not ambient pod-only overrides. The launcher records them in
`run_spec.json`, exports them to each `torchrun` stage, and rejects resume or
relaunch attempts that drift from the original run spec.

Current direct-to-hero controls:

```bash
--optimizer-impl foreach \
--training-dtype float32 \
--mixed-precision-param-dtype bfloat16 \
--mixed-precision-reduce-dtype bfloat16 \
--fsdp-reshard-after-forward never \
--no-detect-anomaly \
--validate-first-step-checkpoint \
--cuda-device-max-connections 1 \
--torch-nccl-async-error-handling 1
```

With `--validate-first-step-checkpoint`, TorchTitan writes and validates a full
DCP checkpoint at optimizer step 1, then writes:

```text
first_step_checkpoint_validation.json
```

Broken checkpoint storage therefore fails after the first stage instead of at
final export.

## Bucket Curriculum

The default `--bucket-curriculum short-to-long` trains shorter non-empty
sequence buckets first, then progressively longer buckets. This is an explicit
engineering choice for the TorchTitan bucketed/CP implementation, not a detail
specified by the paper.

For a no-curriculum ablation, use one configured bucket and
`--bucket-curriculum single-bucket`, for example:

```bash
--buckets 131072 --bucket-cp 131072:8
```

The launcher rejects `single-bucket` when multiple buckets are configured.

## Profiler And Soak Runs

Profiler and memory snapshots are disabled by default.

Use `--max-steps` for a bounded soak without changing dataset, tokenization,
loss mask, optimizer, or bucket plan. The cap applies to total optimizer steps
across launcher stages and is recorded in the immutable run spec.

Short profiler soak:

```bash
scripts/run_midtraining_pod.py train \
  @configs/training/qwen25-coder-7b-direct-to-hero.args \
  --out-dir /workspace/qwen25-coder7b-swehero-profiler-soak \
  --overwrite-output \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --num-examples 64 \
  --max-length 2048 \
  --buckets 1024,2048 \
  --bucket-cp 1024:1,2048:2 \
  --nproc-per-node 8 \
  --global-batch-size 8 \
  --num-train-epochs 8 \
  --max-steps 20 \
  --enable-profiler \
  --profiler-freq 4 \
  --profiler-warmup 1 \
  --profiler-active 1 \
  --profiler-repeat 1 \
  --checkpoint-interval 1000 \
  --checkpoint-async-mode disabled \
  --metrics-log-freq 1 \
  --log-rank 0 \
  --no-compile \
  --no-enable-fp8
```

Profiler traces are written under:

```text
$OUT_DIR/torchtitan/profiling/traces
```

CUDA memory snapshots use `--enable-memory-snapshot` and default to:

```text
$OUT_DIR/torchtitan/profiling/memory_snapshot
```

## Multi-Node Controls

Default single-node contract:

```bash
--nnodes 1 --node-rank 0 --rdzv-backend c10d --rdzv-endpoint localhost:0
```

Multi-node launch is opt-in. For `--nnodes > 1`, provide a stable rendezvous
endpoint and id:

```bash
--nnodes 2 \
--node-rank 0 \
--rdzv-endpoint train-master.example:29400 \
--rdzv-id qwen25-swehero-7b-run-001
```

Each node must use the same run spec, dataset artifact, model assets, bucket
plan, and rendezvous settings, with only `--node-rank` changing per node. The
launcher rejects multi-node settings that still point at `localhost:0` or omit
`--rdzv-id`.
