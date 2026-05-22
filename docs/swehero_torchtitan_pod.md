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

When this wrapper is run from an interactive pod terminal, it creates or
attaches to a `tmux` session named from `--out-dir`. The training process stays
inside that pod-local session if the `kubectl exec` connection drops, and
rerunning the same wrapper command reconnects to the existing session instead
of starting a duplicate job. The wrapper also records the outer launcher
transcript under `/workspace/runlogs/<session>.tmux.log`.

Useful controls:

```bash
# List supervised launches.
tmux ls

# Reattach directly when you know the session name.
tmux attach-session -t swehero-qwen25-coder7b-swehero-torchtitan

# Override the derived session name for a launch.
SWEHERO_POD_TMUX_SESSION=swehero-7b-prod \
  scripts/run_qwen_swehero_torchtitan_pod.sh @configs/swehero-7b.args

# Force a supervised detached launch from a non-interactive exec.
SWEHERO_POD_SUPERVISOR=1 SWEHERO_POD_TMUX_ATTACH=0 \
  scripts/run_qwen_swehero_torchtitan_pod.sh @configs/swehero-7b.args

# Bypass tmux intentionally for non-interactive automation.
SWEHERO_POD_SUPERVISOR=0 \
  scripts/run_qwen_swehero_torchtitan_pod.sh --dry-run
```

The host and container workspace root are both `/workspace`. The pod manifest
uses `hostPath.path: /workspace` with `type: Directory`, so the GPU node must
prepare `/workspace` as a real directory or mountpoint before the pod is
created. Do not rely on a host symlink for this path.

The CUDA base image does not include Python. The pod entrypoint uses the pinned
`/workspace/uv/uv-0.11.16/uv` binary to install CPython 3.10.12 under
`/workspace/python` before idling. It also installs `tmux` when the base image
does not provide it, so reconnectable launches are available before training.
The persisted uv-managed venv under `/workspace/venvs/torchtitan-swehero-cu128`
has a valid interpreter after every pod recreation without relying on
apt-managed Python.

The launcher pins the base checkpoint to
`Qwen/Qwen2.5-Coder-7B-Instruct@c03e6d358207e414f1eca0bb1891e29f1db0e242`.
That revision is passed to Hugging Face asset downloads, recorded in the data
manifest and run spec, and checked during preflight before launch.

Production launches require the canonical workspace root
`/workspace/jaxels-work-trial`. The launcher records the configured root, the
script root, and their resolved physical paths in `run_spec.json`,
`resume_contract.json`, `launcher_plan.json`, and `runtime_metadata.json`.
`launcher_plan.json` and `runtime_metadata.json` also record the current working
directory for debugging. Override `--workspace-root` or `WORKSPACE_ROOT` only
for non-production local tests; `--production-mode` rejects any root other than
the canonical pod path.

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

## Training Dataset

The canonical training dataset artifact is the one-rollout SWE-Hero dataset
generated by `scripts/prepare_swehero_historical_one_rollout.py`. It is ignored
by git locally under `datasets/`, so the pod workflow does not depend on a
developer having copied a 485 MB local folder.

On the pod, the trainer defaults to:

```text
/workspace/datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout
```

If that directory is missing, `scripts/qwen_swehero_train.py` builds it in the
pod from the pinned source dataset:

```text
nvidia/SWE-Hero-openhands-trajectories@5b2ed21270ad773a50163e2999c510f0cbb92cfa
```

The builder writes a Hugging Face-style local Parquet dataset with
`data/*.parquet`, `metadata.json`, and `selection_manifest.jsonl`. The training
manifest records the dataset path, source revision, metadata hashes, selection
manifest hash, and Parquet shard sizes/hashes. Use `--rebuild-source-dataset`
when intentionally replacing the cached pod artifact.

For quick real-data GPU smoke tests, pass `--num-examples 64` or another cap.
Do not combine those smoke caps with `--production-mode`. The default
`--num-examples 0` means materialize all usable examples from the cached
one-rollout dataset.

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

For a launch-path smoke test that deterministically exercises every configured
bucket and CP degree, use `--smoke-synthetic-buckets`. This mode materializes
tiny synthetic tokenized records only; it is not a training dataset and should
not be used for the paper-aligned run.

```bash
cd /workspace/jaxels-work-trial
scripts/run_qwen_swehero_torchtitan_pod.sh \
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

For a small capped smoke test against the real cached SWE traces, replace the
synthetic flags with `--num-examples 64` and use the target bucket plan. That
path is closer to the production data path, but it does not guarantee that
every configured bucket is non-empty.

To exercise the checkpoint/resume/export/validation lifecycle on the GPU pod,
run the lifecycle smoke wrapper. It launches the same TorchTitan trainer with a
single tiny synthetic bucket, requires the step-1 DCP validation report, checks
the final DCP checkpoint plus Hugging Face export, then invokes the same
immutable run spec with `--resume` and verifies the completed run again:

```bash
cd /workspace/jaxels-work-trial
$TORCHTITAN_POD_VENV/bin/python scripts/qwen_swehero_gpu_lifecycle_smoke.py \
  --out-dir /workspace/qwen25-coder7b-swehero-lifecycle-smoke \
  --hf-assets-path /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --nproc-per-node 8
```

This lifecycle smoke intentionally uses `--smoke-synthetic-buckets`,
`--no-compile`, `--no-enable-fp8`, and lowered resource thresholds so it can
run quickly and repeatedly. It does not change the production recipe or the
paper-aligned launch gate.

The production run should use the same wrapper with `--production-mode` and
without `--num-examples`, so the trainer tokenizes the full cached one-rollout
dataset and uses the full bucket plan. The production gate rejects dry runs,
synthetic buckets, subset caps, step caps, shortened context, and alternate
bucket curricula before launch.

Production mode also rejects any bucket stage whose example count is smaller
than its data-parallel degree. Tiny smoke runs may reuse tiny buckets on empty
ranks to exercise distributed code paths, but the paper-aligned data run must
not silently duplicate records because a length bucket is too small for the
configured rank topology.

Production mode also requires `git` to be available in the launch environment
and the repository worktree to be clean. Commit or stash local edits before the
paper-aligned run; non-production smoke runs record whatever Git state is
available but do not enforce cleanliness.

Production mode requires W&B metrics with a durable mode. Include
`--enable-wandb` and do not set `--wandb-mode offline` or
`--wandb-mode disabled` for the paper-aligned run.

Before torchrun starts, the launcher also checks output-disk free space, free
GPU memory, available CPU memory, and write throughput to the run filesystem.
The defaults are conservative launch gates and are recorded in `run_spec.json`:

```bash
--min-free-disk-gb 100 \
--min-free-gpu-memory-gb 60 \
--min-free-cpu-memory-gb 32 \
--min-write-throughput-mb-s 50 \
--write-throughput-probe-mb 64
```

## Launch Argument Files

Long production launch commands can be placed in a reviewed argument file and
passed with argparse's `@file` syntax:

```bash
scripts/run_qwen_swehero_torchtitan_pod.sh @configs/swehero-7b.args
```

Each non-empty line in the argument file is parsed like shell input, and lines
starting with `#` are ignored. Flags written after `@configs/swehero-7b.args`
on the command line override earlier values from the file.

Reviewed argument files for the actual direct-to-hero run should include
`--production-mode` and `--enable-wandb`. Leave production mode out for smoke,
profiler, and bounded soak commands because those intentionally use prototype
settings.

## Output Launch Lock

Every launcher invocation acquires an atomic sidecar lock before reading,
rewriting, or resuming an output directory:

```text
<out-dir>.launch.lock
```

This lock is outside `--out-dir`, so `--overwrite-output` cannot delete a lock
held by another process. If a second launcher targets the same `--out-dir`, it
fails before data prep or TorchTitan startup and reports the lock metadata
(`pid`, `hostname`, and creation time). The lock is removed on normal exit and
on handled exceptions; if the process is killed with `SIGKILL` or the pod dies,
remove the sidecar only after confirming no matching launcher is still running.

## Torchrun Logs

Each bucket-stage attempt writes torchrun stdout and stderr under:

```text
$OUT_DIR/torchrun_logs/
```

The exact per-attempt paths are also recorded in `stage_status.json` under the
stage attempt's `logs` field, so failed stages can be debugged after the
terminal session or pod output scrollback is gone.

## Recorded Training Environment Inputs

Training-affecting TorchTitan environment controls should be set as launcher
arguments, not as ambient pod-only overrides. The launcher records these values
in `run_spec.json`, exports them to each `torchrun` stage, and rejects resume
or relaunch attempts that drift from the original run spec.

The currently explicit controls are:

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

These defaults match the existing direct-to-hero TorchTitan config path. Change
them only for an intentional experiment or debugging run, and keep the reviewed
argument file as the source of truth.

With `--validate-first-step-checkpoint` enabled, TorchTitan writes a full DCP
checkpoint at optimizer step 1, validates its metadata and payload files before
retention cleanup can remove it, and writes
`first_step_checkpoint_validation.json` under the run directory. The launcher
requires that report after the first stage, so broken checkpoint storage fails
early instead of only being discovered at the final export.

## Bucket Curriculum

The launcher defaults to `--bucket-curriculum short-to-long`, which preserves
the current throughput-oriented staging order: shorter non-empty sequence
buckets train first, then progressively longer buckets. This is an explicit
engineering choice for the TorchTitan bucketed/CP implementation, not a
training detail specified by the SWE-ZERO to SWE-HERO paper.

For an ablation that removes the length-bucket curriculum, launch with a single
configured bucket and `--bucket-curriculum single-bucket`, for example
`--buckets 131072 --bucket-cp 131072:8`. The launcher rejects
`single-bucket` when multiple buckets are configured, so the run spec cannot
silently claim a no-curriculum setup while using staged bucket training.

## Profiler And Soak Runs

Profiler and memory snapshot capture are disabled by default, so the
paper-aligned direct-to-hero run does not collect traces or change the training
schedule unless these flags are explicitly provided.

Use `--max-steps` to run a bounded soak without changing the dataset,
tokenization, loss mask, optimizer, or bucket plan. The cap applies to total
optimizer steps across all launcher stages and is recorded in the immutable
run spec.

To collect TorchTitan profiler traces during a short soak, add flags such as:

```bash
cd /workspace/jaxels-work-trial
scripts/run_qwen_swehero_torchtitan_pod.sh \
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

Profiler traces are written under the TorchTitan dump folder, normally
`$OUT_DIR/torchtitan/profiling/traces`. CUDA memory snapshots can be captured
with `--enable-memory-snapshot`; by default those files are written under
`$OUT_DIR/torchtitan/profiling/memory_snapshot`.

## Multi-Node Controls

The launcher defaults to the current single-node pod contract:
`--nnodes 1 --node-rank 0 --rdzv-backend c10d --rdzv-endpoint localhost:0`.
That preserves the existing 8xH100 launch path and the paper-aligned 7B
scale-study recipe.

Multi-node launch is opt-in. For `--nnodes > 1`, provide a stable rendezvous
endpoint and id, for example:

```bash
--nnodes 2 \
--node-rank 0 \
--rdzv-endpoint train-master.example:29400 \
--rdzv-id qwen25-swehero-7b-run-001
```

Each node must use the same run spec, dataset artifact, model assets, bucket
plan, and rendezvous settings, with only `--node-rank` changing per node. The
launcher rejects multi-node settings that still point at `localhost:0` or omit
`--rdzv-id`, so a multi-node attempt cannot silently fall back to a single-node
rendezvous.
