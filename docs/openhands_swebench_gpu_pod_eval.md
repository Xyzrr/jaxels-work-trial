# OpenHands And SWE-Lego SWE-bench Eval on the GPU Pod

## Overview

This is the shared GPU-pod runbook for SWE-bench-style coding evals. It is no
longer SWE-Hero-only: presets choose the OpenHands stack, model-serving
topology, context contract, and grader.

Use this document when launching or reproducing evals. For project-wide
vocabulary and configuration rules, see [`../AGENTS.md`](../AGENTS.md).

The canonical runtime is the privileged `midtraining-dev` pod. Do not run
OpenHands, vLLM, or SWE-bench grading from the laptop.

## Quick Commands

Prebuild OpenHands runtime images:

```bash
scripts/run_midtraining_pod.py prebuild
```

Run a one-instance smoke:

```bash
scripts/run_midtraining_pod.py eval --eval-limit 1
```

Run the full default SWE-bench Verified pass@1 eval:

```bash
scripts/run_midtraining_pod.py eval
```

Run preflight checks only:

```bash
scripts/run_midtraining_pod.py eval --preflight-only --foreground
```

Launch without attaching:

```bash
scripts/run_midtraining_pod.py --no-tty eval --eval-limit 1 --no-attach
```

## Runtime Shape

`scripts/run_midtraining_pod.py eval` and
`scripts/run_midtraining_pod.py prebuild` are workstation `uv` Python
entrypoints. The meta-wrapper pushes the current clean branch, enters
`midtraining-dev` with `tmp/pod-creds/kubeconfig.yaml`, sets the legacy
`SWEHERO_POD_GIT_BRANCH` runtime variable inside the pod, and starts the
lower-level pod wrapper from `/workspace/jaxels-work-trial`.

The pod then runs one of these serving contracts:

- Current Qwen2.5 presets: one vLLM replica per GPU plus a pod-local router.
- SWE-Lego Qwen3 preset: one 8-way tensor-parallel vLLM server, no router.

Inference runs through the preset-selected OpenHands stack. Grading runs
through either OpenHands' bundled SWE-bench path or SWE-Lego's vendored
`SWE-bench-4.0.4` checkout.

## Presets

Eval settings live in `configs/eval/*.args`. The default preset is:

```text
configs/eval/openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args
```

It encodes the paper-aligned Qwen2.5-Coder-7B SWE-bench Verified pass@1 eval:
model path, served model name, OpenHands settings, sampling, 128k YaRN context,
vLLM sizing, and the 4096-token per-turn output cap used for structured
tool-call stability.

Supported presets:

| Preset | Purpose |
| --- | --- |
| `openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args` | Default Qwen2.5-Coder-7B Instruct eval through upstream OpenHands. |
| `openhands-swebench-verified-qwen25-coder-7b-base-native-32k.args` | Released Qwen2.5-Coder-7B base model in native 32k context. |
| `openhands-swebench-verified-qwen25-coder-7b-base-paper-yarn-128k.args` | Released Qwen2.5-Coder-7B base model with paper-style 128k YaRN serving. |
| `openhands-swebench-verified-swe-lego-qwen3-8b.args` | SWE-Lego Qwen3-8B reproduction through the vendored SWE-Lego stack. |

For a new eval, copy a preset, edit the copy, and pass it with `--config PATH`.
Use primitive CLI flags after the preset only for run controls such as
`--eval-limit`, `--eval-ids`, `--output-dir`, `--preflight-only`, and
`--skip-swebench-eval`.

New shared launcher behavior must be selected by explicit preset arguments such
as `--eval-stack`, `--context-mode`, `--vllm-server-count`, and grader flags.
Do not infer stack behavior from model names or historical SWE-Hero defaults.

Environment variables are only for secrets and pod/runtime plumbing:
`LLM_API_KEY`, `WORKSPACE_ROOT`, `VLLM_VENV`, `VLLM_REQUIREMENTS_PATH`,
`VLLM_FORCE_RESTART`, `VLLM_VISIBLE_DEVICES`, `EVAL_VENV`,
`VLLM_NCCL_CUMEM_ENABLE`, `OPENHANDS_EVAL_POETRY_VERSION`,
`REQUIRED_GPU_COUNT`, and tmux/uv path controls.

`LLM_API_KEY` is env-only; set it when the default `local-llm` key is not
appropriate. `SWEHERO_POD_GIT_BRANCH` remains a pod-side legacy compatibility
name; do not add new SWE-Hero-prefixed env vars for general eval controls.

## Pod Requirement

`midtraining-dev` must be created from `manifests/midtraining-hostpath.yaml`.
The manifest makes the pod privileged, installs Docker plus Buildx, and
persists Docker state under `/workspace/pod-docker-data/midtraining-dev`.

Pod security settings are immutable. If an older pod is already running without
privilege, recreate it:

```bash
KUBECONFIG=tmp/pod-creds/kubeconfig.yaml \
  kubectl delete pod -n midtraining midtraining-dev
KUBECONFIG=tmp/pod-creds/kubeconfig.yaml \
  kubectl apply -f manifests/midtraining-hostpath.yaml
KUBECONFIG=tmp/pod-creds/kubeconfig.yaml \
  kubectl wait -n midtraining --for=condition=Ready pod/midtraining-dev --timeout=600s
```

The launcher verifies Docker with both:

```bash
docker run --rm hello-world:latest
docker buildx version
```

`docker info` alone is insufficient because an unprivileged pod can expose a
daemon while still failing container execution.

## Image Prebuild

Prebuild runtime images before eval:

```bash
scripts/run_midtraining_pod.py prebuild
```

For SWE-Lego, prebuild against its vendored `OpenHands-0.53.0`:

```bash
scripts/run_midtraining_pod.py prebuild \
  --config configs/eval/openhands-swebench-verified-swe-lego-qwen3-8b.args
```

Prebuild runs in tmux session `openhands-swebench-image-prebuild` and logs to
`/workspace/runlogs/openhands-swebench-image-prebuild.log`. A rerun attaches to
the existing session. Pass `--replace-session` to kill that session and start a
fresh prebuild.

The launcher skips already-built final OpenHands runtime image tags. Missing
images build in parallel; use `--parallel-builds N` to tune concurrency or
`--parallel-builds 1` for serial behavior. Use `--eval-limit N` only for a
small prebuild smoke.

The launcher writes
`/workspace/runlogs/openhands-swebench-image-prebuild.context.json`. Reruns
with the same session name must match that stored launch context unless
`--replace-session` is set.

## Default Eval Runs

Smoke:

```bash
scripts/run_midtraining_pod.py eval --eval-limit 1
```

Full pass@1:

```bash
scripts/run_midtraining_pod.py eval
```

The eval launcher creates tmux session `openhands-swebench-eval-<timestamp>` by
default and writes `/workspace/runlogs/<session>.log`.

For a healthy Qwen2.5 smoke, preflight should show `used_real_tools: true` and
structured `tool_calls` before pass@1 is reported. Later `loop_errors` describe
the sampled model trajectory quality; they are not by themselves evidence that
vLLM returned plain text instead of tool calls.

## SWE-Lego Eval

The SWE-Lego preset is:

```text
configs/eval/openhands-swebench-verified-swe-lego-qwen3-8b.args
```

It switches from upstream OpenHands to the cloned `SWE-Lego/SWE-Lego` repo at
commit `94704b69aac886e003660e1e0f69f7de163b284e`. That stack uses
`OpenHands-0.53.0` for inference and vendored `SWE-bench-4.0.4` for grading.

SWE-Lego Qwen3 serving contract:

```text
--model-id SWE-Lego/SWE-Lego-Qwen3-8B
--served-model-name Qwen/Qwen3-8B
--context-mode swe-lego-qwen3-160k
--temperature 0.0
--max-input-tokens 147456
--max-output-tokens 16384
--vllm-server-count 1
--no-vllm-use-router
--vllm-tensor-parallel-size 8
--vllm-max-model-len 163840
--vllm-rope-scaling none
--vllm-max-num-seqs 24
--omit-native-tool-calling-config
--no-tool-call-preflight
--num-workers 24
--swebench-cache-level instance
--swebench-timeout 500
--swebench-max-workers 10
```

Do not add the Qwen2.5 YaRN `--rope-scaling` override to this preset; the
Qwen3 long-context settings come from the model `config.json`.

SWE-Lego intentionally differs from current Qwen2.5 OpenHands presets by not
forcing native tool calling or `tool_choice=required`, preserving
`NUM_WORKERS=24`, and grading with `--cache_level instance --timeout 500
--max_workers 10`. Treat those as reproduction-contract settings, not global
defaults.

For multi-GPU vLLM servers, the launcher sets `NCCL_CUMEM_ENABLE=1` by default
because the pod's `/dev/shm` is too small for NCCL per-rank shared-memory
segments when vLLM disables cuMem.

Run the 16-task SWE-Lego infrastructure check:

```bash
eval_ids="django__django-13670,django__django-12663,scikit-learn__scikit-learn-14983,django__django-13279,sphinx-doc__sphinx-7757,django__django-14434,django__django-11999,scikit-learn__scikit-learn-25232,sympy__sympy-15599,astropy__astropy-14309,scikit-learn__scikit-learn-25973,sphinx-doc__sphinx-8551,django__django-12155,sphinx-doc__sphinx-11510,scikit-learn__scikit-learn-13439,django__django-15503"
scripts/run_midtraining_pod.py eval \
  --config configs/eval/openhands-swebench-verified-swe-lego-qwen3-8b.args \
  --eval-ids "$eval_ids" \
  --run-id swe-lego-qwen3-8b-16task
```

After OpenHands finishes, the wrapper converts `output.jsonl` with the vendored
OpenHands SWE-bench converter and runs this from the SWE-Lego
`SWE-bench-4.0.4` environment:

```text
python -m swebench.harness.run_evaluation \
  --cache_level instance \
  --timeout 500 \
  --max_workers 10
```

The final report is copied to `<output-dir>/swebench-results/run_report.json`.

## Base Model Eval Modes

Run both base context modes when comparing a base model against an SFT
checkpoint:

```bash
scripts/run_midtraining_pod.py eval \
  --config configs/eval/openhands-swebench-verified-qwen25-coder-7b-base-native-32k.args \
  --run-id qwen25-coder7b-base-native32k-pass1
```

```bash
scripts/run_midtraining_pod.py eval \
  --config configs/eval/openhands-swebench-verified-qwen25-coder-7b-base-paper-yarn-128k.args \
  --run-id qwen25-coder7b-base-yarn128k-pass1
```

Context modes:

| Mode | Contract |
| --- | --- |
| `base-native-32k` | Released base model at native 32,768-token context: `--max-model-len 32768`, no YaRN, OpenHands `max_input_tokens = 32768`. This may expose real native-context request failures on longer trajectories. |
| `base-paper-yarn-128k` | Same base model with paper-style 131,072-token context and YaRN scaling from native 32k. This is the context-matched base control. |
| `paper-yarn-128k` | Default for SFT/checkpoint evals meant to match the paper's 128k OpenHands setup. |
| `swe-lego-qwen3-160k` | SWE-Lego Qwen3 with OpenHands `max_input_tokens = 147456` and vLLM `--max-model-len 163840`, using model-provided long-context config. |

Changing context mode, vLLM max length, or RoPE scaling changes the vLLM server
contract. The launcher writes `/workspace/runlogs/<vllm-session>.context` and
restarts tmux-managed vLLM servers when the requested signature does not match
the live endpoint. If a non-launcher process owns the port, the launcher stops
instead of reusing it.

## Preflight

Run:

```bash
scripts/run_midtraining_pod.py eval --preflight-only --foreground
```

The Qwen2.5 tool-call preflight must return structured `message.tool_calls`. If
it returns plain assistant text, the eval should not start because OpenHands
cannot reliably execute the model's intended shell/editor action.

The SWE-Lego Qwen3 preset disables this preflight because its OpenHands config
does not force native tool calling or `tool_choice=required`.

## Default Preset Values

The default preset resolves to these argparse values, not environment
variables:

```text
--eval-stack openhands
--model-id /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct
--served-model-name Qwen/Qwen2.5-Coder-7B-Instruct
--litellm-model openai/Qwen/Qwen2.5-Coder-7B-Instruct
--context-mode paper-yarn-128k
--max-output-tokens 4096
--temperature 0.7
--top-p 0.8
--top-k 20
--tool-choice required
--max-iterations 100
--num-workers 192
--openhands-ref 0.62.0
--vllm-max-model-len 131072
--vllm-rope-scaling auto
--vllm-server-count 8
--vllm-use-router
--vllm-agent-tasks-per-server 24
--vllm-router-port 8090
--vllm-gpu-memory-utilization 0.90
--vllm-dtype bfloat16
--vllm-distributed-executor-backend mp
--vllm-enforce-eager
```

The launcher starts replicas on ports `8000..8007` and exposes the router on
port `8090`. `--vllm-agent-tasks-per-server` controls concurrent OpenHands
workers per vLLM replica; the default full-run worker count is `8 * 24 = 192`.

Set `--max-output-tokens none` only in an explicit preset or one-off CLI
override for ablations that intentionally reproduce unbounded-output behavior.

Default output directory:

```text
/workspace/eval-runs/openhands-swebench-verified-pass1/<timestamp>
```

Use `--output-dir PATH` when a stable path is needed.

## Launcher Internals

The launcher:

1. Refuses to push if the local checkout has uncommitted changes, pushes the
   selected branch, and enters `midtraining-dev` with `kubectl exec`.
2. Runs the shared startup guard, which requires `/workspace/jaxels-work-trial`
   to be clean, checked out to `SWEHERO_POD_GIT_BRANCH`, and fast-forwarded to
   `origin/<branch>`.
3. Refuses runtime work on macOS or outside `/workspace`.
4. Bootstraps and verifies pinned `uv 0.11.16` if needed.
5. Starts `dockerd` in pod tmux if needed.
6. Verifies Docker by running a real container and checking Buildx.
7. Creates or repairs the Python 3.12 eval environment, including preset
   Poetry version (`2.1.3` for current OpenHands, `2.1.4` for SWE-Lego).
8. Creates or repairs the Python 3.12 vLLM environment from
   `requirements/openhands-vllm.txt`.
9. Starts the preset vLLM topology: one replica per GPU for current Qwen2.5
   presets, or one tensor-parallel process over GPUs `0..7` for SWE-Lego Qwen3.
10. Starts `scripts/openai_vllm_router.py` only when `--vllm-use-router` is set.
11. Syncs preset-selected OpenHands dependencies: upstream OpenHands for
    current presets or nested `OpenHands-0.53.0` for SWE-Lego.
12. Runs `scripts/openhands_swebench_eval.py` with the selected endpoint and
    preset-defined OpenHands config. Exact subsets should use `--eval-ids`; the
    wrapper also writes OpenHands' benchmark-local `selected_ids` filter so old
    OpenHands versions run the requested IDs exactly.
13. Prints `agent_tool_use` and the SWE-bench pass@1 summary. For SWE-Lego, the
    summary comes from the vendored `SWE-bench-4.0.4` grader report.
