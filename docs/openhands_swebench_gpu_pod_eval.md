# OpenHands SWE-bench Eval on the GPU Pod

The canonical path is a single privileged `midtraining-dev` GPU pod that runs:

- one Qwen2.5-Coder-7B vLLM replica per GPU plus a pod-local router for the
  existing SWE-Hero eval presets; or
- one 8-way tensor-parallel vLLM server without the router for the SWE-Lego
  Qwen3 preset;
- OpenHands inference through the preset-selected eval stack;
- Dockerized SWE-bench grading through either OpenHands' bundled grader path
  or SWE-Lego's vendored `SWE-bench-4.0.4` checkout.

Use `scripts/run_openhands_swebench_eval_pod.sh` from inside that pod. Do not
run OpenHands or SWE-bench from the laptop.

## Configuration Presets

Eval experiment settings live in argparse preset files under `configs/eval/`.
The pod launcher defaults to:

```text
configs/eval/openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args
```

That preset encodes the paper-aligned Qwen2.5-Coder-7B SWE-bench Verified
pass@1 eval: local model path, served model name, OpenHands settings, sampling,
128k YaRN context, vLLM sizing, and the 4096-token per-turn output cap used for
structured tool-call stability. The Python entrypoint also accepts `@...`
argparse files directly, but canonical pod launches should pass presets with
`--config PATH`.

For a different eval, copy a preset, edit the copied file, and swap the
`--config` path. Use primitive CLI flags after the preset only for one-off run
controls such as `--eval-limit`, `--eval-ids`, `--output-dir`,
`--preflight-only`, and `--skip-swebench-eval`. Do not add convenience aliases
for those flags.

Environment variables are reserved for secrets and pod/runtime plumbing:
`LLM_API_KEY`, `WORKSPACE_ROOT`, `VLLM_VENV`, `VLLM_REQUIREMENTS_PATH`,
`VLLM_FORCE_RESTART`, `VLLM_VISIBLE_DEVICES`, `EVAL_VENV`,
`OPENHANDS_EVAL_POETRY_VERSION`, `REQUIRED_GPU_COUNT`,
`SWEHERO_POD_GIT_BRANCH`, and tmux/uv path controls. The API key is env-only;
set `LLM_API_KEY` when the default `local-llm` key is not appropriate.

## Pod Requirement

`midtraining-dev` must be created from `manifests/midtraining-hostpath.yaml`.
That manifest makes the GPU pod privileged, installs Docker plus Buildx, and
persists Docker state under `/workspace/pod-docker-data/midtraining-dev`.

Pod security settings are immutable. If an older `midtraining-dev` pod is
already running without privilege, recreate it before launching this eval:

```bash
kubectl delete pod -n midtraining midtraining-dev
kubectl apply -f manifests/midtraining-hostpath.yaml
kubectl wait -n midtraining --for=condition=Ready pod/midtraining-dev --timeout=600s
```

The launcher intentionally runs both:

```bash
docker run --rm hello-world:latest
docker buildx version
```

`docker info` alone is not enough for this workflow because an unprivileged pod
can report a reachable daemon while still failing container execution.

## Prebuild Runtime Images

Prebuild the per-task OpenHands runtime images before eval:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/prebuild_openhands_swebench_images_pod.sh
'
```

The script runs in tmux session `openhands-swebench-image-prebuild`, logs to
`/workspace/runlogs/openhands-swebench-image-prebuild.log`, and a rerun attaches
to the existing session. Pass `--replace-session` to kill that tmux session and
launch a fresh prebuild instead. It inspects the final OpenHands runtime image
tag before every build, so already-built images are skipped. Missing runtime
images build in parallel; use `--parallel-builds N` to tune concurrency, or
`--parallel-builds 1` to force the old serial behavior. Use `--eval-limit N`
only for a small prebuild smoke.

## Smoke Command

Run a one-instance smoke:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh --eval-limit 1
'
```

For a non-attached launch:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh --eval-limit 1 --no-attach
'
```

The launcher creates a tmux session named
`openhands-swebench-eval-<timestamp>` by default and writes the transcript to
`/workspace/runlogs/<session>.log`.

## Full Pass@1 Command

Run the full SWE-bench Verified split:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh
'
```

## SWE-Lego Eval

The SWE-Lego reproduction preset is:

```text
configs/eval/openhands-swebench-verified-swe-lego-qwen3-8b.args
```

It switches the eval stack from upstream OpenHands to the cloned
`SWE-Lego/SWE-Lego` repository at commit
`94704b69aac886e003660e1e0f69f7de163b284e`. In that checkout the launcher uses
`OpenHands-0.53.0` for inference and installs `SWE-bench-4.0.4` for grading.
The model-serving contract is the SWE-Lego Qwen3 contract:

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

The Qwen3 long-context settings are taken from the model's own `config.json`.
Do not add the Qwen2.5 YaRN `--rope-scaling` override to this preset.

Run SWE-Lego-Qwen3-8B on the 16-task infrastructure check set with:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
eval_ids="django__django-13670,django__django-12663,scikit-learn__scikit-learn-14983,django__django-13279,sphinx-doc__sphinx-7757,django__django-14434,django__django-11999,scikit-learn__scikit-learn-25232,sympy__sympy-15599,astropy__astropy-14309,scikit-learn__scikit-learn-25973,sphinx-doc__sphinx-8551,django__django-12155,sphinx-doc__sphinx-11510,scikit-learn__scikit-learn-13439,django__django-15503"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh \
  --config configs/eval/openhands-swebench-verified-swe-lego-qwen3-8b.args \
  --eval-ids '"$eval_ids"' \
  --run-id swe-lego-qwen3-8b-16task
'
```

After OpenHands finishes, the wrapper converts its `output.jsonl` with the
vendored OpenHands SWE-bench converter and then runs:

```text
python -m swebench.harness.run_evaluation \
  --cache_level instance \
  --timeout 500 \
  --max_workers 10
```

from the SWE-Lego `SWE-bench-4.0.4` environment. The final run report is copied
to `<output-dir>/swebench-results/run_report.json`.

## Base Model Eval Modes

For a full base-model comparison against an SFT checkpoint, run both base
context modes with explicit run IDs:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh \
  --config configs/eval/openhands-swebench-verified-qwen25-coder-7b-base-native-32k.args \
  --run-id qwen25-coder7b-base-native32k-pass1
'
```

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh \
  --config configs/eval/openhands-swebench-verified-qwen25-coder-7b-base-paper-yarn-128k.args \
  --run-id qwen25-coder7b-base-yarn128k-pass1
'
```

The preset's `--context-mode` controls the eval context contract:

- `base-native-32k`: evaluates the released base model inside its native
  32,768-token context window. vLLM starts with `--max-model-len 32768`, no
  YaRN `--rope-scaling`, and OpenHands is configured with
  `max_input_tokens = 32768`. This is the clean "as shipped" base baseline.
  Because OpenHands 0.62.0 sends `max_output_tokens` to the model while its
  `max_input_tokens` field is not a hard truncation mechanism, this mode may
  surface real native-context request failures on longer trajectories.
- `base-paper-yarn-128k`: evaluates the same base model with the paper-style
  131,072-token context budget. vLLM starts with `--max-model-len 131072` plus
  YaRN rope scaling from the native 32k window, and OpenHands is configured
  with `max_input_tokens = 131072`. This is the context-matched base control.
- `paper-yarn-128k`: the default for SFT/checkpoint evals that are meant to
  match the paper's 128k OpenHands setup.
- `swe-lego-qwen3-160k`: evaluates `SWE-Lego/SWE-Lego-Qwen3-8B` with
  OpenHands `max_input_tokens = 147456` and vLLM
  `--max-model-len 163840`, relying on the model-provided Qwen3 long-context
  config instead of a Qwen2.5 YaRN override.

Changing a preset's context mode, vLLM max length, or RoPE scaling changes the
vLLM server contract. The launcher writes a context signature under
`/workspace/runlogs/<vllm-session>.context` and restarts tmux-managed vLLM
servers when the requested signature does not match the live endpoint. If a
non-launcher process is still bound to the port, the launcher stops instead of
silently reusing it.

## Preflight Only

Check the pod runtime, Docker, vLLM, and structured tool calling:

```bash
branch="$(git branch --show-current)"
git push -u origin "$branch"
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
SWEHERO_POD_GIT_BRANCH='"$branch"' scripts/run_openhands_swebench_eval_pod.sh --preflight-only --foreground
'
```

The tool-call preflight must return structured `message.tool_calls` for
Qwen2.5-Coder. If it returns plain assistant text, the eval should not start.
The SWE-Lego Qwen3 preset disables this preflight because its OpenHands config
does not force native tool calling or `tool_choice=required`.

## Preset Defaults

The default eval preset resolves to the experiment settings below. These are
argparse values, not environment variables:

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
port `8090`. `--vllm-agent-tasks-per-server` controls how many concurrent
OpenHands workers are budgeted per vLLM replica; the default preset's full-run
worker count is `8 * 24 = 192`.

Set `--max-output-tokens none` only in an explicit preset or one-off CLI
override for ablations that intentionally reproduce unbounded-output behavior.

Output defaults to:

```text
/workspace/eval-runs/openhands-swebench-verified-pass1/<timestamp>
```

Override with `--output-dir PATH` when a stable path is needed.

## What the Launcher Does

1. Refuses to run on macOS or outside `/workspace`.
2. Refuses to launch a new run unless `SWEHERO_POD_GIT_BRANCH` names the
   current local worktree branch and `/workspace/jaxels-work-trial` is clean,
   checked out to that branch, and fast-forwarded to `origin/<branch>`.
3. Bootstraps and verifies the pinned `uv 0.11.16` binary if the pod does not
   already have it.
4. Starts `dockerd` in a pod tmux session if needed.
5. Verifies Docker by running a real container and checking Buildx.
6. Creates or repairs the Python 3.12 eval environment, including the Poetry
   version requested by the preset (`2.1.3` for the current OpenHands preset,
   `2.1.4` for the SWE-Lego checkout).
7. Creates or repairs the Python 3.12 vLLM environment from
   `requirements/openhands-vllm.txt` before starting any missing vLLM server.
8. Starts the vLLM topology requested by the preset. Current Qwen2.5 presets
   start one replica per GPU. The SWE-Lego Qwen3 preset starts one vLLM process
   with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and tensor parallel size 8.
9. Starts `scripts/openai_vllm_router.py` only when `--vllm-use-router` is set.
10. Syncs the preset-selected OpenHands evaluation dependencies. For SWE-Lego,
    this means the nested `OpenHands-0.53.0`; for current presets, this means
    the configured upstream OpenHands checkout.
11. Runs `scripts/openhands_swebench_eval.py` with the selected endpoint and
    preset-defined OpenHands config. Exact subsets should use `--eval-ids`; the
    wrapper also writes OpenHands' benchmark-local `selected_ids` filter so old
    OpenHands versions run the requested IDs exactly.
12. Prints `agent_tool_use` and the SWE-bench pass@1 summary. For SWE-Lego, the
    summary comes from the vendored `SWE-bench-4.0.4` grader report.

For the 7B smoke, a healthy run should show `used_real_tools: true` and
structured `tool_calls` in the preflight before reporting pass@1. `loop_errors`
then describes the model trajectory quality for the sampled SWE-bench task; it
is not, by itself, evidence that vLLM returned plain text instead of tool calls.
