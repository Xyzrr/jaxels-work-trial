# OpenHands SWE-bench Eval on the GPU Pod

The canonical path is a single privileged `midtraining-dev` GPU pod that runs:

- one Qwen2.5-Coder-7B vLLM replica per GPU;
- a pod-local OpenAI-compatible router across those vLLM replicas;
- OpenHands inference;
- Dockerized SWE-bench grading.

Use `scripts/run_openhands_swebench_eval_pod.sh` from inside that pod. Do not
run OpenHands or SWE-bench from the laptop.

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

## Smoke Command

Run a one-instance smoke:

```bash
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
scripts/run_openhands_swebench_eval_pod.sh --smoke
'
```

For a non-attached launch:

```bash
kubectl exec -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
scripts/run_openhands_swebench_eval_pod.sh --smoke --no-attach
'
```

The launcher creates a tmux session named
`openhands-swebench-eval-<timestamp>` by default and writes the transcript to
`/workspace/runlogs/<session>.log`.

## Full Pass@1 Command

Run the full SWE-bench Verified split:

```bash
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
scripts/run_openhands_swebench_eval_pod.sh --full
'
```

## Preflight Only

Check the pod runtime, Docker, vLLM, and structured tool calling:

```bash
kubectl exec -it -n midtraining midtraining-dev -- bash -lc '
cd /workspace/jaxels-work-trial
scripts/run_openhands_swebench_eval_pod.sh --preflight-only --foreground
'
```

The tool-call preflight must return structured `message.tool_calls` for
Qwen2.5-Coder. If it returns plain assistant text, the eval should not start.

## Defaults

The launcher defaults are:

```text
MODEL_ID=/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct
SERVED_MODEL_NAME=Qwen/Qwen2.5-Coder-7B-Instruct
LITELLM_MODEL=openai/Qwen/Qwen2.5-Coder-7B-Instruct
LLM_API_KEY=local-llm
VLLM_VENV=/workspace/venvs/openhands-vllm
EVAL_VENV=/workspace/venvs/openhands-eval-pod-py312
OPENHANDS_DIR=/workspace/eval-runs/OpenHands
OPENHANDS_REF=0.62.0
MAX_OUTPUT_TOKENS=8192
VLLM_ENFORCE_EAGER=1
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_PIPELINE_PARALLEL_SIZE=1
VLLM_SERVER_COUNT=8
VLLM_AGENT_TASKS_PER_SERVER=24
VLLM_ROUTER_PORT=8090
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_DTYPE=bfloat16
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
REQUIRED_GPU_COUNT=8
```

The launcher starts replicas on ports `8000..8007` and exposes the router on
port `8090`. `VLLM_AGENT_TASKS_PER_SERVER` controls how many concurrent
OpenHands workers are budgeted per vLLM replica; the default full-run worker
count is `8 * 24 = 192`.

Set `MAX_OUTPUT_TOKENS=none` only for ablations that intentionally reproduce
unbounded-output behavior.

Output defaults to:

```text
/workspace/eval-runs/openhands-swebench-verified-pass1/<timestamp>
```

Override with `--output-dir PATH` when a stable path is needed.

## What the Launcher Does

1. Refuses to run on macOS or outside `/workspace`.
2. Starts `dockerd` in a pod tmux session if needed.
3. Verifies Docker by running a real container and checking Buildx.
4. Creates the Python 3.12 eval environment with the pinned `uv` binary.
5. Starts one vLLM tmux session per GPU if the endpoints are not already up.
   The default uses eager execution plus the 8192-token output cap for
   structured tool-call decoding stability.
6. Starts `scripts/openai_vllm_router.py` in a pod tmux session, routing to the
   per-GPU vLLM replicas with the configured per-replica concurrency limit.
7. Installs the OpenHands evaluation dependencies.
8. Runs `scripts/openhands_swebench_eval.py` with the router as the model
   endpoint, `tool_choice=required`, bounded per-turn output, and
   `VLLM_SERVER_COUNT * VLLM_AGENT_TASKS_PER_SERVER` workers for full runs.
9. Prints `agent_tool_use` and the SWE-bench pass@1 summary.

For the 7B smoke, a healthy run should show `used_real_tools: true` and
structured `tool_calls` in the preflight before reporting pass@1. `loop_errors`
then describes the model trajectory quality for the sampled SWE-bench task; it
is not, by itself, evidence that vLLM returned plain text instead of tool calls.
