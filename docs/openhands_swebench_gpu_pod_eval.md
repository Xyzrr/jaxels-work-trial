# OpenHands SWE-bench Eval on the GPU Node

Use this path when the eval should run inside the Kubernetes GPU environment
instead of splitting OpenHands/SWE-bench Docker work onto the local machine.

The existing `midtraining-dev` pod is not sufficient by itself unless it is
recreated as privileged. Installing Docker inside the current unprivileged pod
can make `docker info` pass, but `docker run` still fails with
`unshare: operation not permitted`. The verified path below keeps vLLM on
`midtraining-dev` and runs OpenHands plus SWE-bench grading in a privileged
no-GPU eval-driver pod on the same node with the shared `/workspace` hostPath.

## Start vLLM on the GPU Pod

Run this in `midtraining-dev`:

```bash
cd /workspace/jaxels-work-trial
tmux kill-session -t openhands-vllm-7b 2>/dev/null || true
tmux new-session -d -s openhands-vllm-7b \
  "CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  /workspace/venvs/openhands-vllm/bin/vllm serve \
    /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key local-llm \
    --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct \
    --max-model-len 131072 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > /workspace/runlogs/openhands-vllm-7b.log 2>&1"
```

The required vLLM bits for the smoke were:

- `vllm==0.10.2`
- `torch==2.8.0+cu128`
- `transformers==4.55.4`
- native tool calling enabled with the `hermes` parser

## Create the Eval Driver Pod

Apply the privileged driver pod:

```bash
kubectl apply -f manifests/openhands-eval-driver.yaml
kubectl wait -n midtraining --for=condition=Ready pod/openhands-eval-driver --timeout=300s
```

Verify the Docker failure modes that matter for OpenHands:

```bash
kubectl exec -n midtraining openhands-eval-driver -- bash -lc '
for i in $(seq 1 60); do
  docker info >/tmp/docker-info.out 2>/tmp/docker-info.err && break
  sleep 1
done
docker info | grep -E "Server Version|Storage Driver|Docker Root Dir"
docker run --rm hello-world
docker buildx version
'
```

`docker run --rm hello-world` is intentional. It catches the user namespace
failure that `docker info` missed in the unprivileged GPU pod.

## Prepare Python and OpenHands

Run this once in the eval-driver pod. Use a glibc Ubuntu pod, not Alpine;
OpenHands' lock includes Linux wheels that do not resolve correctly on musl.

```bash
kubectl exec -n midtraining openhands-eval-driver -- bash -lc '
set -euo pipefail
/workspace/uv/uv-0.11.16/uv venv /workspace/venvs/openhands-eval-ubuntu-py312 --python 3.12
/workspace/uv/uv-0.11.16/uv pip install \
  --python /workspace/venvs/openhands-eval-ubuntu-py312/bin/python \
  pip poetry
'
```

Then install the OpenHands eval dependencies:

```bash
kubectl exec -n midtraining openhands-eval-driver -- bash -lc '
set -euo pipefail
cd /workspace/eval-runs/OpenHands
export PATH=/workspace/venvs/openhands-eval-ubuntu-py312/bin:$PATH
export POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-ubuntu
export POETRY_CACHE_DIR=/workspace/.cache/poetry-ubuntu
poetry env use /workspace/venvs/openhands-eval-ubuntu-py312/bin/python
poetry install --with evaluation,test --no-root
'
```

If `/workspace/eval-runs/OpenHands` does not exist yet, the eval scaffold will
clone it on the first run. Run the install command after that clone.

## Run the 7B Smoke

Resolve the vLLM pod IP from the cluster:

```bash
GPU_POD_IP=$(kubectl get pod -n midtraining midtraining-dev -o jsonpath='{.status.podIP}')
```

Preflight from inside the eval-driver pod:

```bash
kubectl exec -n midtraining openhands-eval-driver -- bash -lc "
set -euo pipefail
cd /workspace/jaxels-work-trial
export PATH=/workspace/venvs/openhands-eval-ubuntu-py312/bin:\$PATH
export POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-ubuntu
export POETRY_CACHE_DIR=/workspace/.cache/poetry-ubuntu
python scripts/openhands_swebench_eval.py \
  --model-id /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --litellm-model openai/Qwen/Qwen2.5-Coder-7B-Instruct \
  --base-url http://${GPU_POD_IP}:8000/v1 \
  --api-key local-llm \
  --eval-limit 1 \
  --output-dir /workspace/eval-runs/pod-openhands-swebench-verified-pass1 \
  --openhands-dir /workspace/eval-runs/OpenHands \
  --preflight-only
"
```

Run the smoke:

```bash
kubectl exec -n midtraining openhands-eval-driver -- bash -lc "
set -euo pipefail
cd /workspace/jaxels-work-trial
export PATH=/workspace/venvs/openhands-eval-ubuntu-py312/bin:\$PATH
export POETRY_VIRTUALENVS_PATH=/workspace/venvs/poetry-ubuntu
export POETRY_CACHE_DIR=/workspace/.cache/poetry-ubuntu
rm -rf /workspace/eval-runs/pod-openhands-swebench-verified-pass1/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/Qwen2.5-Coder-7B-Instruct_maxiter_100_N_swehero-qwen25-coder7b-pass1
python scripts/openhands_swebench_eval.py \
  --model-id /workspace/assets/hf/Qwen2.5-Coder-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --litellm-model openai/Qwen/Qwen2.5-Coder-7B-Instruct \
  --base-url http://${GPU_POD_IP}:8000/v1 \
  --api-key local-llm \
  --eval-limit 1 \
  --output-dir /workspace/eval-runs/pod-openhands-swebench-verified-pass1 \
  --openhands-dir /workspace/eval-runs/OpenHands
"
```

## Validation Result

On May 24, 2026, the one-instance 7B smoke completed fully inside the cluster:

```json
{
  "resolved": 0,
  "total": 1,
  "pass_at_1": 0.0,
  "report_path": "/workspace/eval-runs/pod-openhands-swebench-verified-pass1/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/Qwen2.5-Coder-7B-Instruct_maxiter_100_N_swehero-qwen25-coder7b-pass1/report.json"
}
```

The sampled instance was `scikit-learn__scikit-learn-13439`. The agent used
real OpenHands tools instead of repeated message text:

```json
{
  "agent_message_actions": 0,
  "agent_tool_actions": 32,
  "instances": 1,
  "loop_errors": 0,
  "tool_action_counts": {
    "edit": 14,
    "finish": 1,
    "read": 1,
    "run": 5,
    "think": 11
  },
  "used_real_tools": true
}
```

The vLLM native tool-call preflight also returned structured
`message.tool_calls` for Qwen2.5-Coder with `tool_choice=required`.

For a literal single-pod setup, recreate `midtraining-dev` as privileged, add
Docker plus `docker-buildx`, and run the same scaffold from that pod with
`--base-url http://127.0.0.1:8000/v1`. Do not rely on the current
unprivileged `midtraining-dev` pod for Dockerized OpenHands/SWE-bench evals.
