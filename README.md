# Jaxels Midtraining Pipeline

Goal: build a good midtraining pipeline for a small coding model, starting with agentic software-engineering trajectories and scaling toward TorchTitan.

## Layout

- `torchtitan/`: upstream TorchTitan checkout.
- `scripts/qwen_agentic_smoke.py`: minimal Qwen coding-model + agentic dataset loss smoke with optional W&B logging.
- `manifests/midtraining-hostpath.yaml`: one-H100 Kubernetes dev pod manifest.
- `docs/model_dataset_choice.md`: current base-model and dataset rationale.

## Current Starting Point

Base model: `Qwen/Qwen2.5-Coder-1.5B`

Dataset: `nvidia/SWE-Hero-openhands-trajectories`

The model is intentionally a small code-pretrained base, not an agentic coding model. The dataset gives OpenHands-style trajectories for SWE-Bench tasks.

## Environment

Create a local `.env` from `.env.example` and fill in secrets locally. Do not commit `.env`.

Expected pod path:

```bash
/workspace/.env
/workspace/qwen_agentic_smoke.py
/workspace/torchtitan
```

## Smoke Run

Inside the GPU pod:

```bash
cd /workspace
source venv/bin/activate
MODEL_ID=Qwen/Qwen2.5-Coder-1.5B \
DATASET_ID=nvidia/SWE-Hero-openhands-trajectories \
MAX_STEPS=12 \
NUM_EXAMPLES=4 \
MAX_LENGTH=768 \
python qwen_agentic_smoke.py
```

The Together smoke run verified loss decreasing from about `2.25` to `0.13` over 12 steps. That only proves plumbing; it is not a useful trained checkpoint.

## Next Work

Build the real midtraining pipeline around:

- robust trajectory serialization for assistant/tool/code-edit turns;
- loss masking to train only useful assistant/action spans;
- checkpoint saving and eval harnesses;
- W&B metrics, samples, and dataset/version metadata;
- TorchTitan integration once a known-good Torch/TorchTitan pin is selected.
