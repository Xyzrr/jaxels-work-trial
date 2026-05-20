# Midtraining Starting Point

Date: 2026-05-20

## Base model

Use `Qwen/Qwen2.5-Coder-1.5B` as the small open-source starting point.

Why:
- It is a code-specific base model small enough for fast iteration on one H100.
- The Hugging Face model card identifies the training stage as `Pretraining`, not instruction tuning.
- Qwen's newer Qwen3-Coder family is a poor fit for this experiment because its own repo describes it as agentically trained at scale with environment interaction and reinforcement learning.

## Agentic coding dataset

Use `nvidia/SWE-Hero-openhands-trajectories` first.

Why:
- It contains 34k OpenHands agent trajectories for SWE-Bench style software engineering tasks.
- The schema exposes complete trajectories with `system`, `user`, `assistant`, and `tool` roles plus final patches.
- It is much smaller and easier to smoke-test than `AlienKevin/SWE-ZERO-12M-trajectories`, which is useful later for scale.

## TorchTitan note

TorchTitan is set up as the distributed pretraining/midtraining framework target, but its native public examples are centered on Llama-family model definitions and checkpoint formats. The first loss-decrease smoke here uses Hugging Face Transformers against Qwen directly, while TorchTitan is installed/cloned separately as the production-scale training path to adapt next.
