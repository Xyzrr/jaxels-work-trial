# Project Context

## High-Level Overview

Build a prototype mid-training pipeline for a future 100B-500B open source coding model trained from SWE traces.

This repo began as a SWE-Hero reproduction and is becoming a general experiment pipeline. Do not add new hardcoded SWE-Hero assumptions to shared launchers, config parsing, data plumbing, or eval orchestration. Keep experiment-specific behavior in preset files or narrowly named adapters selected by explicit flags such as `--eval-stack`, `--context-mode`, or `@configs/...`.

## Reader Vocabulary

- SFT: supervised fine-tuning, where prompt and observation text usually remain in context but assistant/tool-action target text is used for loss.
- SWE traces: coding-agent execution records containing repository state, prompts, model actions, tools, observations, and patches, not ordinary prompt/answer pairs.
- Context window: the token budget read by the model. Qwen2.5-Coder is native 32k; direct-to-hero presets intentionally use 128k-class YaRN long-context settings.
- One rollout per task: a data-weighting choice. Multiple attempts for one `instance_id` would overweight that task and diverge from the paper setup.
- Eval/runtime stack: vLLM serves an OpenAI-compatible API, OpenHands runs the coding agent, and SWE-bench grades patches. Do not treat these as interchangeable pieces.

## Canonical Workflow Docs

- Training jobs: [`docs/swehero_torchtitan_pod.md`](docs/swehero_torchtitan_pod.md)
- Evals: [`docs/openhands_swebench_gpu_pod_eval.md`](docs/openhands_swebench_gpu_pod_eval.md)
- Local Python/uv development: [`docs/python_uv_project.md`](docs/python_uv_project.md)
- Shell-to-Python conversion proof: [`docs/script_conversion_experiments.md`](docs/script_conversion_experiments.md)
- SWE-Hero public dataset caveat: [`notes/swe-hero-dataset-discrepancy.md`](notes/swe-hero-dataset-discrepancy.md)

## Baseline Snapshot

The original baseline replicates and extends the "direct-to-hero" setup from "From SWE-ZERO to SWE-HERO" (arXiv:2604.01496).

- Base models: `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-14B-Instruct`, and `Qwen2.5-Coder-32B-Instruct`.
- Dataset: `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`, a local public one-rollout approximation. The current artifact has 12,617 selected rows after 128k-context fitting; exact provenance lives in the dataset caveat note.
- Paper caveat: the reported direct-to-hero ablation is for 32B. The 7B and 14B runs are scale-study extensions unless a paper table proves otherwise.
- Eval support: current OpenHands SWE-Hero-style evals use presets under `configs/eval/`; SWE-Lego evals use the vendored stack and serving contract named in the eval doc.

## Operating Mode

Ship the prototype quickly. Skip post-task full validation, parent-branch merges, and PR ceremony unless explicitly asked. When the full task is done, commit task changes and push the current branch.

## Local Resources

- `torchtitan/`: fully vendored TorchTitan base. Do not touch it unless explicitly asked.
- `manifests/midtraining-hostpath.yaml`: GPU pod manifest.
- `tmp/pod-creds/`: local GPU pod credentials; never commit anything under this directory.
- `tmp/pod-creds/kubeconfig.yaml`: kubeconfig for `kubectl`, `helm`, and related project pod commands.

## Working Rules

- Re-read the relevant paper or source workflow before changing pipeline assumptions. For SWE-Lego behavior, inspect the vendored SWE-Lego workflow and preset instead of assuming upstream OpenHands behavior.
- Use pinned `uv 0.11.16` and Python `3.12.13` from `.python-version`; run project tools through `uv run`.
- Create project automation as Python scripts, not bash scripts. Workstation scripts should use `#!/usr/bin/env -S uv run python`; pod bootstrap scripts may use `#!/usr/bin/env python3` only when they cannot assume the project venv exists.
- Do not add new bash scripts outside vendored third-party trees or generated command-record artifacts.
- The default local validation command is `uv run scripts/validate.py`, but the current operating mode skips full post-task validation unless explicitly requested.
- Run training and eval workloads on the GPU pod. Local execution is for editing, lightweight inspection, non-training tests, and Kubernetes orchestration.
- Prefer TorchTitan extension points and existing project scripts over ad-hoc training code.
- Keep secrets, pod credentials, `.env`, checkpoints, datasets, and generated run artifacts out of git.
- Preserve reproducibility metadata for runs: model, dataset revision, tokenizer/chat template, sequence length, loss masking, LR schedule, batch size, hardware, commit, and eval harness revision.
- Verify meaningful behavior with automated tests or a concrete dry run whenever feasible.

## Configuration Principles

- Every public setting has exactly one source: CLI/preset argument or environment variable, never both.
- Use argparse `@preset` files for experiment and reproducibility settings. Paper-faithful recipes belong in swappable preset files under `configs/`.
- Reserve environment variables for secrets, credentials, pod/runtime plumbing, and process supervision. Do not use them for experiment settings such as model, dataset, context, optimizer, eval harness, vLLM sizing, or sampling.
- Avoid aliases and convenience synonyms. Prefer primitive flags such as `--eval-limit 1` over named smoke/full shortcuts.
- When changing training or eval config, update the workflow docs with the canonical preset-based command and preserve runnable behavior through preset contents or explicit CLI flags.
- Existing names such as `SWEHERO_POD_GIT_BRANCH` are legacy compatibility names. Use neutral names for new shared controls.
