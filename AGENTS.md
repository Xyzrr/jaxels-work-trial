# Project Context

Build a prototype mid-training pipeline for a future 100B-500B open source coding model. The pipeline consumes SWE traces.

The project started from a SWE-Hero reproduction, but it is now transitioning into a more general experiment pipeline. Do not introduce new hardcoded SWE-Hero assumptions into shared launchers, config parsing, data plumbing, or eval orchestration. Prefer preset-driven experiment definitions that can support additional trace sources, models, serving topologies, OpenHands versions, and graders.

# ML Reader Notes

- SFT means supervised fine-tuning: continue training a pretrained model so it imitates selected target text. In this repo, the target text is assistant/tool-action content from coding-agent traces; prompt text and tool observations usually stay in the context but are masked out of the loss.
- SWE traces are execution records from coding tasks: repository state, prompts, model actions, shell/editor/tool calls, observations, and produced patches. They are not ordinary prompt/answer pairs, so filtering, ordering, and truncation decisions change what the model learns.
- A context window is the number of tokens the model can read at once. Qwen2.5-Coder is native 32k context, while the direct-to-hero training/eval presets intentionally use 128k-class YaRN long-context settings. Changing context mode, RoPE/YaRN scaling, or vLLM max length changes the ML experiment.
- One rollout per task is a data-weighting choice. Keeping multiple attempts for the same `instance_id` would over-weight that task during SFT and would no longer match the paper's described setup.
- vLLM serves the model behind an OpenAI-compatible API; OpenHands is the coding-agent harness; SWE-bench is the grader. Do not treat those as interchangeable runtime pieces when changing eval presets.

# Mode

Ship prototype ASAP: skip post-task validation, parent-branch merges, and PR ceremony unless explicitly asked. When done, commit task changes and push the current branch.

# Current Baselines And Eval Support

The original training baseline is to replicate and extend the "direct-to-hero" baseline from "From SWE-ZERO to SWE-HERO" (arXiv:2604.01496):

- Base models: `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-14B-Instruct`, `Qwen2.5-Coder-32B-Instruct`.
- Dataset: use the manually filtered local public approximation at `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`. It starts from one rollout per task/`instance_id` from the oldest public `nvidia/SWE-Hero-openhands-trajectories` revision (`5b2ed21270ad773a50163e2999c510f0cbb92cfa`) because that revision has the most public tasks. The raw one-rollout approximation has 12,633 selected rows, then `scripts/refresh_swehero_context_capped_one_rollout.py` replaces over-128k Qwen/OpenHands rows with same-task rollouts that fit the 131,072-token shifted context using the same selection rank, and excludes tasks with no fitting accepted rollout. The current local artifact has 12,617 selected rows: 23 replacements and 16 exclusions from the raw artifact. This is a best-effort match for the paper's "one rollout per task" setup, not the exact internal `~13.2k` paper manifest.
- Paper caveat: the reported direct-to-hero ablation is for 32B; 7B and 14B direct-to-hero runs are a scale-study extension unless a paper table proves otherwise.

Current eval support includes both:

- SWE-Hero/current OpenHands evals: upstream OpenHands through `scripts/run_openhands_swebench_eval_pod.py`, with the Qwen2.5-Coder presets under `configs/eval/`.
- SWE-Lego evals: `configs/eval/openhands-swebench-verified-swe-lego-qwen3-8b.args` uses the vendored `SWE-Lego/SWE-Lego` stack at commit `94704b69aac886e003660e1e0f69f7de163b284e`, nested `OpenHands-0.53.0`, vendored `SWE-bench-4.0.4`, and the single 8-GPU tensor-parallel `SWE-Lego/SWE-Lego-Qwen3-8B` serving contract.

# Primary Workflow Docs

- Training jobs: `docs/swehero_torchtitan_pod.md`.
- Evals: `docs/openhands_swebench_gpu_pod_eval.md`.
- Local Python/uv development: `docs/python_uv_project.md`.
- Shell-to-Python conversion proof: `docs/script_conversion_experiments.md`.

# Local Resources

- `torchtitan/`: fully vendored TorchTitan base for distributed training.
- `manifests/midtraining-hostpath.yaml`: GPU pod manifest.
- `tmp/pod-creds/`: local credentials for the GPU pod, including pod login credentials; never commit anything under this directory.
- `tmp/pod-creds/kubeconfig.yaml`: kubeconfig for the GPU pod. Use this file for Kubernetes access, for example with `KUBECONFIG=tmp/pod-creds/kubeconfig.yaml` or `kubectl --kubeconfig tmp/pod-creds/kubeconfig.yaml ...`.

# Working Rules

- Re-read the relevant paper or source workflow before changing pipeline assumptions. For SWE-Hero-specific training assumptions, use the SWE-Hero paper. For SWE-Lego eval behavior, inspect the vendored SWE-Lego workflow and preset rather than assuming upstream OpenHands behavior.
- This is a `uv` project. Use pinned `uv 0.11.16` and local Python `3.12.13` from `.python-version`; run project tools through `uv run`.
- Create project automation as Python scripts, not bash scripts. Workstation/local scripts that rely on the project environment should use `#!/usr/bin/env -S uv run python` and be runnable as `uv run scripts/name.py`. Pod bootstrap scripts may use `#!/usr/bin/env python3` only when they bootstrap or repair `uv`/Python and therefore cannot assume the project venv exists yet.
- Do not add new bash scripts outside vendored third-party trees or generated command-record artifacts. If a workflow needs shell-like orchestration, implement it in Python with explicit subprocess calls and tests.
- Run `uv run scripts/validate.py` for local validation. It runs pytest, Ruff lint, and Ruff format checks in parallel while keeping each process's stdout/stderr grouped.
- Use `tmp/pod-creds/kubeconfig.yaml` whenever running `kubectl`, `helm`, or other Kubernetes tools for this project.
- Run both training and eval workloads on the GPU pod, not on the local machine. Local execution is only for editing, lightweight inspection, tests that do not require the training/eval stack, and Kubernetes orchestration.
- Prefer TorchTitan's existing extension points and scripts over ad-hoc training code.
- Do not touch `torchtitan/`. It is vendored and intentionally excluded from project `uv`, pytest, Ruff, and CI configuration.
- Keep secrets, pod credentials, `.env`, checkpoints, datasets, and generated run artifacts out of git.
- Preserve enough metadata to reproduce each run: model, dataset revision, tokenizer/chat template, sequence length, loss masking, LR schedule, batch size, hardware, commit, and eval harness revision.
- Verify meaningful behavior with automated tests or a concrete dry run whenever feasible.
- When generalizing the pipeline, keep experiment-specific behavior in preset files or narrowly named adapters. Shared wrappers should select behavior by explicit arguments such as `--eval-stack`, `--context-mode`, or `@configs/...` files, not by inferring from model names or SWE-Hero legacy defaults.

# Configuration Principles

- Every public setting must have exactly one configuration source: either a CLI/preset argument or an environment variable, never both.
- Use argparse `@preset` files for experiment and reproducibility settings. Paper-faithful recipes belong in swappable preset files under `configs/`, not hidden launcher defaults or ambient env defaults.
- Reserve environment variables for secrets, credentials, pod/runtime plumbing, and process supervision. Do not use env vars for model, dataset, context, optimizer, eval harness, vLLM sizing, sampling, or other experiment settings.
- Avoid aliases and convenience synonyms. Prefer primitive flags such as `--eval-limit 1` over named smoke/full shortcuts.
- When adding or changing training/eval config, update the workflow docs with the canonical preset-based command and preserve the existing runnable behavior through preset contents or explicit CLI flags.
- Existing names such as `SWEHERO_POD_GIT_BRANCH` are legacy compatibility names. Do not add new SWE-Hero-prefixed knobs for general functionality; use neutral names for new shared controls.
