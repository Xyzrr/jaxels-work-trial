# Project Context

Build a prototype mid-training pipeline for a future 100B-500B open source coding model. The pipeline consumes SWE traces.

# Mode

Ship prototype ASAP: skip post-task validation, parent-branch merges, and PR ceremony unless explicitly asked. When done, commit task changes and push the current branch.

# Current Baseline

Replicate and extend the "direct-to-hero" baseline from "From SWE-ZERO to SWE-HERO" (arXiv:2604.01496):

- Base models: `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-14B-Instruct`, `Qwen2.5-Coder-32B-Instruct`.
- Dataset: use the manually filtered local public approximation at `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`. It starts from one rollout per task/`instance_id` from the oldest public `nvidia/SWE-Hero-openhands-trajectories` revision (`5b2ed21270ad773a50163e2999c510f0cbb92cfa`) because that revision has the most public tasks. The raw one-rollout approximation has 12,633 selected rows, then `scripts/refresh_swehero_context_capped_one_rollout.py` replaces over-128k Qwen/OpenHands rows with same-task rollouts that fit the 131,072-token shifted context using the same selection rank, and excludes tasks with no fitting accepted rollout. The current local artifact has 12,617 selected rows: 23 replacements and 16 exclusions from the raw artifact. This is a best-effort match for the paper's "one rollout per task" setup, not the exact internal `~13.2k` paper manifest.
- Eval: SWE-bench Verified through the OpenHands harness.
- Paper caveat: the reported direct-to-hero ablation is for 32B; 7B and 14B direct-to-hero runs are a scale-study extension unless a paper table proves otherwise.

# Primary Workflow Docs

- Training jobs: `docs/swehero_torchtitan_pod.md`.
- Evals: `docs/openhands_swebench_gpu_pod_eval.md`.

# Local Resources

- `torchtitan/`: fully vendored TorchTitan base for distributed training.
- `manifests/midtraining-hostpath.yaml`: GPU pod manifest.
- `tmp/pod-creds/`: local credentials for the GPU pod, including pod login credentials; never commit anything under this directory.
- `tmp/pod-creds/kubeconfig.yaml`: kubeconfig for the GPU pod. Use this file for Kubernetes access, for example with `KUBECONFIG=tmp/pod-creds/kubeconfig.yaml` or `kubectl --kubeconfig tmp/pod-creds/kubeconfig.yaml ...`.

# Working Rules

- Re-read the paper before changing pipeline assumptions.
- Use `tmp/pod-creds/kubeconfig.yaml` whenever running `kubectl`, `helm`, or other Kubernetes tools for this project.
- Run both training and eval workloads on the GPU pod, not on the local machine. Local execution is only for editing, lightweight inspection, tests that do not require the training/eval stack, and Kubernetes orchestration.
- Prefer TorchTitan's existing extension points and scripts over ad-hoc training code.
- Keep secrets, pod credentials, `.env`, checkpoints, datasets, and generated run artifacts out of git.
- Preserve enough metadata to reproduce each run: model, dataset revision, tokenizer/chat template, sequence length, loss masking, LR schedule, batch size, hardware, commit, and eval harness revision.
- Verify meaningful behavior with automated tests or a concrete dry run whenever feasible.
