# Project Context

Build a prototype mid-training pipeline for a future 100B-500B open source coding model. The pipeline consumes SWE traces.

# Mode

Ship prototype ASAP: skip post-task validation, parent-branch merges, and PR ceremony unless explicitly asked. When done, commit task changes and push the current branch.

# Current Baseline

Replicate and extend the "direct-to-hero" baseline from "From SWE-ZERO to SWE-HERO" (arXiv:2604.01496):

- Base models: `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-14B-Instruct`, `Qwen2.5-Coder-32B-Instruct`.
- Dataset: use the manually filtered local public approximation at `datasets/swe-hero-openhands-trajectories-5b2ed21-one-rollout/`. It selects one rollout per task/`instance_id` from the oldest public `nvidia/SWE-Hero-openhands-trajectories` revision (`5b2ed21270ad773a50163e2999c510f0cbb92cfa`) because that revision has the most public tasks. This is a best-effort match for the paper's "one rollout per task" setup, not the exact internal `~13.2k` paper manifest; the local artifact has 12,633 selected rows.
- Eval: SWE-bench Verified through the OpenHands harness.
- Paper caveat: the reported direct-to-hero ablation is for 32B; 7B and 14B direct-to-hero runs are a scale-study extension unless a paper table proves otherwise.

# Local Resources

- `torchtitan/`: fully vendored TorchTitan base for distributed training.
- `manifests/midtraining-hostpath.yaml`: GPU pod manifest.
- `tmp/pod-creds/`: local credentials; never commit.

# Working Rules

- Re-read the paper before changing pipeline assumptions.
- Prefer TorchTitan's existing extension points and scripts over ad-hoc training code.
- Keep secrets, pod credentials, `.env`, checkpoints, datasets, and generated run artifacts out of git.
- Preserve enough metadata to reproduce each run: model, dataset revision, tokenizer/chat template, sequence length, loss masking, LR schedule, batch size, hardware, commit, and eval harness revision.
- Verify meaningful behavior with automated tests or a concrete dry run whenever feasible.
