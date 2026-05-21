# Project Context

Build a prototype mid-training pipeline for a future 100B-500B open source coding model. The pipeline consumes SWE traces.

# Mode

Ship prototype ASAP: skip post-task validation, parent-branch merges, and PR ceremony unless explicitly asked. When done, commit task changes and push the current branch.

# Current Baseline

Replicate and extend the "direct-to-hero" baseline from "From SWE-ZERO to SWE-HERO" (arXiv:2604.01496):

- Base models: `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-14B-Instruct`, `Qwen2.5-Coder-32B-Instruct`.
- Dataset: use the public `nvidia/SWE-Hero-openhands-trajectories` Hugging Face release as canonical, even though its current row count differs from the paper's `~13.2k` wording.
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
