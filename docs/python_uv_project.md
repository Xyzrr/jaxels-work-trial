# Python uv Project

## Overview

Read this file when working on local Python automation: launchers, validators,
config parsing, lightweight unit tests, linting, and formatting.

The local `uv` project is not the ML runtime. It must not train models, serve
vLLM, grade SWE-bench, or absorb the full CUDA/PyTorch dependency stack used on
the GPU pod.

The project-wide rules live in [`../AGENTS.md`](../AGENTS.md). This file only
spells out the local Python contract and the commands needed to reproduce it.

## Runtime Boundary

Local `uv.lock` pins developer tooling only. Keeping the local environment small
prevents a routine tooling sync from changing the numerical training stack,
model-serving behavior, or grader environment.

Pod runtimes stay separate:

- TorchTitan training venv: CPython `3.10.12`, built by
  `scripts/setup_torchtitan_pod_venv.py`.
- OpenHands eval and vLLM venvs: Python `3.12`, built by the pod-side eval
  launchers.

Training and eval runtime pins live in `requirements/` and are repaired inside
the pod before runs. Workflow-specific pod commands live in:

- [`swehero_torchtitan_pod.md`](swehero_torchtitan_pod.md)
- [`openhands_swebench_gpu_pod_eval.md`](openhands_swebench_gpu_pod_eval.md)

## Toolchain Contract

- `uv`: `0.11.16`, enforced by `pyproject.toml`
- Python: `3.12.13`, selected by `.python-version`
- Lockfile: `uv.lock`
- Dev tools: `pytest==9.0.3`, `ruff==0.15.14`
- CI command: `.github/workflows/validation.yml` runs
  `uv run scripts/validate.py`

## Local Commands

Sync from the lockfile:

```bash
uv sync --locked
```

Run the full local validation suite:

```bash
uv run scripts/validate.py
```

Run individual checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

`scripts/validate.py` runs pytest, Ruff lint, and Ruff format checks in
parallel while keeping each process's stdout/stderr grouped.

These checks cover project-owned Python automation and configuration plumbing.
They do not validate model architecture, tokenizer assets, sequence length,
precision, distributed training, vLLM sizing, or SWE-bench grading.

## Script Entry Points

Project automation should be Python, not committed shell scripts. See the
conversion rationale and evidence in
[`script_conversion_experiments.md`](script_conversion_experiments.md).

Workstation entry points that depend on the project environment should use:

```python
#!/usr/bin/env -S uv run python
```

Pod bootstrap entry points may use:

```python
#!/usr/bin/env python3
```

Use the bootstrap exception only when a script must install or repair `uv`,
Python, or a pod venv before a project-managed interpreter exists.

Current bootstrap examples:

- `scripts/setup_torchtitan_pod_venv.py`
- `scripts/run_qwen_swehero_torchtitan_pod.py`
- `scripts/run_openhands_swebench_eval_pod.py`
- `scripts/prebuild_openhands_swebench_images_pod.py`

When shell behavior is unavoidable, keep it inside Python subprocess
orchestration and cover meaningful behavior with pytest.

## Exclusions

Do not bring these into the local project package, lint, format, or dependency
scope:

- `torchtitan/`
- model checkpoints
- datasets
- generated run artifacts
- pod credentials and `.env`
- experiment scratch logs

Experiment settings also stay out of `pyproject.toml`, `uv.lock`, and ad hoc
environment variables. Model checkpoints, dataset revisions, context lengths,
batch sizes, optimizers, precision modes, vLLM tensor parallelism, and grader
settings belong in argparse presets under `configs/`.
