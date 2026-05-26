# Python uv Project

The repository is a modern `uv` Python project for local automation, tests,
linting, and formatting. It is not the machine-learning runtime. The local
environment runs launchers, validators, config parsing, and lightweight unit
tests; it does not train models, serve vLLM, or import the full CUDA/PyTorch
stack used in the GPU pod.

The vendored `torchtitan/` tree remains outside this project boundary and must
not be modified or managed by this `pyproject.toml`.

## Pinned Toolchain

- Project `uv`: `0.11.16`
- Project Python: `3.12.13`
- Local Python selector: `.python-version`
- Project dependency lock: `uv.lock`
- Dev tools: `pytest==9.0.3`, `ruff==0.15.14`

The GPU pod runtime still has two runtime-specific Python contracts:

- TorchTitan training venv: CPython `3.10.12`, managed by
  `scripts/setup_torchtitan_pod_venv.py`.
- OpenHands eval and vLLM venvs: Python `3.12`, managed by the pod-side eval
  launchers.

Those pod runtimes are intentionally separate from the local project venv.
PyTorch, CUDA wheels, TorchAO FP8 support, Triton kernels, vLLM, OpenHands, and
SWE-bench have tight and sometimes conflicting dependency constraints. Keeping
them out of the local `uv.lock` prevents a tooling sync from silently changing
the numerical training stack, model-serving behavior, or grader environment.

The local `uv.lock` should therefore stay small: it pins developer tooling.
Training and eval runtime pins live in `requirements/` and are repaired inside
the pod before a run starts.

## Local Setup

Use the pinned `uv` version, then sync from the lockfile:

```bash
uv sync --locked
```

Run individual checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Run the full local validation suite with:

```bash
uv run scripts/validate.py
```

`scripts/validate.py` starts pytest, Ruff lint, and Ruff format checks in
parallel. It captures each subprocess independently and prints one grouped
stdout/stderr block per process, so output stays readable instead of being
interleaved.

GitHub Actions runs the same command in `.github/workflows/validation.yml`.

These checks prove that project-owned Python automation and configuration
plumbing still behave as expected. They are intentionally not a substitute for
GPU-pod validation: changes to model architecture, tokenizer assets, sequence
length, precision, distributed training, vLLM sizing, or SWE-bench grading must
be verified through the workflow-specific pod commands documented in
`docs/swehero_torchtitan_pod.md` and
`docs/openhands_swebench_gpu_pod_eval.md`.

## Script Policy

Project automation should be Python, not bash.

The reason is reproducibility, not syntax preference. Python launchers can parse
the same argparse preset files that define model/data/hyperparameter choices,
record those resolved choices in run specs, and test subprocess assembly without
starting an expensive GPU job. Shell snippets are still fine inside generated
pod commands when they are the interface exposed by `kubectl`, `tmux`, Docker,
or a third-party tool.

Workstation/local entry points that depend on the project environment should
use:

```python
#!/usr/bin/env -S uv run python
```

Pod bootstrap entry points may use:

```python
#!/usr/bin/env python3
```

That exception is only for scripts that have to bootstrap or repair `uv`,
Python, or the pod venv before a project-managed interpreter can be assumed.
Current examples are:

- `scripts/setup_torchtitan_pod_venv.py`
- `scripts/run_qwen_swehero_torchtitan_pod.py`
- `scripts/run_openhands_swebench_eval_pod.py`
- `scripts/prebuild_openhands_swebench_images_pod.py`

Do not add new committed `.sh` scripts outside vendored third-party trees or
generated command-record artifacts. When shell behavior is needed, implement it
as explicit Python subprocess orchestration and cover the behavior with pytest.

## Boundaries

`torchtitan/` is vendored source. The project `pyproject.toml`, pytest
configuration, Ruff configuration, and CI intentionally exclude it. Do not add
`torchtitan/` to project package discovery, lint targets, format targets, or
dependency management.

Experiment configuration has its own boundary as well. Model checkpoints,
dataset revisions, context lengths, batch sizes, optimizers, precision modes,
vLLM tensor parallelism, and grader settings should live in argparse preset
files under `configs/`, not in `pyproject.toml`, `uv.lock`, or ad hoc
environment variables. That keeps each ML run reviewable by reading one preset
plus the run spec it produces.

Generated data, credentials, checkpoints, run artifacts, datasets, and
experiment scratch logs remain out of git. The script conversion experiment
scratch files live under `tmp/uv-conversion-experiments/` and are ignored; the
committed summary is in `docs/script_conversion_experiments.md`.
