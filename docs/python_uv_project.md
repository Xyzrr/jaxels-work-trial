# Python uv Project

The repository is a modern `uv` Python project for all local automation,
tests, linting, and formatting. The vendored `torchtitan/` tree remains outside
this project boundary and must not be modified or managed by this
`pyproject.toml`.

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

## Script Policy

Project automation should be Python, not bash.

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

Generated data, credentials, checkpoints, run artifacts, datasets, and
experiment scratch logs remain out of git. The script conversion experiment
scratch files live under `tmp/uv-conversion-experiments/` and are ignored; the
committed summary is in `docs/script_conversion_experiments.md`.
