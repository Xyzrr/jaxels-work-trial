# Shell-to-Python Script Conversion Experiments

## Overview

This document records the proof that replacing shell launchers with Python
launchers preserved the launch contract for training, eval, and image prebuild
workflows.

The policy rationale lives in [`python_uv_project.md`](python_uv_project.md).
This file is only the evidence record.

The experiments compare ignored scratch logs from:

- Before: `tmp/uv-conversion-experiments/before/`
- After: `tmp/uv-conversion-experiments/after/`

All normalized comparisons passed.

## What Was Verified

The checks focus on contract, not cosmetic output. A launcher contract includes
the command path, argument order, forwarded environment variables, pod script
path, branch guard, worker count, model-serving settings, and eval selectors.

That matters because these workstation commands choose the expensive GPU-pod
workload. A small launcher drift could silently change the model preset,
dataset, context window, vLLM topology, OpenHands worker count, or grader.

## Normalization Rules

Only intentional migration differences were normalized:

- `.sh` entrypoints became `.py`.
- Temporary absolute paths under `tmp/uv-conversion-experiments/...` became
  `<experiment-dir>`.
- After-run commands used the pinned local `tmp/tools/uv-0.11.16/uv run`;
  before-run commands used direct shell execution.
- `UV_PYTHON_INSTALL_DIR` appeared in two after-run fake `kubectl` calls because
  the harness invoked Python launchers through the pinned `uv` binary.

No behavior-specific output was normalized away.

## Result Summary

| Experiment | Contract checked | Result |
| --- | --- | --- |
| `run_midtraining_eval_fake_kubectl` | Eval meta-launcher preserved `kubectl exec` shape, namespace, pod, branch guard env, workspace env, forwarded args, and pod script path except `.sh` to `.py`. | Pass |
| `run_midtraining_prebuild_fake_kubectl` | Prebuild meta-launcher preserved `kubectl exec`, selected env forwarding, namespace/pod defaults, and workload args. | Pass |
| `run_midtraining_help` | Meta-launcher help, options, defaults, and workload mapping were preserved with `.py` names. | Pass |
| `run_midtraining_missing_workload` | Missing workload still prints usage and exits `2`. | Pass |
| `run_qwen_wrapper_fake_venv` | TorchTitan pod wrapper still calls setup with `--venv`, then executes `scripts/qwen_swehero_train.py` through the venv Python with original args. | Pass |
| `run_qwen_wrapper_help` | Local `--help` still fails before training help when pod uv bootstrap runtime is unavailable, prints the same uv requirement, and exits `1`. | Pass |
| `prebuild_invalid_eval_limit` | Prebuild launcher rejects `--eval-limit 0` before pod checks and exits `1`. | Pass |
| `prebuild_invalid_parallel_builds` | Prebuild launcher rejects `--parallel-builds 0` before pod checks and exits `1`. | Pass |
| `prebuild_help` | Prebuild help, options, env docs, and defaults were preserved with `.py` names. | Pass |
| `setup_torchtitan_help` | TorchTitan venv setup help, options, and defaults were preserved with `.py` names. | Pass |
| `setup_torchtitan_unknown` | Unknown setup arg still prints the unknown argument, usage, and exits `2`. | Pass |
| `openhands_eval_conflicting_selectors` | Eval launcher still rejects simultaneous `--eval-limit` and `--eval-ids` before pod checks and exits `1`. | Pass |
| `openhands_eval_help` | Eval launcher help, options, env docs, and defaults were preserved with `.py` names. | Pass |
| `openhands_eval_worker_selection` | Worker selection helper returns `7`, `3`, `3`, `16` for the SWE-Lego/current OpenHands smoke cases. | Pass |
| `openhands_eval_llm_key_selection` | LLM API key helper returns `dummy-key`, `explicit`, `local-llm` for SWE-Lego default, SWE-Lego explicit, and current OpenHands default cases. | Pass |

## Eval Helper Notes

The last two eval cases are high-risk despite being small helpers:

- Worker selection controls how many OpenHands agents submit tasks to the local
  model server at once. Too many workers can exhaust vLLM GPU memory; too few
  can make an eval look slower than the selected configuration.
- LLM key selection keeps local pod inference distinct from provider-backed
  APIs. Dummy or local keys are valid for the OpenAI-compatible vLLM endpoint;
  real secrets should remain explicit runtime inputs.

## Normalized Output

```text
run_midtraining_eval_fake_kubectl: PASS
run_midtraining_prebuild_fake_kubectl: PASS
run_midtraining_help: PASS
run_midtraining_missing_workload: PASS
run_qwen_wrapper_fake_venv: PASS
run_qwen_wrapper_help: PASS
prebuild_invalid_eval_limit: PASS
prebuild_invalid_parallel_builds: PASS
prebuild_help: PASS
setup_torchtitan_help: PASS
setup_torchtitan_unknown: PASS
openhands_eval_conflicting_selectors: PASS
openhands_eval_help: PASS
openhands_eval_worker_selection: PASS
openhands_eval_llm_key_selection: PASS
```
