# Shell-to-Python Script Conversion Experiments

Before replacing the shell launchers, lightweight behavior experiments were
recorded under `tmp/uv-conversion-experiments/before/`. After conversion, the
same experiments were rerun against the Python launchers under
`tmp/uv-conversion-experiments/after/`.

The raw scratch logs are intentionally ignored by git. This document records
the reproducible experiment scope and normalized results.

## Normalization

The comparison normalizes only intentional migration differences:

- `.sh` entrypoint paths became `.py`.
- Temporary absolute paths under `tmp/uv-conversion-experiments/...` were
  normalized to `<experiment-dir>`.
- The after-run command prefix used the pinned local
  `tmp/tools/uv-0.11.16/uv run`; the before-run command prefix used direct
  shell execution.
- `UV_PYTHON_INSTALL_DIR` appeared in two after-run fake `kubectl` calls only
  because the local experiment harness invoked the Python launchers through the
  pinned uv binary with that environment variable set.

No behavior-specific output was normalized away.

## Result Summary

| Experiment | Behavior checked | Result |
| --- | --- | --- |
| `run_midtraining_eval_fake_kubectl` | Workstation meta-launcher builds the same `kubectl exec` command for eval, with the same namespace, pod, branch guard env, workspace env, forwarded workload args, and pod-side script path changed only from `.sh` to `.py`. | Pass |
| `run_midtraining_prebuild_fake_kubectl` | Workstation meta-launcher builds the same prebuild `kubectl exec` command, preserves selected forwarded env vars (`HF_TOKEN`, `VLLM_FORCE_RESTART`), default namespace/pod, and workload args. | Pass |
| `run_midtraining_help` | Help text, options, defaults, and workload mapping are preserved with `.py` script names. | Pass |
| `run_midtraining_missing_workload` | Missing workload still prints usage and exits `2`. | Pass |
| `run_qwen_wrapper_fake_venv` | TorchTitan pod wrapper still calls setup with `--venv` and then executes `scripts/qwen_swehero_train.py` through the venv Python with original args unchanged. | Pass |
| `run_qwen_wrapper_help` | Without a fake setup override, local `--help` still fails before training help because the pod uv bootstrap runtime is unavailable on the workstation, prints the same uv requirement, and exits `1`. | Pass |
| `prebuild_invalid_eval_limit` | Prebuild launcher rejects `--eval-limit 0` before pod checks and exits `1`. | Pass |
| `prebuild_invalid_parallel_builds` | Prebuild launcher rejects `--parallel-builds 0` before pod checks and exits `1`. | Pass |
| `prebuild_help` | Help text, options, env docs, and defaults are preserved with `.py` script names. | Pass |
| `setup_torchtitan_help` | TorchTitan venv setup help text, options, and defaults are preserved with `.py` script names. | Pass |
| `setup_torchtitan_unknown` | Unknown setup arg still prints the unknown argument, usage, and exits `2`. | Pass |
| `openhands_eval_conflicting_selectors` | Eval launcher still rejects simultaneous `--eval-limit` and `--eval-ids` before pod checks and exits `1`. | Pass |
| `openhands_eval_help` | Eval launcher help text, options, env docs, and defaults are preserved with `.py` script names. | Pass |
| `openhands_eval_worker_selection` | Worker selection helper returns `7`, `3`, `3`, `16` for the SWE-Lego/current OpenHands smoke cases. | Pass |
| `openhands_eval_llm_key_selection` | LLM API key helper returns `dummy-key`, `explicit`, `local-llm` for SWE-Lego default, SWE-Lego explicit, and current OpenHands default cases. | Pass |

## Normalized Comparison Output

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
