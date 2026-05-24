"""Run a paper-aligned OpenHands pass@1 eval on SWE-bench Verified.

This script is intentionally a thin scaffold around the OpenHands SWE-bench
evaluation harness. It prepares the LLM config, records the exact commands, and
optionally executes both phases:

1. OpenHands inference: generate one patch per SWE-bench Verified task.
2. SWE-bench grading: run the official dockerized evaluator on those patches.

The defaults match the SWE-HERO paper's reported evaluation setup where the
paper is explicit: OpenHands, SWE-bench Verified, CodeActAgent, 100 interaction
rounds, 128k context, temperature 0.7, top-p 0.8, and top-k 20. The paper does
not publish an exact OpenHands git commit, so the harness ref is a required
piece of recorded metadata and defaults to the latest V0 release line that still
contains the legacy SWE-bench scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import qwen_swehero_smoke as smoke


DEFAULT_OPENHANDS_REPO = "https://github.com/OpenHands/OpenHands.git"
DEFAULT_OPENHANDS_REF = "0.62.0"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_AGENT = "CodeActAgent"
DEFAULT_MODEL_CONFIG_NAME = "llm.swehero_qwen25_coder7b"
DEFAULT_AGENT_CONFIG_NAME = "agent.swehero_openhands"
DEFAULT_BASE_URL = ""
DEFAULT_API_KEY = "local-llm"
DEFAULT_OUTPUT_DIR = Path("eval-runs/openhands-swebench-verified-pass1")
# OpenHands' SWE-bench script names its Docker execution backend "local".
# The canonical wrapper invokes that backend inside the GPU pod.
SWE_BENCH_DOCKER_ENVIRONMENT = "local"
PAPER_CONTEXT_LENGTH = 131_072
PAPER_MAX_ITERATIONS = 100
PAPER_TEMPERATURE = 0.7
PAPER_TOP_P = 0.8
PAPER_TOP_K = 20
DEFAULT_TOOL_CHOICE = "required"
DEFAULT_DOCKER_SMOKE_IMAGE = "hello-world:latest"
TOOL_CALL_PREFLIGHT_NAME = "report_eval_preflight"
TOOL_CALL_PREFLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_CALL_PREFLIGHT_NAME,
        "description": "Report that the model server emitted a structured tool call.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Set to ok when the tool call path works.",
                    "enum": ["ok"],
                }
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class EvalPaths:
    run_dir: Path
    config_path: Path
    metadata_path: Path
    commands_path: Path
    openhands_output_dir: Path
    expected_output_jsonl: Path
    expected_report_json: Path


@dataclass(frozen=True)
class EvalCommands:
    prepare_openhands: list[str]
    run_infer: list[str]
    run_eval: list[str]
    serve_vllm: list[str]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def litellm_model_name(args: argparse.Namespace) -> str:
    if args.litellm_model:
        return args.litellm_model
    served_model_name = args.served_model_name or args.model_id
    if served_model_name.startswith("openai/"):
        return served_model_name
    return f"openai/{served_model_name}"


def model_output_path_component(litellm_model: str) -> str:
    model_name = litellm_model.split("/")[-1]
    return model_name.replace(":", "_").replace("@", "-")


def dataset_description(dataset: str, split: str) -> str:
    return dataset.replace("/", "__") + "-" + split.replace("/", "__")


def expected_output_jsonl(args: argparse.Namespace) -> Path:
    litellm_model = litellm_model_name(args)
    eval_note = f"_N_{args.eval_note}" if args.eval_note else ""
    return (
        args.output_dir
        / "outputs"
        / dataset_description(args.dataset, args.split)
        / args.agent
        / f"{model_output_path_component(litellm_model)}_maxiter_{args.max_iterations}{eval_note}"
        / "output.jsonl"
    )


def eval_paths(args: argparse.Namespace) -> EvalPaths:
    run_dir = args.output_dir
    output_jsonl = expected_output_jsonl(args)
    return EvalPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.toml",
        metadata_path=run_dir / "eval_metadata.json",
        commands_path=run_dir / "commands.sh",
        openhands_output_dir=run_dir / "outputs",
        expected_output_jsonl=output_jsonl,
        expected_report_json=output_jsonl.parent / "report.json",
    )


def build_openhands_config(args: argparse.Namespace) -> str:
    litellm_model = litellm_model_name(args)
    config_lines = [
        "# Generated by scripts/openhands_swebench_eval.py.",
        "# Contains only eval/runtime config; generated files live under eval-runs/.",
        "",
        f"[{args.model_config_name}]",
        f"model = {_toml_string(litellm_model)}",
        f"base_url = {_toml_string(args.base_url)}",
        f"api_key = {_toml_string(args.api_key)}",
        f"temperature = {args.temperature}",
        f"top_p = {args.top_p}",
        f"top_k = {args.top_k}",
        f"max_input_tokens = {args.max_input_tokens}",
        f"timeout = {args.timeout}",
        f"drop_params = {_toml_bool(args.drop_params)}",
        f"disable_vision = {_toml_bool(args.disable_vision)}",
        f"native_tool_calling = {_toml_bool(args.native_tool_calling)}",
    ]
    if args.custom_llm_provider:
        config_lines.append(
            f"custom_llm_provider = {_toml_string(args.custom_llm_provider)}"
        )
    if args.max_output_tokens is not None:
        config_lines.append(f"max_output_tokens = {args.max_output_tokens}")
    if args.native_tool_calling:
        config_lines.append(
            "completion_kwargs = "
            f"{{ tool_choice = {_toml_string(args.tool_choice)} }}"
        )

    config_lines.extend(
        [
            "",
            f"[{args.agent_config_name}]",
            "enable_jupyter = false",
            "enable_browsing = false",
            "enable_llm_editor = false",
            "enable_mcp = false",
            "enable_prompt_extensions = false",
            "",
            "[condenser]",
            'type = "noop"',
            "",
        ]
    )
    return "\n".join(config_lines)


def build_commands(args: argparse.Namespace, paths: EvalPaths) -> EvalCommands:
    prepare_openhands = [
        "git",
        "clone",
        "--branch",
        args.openhands_ref,
        "--depth",
        "1",
        args.openhands_repo,
        str(args.openhands_dir),
    ]

    run_infer = [
        "poetry",
        "run",
        "python",
        "evaluation/benchmarks/swe_bench/run_infer.py",
        "--config-file",
        str(paths.config_path),
        "--agent-cls",
        args.agent,
        "--agent-config",
        args.agent_config_name,
        "--llm-config",
        args.model_config_name,
        "--max-iterations",
        str(args.max_iterations),
        "--eval-num-workers",
        str(args.num_workers),
        "--eval-note",
        args.eval_note,
        "--eval-output-dir",
        str(paths.openhands_output_dir),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--mode",
        "swe",
    ]
    if args.eval_limit is not None:
        run_infer.extend(["--eval-n-limit", str(args.eval_limit)])

    run_eval = [
        "bash",
        "evaluation/benchmarks/swe_bench/scripts/eval_infer.sh",
        str(paths.expected_output_jsonl),
        "",
        args.dataset,
        args.split,
        SWE_BENCH_DOCKER_ENVIRONMENT,
    ]

    served_model_name = args.served_model_name or args.model_id
    serve_vllm = [
        "vllm",
        "serve",
        args.model_id,
        "--host",
        args.vllm_host,
        "--port",
        str(args.vllm_port),
        "--api-key",
        args.api_key,
        "--served-model-name",
        served_model_name,
        "--max-model-len",
        str(args.max_input_tokens),
    ]
    if args.vllm_tensor_parallel_size:
        serve_vllm.extend(
            ["--tensor-parallel-size", str(args.vllm_tensor_parallel_size)]
        )
    if args.native_tool_calling and args.vllm_enable_auto_tool_choice:
        serve_vllm.append("--enable-auto-tool-choice")
        if args.vllm_tool_call_parser:
            serve_vllm.extend(["--tool-call-parser", args.vllm_tool_call_parser])

    return EvalCommands(
        prepare_openhands=prepare_openhands,
        run_infer=run_infer,
        run_eval=run_eval,
        serve_vllm=serve_vllm,
    )


def write_scaffold(args: argparse.Namespace) -> tuple[EvalPaths, EvalCommands]:
    paths = eval_paths(args)
    commands = build_commands(args, paths)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.config_path.write_text(build_openhands_config(args) + "\n")

    redacted_commands = {
        name: [
            "<redacted>"
            if part == args.api_key and args.api_key != DEFAULT_API_KEY
            else part
            for part in command
        ]
        for name, command in {
            "serve_vllm": commands.serve_vllm,
            "prepare_openhands": commands.prepare_openhands,
            "run_infer": commands.run_infer,
            "run_eval": commands.run_eval,
        }.items()
    }
    metadata = {
        "intent": "SWE-HERO paper-aligned OpenHands pass@1 evaluation scaffold",
        "paper_eval_settings": {
            "benchmark": "SWE-bench Verified",
            "metric": "resolved rate / pass@1",
            "scaffold": "OpenHands",
            "agent": args.agent,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_input_tokens": args.max_input_tokens,
            "max_interaction_rounds": args.max_iterations,
            "tts": "disabled",
            "n_runs": 1,
        },
        "paper_caveats": [
            "The paper says reported results average three evaluation passes; this scaffold defaults to one pass@1 run per user request.",
            "The paper does not publish an exact OpenHands git commit; openhands_ref is recorded explicitly.",
        ],
        "openhands": {
            "repo": args.openhands_repo,
            "ref": args.openhands_ref,
            "dir": str(args.openhands_dir),
        },
        "model": {
            "model_id": args.model_id,
            "served_model_name": args.served_model_name or args.model_id,
            "litellm_model": litellm_model_name(args),
            "base_url": args.base_url,
            "api_key_source": args.api_key_source,
            "custom_llm_provider": args.custom_llm_provider,
            "native_tool_calling": args.native_tool_calling,
            "tool_choice": args.tool_choice if args.native_tool_calling else None,
            "tool_call_preflight": args.tool_call_preflight,
        },
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "eval_limit": args.eval_limit,
        },
        "runtime": {
            "openhands_runtime": args.runtime,
            "swebench_grader": "dockerized SWE-bench harness inside the GPU pod",
            "docker_smoke_image": args.docker_smoke_image,
            "skip_docker_run_check": args.skip_docker_run_check,
            "skip_docker_buildx_check": args.skip_docker_buildx_check,
        },
        "paths": {
            "config": str(paths.config_path),
            "commands": str(paths.commands_path),
            "expected_output_jsonl": str(paths.expected_output_jsonl),
            "expected_report_json": str(paths.expected_report_json),
        },
        "commands": redacted_commands,
        "generated_at_unix": time.time(),
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    commands_text = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Generated command record. The supported launcher is:",
            "# scripts/run_openhands_swebench_eval_pod.sh",
            "",
            "# 1. Serve Qwen2.5-Coder-7B-Instruct from the GPU pod.",
            "# " + _shell_join(commands.serve_vllm),
            "",
            "# 2. Prepare the legacy OpenHands SWE-bench harness if needed.",
            f'test -d {shlex.quote(str(args.openhands_dir / ".git"))} || '
            + _shell_join(commands.prepare_openhands),
            "",
            "# 3. Run one OpenHands pass@1 rollout on SWE-bench Verified.",
            f"cd {shlex.quote(str(args.openhands_dir))}",
            f"export RUNTIME={shlex.quote(args.runtime)}",
            "export RUN_WITH_BROWSING=false",
            "export USE_HINT_TEXT=false",
            "export ITERATIVE_EVAL_MODE=false",
            "export ENABLE_LLM_EDITOR=false",
            _shell_join(commands.run_infer),
            "",
            "# 4. Grade generated patches with the SWE-bench dockerized harness.",
            _shell_join(commands.run_eval),
            "",
        ]
    )
    paths.commands_path.write_text(commands_text)
    paths.commands_path.chmod(0o755)
    return paths, commands


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print(f"$ {_shell_join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def git_output(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def ensure_openhands_checkout(args: argparse.Namespace) -> None:
    if args.openhands_dir.exists():
        if not (args.openhands_dir / ".git").exists():
            raise RuntimeError(f"{args.openhands_dir} exists but is not a git checkout")
        status = git_output(["status", "--porcelain"], cwd=args.openhands_dir)
        if status:
            raise RuntimeError(
                f"{args.openhands_dir} has local changes; refusing to change refs"
            )
        run_command(
            ["git", "fetch", "--tags", "--depth", "1", "origin", args.openhands_ref],
            cwd=args.openhands_dir,
        )
        run_command(
            ["git", "checkout", "--detach", args.openhands_ref],
            cwd=args.openhands_dir,
        )
        return

    args.openhands_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_commands(args, eval_paths(args)).prepare_openhands)


def check_runtime(args: argparse.Namespace) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = []
    checks.append(
        RuntimeCheck(
            "git",
            shutil.which("git") is not None,
            shutil.which("git") or "git not found on PATH",
        )
    )
    checks.append(
        RuntimeCheck(
            "poetry",
            shutil.which("poetry") is not None,
            shutil.which("poetry") or "poetry not found on PATH",
        )
    )
    if not args.skip_llm_endpoint_check:
        checks.append(check_llm_endpoint(args))
    if args.native_tool_calling and args.tool_call_preflight:
        checks.append(check_tool_call_endpoint(args))
    if args.runtime == "docker":
        checks.extend(check_docker_runtime(args))
    return checks


def _last_output_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else "no output"


def check_docker_runtime(args: argparse.Namespace) -> list[RuntimeCheck]:
    docker_path = shutil.which("docker")
    if docker_path is None:
        return [RuntimeCheck("docker", False, "docker not found on PATH")]

    checks: list[RuntimeCheck] = []
    info = subprocess.run(
        ["docker", "info"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    checks.append(
        RuntimeCheck(
            "docker_daemon",
            info.returncode == 0,
            "docker daemon reachable"
            if info.returncode == 0
            else _last_output_line(info.stdout),
        )
    )

    if not args.skip_docker_run_check:
        if info.returncode != 0:
            checks.append(
                RuntimeCheck(
                    "docker_run",
                    False,
                    "skipped because docker daemon check failed",
                )
            )
        else:
            run = subprocess.run(
                ["docker", "run", "--rm", args.docker_smoke_image],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            checks.append(
                RuntimeCheck(
                    "docker_run",
                    run.returncode == 0,
                    f"docker run --rm {args.docker_smoke_image} succeeded"
                    if run.returncode == 0
                    else _last_output_line(run.stdout),
                )
            )

    if not args.skip_docker_buildx_check:
        buildx = subprocess.run(
            ["docker", "buildx", "version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        checks.append(
            RuntimeCheck(
                "docker_buildx",
                buildx.returncode == 0,
                _last_output_line(buildx.stdout)
                if buildx.returncode == 0
                else f"docker buildx unavailable: {_last_output_line(buildx.stdout)}",
            )
        )

    return checks


def check_llm_endpoint(args: argparse.Namespace) -> RuntimeCheck:
    url = args.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {args.api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return RuntimeCheck(
                "llm_endpoint",
                200 <= response.status < 500,
                f"{url} returned HTTP {response.status}",
            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return RuntimeCheck("llm_endpoint", False, f"{url} unreachable: {exc}")


def tool_call_preflight_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.served_model_name or args.model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are checking an OpenAI-compatible tool calling endpoint. "
                    "Use the provided tool; do not answer in plain text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Call {TOOL_CALL_PREFLIGHT_NAME} with status set to ok."
                ),
            },
        ],
        "tools": [TOOL_CALL_PREFLIGHT_TOOL],
        "tool_choice": args.tool_choice,
        "temperature": 0,
        "max_tokens": 128,
    }


def validate_tool_call_response(response: dict[str, object]) -> RuntimeCheck:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return RuntimeCheck("tool_call_preflight", False, "response has no choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return RuntimeCheck(
            "tool_call_preflight", False, "first choice is not an object"
        )
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return RuntimeCheck(
            "tool_call_preflight", False, "first choice has no message object"
        )
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        content = message.get("content")
        preview = "" if content is None else str(content).replace("\n", " ")[:200]
        return RuntimeCheck(
            "tool_call_preflight",
            False,
            f"message.tool_calls missing or empty; content_preview={preview!r}",
        )

    names: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    if TOOL_CALL_PREFLIGHT_NAME not in names:
        return RuntimeCheck(
            "tool_call_preflight",
            False,
            f"structured tool_calls present but expected {TOOL_CALL_PREFLIGHT_NAME!r}; got {names!r}",
        )
    return RuntimeCheck(
        "tool_call_preflight",
        True,
        f"structured tool_calls returned: {', '.join(names)}",
    )


def check_tool_call_endpoint(args: argparse.Namespace) -> RuntimeCheck:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(tool_call_preflight_payload(args)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.tool_call_preflight_timeout) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            if not (200 <= response.status < 300):
                return RuntimeCheck(
                    "tool_call_preflight",
                    False,
                    f"{url} returned HTTP {response.status}: {response_body[:300]}",
                )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return RuntimeCheck(
            "tool_call_preflight",
            False,
            f"{url} returned HTTP {exc.code}: {body[:300]}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return RuntimeCheck("tool_call_preflight", False, f"{url} unreachable: {exc}")

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        return RuntimeCheck(
            "tool_call_preflight", False, f"response is not JSON: {exc}"
        )
    if not isinstance(parsed, dict):
        return RuntimeCheck(
            "tool_call_preflight", False, "response JSON is not an object"
        )
    check = validate_tool_call_response(parsed)
    return RuntimeCheck(check.name, check.ok, f"{url}: {check.detail}")


def assert_preflight_ok(checks: list[RuntimeCheck]) -> None:
    failed = [check for check in checks if not check.ok]
    if not failed:
        return
    details = "\n".join(f"- {check.name}: {check.detail}" for check in failed)
    raise RuntimeError(f"Preflight failed:\n{details}")


def summarize_report(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    resolved_ids = report.get("resolved_ids") or []
    unresolved_ids = report.get("unresolved_ids") or []
    error_ids = report.get("error_ids") or []
    resolved = report.get("resolved_instances")
    if resolved is None:
        resolved = len(resolved_ids)
    total = report.get("submitted_instances", report.get("total_instances"))
    if total is None:
        total = len(set(resolved_ids) | set(unresolved_ids) | set(error_ids))
    pass_at_1 = None if not total else resolved / total
    return {
        "resolved": resolved,
        "total": total,
        "pass_at_1": pass_at_1,
        "report_path": str(report_path),
    }


def summarize_agent_tool_use(output_jsonl: Path) -> dict[str, Any]:
    instances = 0
    agent_messages = 0
    agent_tool_actions = 0
    tool_action_counts: dict[str, int] = {}
    loop_errors = 0

    for line in output_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        instances += 1
        row = json.loads(line)
        error = row.get("error")
        if isinstance(error, str) and "AgentStuckInLoopError" in error:
            loop_errors += 1
        history = row.get("history")
        if not isinstance(history, list):
            continue
        for event in history:
            if not isinstance(event, dict) or event.get("source") != "agent":
                continue
            action = event.get("action")
            if action in {"message", "system", None}:
                if action == "message":
                    agent_messages += 1
                continue
            action_name = str(action)
            agent_tool_actions += 1
            tool_action_counts[action_name] = tool_action_counts.get(action_name, 0) + 1

    return {
        "instances": instances,
        "agent_tool_actions": agent_tool_actions,
        "agent_message_actions": agent_messages,
        "tool_action_counts": tool_action_counts,
        "loop_errors": loop_errors,
        "used_real_tools": agent_tool_actions > 0,
    }


def run_eval(args: argparse.Namespace, paths: EvalPaths, commands: EvalCommands) -> None:
    if not args.skip_preflight:
        checks = check_runtime(args)
        for check in checks:
            status = "ok" if check.ok else "fail"
            print(f"preflight.{check.name}={status} ({check.detail})")
        assert_preflight_ok(checks)

    ensure_openhands_checkout(args)
    env = os.environ.copy()
    env["RUNTIME"] = args.runtime
    env["RUN_WITH_BROWSING"] = "false"
    env["USE_HINT_TEXT"] = "false"
    env["ITERATIVE_EVAL_MODE"] = "false"
    env["ENABLE_LLM_EDITOR"] = "false"
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(args.openhands_dir)
        if not existing_pythonpath
        else f"{args.openhands_dir}{os.pathsep}{existing_pythonpath}"
    )

    run_command(commands.run_infer, cwd=args.openhands_dir, env=env)
    if not paths.expected_output_jsonl.exists():
        raise RuntimeError(
            f"OpenHands finished but expected output was not found: {paths.expected_output_jsonl}"
        )
    print(
        "agent_tool_use="
        + json.dumps(
            summarize_agent_tool_use(paths.expected_output_jsonl), sort_keys=True
        )
    )
    if not args.skip_swebench_eval:
        run_command(commands.run_eval, cwd=args.openhands_dir, env=env)
        if paths.expected_report_json.exists():
            print(json.dumps(summarize_report(paths.expected_report_json), indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run/prepare OpenHands SWE-bench Verified pass@1 eval for Qwen2.5-Coder-7B."
    )
    parser.add_argument("--model-id", default=_env("MODEL_ID", smoke.MODEL_ID))
    parser.add_argument(
        "--served-model-name",
        default=_env("SERVED_MODEL_NAME", ""),
        help="Name exposed by vLLM/SGLang. Defaults to --model-id.",
    )
    parser.add_argument(
        "--litellm-model",
        default=_env("LITELLM_MODEL", ""),
        help="Full LiteLLM model string. Defaults to openai/<served-model-name>.",
    )
    parser.add_argument(
        "--base-url",
        default=_env("LLM_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible endpoint. The pod launcher sets this to the GPU pod IP.",
    )
    parser.add_argument(
        "--api-key",
        default=_env("LLM_API_KEY", _env("OPENAI_API_KEY", DEFAULT_API_KEY)),
    )
    parser.add_argument(
        "--api-key-source",
        default="LLM_API_KEY/OPENAI_API_KEY/default",
        help="Metadata-only description; the actual API key is not written there.",
    )
    parser.add_argument(
        "--custom-llm-provider",
        default=_env("CUSTOM_LLM_PROVIDER", ""),
    )
    parser.add_argument(
        "--native-tool-calling",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("NATIVE_TOOL_CALLING", True),
    )
    parser.add_argument(
        "--tool-choice",
        choices=("required", "auto", "none"),
        default=_env("TOOL_CHOICE", DEFAULT_TOOL_CHOICE),
        help=(
            "OpenAI tool_choice sent through OpenHands completion_kwargs. "
            "Qwen2.5-Coder loops under OpenHands with auto in smoke tests, "
            "while required forces structured CodeAct tool calls."
        ),
    )
    parser.add_argument(
        "--drop-params",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("DROP_PARAMS", False),
    )
    parser.add_argument(
        "--disable-vision",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("DISABLE_VISION", True),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_env_float("TEMPERATURE", PAPER_TEMPERATURE),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=_env_float("TOP_P", PAPER_TOP_P),
    )
    parser.add_argument("--top-k", type=int, default=_env_int("TOP_K", PAPER_TOP_K))
    parser.add_argument(
        "--tool-call-preflight",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("TOOL_CALL_PREFLIGHT", True),
        help=(
            "Require an OpenAI-compatible /chat/completions request to return "
            "structured message.tool_calls before eval."
        ),
    )
    parser.add_argument(
        "--tool-call-preflight-timeout",
        type=int,
        default=_env_int("TOOL_CALL_PREFLIGHT_TIMEOUT", 120),
        help="Seconds to wait for the vLLM tool-call preflight request.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=_env_int("MAX_INPUT_TOKENS", PAPER_CONTEXT_LENGTH),
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=_env_int("LLM_TIMEOUT", 300))
    parser.add_argument("--dataset", default=_env("DATASET", DEFAULT_DATASET))
    parser.add_argument("--split", default=_env("SPLIT", DEFAULT_SPLIT))
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Limit tasks for smoke runs. Omit for all 500 Verified tasks.",
    )
    parser.add_argument("--agent", default=_env("AGENT", DEFAULT_AGENT))
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=_env_int("MAX_ITERATIONS", PAPER_MAX_ITERATIONS),
    )
    parser.add_argument("--num-workers", type=int, default=_env_int("NUM_WORKERS", 1))
    parser.add_argument(
        "--eval-note",
        default=_env("EVAL_NOTE", "swehero-qwen25-coder7b-pass1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_env("OUT_DIR", str(DEFAULT_OUTPUT_DIR))),
    )
    parser.add_argument(
        "--openhands-repo", default=_env("OPENHANDS_REPO", DEFAULT_OPENHANDS_REPO)
    )
    parser.add_argument(
        "--openhands-ref", default=_env("OPENHANDS_REF", DEFAULT_OPENHANDS_REF)
    )
    parser.add_argument(
        "--openhands-dir",
        type=Path,
        default=Path(_env("OPENHANDS_DIR", "eval-runs/OpenHands")),
    )
    parser.add_argument("--model-config-name", default=DEFAULT_MODEL_CONFIG_NAME)
    parser.add_argument("--agent-config-name", default=DEFAULT_AGENT_CONFIG_NAME)
    parser.add_argument(
        "--runtime",
        choices=("docker", "remote"),
        default=_env("RUNTIME", "docker"),
        help="OpenHands runtime backend. Docker is the paper default.",
    )
    parser.add_argument(
        "--docker-smoke-image",
        default=_env("DOCKER_SMOKE_IMAGE", DEFAULT_DOCKER_SMOKE_IMAGE),
        help=(
            "Image used by the Docker preflight container-run check. "
            "The default is tiny and catches unprivileged pod/userns failures."
        ),
    )
    parser.add_argument(
        "--skip-docker-run-check",
        action="store_true",
        default=_env_bool("SKIP_DOCKER_RUN_CHECK", False),
        help="Skip the preflight docker run --rm smoke container.",
    )
    parser.add_argument(
        "--skip-docker-buildx-check",
        action="store_true",
        default=_env_bool("SKIP_DOCKER_BUILDX_CHECK", False),
        help="Skip the preflight docker buildx version check.",
    )
    parser.add_argument("--vllm-host", default=_env("VLLM_HOST", "0.0.0.0"))
    parser.add_argument("--vllm-port", type=int, default=_env_int("VLLM_PORT", 8000))
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=_env_int("VLLM_TENSOR_PARALLEL_SIZE", 1),
    )
    parser.add_argument(
        "--vllm-enable-auto-tool-choice",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("VLLM_ENABLE_AUTO_TOOL_CHOICE", True),
    )
    parser.add_argument(
        "--vllm-tool-call-parser",
        default=_env("VLLM_TOOL_CALL_PARSER", "hermes"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-llm-endpoint-check", action="store_true")
    parser.add_argument("--skip-swebench-eval", action="store_true")
    args = parser.parse_args(argv)

    if not args.served_model_name:
        args.served_model_name = args.model_id
    if not args.litellm_model:
        args.litellm_model = None
    if args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive when provided")
    if not args.base_url and not args.dry_run:
        raise ValueError(
            "--base-url is required; use scripts/run_openhands_swebench_eval_pod.sh "
            "to launch the eval from the GPU pod."
        )
    args.output_dir = args.output_dir.resolve()
    args.openhands_dir = args.openhands_dir.resolve()
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths, commands = write_scaffold(args)
    print(f"wrote_config={paths.config_path}")
    print(f"wrote_metadata={paths.metadata_path}")
    print(f"wrote_commands={paths.commands_path}")
    print(f"expected_output_jsonl={paths.expected_output_jsonl}")

    if args.preflight_only:
        checks = check_runtime(args)
        for check in checks:
            status = "ok" if check.ok else "fail"
            print(f"{check.name}: {status} - {check.detail}")
        assert_preflight_ok(checks)
        return

    if args.dry_run:
        print("serve_vllm:", _shell_join(commands.serve_vllm))
        print("prepare_openhands:", _shell_join(commands.prepare_openhands))
        print("run_infer:", _shell_join(commands.run_infer))
        print("run_eval:", _shell_join(commands.run_eval))
        return

    run_eval(args, paths, commands)


if __name__ == "__main__":
    main()
