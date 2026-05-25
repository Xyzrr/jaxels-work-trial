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
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

DEFAULT_OPENHANDS_REPO = "https://github.com/OpenHands/OpenHands.git"
DEFAULT_OPENHANDS_REF = "0.62.0"
DEFAULT_SWE_LEGO_REPO = "https://github.com/SWE-Lego/SWE-Lego.git"
DEFAULT_SWE_LEGO_REF = "94704b69aac886e003660e1e0f69f7de163b284e"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
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
QWEN_NATIVE_CONTEXT_LENGTH = 32_768
PAPER_YARN_FACTOR = PAPER_CONTEXT_LENGTH / QWEN_NATIVE_CONTEXT_LENGTH
PAPER_MAX_ITERATIONS = 100
PAPER_TEMPERATURE = 0.7
PAPER_TOP_P = 0.8
PAPER_TOP_K = 20
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_TOOL_CHOICE = "required"
DEFAULT_DOCKER_SMOKE_IMAGE = "hello-world:latest"
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_VLLM_PIPELINE_PARALLEL_SIZE = 1
DEFAULT_VLLM_SERVER_COUNT = 8
DEFAULT_VLLM_AGENT_TASKS_PER_SERVER = 24
DEFAULT_VLLM_ROUTER_PORT = 8090
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_VLLM_DTYPE = "bfloat16"
DEFAULT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND = "mp"
SWE_LEGO_MAX_INPUT_TOKENS = 147_456
SWE_LEGO_VLLM_MAX_MODEL_LEN = 163_840
SWE_LEGO_MAX_OUTPUT_TOKENS = 16_384
SWE_LEGO_TEMPERATURE = 0.0
SWE_LEGO_GRADER_MAX_WORKERS = 10
SWE_LEGO_GRADER_TIMEOUT = 500
SWE_LEGO_GRADER_CACHE_LEVEL = "instance"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_PRESET = (
    REPO_ROOT
    / "configs"
    / "eval"
    / "openhands-swebench-verified-qwen25-coder-7b-paper-yarn-128k.args"
)
CONTEXT_MODE_PAPER_YARN_128K = "paper-yarn-128k"
CONTEXT_MODE_BASE_NATIVE_32K = "base-native-32k"
CONTEXT_MODE_BASE_PAPER_YARN_128K = "base-paper-yarn-128k"
CONTEXT_MODE_SWE_LEGO_QWEN3_160K = "swe-lego-qwen3-160k"
CONTEXT_MODES = (
    CONTEXT_MODE_PAPER_YARN_128K,
    CONTEXT_MODE_BASE_NATIVE_32K,
    CONTEXT_MODE_BASE_PAPER_YARN_128K,
    CONTEXT_MODE_SWE_LEGO_QWEN3_160K,
)
DEFAULT_CONTEXT_MODE = CONTEXT_MODE_PAPER_YARN_128K
EVAL_STACK_OPENHANDS = "openhands"
EVAL_STACK_SWE_LEGO = "swe-lego"
EVAL_STACKS = (EVAL_STACK_OPENHANDS, EVAL_STACK_SWE_LEGO)
PAPER_YARN_ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": PAPER_YARN_FACTOR,
    "original_max_position_embeddings": QWEN_NATIVE_CONTEXT_LENGTH,
}
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
    converted_swebench_jsonl: Path
    expected_report_json: Path
    swebench_report_dir: Path


@dataclass(frozen=True)
class EvalCommands:
    prepare_openhands: list[str]
    run_infer: list[str]
    convert_output: list[str] | None
    run_eval: list[str]
    serve_vllm: list[str]


@dataclass(frozen=True)
class ContextModeSpec:
    mode: str
    description: str
    max_input_tokens: int
    vllm_max_model_len: int
    vllm_rope_scaling: str | None
    default_eval_note: str
    requires_rope_scaling: bool = True
    allow_long_max_model_len: bool = True


def paper_yarn_rope_scaling_json() -> str:
    return json.dumps(PAPER_YARN_ROPE_SCALING, separators=(",", ":"))


def context_mode_spec(mode: str) -> ContextModeSpec:
    if mode == CONTEXT_MODE_PAPER_YARN_128K:
        return ContextModeSpec(
            mode=mode,
            description=(
                "Paper-aligned 128k context using YaRN scaling from the "
                "Qwen2.5-Coder native 32k window."
            ),
            max_input_tokens=PAPER_CONTEXT_LENGTH,
            vllm_max_model_len=PAPER_CONTEXT_LENGTH,
            vllm_rope_scaling=paper_yarn_rope_scaling_json(),
            default_eval_note="swehero-qwen25-coder7b-pass1",
        )
    if mode == CONTEXT_MODE_BASE_NATIVE_32K:
        return ContextModeSpec(
            mode=mode,
            description=(
                "Released base-model baseline at Qwen2.5-Coder's native 32k "
                "context, without long-context rope overrides."
            ),
            max_input_tokens=QWEN_NATIVE_CONTEXT_LENGTH,
            vllm_max_model_len=QWEN_NATIVE_CONTEXT_LENGTH,
            vllm_rope_scaling=None,
            default_eval_note="base-native-32k-pass1",
            requires_rope_scaling=False,
            allow_long_max_model_len=False,
        )
    if mode == CONTEXT_MODE_BASE_PAPER_YARN_128K:
        return ContextModeSpec(
            mode=mode,
            description=(
                "Base-model control with the same 128k YaRN context budget as "
                "the paper-aligned SFT evaluation."
            ),
            max_input_tokens=PAPER_CONTEXT_LENGTH,
            vllm_max_model_len=PAPER_CONTEXT_LENGTH,
            vllm_rope_scaling=paper_yarn_rope_scaling_json(),
            default_eval_note="base-paper-yarn-128k-pass1",
        )
    if mode == CONTEXT_MODE_SWE_LEGO_QWEN3_160K:
        return ContextModeSpec(
            mode=mode,
            description=(
                "SWE-Lego Qwen3 long-context serving contract: 147456 "
                "OpenHands input tokens against a 163840-token model context. "
                "The SWE-Lego checkpoint carries its Qwen3 YaRN long-context "
                "configuration in config.json, so no vLLM rope override is passed."
            ),
            max_input_tokens=SWE_LEGO_MAX_INPUT_TOKENS,
            vllm_max_model_len=SWE_LEGO_VLLM_MAX_MODEL_LEN,
            vllm_rope_scaling=None,
            default_eval_note="swe-lego-qwen3-8b-pass1",
            requires_rope_scaling=False,
            allow_long_max_model_len=False,
        )
    raise ValueError(f"unknown context mode: {mode}")


def normalize_optional_value(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"", "none", "null", "off", "false", "disabled"}:
        return None
    return value


def resolve_vllm_rope_scaling(
    requested: str | None, context_spec: ContextModeSpec
) -> str | None:
    requested = normalize_optional_value(requested)
    if requested is None:
        return None
    if requested.lower() == "auto":
        return context_spec.vllm_rope_scaling
    return requested


def decoded_vllm_rope_scaling(value: str | None) -> object:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def validate_context_args(args: argparse.Namespace) -> None:
    context_spec = context_mode_spec(args.context_mode)
    if args.context_mode == CONTEXT_MODE_BASE_NATIVE_32K:
        if args.max_input_tokens > QWEN_NATIVE_CONTEXT_LENGTH:
            raise ValueError(
                f"{CONTEXT_MODE_BASE_NATIVE_32K} requires "
                f"--max-input-tokens <= {QWEN_NATIVE_CONTEXT_LENGTH}; use "
                f"{CONTEXT_MODE_BASE_PAPER_YARN_128K} for the 128k base-model "
                "control."
            )
        if args.vllm_max_model_len > QWEN_NATIVE_CONTEXT_LENGTH:
            raise ValueError(
                f"{CONTEXT_MODE_BASE_NATIVE_32K} requires "
                f"--vllm-max-model-len <= {QWEN_NATIVE_CONTEXT_LENGTH}; use "
                f"{CONTEXT_MODE_BASE_PAPER_YARN_128K} for the 128k base-model "
                "control."
            )
        if args.vllm_rope_scaling is not None:
            raise ValueError(
                f"{CONTEXT_MODE_BASE_NATIVE_32K} must not set "
                "--vllm-rope-scaling; it is the no-YaRN native-context "
                "baseline."
            )
        if args.vllm_max_model_len < args.max_input_tokens:
            raise ValueError(
                "--vllm-max-model-len must be greater than or equal to "
                "--max-input-tokens"
            )
        return

    if args.vllm_max_model_len < args.max_input_tokens:
        raise ValueError(
            "--vllm-max-model-len must be greater than or equal to "
            "--max-input-tokens"
        )

    if (
        context_spec.requires_rope_scaling
        and args.vllm_max_model_len > QWEN_NATIVE_CONTEXT_LENGTH
        and not args.vllm_rope_scaling
    ):
        raise ValueError(
            f"{args.context_mode} extends Qwen2.5-Coder beyond "
            f"{QWEN_NATIVE_CONTEXT_LENGTH} tokens and requires explicit "
            "vLLM YaRN rope scaling."
        )


class EvalArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        stripped = arg_line.strip()
        if not stripped or stripped.startswith("#"):
            return []
        return shlex.split(stripped)


def _argv_with_default_preset(
    argv: list[str] | None,
    *,
    default_preset: Path = DEFAULT_EVAL_PRESET,
) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if any(value.startswith("@") for value in values):
        return values
    return [f"@{default_preset}", *values]


def _optional_positive_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "off", "false", "disabled", "unbounded"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'none'")
    return parsed


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "off", "false", "disabled", "omit"}:
        return None
    return float(value)


def _optional_int(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "off", "false", "disabled", "omit"}:
        return None
    return int(value)


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


def parse_eval_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def effective_openhands_dir(args: argparse.Namespace) -> Path:
    if args.eval_stack == EVAL_STACK_SWE_LEGO:
        return args.swe_lego_dir / "OpenHands-0.53.0"
    return args.openhands_dir


def effective_swebench_dir(args: argparse.Namespace) -> Path | None:
    if args.eval_stack == EVAL_STACK_SWE_LEGO:
        return args.swe_lego_dir / "SWE-bench-4.0.4"
    return None


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
    converted_jsonl = output_jsonl.with_name(output_jsonl.name.replace(".jsonl", ".swebench.jsonl"))
    swebench_report_dir = run_dir / "swebench-results"
    return EvalPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.toml",
        metadata_path=run_dir / "eval_metadata.json",
        commands_path=run_dir / "commands.sh",
        openhands_output_dir=run_dir / "outputs",
        expected_output_jsonl=output_jsonl,
        converted_swebench_jsonl=converted_jsonl,
        expected_report_json=(
            output_jsonl.parent / "report.json"
            if args.eval_stack == EVAL_STACK_OPENHANDS
            else swebench_report_dir / "run_report.json"
        ),
        swebench_report_dir=swebench_report_dir,
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
        f"max_input_tokens = {args.max_input_tokens}",
        f"timeout = {args.timeout}",
        f"drop_params = {_toml_bool(args.drop_params)}",
        f"disable_vision = {_toml_bool(args.disable_vision)}",
    ]
    if args.top_p is not None:
        config_lines.append(f"top_p = {args.top_p}")
    if args.top_k is not None:
        config_lines.append(f"top_k = {args.top_k}")
    if not args.omit_native_tool_calling_config:
        config_lines.append(
            f"native_tool_calling = {_toml_bool(args.native_tool_calling)}"
        )
    if args.custom_llm_provider:
        config_lines.append(
            f"custom_llm_provider = {_toml_string(args.custom_llm_provider)}"
        )
    if args.max_output_tokens is not None:
        config_lines.append(f"max_output_tokens = {args.max_output_tokens}")
    if args.native_tool_calling and not args.omit_native_tool_calling_config:
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
    if args.eval_stack == EVAL_STACK_SWE_LEGO:
        prepare_openhands = [
            "git",
            "clone",
            args.swe_lego_repo,
            str(args.swe_lego_dir),
        ]
    else:
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
    if args.eval_ids:
        run_infer.extend(["--eval-ids", args.eval_ids])

    convert_output = None
    if args.eval_stack == EVAL_STACK_SWE_LEGO:
        convert_output = [
            "poetry",
            "run",
            "python",
            "evaluation/benchmarks/swe_bench/scripts/eval/convert_oh_output_to_swe_json.py",
            str(paths.expected_output_jsonl),
        ]
        run_eval = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--max_workers",
            str(args.swebench_max_workers),
            "--dataset_name",
            args.dataset,
            "--split",
            args.split,
            "--report_dir",
            str(paths.swebench_report_dir),
            "--cache_level",
            args.swebench_cache_level,
            "--predictions_path",
            str(paths.converted_swebench_jsonl),
            "--run_id",
            args.swebench_run_id,
            "--timeout",
            str(args.swebench_timeout),
        ]
    else:
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
        str(args.vllm_max_model_len),
    ]
    if args.vllm_rope_scaling:
        serve_vllm.extend(["--rope-scaling", args.vllm_rope_scaling])
    if args.vllm_tensor_parallel_size:
        serve_vllm.extend(
            ["--tensor-parallel-size", str(args.vllm_tensor_parallel_size)]
        )
    if args.vllm_pipeline_parallel_size:
        serve_vllm.extend(
            ["--pipeline-parallel-size", str(args.vllm_pipeline_parallel_size)]
        )
    if args.vllm_max_num_seqs is not None:
        serve_vllm.extend(["--max-num-seqs", str(args.vllm_max_num_seqs)])
    serve_vllm.extend(
        ["--gpu-memory-utilization", str(args.vllm_gpu_memory_utilization)]
    )
    if args.vllm_dtype:
        serve_vllm.extend(["--dtype", args.vllm_dtype])
    if args.native_tool_calling and args.vllm_enable_auto_tool_choice:
        serve_vllm.append("--enable-auto-tool-choice")
        if args.vllm_tool_call_parser:
            serve_vllm.extend(["--tool-call-parser", args.vllm_tool_call_parser])
    if args.vllm_distributed_executor_backend:
        serve_vllm.extend(
            [
                "--distributed-executor-backend",
                args.vllm_distributed_executor_backend,
            ]
        )
    if args.vllm_enforce_eager:
        serve_vllm.append("--enforce-eager")

    return EvalCommands(
        prepare_openhands=prepare_openhands,
        run_infer=run_infer,
        convert_output=convert_output,
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
            "convert_output": commands.convert_output or [],
            "run_eval": commands.run_eval,
        }.items()
    }
    metadata = {
        "intent": (
            "SWE-Lego vendored OpenHands/SWE-bench pass@1 evaluation scaffold"
            if args.eval_stack == EVAL_STACK_SWE_LEGO
            else "SWE-HERO paper-aligned OpenHands pass@1 evaluation scaffold"
        ),
        "eval_stack": args.eval_stack,
        "paper_eval_settings": {
            "benchmark": "SWE-bench Verified",
            "metric": "resolved rate / pass@1",
            "scaffold": "OpenHands",
            "agent": args.agent,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "context_mode": args.context_mode,
            "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": args.max_output_tokens,
            "max_interaction_rounds": args.max_iterations,
            "tts": "disabled",
            "n_runs": 1,
            "eval_ids": parse_eval_ids(args.eval_ids),
        },
        "paper_caveats": [
            "The paper says reported results average three evaluation passes; this scaffold defaults to one pass@1 run per user request.",
            "The paper does not publish an exact OpenHands git commit; openhands_ref is recorded explicitly.",
            "The base-native-32k and base-paper-yarn-128k modes are controls for base-model comparison; the paper's final eval description fixes 128k context for trained models.",
        ],
        "openhands": {
            "repo": args.openhands_repo,
            "ref": args.openhands_ref,
            "dir": str(effective_openhands_dir(args)),
        },
        "swe_lego": {
            "repo": args.swe_lego_repo,
            "ref": args.swe_lego_ref,
            "dir": str(args.swe_lego_dir),
            "openhands_dir": str(effective_openhands_dir(args)),
            "swebench_dir": (
                str(effective_swebench_dir(args))
                if effective_swebench_dir(args) is not None
                else None
            ),
        },
        "model": {
            "model_id": args.model_id,
            "served_model_name": args.served_model_name or args.model_id,
            "litellm_model": litellm_model_name(args),
            "base_url": args.base_url,
            "api_key_source": args.api_key_source,
            "custom_llm_provider": args.custom_llm_provider,
            "native_tool_calling": (
                None
                if args.omit_native_tool_calling_config
                else args.native_tool_calling
            ),
            "tool_choice": (
                args.tool_choice
                if args.native_tool_calling
                and not args.omit_native_tool_calling_config
                else None
            ),
            "tool_call_preflight": args.tool_call_preflight,
        },
        "context": {
            "mode": args.context_mode,
            "description": context_mode_spec(args.context_mode).description,
            "qwen_native_context_tokens": QWEN_NATIVE_CONTEXT_LENGTH,
            "paper_context_tokens": PAPER_CONTEXT_LENGTH,
            "max_input_tokens": args.max_input_tokens,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_rope_scaling": decoded_vllm_rope_scaling(
                args.vllm_rope_scaling
            ),
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
            "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
            "vllm_pipeline_parallel_size": args.vllm_pipeline_parallel_size,
            "vllm_server_count": args.vllm_server_count,
            "vllm_agent_tasks_per_server": args.vllm_agent_tasks_per_server,
            "vllm_use_router": args.vllm_use_router,
            "vllm_router_port": args.vllm_router_port,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
            "vllm_rope_scaling": args.vllm_rope_scaling,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_dtype": args.vllm_dtype,
            "vllm_distributed_executor_backend": (
                args.vllm_distributed_executor_backend
            ),
            "vllm_enforce_eager": args.vllm_enforce_eager,
        },
        "paths": {
            "config": str(paths.config_path),
            "commands": str(paths.commands_path),
            "expected_output_jsonl": str(paths.expected_output_jsonl),
            "converted_swebench_jsonl": str(paths.converted_swebench_jsonl),
            "expected_report_json": str(paths.expected_report_json),
            "swebench_report_dir": str(paths.swebench_report_dir),
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
            "# scripts/run_midtraining_pod.sh eval",
            "",
            "# 1. Serve the requested model from the GPU pod.",
            "# " + _shell_join(commands.serve_vllm),
            "",
            "# 2. Prepare the legacy OpenHands SWE-bench harness if needed.",
            f'test -d {shlex.quote(str(effective_openhands_dir(args) / ".git"))} || '
            + _shell_join(commands.prepare_openhands),
            "",
            "# 3. Run one OpenHands pass@1 rollout on SWE-bench Verified.",
            f"cd {shlex.quote(str(effective_openhands_dir(args)))}",
            f"export RUNTIME={shlex.quote(args.runtime)}",
            "export RUN_WITH_BROWSING=false",
            f"export USE_HINT_TEXT={shlex.quote(str(args.use_hint_text).lower())}",
            f"export ENABLE_PLAN_MODE={shlex.quote(str(args.enable_plan_mode).lower())}",
            "export ADD_IN_CONTEXT_LEARNING_EXAMPLE="
            f"{shlex.quote(str(args.add_in_context_learning_example).lower())}",
            f"export INSTRUCTION_TEMPLATE_NAME={shlex.quote(args.instruction_template_name)}",
            f"export ITERATIVE_EVAL_MODE={shlex.quote(str(args.iterative_eval_mode).lower())}",
            f"export ENABLE_LLM_EDITOR={shlex.quote(str(args.enable_llm_editor).lower())}",
            _shell_join(commands.run_infer),
            "",
            "# 4. Grade generated patches with the SWE-bench dockerized harness.",
            *(
                [
                    "# Convert OpenHands output to SWE-bench prediction JSONL.",
                    _shell_join(commands.convert_output),
                ]
                if commands.convert_output
                else []
            ),
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


def write_swebench_filter_config(
    openhands_dir: Path, eval_ids: list[str]
) -> tuple[Path, str | None]:
    config_path = (
        openhands_dir / "evaluation" / "benchmarks" / "swe_bench" / "config.toml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    original = config_path.read_text() if config_path.exists() else None
    selected = ", ".join(_toml_string(instance_id) for instance_id in eval_ids)
    config_path.write_text(f"selected_ids = [{selected}]\n")
    return config_path, original


def restore_swebench_filter_config(config_path: Path, original: str | None) -> None:
    if original is None:
        config_path.unlink(missing_ok=True)
    else:
        config_path.write_text(original)


def git_output(args: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def ensure_openhands_checkout(args: argparse.Namespace) -> None:
    if args.eval_stack == EVAL_STACK_SWE_LEGO:
        if args.swe_lego_dir.exists():
            if not (args.swe_lego_dir / ".git").exists():
                raise RuntimeError(
                    f"{args.swe_lego_dir} exists but is not a git checkout"
                )
            status = git_output(["status", "--porcelain"], cwd=args.swe_lego_dir)
            if status:
                raise RuntimeError(
                    f"{args.swe_lego_dir} has local changes; refusing to change refs"
                )
            run_command(["git", "fetch", "--depth", "1", "origin", args.swe_lego_ref], cwd=args.swe_lego_dir)
            run_command(["git", "checkout", "--detach", args.swe_lego_ref], cwd=args.swe_lego_dir)
        else:
            args.swe_lego_dir.parent.mkdir(parents=True, exist_ok=True)
            run_command(build_commands(args, eval_paths(args)).prepare_openhands)
            run_command(["git", "checkout", "--detach", args.swe_lego_ref], cwd=args.swe_lego_dir)
        openhands_dir = effective_openhands_dir(args)
        swebench_dir = effective_swebench_dir(args)
        if not openhands_dir.is_dir():
            raise RuntimeError(f"SWE-Lego OpenHands checkout missing: {openhands_dir}")
        if swebench_dir is None or not swebench_dir.is_dir():
            raise RuntimeError(f"SWE-Lego SWE-bench checkout missing: {swebench_dir}")
        return

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
    if (
        args.native_tool_calling
        and not args.omit_native_tool_calling_config
        and args.tool_call_preflight
    ):
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
    openhands_dir = effective_openhands_dir(args)
    env = os.environ.copy()
    env["RUNTIME"] = args.runtime
    env["RUN_WITH_BROWSING"] = "false"
    env["USE_HINT_TEXT"] = str(args.use_hint_text).lower()
    env["ENABLE_PLAN_MODE"] = str(args.enable_plan_mode).lower()
    env["ADD_IN_CONTEXT_LEARNING_EXAMPLE"] = str(
        args.add_in_context_learning_example
    ).lower()
    env["INSTRUCTION_TEMPLATE_NAME"] = args.instruction_template_name
    env["ITERATIVE_EVAL_MODE"] = str(args.iterative_eval_mode).lower()
    env["ENABLE_LLM_EDITOR"] = str(args.enable_llm_editor).lower()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(openhands_dir)
        if not existing_pythonpath
        else f"{openhands_dir}{os.pathsep}{existing_pythonpath}"
    )

    eval_ids = parse_eval_ids(args.eval_ids)
    filter_state: tuple[Path, str | None] | None = None
    if eval_ids:
        filter_state = write_swebench_filter_config(openhands_dir, eval_ids)
    try:
        run_command(commands.run_infer, cwd=openhands_dir, env=env)
        if not paths.expected_output_jsonl.exists():
            raise RuntimeError(
                "OpenHands finished but expected output was not found: "
                f"{paths.expected_output_jsonl}"
            )
        print(
            "agent_tool_use="
            + json.dumps(
                summarize_agent_tool_use(paths.expected_output_jsonl),
                sort_keys=True,
            )
        )
        if not args.skip_swebench_eval:
            if commands.convert_output is not None:
                run_command(commands.convert_output, cwd=openhands_dir, env=env)
            if args.eval_stack == EVAL_STACK_SWE_LEGO:
                swebench_dir = effective_swebench_dir(args)
                if swebench_dir is None:
                    raise RuntimeError("SWE-Lego SWE-bench directory is not configured")
                paths.swebench_report_dir.mkdir(parents=True, exist_ok=True)
                grader_env = env.copy()
                grader_pythonpath = grader_env.get("PYTHONPATH")
                grader_env["PYTHONPATH"] = (
                    str(swebench_dir)
                    if not grader_pythonpath
                    else f"{swebench_dir}{os.pathsep}{grader_pythonpath}"
                )
                run_command(commands.run_eval, cwd=paths.swebench_report_dir, env=grader_env)
                report_candidates = sorted(paths.swebench_report_dir.glob("*.json"))
                if report_candidates:
                    report_candidates[-1].replace(paths.expected_report_json)
            else:
                run_command(commands.run_eval, cwd=openhands_dir, env=env)
            if paths.expected_report_json.exists():
                print(json.dumps(summarize_report(paths.expected_report_json), indent=2))
    finally:
        if filter_state is not None:
            restore_swebench_filter_config(*filter_state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = _argv_with_default_preset(argv)
    parser = EvalArgumentParser(
        description="Run/prepare OpenHands SWE-bench Verified pass@1 eval for Qwen2.5-Coder-7B.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--eval-stack", choices=EVAL_STACKS, default=EVAL_STACK_OPENHANDS)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--served-model-name",
        default="",
        help="Name exposed by vLLM/SGLang. Defaults to --model-id.",
    )
    parser.add_argument(
        "--litellm-model",
        default="",
        help="Full LiteLLM model string. Defaults to openai/<served-model-name>.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible endpoint. The pod launcher sets this to the GPU pod IP.",
    )
    parser.add_argument(
        "--custom-llm-provider",
        default="",
    )
    parser.add_argument(
        "--native-tool-calling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--omit-native-tool-calling-config",
        action="store_true",
        default=False,
        help=(
            "Do not write native_tool_calling or tool_choice to config.toml. "
            "Use for eval stacks whose checked-in config intentionally leaves "
            "OpenHands on its default non-native tool-calling path."
        ),
    )
    parser.add_argument(
        "--tool-choice",
        choices=("required", "auto", "none"),
        default=DEFAULT_TOOL_CHOICE,
        help=(
            "OpenAI tool_choice sent through OpenHands completion_kwargs. "
            "Qwen2.5-Coder loops under OpenHands with auto in smoke tests, "
            "while required forces structured CodeAct tool calls."
        ),
    )
    parser.add_argument(
        "--drop-params",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--disable-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=PAPER_TEMPERATURE,
    )
    parser.add_argument(
        "--top-p",
        type=_optional_float,
        default=PAPER_TOP_P,
    )
    parser.add_argument("--top-k", type=_optional_int, default=PAPER_TOP_K)
    parser.add_argument(
        "--tool-call-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require an OpenAI-compatible /chat/completions request to return "
            "structured message.tool_calls before eval."
        ),
    )
    parser.add_argument(
        "--tool-call-preflight-timeout",
        type=int,
        default=120,
        help="Seconds to wait for the vLLM tool-call preflight request.",
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_MODES,
        default=DEFAULT_CONTEXT_MODE,
        help=(
            "Evaluation context budget. Use base-native-32k for an as-released "
            "base-model baseline, or base-paper-yarn-128k for a context-matched "
            "base-model control."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-output-tokens",
        type=_optional_positive_int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "Bound each OpenHands model turn. Leaving this unbounded lets "
            "LiteLLM request the rest of the 128k context as output tokens, "
            "which is unstable with vLLM structured decoding."
        ),
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Limit tasks for smoke runs. Omit for all 500 Verified tasks.",
    )
    parser.add_argument(
        "--eval-ids",
        default="",
        help=(
            "Comma-separated SWE-bench instance IDs to evaluate. The launcher "
            "also writes OpenHands' benchmark-local selected_ids filter so old "
            "OpenHands releases that ignore --eval-ids still run exactly this set."
        ),
    )
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=PAPER_MAX_ITERATIONS,
    )
    default_workers = DEFAULT_VLLM_SERVER_COUNT * DEFAULT_VLLM_AGENT_TASKS_PER_SERVER
    parser.add_argument("--num-workers", type=int, default=default_workers)
    parser.add_argument(
        "--eval-note",
        default="",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--openhands-repo", default=DEFAULT_OPENHANDS_REPO
    )
    parser.add_argument(
        "--openhands-ref", default=DEFAULT_OPENHANDS_REF
    )
    parser.add_argument(
        "--openhands-dir",
        type=Path,
        default=Path("eval-runs/OpenHands"),
    )
    parser.add_argument("--swe-lego-repo", default=DEFAULT_SWE_LEGO_REPO)
    parser.add_argument("--swe-lego-ref", default=DEFAULT_SWE_LEGO_REF)
    parser.add_argument(
        "--swe-lego-dir",
        type=Path,
        default=Path("eval-runs/SWE-Lego"),
    )
    parser.add_argument(
        "--openhands-poetry-version",
        default="",
        help=(
            "Poetry version expected by the eval checkout. The pod launcher "
            "uses this to provision the OpenHands environment."
        ),
    )
    parser.add_argument("--model-config-name", default=DEFAULT_MODEL_CONFIG_NAME)
    parser.add_argument("--agent-config-name", default=DEFAULT_AGENT_CONFIG_NAME)
    parser.add_argument(
        "--runtime",
        choices=("docker", "remote"),
        default="docker",
        help="OpenHands runtime backend. Docker is the paper default.",
    )
    parser.add_argument(
        "--use-hint-text",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-plan-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--add-in-context-learning-example",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--instruction-template-name",
        default="swe_default.j2",
    )
    parser.add_argument(
        "--iterative-eval-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-llm-editor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--docker-smoke-image",
        default=DEFAULT_DOCKER_SMOKE_IMAGE,
        help=(
            "Image used by the Docker preflight container-run check. "
            "The default is tiny and catches unprivileged pod/userns failures."
        ),
    )
    parser.add_argument(
        "--skip-docker-run-check",
        action="store_true",
        default=False,
        help="Skip the preflight docker run --rm smoke container.",
    )
    parser.add_argument(
        "--skip-docker-buildx-check",
        action="store_true",
        default=False,
        help="Skip the preflight docker buildx version check.",
    )
    parser.add_argument("--vllm-host", default="0.0.0.0")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
        help=(
            "Tensor-parallel degree for each vLLM server. The canonical pod "
            "launcher keeps this at 1 and runs one vLLM replica per GPU."
        ),
    )
    parser.add_argument(
        "--vllm-pipeline-parallel-size",
        type=int,
        default=DEFAULT_VLLM_PIPELINE_PARALLEL_SIZE,
        help=(
            "Pipeline-parallel degree for each vLLM server. The canonical pod "
            "launcher keeps this at 1 and runs one vLLM replica per GPU."
        ),
    )
    parser.add_argument(
        "--vllm-server-count",
        type=int,
        default=DEFAULT_VLLM_SERVER_COUNT,
    )
    parser.add_argument(
        "--vllm-agent-tasks-per-server",
        type=int,
        default=DEFAULT_VLLM_AGENT_TASKS_PER_SERVER,
        help="Concurrent OpenHands workers allocated per vLLM replica.",
    )
    parser.add_argument(
        "--vllm-use-router",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vllm-router-port",
        type=int,
        default=DEFAULT_VLLM_ROUTER_PORT,
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help=(
            "vLLM server context limit. Defaults to the selected context mode "
            "and must be at least --max-input-tokens."
        ),
    )
    parser.add_argument("--vllm-max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--vllm-rope-scaling",
        default="auto",
        help=(
            "RoPE scaling JSON passed to vLLM. The default, auto, uses YaRN "
            "for the 128k modes and no override for base-native-32k."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--vllm-dtype",
        default=DEFAULT_VLLM_DTYPE,
    )
    parser.add_argument(
        "--vllm-enable-auto-tool-choice",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vllm-tool-call-parser",
        default="hermes",
    )
    parser.add_argument(
        "--vllm-distributed-executor-backend",
        default=DEFAULT_VLLM_DISTRIBUTED_EXECUTOR_BACKEND,
    )
    parser.add_argument(
        "--vllm-enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--swebench-cache-level",
        default=SWE_LEGO_GRADER_CACHE_LEVEL,
        choices=("none", "base", "env", "instance"),
    )
    parser.add_argument(
        "--swebench-timeout",
        type=int,
        default=SWE_LEGO_GRADER_TIMEOUT,
    )
    parser.add_argument(
        "--swebench-max-workers",
        type=int,
        default=SWE_LEGO_GRADER_MAX_WORKERS,
    )
    parser.add_argument("--swebench-run-id", default="openhands")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--preflight-only", action="store_true", default=False)
    parser.add_argument("--skip-preflight", action="store_true", default=False)
    parser.add_argument("--skip-llm-endpoint-check", action="store_true", default=False)
    parser.add_argument("--skip-swebench-eval", action="store_true", default=False)
    args = parser.parse_args(argv)

    args.api_key = os.environ.get("LLM_API_KEY") or DEFAULT_API_KEY
    if os.environ.get("LLM_API_KEY"):
        args.api_key_source = "LLM_API_KEY"
    else:
        args.api_key_source = "default"
    if not args.served_model_name:
        args.served_model_name = args.model_id
    if not args.litellm_model:
        args.litellm_model = None
    context_spec = context_mode_spec(args.context_mode)
    if args.max_input_tokens is None:
        args.max_input_tokens = context_spec.max_input_tokens
    if args.vllm_max_model_len is None:
        args.vllm_max_model_len = context_spec.vllm_max_model_len
    args.vllm_rope_scaling = resolve_vllm_rope_scaling(
        args.vllm_rope_scaling, context_spec
    )
    args.vllm_distributed_executor_backend = normalize_optional_value(
        args.vllm_distributed_executor_backend
    )
    if not args.eval_note:
        args.eval_note = context_spec.default_eval_note
    validate_context_args(args)
    if args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError("--eval-limit must be positive when provided")
    if args.eval_limit is not None and args.eval_ids:
        raise ValueError("--eval-limit and --eval-ids are mutually exclusive")
    if args.vllm_server_count <= 0:
        raise ValueError("--vllm-server-count must be positive")
    if args.vllm_tensor_parallel_size <= 0:
        raise ValueError("--vllm-tensor-parallel-size must be positive")
    if args.vllm_pipeline_parallel_size <= 0:
        raise ValueError("--vllm-pipeline-parallel-size must be positive")
    if not args.vllm_use_router and args.vllm_server_count != 1:
        raise ValueError("--no-vllm-use-router requires --vllm-server-count 1")
    if args.vllm_max_num_seqs is not None and args.vllm_max_num_seqs <= 0:
        raise ValueError("--vllm-max-num-seqs must be positive when provided")
    if args.swebench_timeout <= 0:
        raise ValueError("--swebench-timeout must be positive")
    if args.swebench_max_workers <= 0:
        raise ValueError("--swebench-max-workers must be positive")
    if not args.base_url and not args.dry_run:
        raise ValueError(
            "--base-url is required; use scripts/run_midtraining_pod.sh eval "
            "to launch the eval on the GPU pod."
        )
    args.output_dir = args.output_dir.resolve()
    args.openhands_dir = args.openhands_dir.resolve()
    args.swe_lego_dir = args.swe_lego_dir.resolve()
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
        if commands.convert_output:
            print("convert_output:", _shell_join(commands.convert_output))
        print("run_eval:", _shell_join(commands.run_eval))
        return

    run_eval(args, paths, commands)


if __name__ == "__main__":
    main()
