"""Launch the TorchTitan SWE-HERO direct-to-hero 7B training job.

This is the production entrypoint for the Qwen2.5-Coder-7B SWE-HERO
scale-study run. It intentionally keeps the paper-facing recipe visible:

* Qwen2.5-Coder-7B-Instruct initialized from the Hugging Face checkpoint;
* one-rollout SWE-Hero training artifact generated from the pinned public
  historical ``nvidia/SWE-Hero-openhands-trajectories`` revision;
* three SFT epochs, global batch size 32, cosine LR 1e-5 -> 1e-8 with 0.1
  warmup;
* 128k YaRN context extension from Qwen2.5's native 32k context;
* assistant/action-only loss masking, with tool observations masked.

The important engineering deltas are explicit and recorded in the generated
manifest: this uses TorchTitan distributed training, BF16 mixed-precision
FSDP parameters/reductions plus FP8 linear training where TorchTitan supports
it, and length buckets with bucket-specific context parallelism instead of
padding every example to 128k.
TorchTitan's native varlen attention currently rejects CP, so CP stages use
the supported SDPA/Flex attention path and get the padding reduction from
bucketed sequence lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import prepare_swehero_historical_one_rollout as one_rollout
from scripts import qwen_swehero_smoke as smoke


IGNORE_INDEX = -100

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
TRAINING_DATASET_NAME = "swe-hero-openhands-trajectories-5b2ed21-one-rollout"
DATASET_ID = TRAINING_DATASET_NAME
SOURCE_DATASET_ID = one_rollout.DATASET_ID
SOURCE_DATASET_REVISION = one_rollout.HISTORICAL_REVISION
PAPER_CONTEXT_LENGTH = 131_072
QWEN_NATIVE_CONTEXT_LENGTH = 32_768
DEFAULT_OUT_DIR = Path("/workspace/qwen25-coder7b-swehero-torchtitan")
DEFAULT_HF_ASSETS_PATH = Path("/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct")
DEFAULT_DATASET_PATH = Path("/workspace/datasets") / TRAINING_DATASET_NAME
CANONICAL_WORKSPACE_ROOT = Path("/workspace/jaxels-work-trial")
DEFAULT_NUM_EXAMPLES = 0
DEFAULT_MAX_STREAMED_EXAMPLES = 0
DEFAULT_BUCKETS = (8_192, 16_384, 32_768, 65_536, PAPER_CONTEXT_LENGTH)
DEFAULT_BUCKET_CP = {
    8_192: 1,
    16_384: 1,
    32_768: 2,
    65_536: 4,
    PAPER_CONTEXT_LENGTH: 8,
}
DEFAULT_BUCKET_CURRICULUM = "short-to-long"
DEFAULT_LONG_EXAMPLE_POLICY = "error"
DEFAULT_MIN_TRAINABLE_TOKENS = 1
DEFAULT_INCLUDE_MODEL_PATCH = False
DEFAULT_NUM_TRAIN_EPOCHS = 3.0
DEFAULT_MAX_STEPS = 0
DEFAULT_GLOBAL_BATCH_SIZE = 32
DEFAULT_LOCAL_BATCH_SIZE = 1
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MIN_LEARNING_RATE = 1e-8
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MIN_FREE_DISK_GB = 100.0
DEFAULT_MIN_FREE_GPU_MEMORY_GB = 60.0
DEFAULT_MIN_FREE_CPU_MEMORY_GB = 32.0
DEFAULT_MIN_WRITE_THROUGHPUT_MB_S = 50.0
DEFAULT_WRITE_THROUGHPUT_PROBE_MB = 64
BUCKET_CURRICULUM_CHOICES = (
    "short-to-long",
    "long-to-short",
    "single-bucket",
)
QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
QWEN_ROPE_THETA = 1_000_000.0
QWEN_YARN_BETA_FAST = 32.0
QWEN_YARN_BETA_SLOW = 1.0
MATERIALIZED_DATA_SCHEMA_VERSION = 1
MODEL_ASSET_PROVENANCE_SCHEMA_VERSION = 1
DATA_PROVENANCE_SCHEMA_VERSION = 1
GIT_STATE_SCHEMA_VERSION = 1
RUN_SPEC_SCHEMA_VERSION = 1
RUN_SPEC_FILENAME = "run_spec.json"
RUN_SPEC_SHA256_FILENAME = "run_spec.sha256"
RESUME_CONTRACT_SCHEMA_VERSION = 1
RESUME_CONTRACT_FILENAME = "resume_contract.json"
REQUIRED_HF_ASSET_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
FINAL_MODEL_EXPORT_FOLDER = "final_export"
DEFAULT_VALIDATE_FIRST_STEP_CHECKPOINT = True
FIRST_STEP_CHECKPOINT_VALIDATION_SCHEMA_VERSION = 1
FIRST_STEP_CHECKPOINT_VALIDATION_FILENAME = "first_step_checkpoint_validation.json"
FINAL_ARTIFACT_VALIDATION_SCHEMA_VERSION = 1
FINAL_ARTIFACT_VALIDATION_FILENAME = "final_artifact_validation.json"
POST_TRAINING_EVAL_STATUS_SCHEMA_VERSION = 1
POST_TRAINING_EVAL_STATUS_FILENAME = "post_training_eval_status.json"
RUNTIME_METADATA_SCHEMA_VERSION = 1
RUNTIME_METADATA_FILENAME = "runtime_metadata.json"
STAGE_STATUS_SCHEMA_VERSION = 1
STAGE_STATUS_FILENAME = "stage_status.json"
LAUNCH_LOCK_SCHEMA_VERSION = 1
LAUNCH_LOCK_SUFFIX = ".launch.lock"
WANDB_IDENTITY_SCHEMA_VERSION = 1
WANDB_IDENTITY_FILENAME = "wandb_identity.json"
WANDB_RESUME_CHOICES = ("allow", "never", "must", "auto")
WANDB_MODE_CHOICES = ("online", "offline", "disabled", "shared")
WANDB_RUN_ID_FORBIDDEN_CHARS = frozenset("/\\#?%:")
OPTIMIZER_IMPL_CHOICES = ("for-loop", "foreach", "fused", "fused_opt_states_bf16")
TORCH_DTYPE_CHOICES = ("float32", "bfloat16")
FSDP_RESHARD_AFTER_FORWARD_CHOICES = ("default", "always", "never")
TERMINATION_SIGNALS = tuple(
    signal_number
    for signal_number in (
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
    )
    if signal_number is not None
)
SIGNAL_FORWARD_GRACE_SECONDS = 30.0
CONTROLLED_WANDB_ENV_KEYS = (
    "WANDB_PROJECT",
    "WANDB_TEAM",
    "WANDB_ENTITY",
    "WANDB_RUN_NAME",
    "WANDB_NAME",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_RESUME_FROM",
    "WANDB_FORK_FROM",
    "WANDB_RUN_GROUP",
    "WANDB_RUN_JOB_TYPE",
    "WANDB_JOB_TYPE",
    "WANDB_RUN_TAGS",
    "WANDB_TAGS",
    "WANDB_RUN_NOTES",
    "WANDB_NOTES",
    "WANDB_MODE",
)
RESUME_ARG_FIELDS = (
    "workspace_root",
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_path",
    "source_dataset_id",
    "source_dataset_revision",
    "hf_assets_path",
    "production_mode",
    "production_acceptance_smoke",
    "num_examples",
    "max_streamed_examples",
    "shuffle_buffer",
    "seed",
    "max_length",
    "long_example_policy",
    "buckets",
    "bucket_cp",
    "min_trainable_tokens",
    "include_model_patch",
    "num_train_epochs",
    "max_steps",
    "global_batch_size",
    "local_batch_size",
    "learning_rate",
    "min_learning_rate",
    "warmup_ratio",
    "weight_decay",
    "max_grad_norm",
    "optimizer_impl",
    "attention_backend",
    "enable_fp8",
    "fp8_recipe",
    "compile",
    "training_dtype",
    "mixed_precision_param_dtype",
    "mixed_precision_reduce_dtype",
    "fsdp_reshard_after_forward",
    "activation_checkpoint_mode",
    "chunked_ce_chunks",
    "detect_anomaly",
    "validate_first_step_checkpoint",
    "min_free_disk_gb",
    "min_free_gpu_memory_gb",
    "min_free_cpu_memory_gb",
    "min_write_throughput_mb_s",
    "write_throughput_probe_mb",
    "enable_wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_run_id",
    "wandb_resume",
    "wandb_resume_from",
    "wandb_fork_from",
    "wandb_run_group",
    "wandb_run_job_type",
    "wandb_run_tags",
    "wandb_run_notes",
    "wandb_mode",
    "nproc_per_node",
    "nnodes",
    "node_rank",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
    "cuda_device_max_connections",
    "torch_nccl_async_error_handling",
)
RUN_SPEC_ARG_FIELDS = (
    "workspace_root",
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_path",
    "source_dataset_id",
    "source_dataset_revision",
    "build_dataset_if_missing",
    "source_dataset_rows_per_shard",
    "source_dataset_build_batch_size",
    "hf_assets_path",
    "production_mode",
    "production_acceptance_smoke",
    "num_examples",
    "max_streamed_examples",
    "shuffle_buffer",
    "seed",
    "max_length",
    "long_example_policy",
    "smoke_synthetic_buckets",
    "smoke_synthetic_examples_per_bucket",
    "bucket_curriculum",
    "buckets",
    "bucket_cp",
    "min_trainable_tokens",
    "include_model_patch",
    "num_train_epochs",
    "max_steps",
    "global_batch_size",
    "local_batch_size",
    "learning_rate",
    "min_learning_rate",
    "warmup_ratio",
    "weight_decay",
    "max_grad_norm",
    "optimizer_impl",
    "attention_backend",
    "enable_fp8",
    "fp8_recipe",
    "compile",
    "training_dtype",
    "mixed_precision_param_dtype",
    "mixed_precision_reduce_dtype",
    "fsdp_reshard_after_forward",
    "activation_checkpoint_mode",
    "chunked_ce_chunks",
    "detect_anomaly",
    "checkpoint_interval",
    "checkpoint_async_mode",
    "validate_first_step_checkpoint",
    "metrics_log_freq",
    "min_free_disk_gb",
    "min_free_gpu_memory_gb",
    "min_free_cpu_memory_gb",
    "min_write_throughput_mb_s",
    "write_throughput_probe_mb",
    "enable_profiler",
    "profiler_trace_folder",
    "profiler_freq",
    "profiler_active",
    "profiler_warmup",
    "profiler_repeat",
    "profiler_skip_first",
    "profiler_skip_first_wait",
    "enable_memory_snapshot",
    "memory_snapshot_folder",
    "enable_wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_run_id",
    "wandb_resume",
    "wandb_resume_from",
    "wandb_fork_from",
    "wandb_run_group",
    "wandb_run_job_type",
    "wandb_run_tags",
    "wandb_run_notes",
    "wandb_mode",
    "post_training_eval_command",
    "nproc_per_node",
    "nnodes",
    "node_rank",
    "rdzv_backend",
    "rdzv_endpoint",
    "rdzv_id",
    "torchrun_bin",
    "log_rank",
    "torchrun_log_rank_filter",
    "cuda_device_max_connections",
    "torch_nccl_async_error_handling",
)
RESUME_STAGE_ENV_KEYS = (
    "PYTHONPATH",
    "TOKENIZERS_PARALLELISM",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "SWEHERO_WORKSPACE_ROOT",
    "SWEHERO_MODEL_ID",
    "SWEHERO_MODEL_REVISION",
    "SWEHERO_DATASET_ID",
    "SWEHERO_DATASET_PATH",
    "SWEHERO_BUCKET_FILE",
    "SWEHERO_BUCKET_SEQ_LEN",
    "SWEHERO_BUCKET_CP",
    "SWEHERO_ALLOW_EMPTY_RANK_REUSE",
    "SWEHERO_TOTAL_STEPS",
    "SWEHERO_CUMULATIVE_STEPS",
    "SWEHERO_WARMUP_STEPS",
    "SWEHERO_HF_ASSETS_PATH",
    "SWEHERO_TORCHTITAN_DUMP_FOLDER",
    "SWEHERO_FINAL_EXPORT_FOLDER",
    "SWEHERO_PAD_TOKEN_ID",
    "SWEHERO_SEED",
    "SWEHERO_GLOBAL_BATCH_SIZE",
    "SWEHERO_LOCAL_BATCH_SIZE",
    "SWEHERO_LEARNING_RATE",
    "SWEHERO_MIN_LEARNING_RATE",
    "SWEHERO_WEIGHT_DECAY",
    "SWEHERO_MAX_GRAD_NORM",
    "SWEHERO_OPTIMIZER_IMPL",
    "SWEHERO_ATTENTION_BACKEND",
    "SWEHERO_ENABLE_FP8",
    "SWEHERO_FP8_RECIPE",
    "SWEHERO_COMPILE",
    "SWEHERO_TRAINING_DTYPE",
    "SWEHERO_MP_PARAM_DTYPE",
    "SWEHERO_MP_REDUCE_DTYPE",
    "SWEHERO_FSDP_RESHARD_AFTER_FORWARD",
    "SWEHERO_AC_MODE",
    "SWEHERO_CHUNKED_CE_CHUNKS",
    "SWEHERO_DETECT_ANOMALY",
    "SWEHERO_SAVE_FINAL_FULL_CHECKPOINT",
    "SWEHERO_ENABLE_FIRST_STEP_CHECKPOINT",
    "SWEHERO_FIRST_STEP_CHECKPOINT_VALIDATION_REPORT",
)
LAUNCH_STAGE_ENV_KEYS = (
    *RESUME_STAGE_ENV_KEYS,
    "SWEHERO_CHECKPOINT_INTERVAL",
    "SWEHERO_CHECKPOINT_ASYNC_MODE",
    "SWEHERO_METRICS_LOG_FREQ",
    "SWEHERO_ENABLE_PROFILER",
    "SWEHERO_PROFILER_TRACE_FOLDER",
    "SWEHERO_PROFILER_FREQ",
    "SWEHERO_PROFILER_ACTIVE",
    "SWEHERO_PROFILER_WARMUP",
    "SWEHERO_PROFILER_REPEAT",
    "SWEHERO_PROFILER_SKIP_FIRST",
    "SWEHERO_PROFILER_SKIP_FIRST_WAIT",
    "SWEHERO_ENABLE_MEMORY_SNAPSHOT",
    "SWEHERO_MEMORY_SNAPSHOT_FOLDER",
    "SWEHERO_LOAD_DATALOADER_STATE",
    "SWEHERO_ENABLE_WANDB",
    "LOG_RANK",
    "WANDB_PROJECT",
    "WANDB_TEAM",
    "WANDB_ENTITY",
    "WANDB_RUN_NAME",
    "WANDB_NAME",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_RESUME_FROM",
    "WANDB_FORK_FROM",
    "WANDB_RUN_GROUP",
    "WANDB_RUN_JOB_TYPE",
    "WANDB_JOB_TYPE",
    "WANDB_RUN_TAGS",
    "WANDB_TAGS",
    "WANDB_RUN_NOTES",
    "WANDB_NOTES",
    "WANDB_MODE",
)


@dataclass(frozen=True)
class BucketStage:
    bucket: int
    cp_degree: int
    example_count: int
    steps: int
    cumulative_steps: int
    bucket_file: Path


@dataclass(frozen=True)
class BucketPlan:
    stages: tuple[BucketStage, ...]
    total_steps: int
    warmup_steps: int


@dataclass(frozen=True)
class ResumeCheckpointState:
    checkpoint_dir: Path
    final_export_dir: Path
    latest_resumable_step: int | None
    latest_model_export_step: int | None
    latest_any_step: int


class LongExampleError(ValueError):
    def __init__(self, *, token_count: int, max_length: int) -> None:
        self.token_count = token_count
        self.max_length = max_length
        self.max_token_count = max_length + 1
        self.shifted_input_length = token_count - 1
        super().__init__(
            f"encoded example has shifted input length {self.shifted_input_length}, "
            f"which exceeds --max-length={max_length}"
        )


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


class SignalTerminationError(RuntimeError):
    def __init__(
        self,
        *,
        signum: int,
        command: list[str],
        returncode: int | None,
        killed_after_grace: bool = False,
    ) -> None:
        self.signum = int(signum)
        self.signal_name = _signal_name(self.signum)
        self.command = list(command)
        self.returncode = returncode
        self.killed_after_grace = killed_after_grace
        message = (
            f"received {self.signal_name}; forwarded it to the torchrun process group"
        )
        if killed_after_grace:
            message += " and sent SIGKILL after the grace period"
        super().__init__(message)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be a boolean env value "
        "(one of 1/0, true/false, yes/no, on/off)"
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite float; got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {raw!r}")
    return value


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return default if raw is None else Path(raw)


def _env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None:
            return raw
    return default


def _default_torchrun_bin() -> str:
    candidate = Path(sys.executable).with_name("torchrun")
    if candidate.exists():
        return str(candidate)
    return "torchrun"


class LaunchArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        stripped = arg_line.strip()
        if not stripped or stripped.startswith("#"):
            return []
        return shlex.split(stripped)


def _argv_for_env_file_scan(argv: list[str] | None) -> list[str]:
    return list(sys.argv[1:] if argv is None else argv)


def _env_file_from_argv(argv: list[str] | None) -> tuple[str, bool]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file")
    namespace, _unknown = pre_parser.parse_known_args(_argv_for_env_file_scan(argv))
    if namespace.env_file is not None:
        if not namespace.env_file.strip():
            raise ValueError("--env-file cannot be empty")
        return namespace.env_file, True

    env_file = os.environ.get("ENV_FILE")
    if env_file is not None:
        if not env_file.strip():
            raise ValueError("ENV_FILE cannot be empty")
        return env_file, True
    return smoke.ENV_FILE, False


def load_launch_env_file(argv: list[str] | None = None) -> str:
    env_file, required = _env_file_from_argv(argv)
    smoke.load_env_file(env_file, required=required)
    return env_file


def parse_bucket_list(raw: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = []
        seen = set()
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"invalid bucket size in --buckets: {part!r}"
                ) from exc
            if value in seen:
                raise ValueError(f"duplicate bucket size in --buckets: {value}")
            seen.add(value)
            values.append(value)
    else:
        values = []
        seen = set()
        for part in raw:
            value = int(part)
            if value in seen:
                raise ValueError(f"duplicate bucket size in --buckets: {value}")
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("at least one sequence bucket is required")
    if any(value <= 0 for value in values):
        raise ValueError(f"bucket sizes must be positive: {values}")
    values = sorted(values)
    return tuple(values)


def parse_bucket_cp_map(raw: str | Mapping[int, int]) -> dict[int, int]:
    if isinstance(raw, Mapping):
        parsed = {int(bucket): int(cp) for bucket, cp in raw.items()}
    else:
        parsed = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    "bucket CP map entries must look like '<bucket>:<cp>'"
                )
            bucket, cp = part.split(":", 1)
            try:
                parsed_bucket = int(bucket.strip())
                parsed_cp = int(cp.strip())
            except ValueError as exc:
                raise ValueError(
                    f"invalid bucket CP map entry in --bucket-cp: {part!r}"
                ) from exc
            if parsed_bucket in parsed:
                raise ValueError(
                    f"duplicate bucket in --bucket-cp: {parsed_bucket}"
                )
            parsed[parsed_bucket] = parsed_cp
    if not parsed:
        raise ValueError("at least one bucket:cp entry is required")
    bad = {bucket: cp for bucket, cp in parsed.items() if bucket <= 0 or cp <= 0}
    if bad:
        raise ValueError(f"bucket and CP values must be positive: {bad}")
    return parsed


def _format_bucket_cp_map(bucket_cp: Mapping[int, int]) -> str:
    return ",".join(f"{bucket}:{bucket_cp[bucket]}" for bucket in sorted(bucket_cp))


def expected_qwen_yarn_rope_config() -> dict[str, Any]:
    return {
        "rope_type": "yarn",
        "max_position_embeddings": PAPER_CONTEXT_LENGTH,
        "original_max_position_embeddings": QWEN_NATIVE_CONTEXT_LENGTH,
        "factor": PAPER_CONTEXT_LENGTH / QWEN_NATIVE_CONTEXT_LENGTH,
        "rope_theta": QWEN_ROPE_THETA,
        "beta_fast": QWEN_YARN_BETA_FAST,
        "beta_slow": QWEN_YARN_BETA_SLOW,
        "backend": "cos_sin",
    }


def choose_bucket(length: int, buckets: Iterable[int]) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    for bucket in sorted(buckets):
        if length <= bucket:
            return bucket
    raise ValueError(f"length {length} exceeds largest bucket {max(buckets)}")


def _add_min_error(
    errors: list[str],
    *,
    name: str,
    value: int | float,
    minimum: int | float,
    inclusive: bool = True,
) -> None:
    invalid = value < minimum if inclusive else value <= minimum
    if invalid:
        comparator = ">=" if inclusive else ">"
        errors.append(f"{name} must be {comparator} {minimum}; got {value!r}")


def _add_max_error(
    errors: list[str],
    *,
    name: str,
    value: int | float,
    maximum: int | float,
    inclusive: bool = True,
) -> None:
    invalid = value > maximum if inclusive else value >= maximum
    if invalid:
        comparator = "<=" if inclusive else "<"
        errors.append(f"{name} must be {comparator} {maximum}; got {value!r}")


def _add_finite_float_error(
    errors: list[str],
    *,
    name: str,
    value: float,
) -> None:
    if not math.isfinite(value):
        errors.append(f"{name} must be finite; got {value!r}")


def _validate_rank_filter(
    errors: list[str],
    *,
    name: str,
    value: str | None,
) -> None:
    if value is None or value == "":
        return
    for part in value.split(","):
        part = part.strip()
        if not part:
            errors.append(f"{name} must be a comma-separated list of ranks")
            return
        try:
            rank = int(part)
        except ValueError:
            errors.append(f"{name} contains a non-integer rank: {part!r}")
            return
        if rank < 0:
            errors.append(f"{name} ranks must be non-negative; got {rank}")
            return


def _rdzv_endpoint_is_local_or_ephemeral(endpoint: str | None) -> bool:
    if endpoint is None or not endpoint.strip():
        return True
    normalized = endpoint.strip()
    host = normalized
    port = ""
    if normalized.startswith("[") and "]:" in normalized:
        host, port = normalized[1:].split("]:", 1)
    elif ":" in normalized:
        host, port = normalized.rsplit(":", 1)
    host = host.strip().lower()
    port = port.strip()
    return host in {"localhost", "127.0.0.1", "::1"} or port in {"", "0"}


def _resolve_for_safety(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path(os.environ.get("PWD") or Path.cwd()) / expanded
    return Path(os.path.normpath(os.fspath(expanded)))


def _detected_workspace_root() -> Path:
    script_path = Path(__file__)
    script_resolved = script_path.resolve(strict=False)
    canonical_root = _absolute_without_symlink_resolution(CANONICAL_WORKSPACE_ROOT)
    canonical_script = canonical_root / "scripts" / script_path.name
    if canonical_script.resolve(strict=False) == script_resolved:
        return canonical_root
    logical_pwd = os.environ.get("PWD")
    if logical_pwd:
        logical_root = _absolute_without_symlink_resolution(Path(logical_pwd))
        logical_script = logical_root / "scripts" / script_path.name
        if logical_script.resolve(strict=False) == script_resolved:
            return logical_root
    if not script_path.is_absolute():
        script_path = Path(os.environ.get("PWD") or Path.cwd()) / script_path
    return _absolute_without_symlink_resolution(script_path.parent.parent)


def _default_workspace_root() -> Path:
    configured = os.environ.get("WORKSPACE_ROOT") or os.environ.get(
        "SWEHERO_WORKSPACE_ROOT"
    )
    if configured:
        return _absolute_without_symlink_resolution(Path(configured))
    return _detected_workspace_root()


def _configured_workspace_root(args: argparse.Namespace) -> Path:
    return _absolute_without_symlink_resolution(args.workspace_root)


def workspace_root_metadata(
    args: argparse.Namespace,
    *,
    include_cwd: bool = False,
) -> dict[str, Any]:
    configured_root = _configured_workspace_root(args)
    script_root = _detected_workspace_root()
    canonical_root = _absolute_without_symlink_resolution(CANONICAL_WORKSPACE_ROOT)
    metadata = {
        "canonical_root": str(canonical_root),
        "configured_root": str(configured_root),
        "script_root": str(script_root),
        "configured_root_resolved": str(_resolve_for_safety(configured_root)),
        "script_root_resolved": str(_resolve_for_safety(script_root)),
        "matches_canonical": (
            configured_root == canonical_root and script_root == canonical_root
        ),
    }
    if include_cwd:
        cwd = _absolute_without_symlink_resolution(Path.cwd())
        logical_cwd = _absolute_without_symlink_resolution(
            Path(os.environ.get("PWD") or cwd)
        )
        metadata.update(
            {
                "cwd": str(cwd),
                "logical_cwd": str(logical_cwd),
                "cwd_resolved": str(_resolve_for_safety(cwd)),
            }
        )
    return metadata


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = _resolve_for_safety(left)
    right_resolved = _resolve_for_safety(right)
    return _path_is_relative_to(left_resolved, right_resolved) or _path_is_relative_to(
        right_resolved,
        left_resolved,
    )


def _destructive_path_safety_errors(name: str, path: Path) -> list[str]:
    resolved = _resolve_for_safety(path)
    repo_root = Path(__file__).resolve().parents[1].resolve()
    protected_paths = [
        Path("/").resolve(),
        Path.home().resolve(),
        repo_root,
        repo_root / ".git",
        repo_root / "scripts",
        repo_root / "tests",
        repo_root / "torchtitan",
        repo_root / "requirements",
        repo_root / "manifests",
        repo_root / "tmp" / "pod-creds",
        Path("/workspace"),
        Path("/workspace/assets"),
        DEFAULT_HF_ASSETS_PATH,
    ]

    errors = []
    if (resolved / ".git").exists():
        errors.append(
            f"dangerous {name} overwrite target {resolved} contains a .git directory"
        )
    for protected in protected_paths:
        protected_resolved = _resolve_for_safety(protected)
        if resolved == protected_resolved or _path_is_relative_to(
            protected_resolved,
            resolved,
        ):
            errors.append(
                f"dangerous {name} overwrite target {resolved} would remove "
                f"protected path {protected_resolved}"
            )
            break
    return errors


def _ensure_safe_destructive_path(name: str, path: Path) -> None:
    errors = _destructive_path_safety_errors(name, path)
    if errors:
        raise ValueError("\n".join(errors))


def _artifact_path_safety_errors(args: argparse.Namespace) -> list[str]:
    errors = []
    path_pairs = (
        ("--out-dir", args.out_dir, "--dataset-path", args.dataset_path),
        ("--out-dir", args.out_dir, "--hf-assets-path", args.hf_assets_path),
        ("--dataset-path", args.dataset_path, "--hf-assets-path", args.hf_assets_path),
    )
    for left_name, left_path, right_name, right_path in path_pairs:
        if _paths_overlap(left_path, right_path):
            errors.append(
                f"{left_name}={_resolve_for_safety(left_path)} overlaps "
                f"{right_name}={_resolve_for_safety(right_path)}"
            )

    if args.overwrite_output:
        errors.extend(_destructive_path_safety_errors("--out-dir", args.out_dir))
    if args.rebuild_source_dataset:
        errors.extend(
            _destructive_path_safety_errors("--dataset-path", args.dataset_path)
        )
    return errors


def _values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    return actual == expected


def _production_mode_errors(
    args: argparse.Namespace,
    *,
    buckets: tuple[int, ...],
    bucket_cp: Mapping[int, int],
) -> list[str]:
    if not args.production_mode:
        return []

    acceptance_smoke = bool(args.production_acceptance_smoke)
    errors: list[str] = []
    forbidden_flags = (
        ("--dry-run", args.dry_run, "it does not launch training"),
        ("--prepare-data-only", args.prepare_data_only, "it exits before training"),
        (
            "--smoke-synthetic-buckets",
            args.smoke_synthetic_buckets,
            "it trains on synthetic launcher records",
        ),
    )
    for name, enabled, reason in forbidden_flags:
        if enabled:
            errors.append(f"--production-mode rejects {name} because {reason}")

    if args.skip_data_prep and not args.resume:
        errors.append(
            "--production-mode rejects --skip-data-prep for fresh launches; "
            "materialize and validate data in the launch or use --resume for "
            "checkpoint continuation"
        )

    configured_workspace_root = _configured_workspace_root(args)
    detected_workspace_root = _detected_workspace_root()
    canonical_workspace_root = _absolute_without_symlink_resolution(
        CANONICAL_WORKSPACE_ROOT
    )
    if configured_workspace_root != canonical_workspace_root:
        errors.append(
            "--production-mode requires canonical workspace root "
            f"--workspace-root={canonical_workspace_root}; got "
            f"{configured_workspace_root}"
        )
    if detected_workspace_root != canonical_workspace_root:
        errors.append(
            "--production-mode requires the launcher script to run from the "
            f"canonical workspace root {canonical_workspace_root}; detected "
            f"{detected_workspace_root}"
        )

    required_values = (
        ("--dataset-id", args.dataset_id, DATASET_ID),
        ("--source-dataset-id", args.source_dataset_id, SOURCE_DATASET_ID),
        (
            "--source-dataset-revision",
            args.source_dataset_revision,
            SOURCE_DATASET_REVISION,
        ),
        ("--num-examples", args.num_examples, DEFAULT_NUM_EXAMPLES),
        (
            "--max-streamed-examples",
            args.max_streamed_examples,
            DEFAULT_MAX_STREAMED_EXAMPLES,
        ),
        ("--max-length", args.max_length, PAPER_CONTEXT_LENGTH),
        (
            "--long-example-policy",
            args.long_example_policy,
            DEFAULT_LONG_EXAMPLE_POLICY,
        ),
        (
            "--min-trainable-tokens",
            args.min_trainable_tokens,
            DEFAULT_MIN_TRAINABLE_TOKENS,
        ),
        (
            "--include-model-patch",
            args.include_model_patch,
            DEFAULT_INCLUDE_MODEL_PATCH,
        ),
        ("--num-train-epochs", args.num_train_epochs, DEFAULT_NUM_TRAIN_EPOCHS),
        ("--max-steps", args.max_steps, DEFAULT_MAX_STEPS),
        ("--global-batch-size", args.global_batch_size, DEFAULT_GLOBAL_BATCH_SIZE),
        ("--local-batch-size", args.local_batch_size, DEFAULT_LOCAL_BATCH_SIZE),
        ("--learning-rate", args.learning_rate, DEFAULT_LEARNING_RATE),
        ("--min-learning-rate", args.min_learning_rate, DEFAULT_MIN_LEARNING_RATE),
        ("--warmup-ratio", args.warmup_ratio, DEFAULT_WARMUP_RATIO),
        ("--weight-decay", args.weight_decay, DEFAULT_WEIGHT_DECAY),
        (
            "--validate-first-step-checkpoint",
            args.validate_first_step_checkpoint,
            DEFAULT_VALIDATE_FIRST_STEP_CHECKPOINT,
        ),
    )
    if acceptance_smoke:
        required_values = required_values[:3] + required_values[-1:]
    for name, actual, expected in required_values:
        if not _values_match(actual, expected):
            errors.append(
                f"--production-mode requires {name}={expected!r}; got "
                f"{actual!r}. This prevents a smoke/prototype launch from "
                "being recorded as the production direct-to-hero run."
            )

    if acceptance_smoke:
        if args.num_examples <= 0:
            errors.append(
                "--production-acceptance-smoke requires --num-examples > 0 "
                "so the run is a bounded real dataset subset."
            )
        if args.max_streamed_examples <= 0:
            errors.append(
                "--production-acceptance-smoke requires "
                "--max-streamed-examples > 0 so the real-data smoke is bounded."
            )
        elif args.num_examples > 0 and args.max_streamed_examples < args.num_examples:
            errors.append(
                "--production-acceptance-smoke requires "
                "--max-streamed-examples >= --num-examples."
            )
        if args.max_steps <= 0:
            errors.append(
                "--production-acceptance-smoke requires --max-steps > 0 "
                "so acceptance cannot accidentally launch the full training run."
            )
    elif buckets != DEFAULT_BUCKETS:
        errors.append(
            "--production-mode requires "
            f"--buckets={','.join(str(bucket) for bucket in DEFAULT_BUCKETS)}; "
            f"got {','.join(str(bucket) for bucket in buckets)}. This preserves "
            "the reviewed full-context bucket plan."
        )
    if not acceptance_smoke and dict(bucket_cp) != DEFAULT_BUCKET_CP:
        errors.append(
            "--production-mode requires "
            f"--bucket-cp={_format_bucket_cp_map(DEFAULT_BUCKET_CP)}; got "
            f"{_format_bucket_cp_map(bucket_cp)}. This preserves the reviewed "
            "per-bucket context-parallel plan."
        )
    if not acceptance_smoke and args.bucket_curriculum != DEFAULT_BUCKET_CURRICULUM:
        errors.append(
            "--production-mode requires "
            f"--bucket-curriculum={DEFAULT_BUCKET_CURRICULUM!r}; got "
            f"{args.bucket_curriculum!r}. Alternate curricula must be launched "
            "as explicit non-production ablations."
        )

    if not args.enable_wandb:
        errors.append(
            "--production-mode requires --enable-wandb so training metrics are "
            "written to a durable backend, not only local stdout/log files."
        )
    elif args.wandb_mode in {"offline", "disabled"}:
        errors.append(
            "--production-mode requires a durable W&B mode; "
            f"--wandb-mode={args.wandb_mode!r} is not durable during training."
        )
    return errors


def validate_launch_inputs(
    args: argparse.Namespace,
    *,
    buckets: tuple[int, ...],
    bucket_cp: Mapping[int, int],
) -> None:
    errors: list[str] = []
    errors.extend(_artifact_path_safety_errors(args))
    if args.production_acceptance_smoke and not args.production_mode:
        errors.append("--production-acceptance-smoke requires --production-mode")
    if args.model_id != MODEL_ID:
        errors.append(
            "--model-id must be "
            f"{MODEL_ID!r} because this launcher always starts the hardcoded "
            "TorchTitan qwen25_coder7b_direct_to_hero config "
            "(model_registry('coder7b')). Use a separate launcher/config pair "
            "for 14B or 32B runs."
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(args.model_revision)):
        errors.append(
            "--model-revision must be an exact 40-character lowercase "
            f"Hugging Face commit SHA; got {args.model_revision!r}"
        )
    elif args.model_revision != MODEL_REVISION:
        errors.append(
            "--model-revision must be the pinned "
            f"{MODEL_ID}@{MODEL_REVISION} revision for this production "
            f"direct-to-hero 7B launcher; got {args.model_revision!r}"
        )

    int_minima = (
        ("--source-dataset-rows-per-shard", args.source_dataset_rows_per_shard, 1),
        ("--source-dataset-build-batch-size", args.source_dataset_build_batch_size, 1),
        ("--num-examples", args.num_examples, 0),
        ("--max-streamed-examples", args.max_streamed_examples, 0),
        ("--shuffle-buffer", args.shuffle_buffer, 0),
        ("--seed", args.seed, 0),
        ("--max-length", args.max_length, 1),
        ("--min-trainable-tokens", args.min_trainable_tokens, 1),
        (
            "--smoke-synthetic-examples-per-bucket",
            args.smoke_synthetic_examples_per_bucket,
            1,
        ),
        ("--max-steps", args.max_steps, 0),
        ("--global-batch-size", args.global_batch_size, 1),
        ("--local-batch-size", args.local_batch_size, 1),
        ("--chunked-ce-chunks", args.chunked_ce_chunks, 1),
        ("--checkpoint-interval", args.checkpoint_interval, 1),
        ("--metrics-log-freq", args.metrics_log_freq, 1),
        ("--profiler-freq", args.profiler_freq, 1),
        ("--profiler-active", args.profiler_active, 1),
        ("--profiler-warmup", args.profiler_warmup, 0),
        ("--write-throughput-probe-mb", args.write_throughput_probe_mb, 1),
        ("--nproc-per-node", args.nproc_per_node, 1),
        ("--nnodes", args.nnodes, 1),
        ("--node-rank", args.node_rank, 0),
    )
    for name, value, minimum in int_minima:
        _add_min_error(errors, name=name, value=value, minimum=minimum)

    optional_int_minima = (
        ("--profiler-repeat", args.profiler_repeat, 1),
        ("--profiler-skip-first", args.profiler_skip_first, 0),
        ("--profiler-skip-first-wait", args.profiler_skip_first_wait, 0),
    )
    for name, value, minimum in optional_int_minima:
        if value is not None:
            _add_min_error(errors, name=name, value=value, minimum=minimum)

    _add_max_error(
        errors,
        name="--seed",
        value=args.seed,
        maximum=2**63 - 1,
    )

    finite_float_fields = (
        ("--num-train-epochs", args.num_train_epochs),
        ("--learning-rate", args.learning_rate),
        ("--min-learning-rate", args.min_learning_rate),
        ("--warmup-ratio", args.warmup_ratio),
        ("--weight-decay", args.weight_decay),
        ("--max-grad-norm", args.max_grad_norm),
        ("--min-free-disk-gb", args.min_free_disk_gb),
        ("--min-free-gpu-memory-gb", args.min_free_gpu_memory_gb),
        ("--min-free-cpu-memory-gb", args.min_free_cpu_memory_gb),
        ("--min-write-throughput-mb-s", args.min_write_throughput_mb_s),
    )
    for name, value in finite_float_fields:
        _add_finite_float_error(errors, name=name, value=value)

    _add_min_error(
        errors,
        name="--num-train-epochs",
        value=args.num_train_epochs,
        minimum=0.0,
        inclusive=False,
    )
    _add_min_error(
        errors,
        name="--learning-rate",
        value=args.learning_rate,
        minimum=0.0,
        inclusive=False,
    )
    _add_min_error(
        errors,
        name="--min-learning-rate",
        value=args.min_learning_rate,
        minimum=0.0,
    )
    _add_min_error(
        errors,
        name="--warmup-ratio",
        value=args.warmup_ratio,
        minimum=0.0,
    )
    _add_max_error(
        errors,
        name="--warmup-ratio",
        value=args.warmup_ratio,
        maximum=1.0,
    )
    _add_min_error(
        errors,
        name="--weight-decay",
        value=args.weight_decay,
        minimum=0.0,
    )
    _add_min_error(
        errors,
        name="--max-grad-norm",
        value=args.max_grad_norm,
        minimum=0.0,
        inclusive=False,
    )
    _add_min_error(
        errors,
        name="--min-free-disk-gb",
        value=args.min_free_disk_gb,
        minimum=0.0,
    )
    _add_min_error(
        errors,
        name="--min-free-gpu-memory-gb",
        value=args.min_free_gpu_memory_gb,
        minimum=0.0,
    )
    _add_min_error(
        errors,
        name="--min-free-cpu-memory-gb",
        value=args.min_free_cpu_memory_gb,
        minimum=0.0,
    )
    _add_min_error(
        errors,
        name="--min-write-throughput-mb-s",
        value=args.min_write_throughput_mb_s,
        minimum=0.0,
    )

    if (
        math.isfinite(args.min_learning_rate)
        and math.isfinite(args.learning_rate)
        and args.min_learning_rate > args.learning_rate
    ):
        errors.append(
            "--min-learning-rate cannot exceed --learning-rate; got "
            f"{args.min_learning_rate!r} > {args.learning_rate!r}"
        )

    choice_fields = (
        ("--optimizer-impl", args.optimizer_impl, OPTIMIZER_IMPL_CHOICES),
        ("--training-dtype", args.training_dtype, TORCH_DTYPE_CHOICES),
        (
            "--mixed-precision-param-dtype",
            args.mixed_precision_param_dtype,
            TORCH_DTYPE_CHOICES,
        ),
        (
            "--mixed-precision-reduce-dtype",
            args.mixed_precision_reduce_dtype,
            TORCH_DTYPE_CHOICES,
        ),
        (
            "--fsdp-reshard-after-forward",
            args.fsdp_reshard_after_forward,
            FSDP_RESHARD_AFTER_FORWARD_CHOICES,
        ),
    )
    for name, value, choices in choice_fields:
        if value not in choices:
            errors.append(
                f"{name} must be one of {', '.join(choices)}; got {value!r}"
            )

    try:
        cuda_device_max_connections = int(args.cuda_device_max_connections)
    except ValueError:
        errors.append(
            "--cuda-device-max-connections must be an integer; got "
            f"{args.cuda_device_max_connections!r}"
        )
    else:
        _add_min_error(
            errors,
            name="--cuda-device-max-connections",
            value=cuda_device_max_connections,
            minimum=1,
        )
    if not str(args.torch_nccl_async_error_handling).strip():
        errors.append("--torch-nccl-async-error-handling cannot be empty")

    if args.max_length > PAPER_CONTEXT_LENGTH:
        errors.append(
            f"--max-length={args.max_length} exceeds paper context "
            f"{PAPER_CONTEXT_LENGTH}"
        )
    if buckets and args.max_length > max(buckets):
        errors.append("--max-length cannot exceed the largest bucket")

    extra_cp_buckets = sorted(set(bucket_cp) - set(buckets))
    if extra_cp_buckets:
        errors.append(
            "--bucket-cp contains buckets not present in --buckets: "
            f"{extra_cp_buckets}"
        )
    if args.bucket_curriculum not in BUCKET_CURRICULUM_CHOICES:
        errors.append(
            "--bucket-curriculum must be one of "
            f"{', '.join(BUCKET_CURRICULUM_CHOICES)}; got "
            f"{args.bucket_curriculum!r}"
        )
    if args.bucket_curriculum == "single-bucket" and len(buckets) != 1:
        errors.append(
            "--bucket-curriculum=single-bucket requires exactly one configured "
            "--buckets entry"
        )

    if args.smoke_synthetic_buckets and args.num_examples != 0:
        errors.append(
            "--smoke-synthetic-buckets cannot be combined with --num-examples; "
            "use --smoke-synthetic-examples-per-bucket instead"
        )
    if args.smoke_synthetic_buckets and args.max_streamed_examples != 0:
        errors.append(
            "--smoke-synthetic-buckets cannot be combined with "
            "--max-streamed-examples because no SWE rows are streamed"
        )
    if (
        not args.smoke_synthetic_buckets
        and args.smoke_synthetic_examples_per_bucket != 1
    ):
        errors.append(
            "--smoke-synthetic-examples-per-bucket only applies when "
            "--smoke-synthetic-buckets is set"
        )
    errors.extend(_production_mode_errors(args, buckets=buckets, bucket_cp=bucket_cp))
    errors.extend(_production_git_state_errors(args))

    if args.profiler_freq < args.profiler_active + args.profiler_warmup:
        errors.append(
            "--profiler-freq must be greater than or equal to "
            "--profiler-active + --profiler-warmup"
        )

    if args.nnodes == 1 and args.node_rank != 0:
        errors.append("--node-rank must be 0 when --nnodes=1")
    if args.nnodes > 1:
        if args.node_rank >= args.nnodes:
            errors.append("--node-rank must be less than --nnodes")
        if not args.rdzv_id:
            errors.append("--rdzv-id is required when --nnodes > 1")
        if _rdzv_endpoint_is_local_or_ephemeral(args.rdzv_endpoint):
            errors.append(
                "--rdzv-endpoint must be a stable host:port reachable from all "
                "nodes when --nnodes > 1"
            )

    if args.nproc_per_node > 0 and args.local_batch_size > 0:
        for bucket in buckets:
            cp = bucket_cp.get(bucket)
            if cp is None or cp <= 0 or args.nproc_per_node % cp != 0:
                continue
            data_parallel_degree = args.nproc_per_node // cp
            microbatch = args.local_batch_size * data_parallel_degree
            if microbatch > 0 and args.global_batch_size % microbatch != 0:
                errors.append(
                    "--global-batch-size must be divisible by "
                    "--local-batch-size * data_parallel_degree for every bucket; "
                    f"bucket {bucket} uses CP={cp}, data_parallel_degree="
                    f"{data_parallel_degree}, so {args.global_batch_size} % "
                    f"{microbatch} != 0"
                )

    _validate_rank_filter(errors, name="--log-rank", value=args.log_rank)
    _validate_rank_filter(
        errors,
        name="--torchrun-log-rank-filter",
        value=args.torchrun_log_rank_filter,
    )

    if errors:
        raise ValueError(
            "Invalid launch inputs:\n" + "\n".join(f"- {error}" for error in errors)
        )


def validate_bucket_config(
    *,
    buckets: tuple[int, ...],
    bucket_cp: Mapping[int, int],
    nproc_per_node: int,
    attention_backend: str,
) -> None:
    missing = [bucket for bucket in buckets if bucket not in bucket_cp]
    if missing:
        raise ValueError(f"missing CP degree for buckets: {missing}")
    for bucket in buckets:
        cp = bucket_cp[bucket]
        if nproc_per_node % cp != 0:
            raise ValueError(
                f"bucket {bucket} uses CP={cp}, which must divide "
                f"--nproc-per-node={nproc_per_node}"
            )
        divisor = 2 * cp
        if bucket % divisor != 0:
            raise ValueError(
                f"bucket {bucket} must be divisible by 2 * CP ({divisor}) "
                "for TorchTitan context parallelism"
            )
    if attention_backend == "varlen" and any(bucket_cp[b] > 1 for b in buckets):
        raise ValueError(
            "TorchTitan VarlenAttention does not support Context Parallelism; "
            "use --attention-backend sdpa/flex/flex_flash for bucketed CP."
        )


def _ceil_steps(example_count: int, epochs: float, global_batch_size: int) -> int:
    if example_count <= 0:
        return 0
    return max(1, math.ceil(example_count * epochs / global_batch_size))


def ordered_buckets_for_curriculum(
    bucket_counts: Mapping[int, int],
    bucket_curriculum: str = DEFAULT_BUCKET_CURRICULUM,
) -> tuple[int, ...]:
    non_empty_buckets = [
        bucket
        for bucket, count in bucket_counts.items()
        if int(count) > 0
    ]
    if bucket_curriculum == "short-to-long":
        return tuple(sorted(non_empty_buckets))
    if bucket_curriculum == "long-to-short":
        return tuple(sorted(non_empty_buckets, reverse=True))
    if bucket_curriculum == "single-bucket":
        if len(non_empty_buckets) != 1:
            raise ValueError(
                "single-bucket curriculum requires exactly one non-empty bucket; "
                f"found {len(non_empty_buckets)}"
            )
        return tuple(sorted(non_empty_buckets))
    raise ValueError(
        "unknown bucket curriculum "
        f"{bucket_curriculum!r}; expected one of {BUCKET_CURRICULUM_CHOICES}"
    )


def build_bucket_plan(
    *,
    bucket_counts: Mapping[int, int],
    bucket_files: Mapping[int, Path],
    bucket_cp: Mapping[int, int],
    epochs: float,
    global_batch_size: int,
    warmup_ratio: float,
    max_steps: int = 0,
    bucket_curriculum: str = DEFAULT_BUCKET_CURRICULUM,
) -> BucketPlan:
    if epochs <= 0 and max_steps <= 0:
        raise ValueError("epochs must be positive unless max_steps is set")
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")

    natural: list[tuple[int, int]] = []
    for bucket in ordered_buckets_for_curriculum(bucket_counts, bucket_curriculum):
        steps = _ceil_steps(bucket_counts[bucket], epochs, global_batch_size)
        if steps > 0:
            natural.append((bucket, steps))
    if not natural:
        raise ValueError("no non-empty buckets found")

    if max_steps > 0:
        remaining = max_steps
        limited: list[tuple[int, int]] = []
        for bucket, steps in natural:
            if remaining <= 0:
                break
            stage_steps = min(steps, remaining)
            limited.append((bucket, stage_steps))
            remaining -= stage_steps
        natural = limited

    cumulative = 0
    stages = []
    for bucket, steps in natural:
        cumulative += steps
        stages.append(
            BucketStage(
                bucket=bucket,
                cp_degree=bucket_cp[bucket],
                example_count=bucket_counts[bucket],
                steps=steps,
                cumulative_steps=cumulative,
                bucket_file=bucket_files[bucket],
            )
        )

    warmup_steps = (
        max(1, math.ceil(cumulative * warmup_ratio))
        if warmup_ratio > 0 and cumulative > 1
        else 0
    )
    return BucketPlan(tuple(stages), cumulative, warmup_steps)


def _resume_contract_path(out_dir: Path) -> Path:
    return out_dir / RESUME_CONTRACT_FILENAME


def _run_spec_path(out_dir: Path) -> Path:
    return out_dir / RUN_SPEC_FILENAME


def _run_spec_sha256_path(out_dir: Path) -> Path:
    return out_dir / RUN_SPEC_SHA256_FILENAME


def _final_artifact_validation_path(out_dir: Path) -> Path:
    return out_dir / FINAL_ARTIFACT_VALIDATION_FILENAME


def _first_step_checkpoint_validation_path(out_dir: Path) -> Path:
    return out_dir / FIRST_STEP_CHECKPOINT_VALIDATION_FILENAME


def _post_training_eval_status_path(out_dir: Path) -> Path:
    return out_dir / POST_TRAINING_EVAL_STATUS_FILENAME


def _runtime_metadata_path(out_dir: Path) -> Path:
    return out_dir / RUNTIME_METADATA_FILENAME


def _torchrun_logs_dir(out_dir: Path) -> Path:
    return out_dir / "torchrun_logs"


def _stage_attempt_log_paths(
    out_dir: Path,
    *,
    stage_id: str,
    attempt_number: int,
) -> dict[str, str]:
    base = _torchrun_logs_dir(out_dir) / f"{stage_id}-attempt-{attempt_number:02d}"
    return {
        "stdout": str(base.with_suffix(".stdout.log")),
        "stderr": str(base.with_suffix(".stderr.log")),
    }


def _launch_lock_path(out_dir: Path) -> Path:
    return out_dir.with_name(out_dir.name + LAUNCH_LOCK_SUFFIX)


def _stage_status_path(out_dir: Path) -> Path:
    return out_dir / STAGE_STATUS_FILENAME


def _wandb_identity_path(out_dir: Path) -> Path:
    return out_dir / WANDB_IDENTITY_FILENAME


def _torchtitan_dump_dir(out_dir: Path) -> Path:
    return out_dir / "torchtitan"


def _checkpoint_dir(out_dir: Path) -> Path:
    return _torchtitan_dump_dir(out_dir) / "checkpoint"


def _final_model_export_dir(out_dir: Path) -> Path:
    return _torchtitan_dump_dir(out_dir) / FINAL_MODEL_EXPORT_FOLDER


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _launch_lock_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_LOCK_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "pid": os.getpid(),
        "hostname": platform.node(),
        "out_dir": str(args.out_dir),
        "lock_path": str(_launch_lock_path(args.out_dir)),
        "workspace_root": str(_configured_workspace_root(args)),
        "production_mode": bool(args.production_mode),
        "production_acceptance_smoke": bool(args.production_acceptance_smoke),
        "resume": bool(args.resume),
        "overwrite_output": bool(args.overwrite_output),
        "argv": sys.argv,
    }


def _read_launch_lock(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


class OutDirLaunchLock:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.path = _launch_lock_path(args.out_dir)
        self._acquired = False

    def __enter__(self) -> "OutDirLaunchLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _launch_lock_payload(self.args)
        payload_text = _canonical_json_text(payload)
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            existing = _read_launch_lock(self.path)
            if existing is None:
                detail = "existing lock could not be parsed"
            else:
                detail = (
                    f"pid={existing.get('pid')!r}, "
                    f"hostname={existing.get('hostname')!r}, "
                    f"created_at_unix={existing.get('created_at_unix')!r}"
                )
            raise RuntimeError(
                "Launch lock already exists for this output directory: "
                f"{self.path} ({detail}). Another launcher may be using "
                "the same --out-dir; remove the lock only after confirming "
                "no launch is still running."
            ) from exc

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload_text)
        except BaseException:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            raise
        self._acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._acquired = False


def launch_lock(args: argparse.Namespace) -> OutDirLaunchLock:
    return OutDirLaunchLock(args)


def _has_dcp_checkpoint_metadata(path: Path) -> bool:
    return (path / ".metadata").is_file()


def _checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []

    steps = []
    for path in checkpoint_dir.iterdir():
        step = _checkpoint_step(path)
        if step is None:
            continue
        if _has_dcp_checkpoint_metadata(path):
            steps.append(step)
    return sorted(steps)


def _model_export_steps(export_dir: Path) -> list[int]:
    if not export_dir.is_dir():
        return []

    steps = []
    for path in export_dir.iterdir():
        step = _checkpoint_step(path)
        if step is not None and (path / "model.safetensors.index.json").is_file():
            steps.append(step)
    return sorted(steps)


def _legacy_model_export_steps(checkpoint_dir: Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []

    steps = []
    for path in checkpoint_dir.iterdir():
        step = _checkpoint_step(path)
        if step is None:
            continue
        if not _has_dcp_checkpoint_metadata(path) and (
            path / "model.safetensors.index.json"
        ).is_file():
            steps.append(step)
    return sorted(steps)


def validate_resume_request(args: argparse.Namespace) -> ResumeCheckpointState | None:
    if not args.resume:
        return None
    if args.overwrite_output:
        raise ValueError("--resume cannot be combined with --overwrite-output")
    if args.rebuild_source_dataset:
        raise ValueError("--resume cannot be combined with --rebuild-source-dataset")
    if args.download_hf_assets:
        raise ValueError("--resume cannot be combined with --download-hf-assets")

    checkpoint_dir = _checkpoint_dir(args.out_dir)
    final_export_dir = _final_model_export_dir(args.out_dir)
    resumable_steps = _checkpoint_steps(checkpoint_dir)
    model_export_steps = _model_export_steps(final_export_dir)
    legacy_export_steps = _legacy_model_export_steps(checkpoint_dir)
    all_steps = sorted(
        {*resumable_steps, *model_export_steps, *legacy_export_steps}
    )
    if not all_steps:
        if not checkpoint_dir.is_dir() and not final_export_dir.is_dir():
            raise FileNotFoundError(
                "--resume requires an existing TorchTitan checkpoint directory "
                f"or final export directory: {checkpoint_dir} or {final_export_dir}"
            )
        raise FileNotFoundError(
            "--resume found no DCP checkpoints or final model exports under "
            f"{checkpoint_dir} or {final_export_dir}"
        )

    latest_model_export_step = (
        max([*model_export_steps, *legacy_export_steps])
        if (model_export_steps or legacy_export_steps)
        else None
    )
    return ResumeCheckpointState(
        checkpoint_dir=checkpoint_dir,
        final_export_dir=final_export_dir,
        latest_resumable_step=max(resumable_steps) if resumable_steps else None,
        latest_model_export_step=latest_model_export_step,
        latest_any_step=max(all_steps),
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    return value


def _canonical_json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256_text(text)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _resume_manifest_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tokenizer = manifest.get("tokenizer")
    tokenizer_contract = {}
    if isinstance(tokenizer, Mapping):
        tokenizer_contract = {
            "hf_assets_path": tokenizer.get("hf_assets_path"),
            "tokenizer_json_sha256": tokenizer.get("tokenizer_json_sha256"),
            "tokenizer_config_sha256": tokenizer.get("tokenizer_config_sha256"),
            "chat_template_sha256": tokenizer.get("chat_template_sha256"),
            "bos_id": tokenizer.get("bos_id"),
            "eos_id": tokenizer.get("eos_id"),
            "pad_id": tokenizer.get("pad_id"),
            "trace_serializer": tokenizer.get("trace_serializer"),
        }

    return {
        "materialized_data_schema_version": manifest.get(
            "materialized_data_schema_version"
        ),
        "model_id": manifest.get("model_id"),
        "model_revision": manifest.get("model_revision"),
        "dataset_id": manifest.get("dataset_id"),
        "dataset_path": manifest.get("dataset_path"),
        "dataset_artifact": manifest.get("dataset_artifact"),
        "source_dataset_id": manifest.get("source_dataset_id"),
        "source_dataset_revision": manifest.get("source_dataset_revision"),
        "model_assets": manifest.get("model_assets"),
        "data_provenance": manifest.get("data_provenance"),
        "tokenizer": tokenizer_contract,
        "pad_token_id": manifest.get("pad_token_id"),
        "max_length": manifest.get("max_length"),
        "long_example_policy": manifest.get("long_example_policy"),
        "buckets": manifest.get("buckets"),
        "bucket_files": manifest.get("bucket_files"),
        "bucket_file_integrity": manifest.get("bucket_file_integrity"),
        "bucket_counts": manifest.get("bucket_counts"),
        "num_usable_examples": manifest.get("num_usable_examples"),
        "streamed_examples_scanned": manifest.get("streamed_examples_scanned"),
        "skipped": manifest.get("skipped"),
        "include_model_patch": manifest.get("include_model_patch"),
    }


def build_resume_contract(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    pad_token_id = int(manifest["pad_token_id"])
    stages = []
    for stage in plan.stages:
        env = build_stage_env(
            args,
            stage=stage,
            total_steps=plan.total_steps,
            warmup_steps=plan.warmup_steps,
            pad_token_id=pad_token_id,
        )
        stages.append(
            {
                "stage": {
                    **asdict(stage),
                    "bucket_file": str(stage.bucket_file),
                },
                "env": {
                    key: env[key]
                    for key in RESUME_STAGE_ENV_KEYS
                    if key in env
                },
            }
        )

    return {
        "schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "args": {
            field: _jsonable(getattr(args, field))
            for field in RESUME_ARG_FIELDS
        },
        "workspace": workspace_root_metadata(args),
        "git": git_state_for_workspace(_configured_workspace_root(args)),
        "manifest": _resume_manifest_contract(manifest),
        "plan": {
            "total_steps": plan.total_steps,
            "warmup_steps": plan.warmup_steps,
            "stages": [
                {
                    **asdict(stage),
                    "bucket_file": str(stage.bucket_file),
                }
                for stage in plan.stages
            ],
        },
        "stage_env": stages,
    }


def _write_resume_contract(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
) -> None:
    contract = build_resume_contract(args, plan, manifest)
    _resume_contract_path(args.out_dir).write_text(json.dumps(contract, indent=2))


def _contract_diffs(expected: object, actual: object, path: str = "$") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        diffs = []
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            child_path = f"{path}.{key}"
            if key not in expected:
                diffs.append(f"{child_path}: unexpected current value {actual[key]!r}")
            elif key not in actual:
                diffs.append(
                    f"{child_path}: missing current value; expected {expected[key]!r}"
                )
            else:
                diffs.extend(_contract_diffs(expected[key], actual[key], child_path))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        diffs = []
        if len(expected) != len(actual):
            diffs.append(
                f"{path}: expected list length {len(expected)}, found {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            diffs.extend(
                _contract_diffs(expected_item, actual_item, f"{path}[{index}]")
            )
        return diffs
    if expected != actual:
        return [f"{path}: expected {expected!r}, found {actual!r}"]
    return []


def validate_resume_contract(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
) -> None:
    contract_path = _resume_contract_path(args.out_dir)
    if not contract_path.exists():
        raise RuntimeError(
            f"--resume cannot be validated because {contract_path} is missing. "
            "Start a fresh run with the current launcher so future resumes have "
            "a recorded config contract."
        )

    expected = json.loads(contract_path.read_text())
    actual = build_resume_contract(args, plan, manifest)
    diffs = _contract_diffs(expected, actual)
    if diffs:
        preview = "\n".join(f"- {diff}" for diff in diffs[:20])
        extra = "" if len(diffs) <= 20 else f"\n... and {len(diffs) - 20} more"
        raise RuntimeError(
            "--resume launch config does not match the original run contract:\n"
            f"{preview}{extra}"
        )


def validate_resume_progress(
    plan: BucketPlan,
    resume_state: ResumeCheckpointState,
) -> None:
    if resume_state.latest_any_step > plan.total_steps:
        raise RuntimeError(
            f"latest checkpoint step {resume_state.latest_any_step} exceeds the "
            f"current plan total_steps {plan.total_steps}; refusing incompatible resume"
        )

    if resume_state.latest_model_export_step is not None:
        if resume_state.latest_model_export_step == plan.total_steps:
            if resume_state.latest_resumable_step == plan.total_steps:
                return
            raise RuntimeError(
                "final model export exists at the plan total step, but the "
                "matching final full DCP checkpoint is missing; refusing to "
                "treat a non-resumable export as a complete production run"
            )
        if (
            resume_state.latest_resumable_step is not None
            and resume_state.latest_resumable_step == plan.total_steps
        ):
            return
        latest_resumable = resume_state.latest_resumable_step or -1
        if resume_state.latest_model_export_step > latest_resumable:
            raise RuntimeError(
                "latest checkpoint is a non-resumable export without optimizer/train-state "
                f"metadata at step {resume_state.latest_model_export_step}, while "
                f"latest full DCP checkpoint is step {resume_state.latest_resumable_step}. "
                "Refusing because TorchTitan can only resume from full DCP "
                "checkpoints."
            )

    if resume_state.latest_resumable_step is None:
        raise RuntimeError(
            "--resume requires at least one full DCP checkpoint with optimizer, "
            "scheduler, and train-state metadata unless the separated final model "
            "export already completes the run."
        )


def stages_to_run_for_resume(
    plan: BucketPlan,
    resume_state: ResumeCheckpointState | None,
) -> tuple[BucketStage, ...]:
    if resume_state is None:
        return plan.stages
    if resume_state.latest_model_export_step == plan.total_steps:
        return ()
    progress_step = resume_state.latest_resumable_step or 0
    return tuple(
        stage
        for stage in plan.stages
        if stage.cumulative_steps > progress_step
    )


def should_load_dataloader_state_for_stage(
    *,
    stage_start_step: int,
    stage: BucketStage,
    resume_state: ResumeCheckpointState | None,
) -> bool:
    if resume_state is None or resume_state.latest_resumable_step is None:
        return False
    checkpoint_step = resume_state.latest_resumable_step
    return stage_start_step < checkpoint_step < stage.cumulative_steps


def dataloader_resume_flags_by_stage(
    plan: BucketPlan,
    resume_state: ResumeCheckpointState | None,
) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    stage_start_step = 0
    for stage in plan.stages:
        flags[stage.cumulative_steps] = should_load_dataloader_state_for_stage(
            stage_start_step=stage_start_step,
            stage=stage,
            resume_state=resume_state,
        )
        stage_start_step = stage.cumulative_steps
    return flags


def parse_args(
    argv: list[str] | None = None,
    *,
    env_file_default: str | None = None,
) -> argparse.Namespace:
    env_file_default = (
        os.environ.get("ENV_FILE", smoke.ENV_FILE)
        if env_file_default is None
        else env_file_default
    )
    parser = LaunchArgumentParser(
        description="Materialize bucketed SWE-HERO training data and launch TorchTitan.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", MODEL_ID))
    parser.add_argument(
        "--model-revision",
        default=(
            os.environ.get("MODEL_REVISION")
            or os.environ.get("HF_MODEL_REVISION")
            or MODEL_REVISION
        ),
        help=(
            "Pinned Hugging Face model commit SHA. This production 7B launcher "
            f"requires {MODEL_ID}@{MODEL_REVISION} so asset downloads cannot float."
        ),
    )
    parser.add_argument(
        "--dataset-id", default=os.environ.get("DATASET_ID", DATASET_ID)
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("DATASET_REVISION"),
        help=(
            "Deprecated alias for --source-dataset-revision. Kept so older "
            "pod commands still pin the same source revision."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=_env_path("DATASET_PATH", DEFAULT_DATASET_PATH),
        help=(
            "Local one-rollout SWE-Hero dataset artifact. On pods this defaults "
            "to /workspace/datasets/... and is built automatically when missing."
        ),
    )
    parser.add_argument(
        "--source-dataset-id",
        default=os.environ.get("SOURCE_DATASET_ID", SOURCE_DATASET_ID),
        help="Public source dataset used to build --dataset-path when missing.",
    )
    parser.add_argument(
        "--source-dataset-revision",
        default=(
            os.environ.get("SOURCE_DATASET_REVISION")
            or os.environ.get("DATASET_REVISION")
            or SOURCE_DATASET_REVISION
        ),
        help="Pinned source dataset revision used to build --dataset-path.",
    )
    parser.add_argument(
        "--build-dataset-if-missing",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("BUILD_DATASET_IF_MISSING", True),
        help="Build --dataset-path from the pinned source dataset when it is absent.",
    )
    parser.add_argument(
        "--rebuild-source-dataset",
        action="store_true",
        default=_env_flag("REBUILD_SOURCE_DATASET", False),
        help="Rebuild --dataset-path from the pinned source dataset before tokenizing.",
    )
    parser.add_argument(
        "--source-dataset-rows-per-shard",
        type=int,
        default=_env_int(
            "SOURCE_DATASET_ROWS_PER_SHARD", one_rollout.DEFAULT_ROWS_PER_SHARD
        ),
        help="Rows per Parquet shard when building --dataset-path.",
    )
    parser.add_argument(
        "--source-dataset-build-batch-size",
        type=int,
        default=_env_int("SOURCE_DATASET_BUILD_BATCH_SIZE", 64),
        help="Parquet read batch size when building --dataset-path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_env_path("OUT_DIR", DEFAULT_OUT_DIR),
        help="Run folder on the pod. Bucket JSONL, manifests, and checkpoints live here.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=_default_workspace_root(),
        help=(
            "Repository workspace root used for launcher subprocesses and "
            "provenance. Production mode requires /workspace/jaxels-work-trial."
        ),
    )
    parser.add_argument(
        "--hf-assets-path",
        type=Path,
        default=_env_path("HF_ASSETS_PATH", DEFAULT_HF_ASSETS_PATH),
        help="Local directory containing Qwen tokenizer/config/safetensors assets.",
    )
    parser.add_argument(
        "--env-file",
        default=env_file_default,
        help=(
            "Optional dotenv-style file loaded before argument defaults are "
            "resolved. CLI flags override process env, and process env overrides "
            "values in this file."
        ),
    )
    parser.add_argument(
        "--download-hf-assets",
        action="store_true",
        default=_env_flag("DOWNLOAD_HF_ASSETS", False),
        help="Download tokenizer/config/safetensors into --hf-assets-path parent first.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Optional Hugging Face token for asset download.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=_env_int("NUM_EXAMPLES", DEFAULT_NUM_EXAMPLES),
        help="Usable examples to keep. Defaults to 0, meaning all examples.",
    )
    parser.add_argument(
        "--max-streamed-examples",
        type=int,
        default=_env_int("MAX_STREAMED_EXAMPLES", DEFAULT_MAX_STREAMED_EXAMPLES),
        help="Maximum raw streamed rows to inspect while finding usable examples.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=_env_int("SHUFFLE_BUFFER", 2_048),
        help="Streaming shuffle buffer. Set 0 to keep HF order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("SEED", 17),
        help="Dataset shuffle and TorchTitan seed.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=_env_int("MAX_LENGTH", PAPER_CONTEXT_LENGTH),
        help="Maximum shifted input length. Defaults to the paper 128k context.",
    )
    parser.add_argument(
        "--long-example-policy",
        choices=("error", "skip"),
        default=os.environ.get("LONG_EXAMPLE_POLICY", DEFAULT_LONG_EXAMPLE_POLICY),
        help=(
            "How to handle examples whose shifted input length exceeds "
            "--max-length. The production default is error so over-context "
            "trajectories cannot be silently truncated."
        ),
    )
    parser.add_argument(
        "--smoke-synthetic-buckets",
        action="store_true",
        default=_env_flag("SMOKE_SYNTHETIC_BUCKETS", False),
        help=(
            "Smoke-only mode: materialize synthetic tokenized records so every "
            "configured bucket/CP stage is non-empty. Do not use for a real "
            "training run."
        ),
    )
    parser.add_argument(
        "--smoke-synthetic-examples-per-bucket",
        type=int,
        default=_env_int("SMOKE_SYNTHETIC_EXAMPLES_PER_BUCKET", 1),
        help=(
            "Synthetic records per bucket when --smoke-synthetic-buckets is set."
        ),
    )
    parser.add_argument(
        "--buckets",
        default=os.environ.get(
            "SWEHERO_BUCKETS", ",".join(str(b) for b in DEFAULT_BUCKETS)
        ),
        help="Comma-separated sequence buckets.",
    )
    parser.add_argument(
        "--bucket-curriculum",
        default=os.environ.get(
            "SWEHERO_BUCKET_CURRICULUM", DEFAULT_BUCKET_CURRICULUM
        ),
        choices=BUCKET_CURRICULUM_CHOICES,
        help=(
            "Order non-empty bucket stages. short-to-long preserves the "
            "current throughput-oriented default; single-bucket documents a "
            "no-length-curriculum run and requires exactly one bucket."
        ),
    )
    parser.add_argument(
        "--bucket-cp",
        default=os.environ.get(
            "SWEHERO_BUCKET_CP", _format_bucket_cp_map(DEFAULT_BUCKET_CP)
        ),
        help="Comma-separated '<bucket>:<context-parallel-degree>' entries.",
    )
    parser.add_argument(
        "--min-trainable-tokens",
        type=int,
        default=_env_int("MIN_TRAINABLE_TOKENS", DEFAULT_MIN_TRAINABLE_TOKENS),
        help="Drop examples with fewer trainable shifted labels than this.",
    )
    parser.add_argument(
        "--include-model-patch",
        action="store_true",
        default=_env_flag("INCLUDE_MODEL_PATCH", DEFAULT_INCLUDE_MODEL_PATCH),
        help="Also train on the model_patch field when present.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=_env_float("NUM_TRAIN_EPOCHS", DEFAULT_NUM_TRAIN_EPOCHS),
        help="SFT epochs over the materialized subset. Paper uses up to 3.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int("MAX_STEPS", DEFAULT_MAX_STEPS),
        help="Optional total optimizer-step cap across all bucket stages.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=_env_int("GLOBAL_BATCH_SIZE", DEFAULT_GLOBAL_BATCH_SIZE),
        help="Paper global batch size is 32.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=_env_int("LOCAL_BATCH_SIZE", DEFAULT_LOCAL_BATCH_SIZE),
        help="TorchTitan local microbatch size per data-parallel rank.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_env_float("LEARNING_RATE", DEFAULT_LEARNING_RATE),
        help="Peak AdamW learning rate. Paper uses 1e-5.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=_env_float("MIN_LEARNING_RATE", DEFAULT_MIN_LEARNING_RATE),
        help="Cosine floor learning rate. Paper uses 1e-8.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=_env_float("WARMUP_RATIO", DEFAULT_WARMUP_RATIO),
        help="Warmup ratio. Paper uses 0.1.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=_env_float("WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY),
        help="AdamW weight decay. The paper does not report this.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=_env_float("MAX_GRAD_NORM", 1.0),
    )
    parser.add_argument(
        "--optimizer-impl",
        choices=OPTIMIZER_IMPL_CHOICES,
        default=os.environ.get("SWEHERO_OPTIMIZER_IMPL", "foreach"),
        help=(
            "TorchTitan AdamW implementation. This used to be an implicit "
            "SWEHERO_OPTIMIZER_IMPL environment input; it is now part of the "
            "recorded launch contract."
        ),
    )
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "flex", "flex_flash", "varlen"),
        default=os.environ.get("ATTENTION_BACKEND", "sdpa"),
        help="Use sdpa/flex/flex_flash for CP. Varlen is allowed only when CP=1.",
    )
    parser.add_argument(
        "--enable-fp8",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ENABLE_FP8", True),
        help="Use TorchTitan float8 linear training where safe and supported.",
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("rowwise", "rowwise_with_gw_hp"),
        default=os.environ.get("FP8_RECIPE", "rowwise"),
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("COMPILE", True),
        help="Torch compile model/loss. Keep enabled for FP8 performance.",
    )
    parser.add_argument(
        "--training-dtype",
        choices=TORCH_DTYPE_CHOICES,
        default=os.environ.get("SWEHERO_TRAINING_DTYPE", "float32"),
        help=(
            "TorchTitan training dtype. Defaults to the current direct-to-hero "
            "config value and is exported as SWEHERO_TRAINING_DTYPE."
        ),
    )
    parser.add_argument(
        "--mixed-precision-param-dtype",
        choices=TORCH_DTYPE_CHOICES,
        default=os.environ.get("SWEHERO_MP_PARAM_DTYPE", "bfloat16"),
        help="FSDP/autocast parameter dtype exported as SWEHERO_MP_PARAM_DTYPE.",
    )
    parser.add_argument(
        "--mixed-precision-reduce-dtype",
        choices=TORCH_DTYPE_CHOICES,
        default=os.environ.get("SWEHERO_MP_REDUCE_DTYPE", "bfloat16"),
        help="FSDP reduction dtype exported as SWEHERO_MP_REDUCE_DTYPE.",
    )
    parser.add_argument(
        "--fsdp-reshard-after-forward",
        choices=FSDP_RESHARD_AFTER_FORWARD_CHOICES,
        default=os.environ.get("SWEHERO_FSDP_RESHARD_AFTER_FORWARD", "never"),
        help=(
            "TorchTitan FSDP reshard policy. Defaults to the current "
            "direct-to-hero config value and is recorded in the run spec."
        ),
    )
    parser.add_argument(
        "--activation-checkpoint-mode",
        choices=("full", "selective", "memory_budget", "none"),
        default=os.environ.get("ACTIVATION_CHECKPOINT_MODE", "full"),
    )
    parser.add_argument(
        "--chunked-ce-chunks",
        type=int,
        default=_env_int("CHUNKED_CE_CHUNKS", 8),
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=_env_int("CHECKPOINT_INTERVAL", 25),
    )
    parser.add_argument(
        "--checkpoint-async-mode",
        choices=("disabled", "async", "async_with_pinned_mem"),
        default=os.environ.get("CHECKPOINT_ASYNC_MODE", "async"),
    )
    parser.add_argument(
        "--validate-first-step-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=_env_flag(
            "VALIDATE_FIRST_STEP_CHECKPOINT",
            DEFAULT_VALIDATE_FIRST_STEP_CHECKPOINT,
        ),
        help=(
            "Ask TorchTitan to save and validate a full DCP checkpoint after "
            "optimizer step 1, then require the validation report before the "
            "launcher continues. Keep enabled for production."
        ),
    )
    parser.add_argument(
        "--metrics-log-freq",
        type=int,
        default=_env_int("METRICS_LOG_FREQ", 1),
    )
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=float(os.environ.get("MIN_FREE_DISK_GB", DEFAULT_MIN_FREE_DISK_GB)),
        help="Minimum free GiB required on --out-dir filesystem before launch.",
    )
    parser.add_argument(
        "--min-free-gpu-memory-gb",
        type=float,
        default=float(
            os.environ.get(
                "MIN_FREE_GPU_MEMORY_GB",
                DEFAULT_MIN_FREE_GPU_MEMORY_GB,
            )
        ),
        help="Minimum free GiB required on each visible training GPU.",
    )
    parser.add_argument(
        "--min-free-cpu-memory-gb",
        type=float,
        default=float(
            os.environ.get(
                "MIN_FREE_CPU_MEMORY_GB",
                DEFAULT_MIN_FREE_CPU_MEMORY_GB,
            )
        ),
        help="Minimum available system memory GiB required before launch.",
    )
    parser.add_argument(
        "--min-write-throughput-mb-s",
        type=float,
        default=float(
            os.environ.get(
                "MIN_WRITE_THROUGHPUT_MB_S",
                DEFAULT_MIN_WRITE_THROUGHPUT_MB_S,
            )
        ),
        help="Minimum write throughput MiB/s required on --out-dir filesystem.",
    )
    parser.add_argument(
        "--write-throughput-probe-mb",
        type=int,
        default=_env_int(
            "WRITE_THROUGHPUT_PROBE_MB",
            DEFAULT_WRITE_THROUGHPUT_PROBE_MB,
        ),
        help="MiB to write for the output filesystem throughput probe.",
    )
    parser.add_argument(
        "--enable-profiler",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ENABLE_PROFILER", False),
        help="Enable TorchTitan torch.profiler traces. Default is disabled.",
    )
    parser.add_argument(
        "--profiler-trace-folder",
        default=os.environ.get("PROFILER_TRACE_FOLDER", "profiling/traces"),
        help="Profiler trace folder relative to the TorchTitan dump folder.",
    )
    parser.add_argument(
        "--profiler-freq",
        type=int,
        default=_env_int("PROFILER_FREQ", 10),
        help="Torch profiler schedule period in training steps.",
    )
    parser.add_argument(
        "--profiler-active",
        type=int,
        default=_env_int("PROFILER_ACTIVE", 1),
        help="Active profiler steps per profiling period.",
    )
    parser.add_argument(
        "--profiler-warmup",
        type=int,
        default=_env_int("PROFILER_WARMUP", 3),
        help="Warmup profiler steps per profiling period.",
    )
    parser.add_argument(
        "--profiler-repeat",
        type=int,
        default=_env_optional_int("PROFILER_REPEAT"),
        help="Optional torch.profiler schedule repeat count.",
    )
    parser.add_argument(
        "--profiler-skip-first",
        type=int,
        default=_env_optional_int("PROFILER_SKIP_FIRST"),
        help="Optional initial profiler cycles to skip.",
    )
    parser.add_argument(
        "--profiler-skip-first-wait",
        type=int,
        default=_env_optional_int("PROFILER_SKIP_FIRST_WAIT"),
        help="Optional initial profiler wait cycles to skip.",
    )
    parser.add_argument(
        "--enable-memory-snapshot",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ENABLE_MEMORY_SNAPSHOT", False),
        help="Enable TorchTitan CUDA memory snapshots. Default is disabled.",
    )
    parser.add_argument(
        "--memory-snapshot-folder",
        default=os.environ.get(
            "MEMORY_SNAPSHOT_FOLDER",
            "profiling/memory_snapshot",
        ),
        help="Memory snapshot folder relative to the TorchTitan dump folder.",
    )
    parser.add_argument(
        "--detect-anomaly",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SWEHERO_DETECT_ANOMALY", False),
        help=(
            "Enable Torch autograd anomaly detection through TorchTitan. This "
            "used to be an unrecorded SWEHERO_DETECT_ANOMALY environment input."
        ),
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        default=_env_flag("ENABLE_WANDB", False),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", smoke.WANDB_PROJECT),
    )
    parser.add_argument(
        "--wandb-entity",
        "--wandb-team",
        dest="wandb_entity",
        default=_env_first("WANDB_TEAM", "WANDB_ENTITY"),
        help=(
            "Optional W&B entity/team. Passed to TorchTitan through WANDB_TEAM "
            "and also exported as WANDB_ENTITY for SDK compatibility."
        ),
    )
    parser.add_argument(
        "--wandb-run-name",
        default=_env_first(
            "WANDB_RUN_NAME",
            "WANDB_NAME",
            default="qwen25-coder7b-swehero-tt",
        ),
    )
    parser.add_argument(
        "--wandb-run-id",
        default=os.environ.get("WANDB_RUN_ID"),
        help=(
            "Stable W&B run id. If W&B is enabled and omitted for a fresh "
            "launch, the launcher generates one and persists it in "
            f"{WANDB_IDENTITY_FILENAME}."
        ),
    )
    parser.add_argument(
        "--wandb-resume",
        choices=WANDB_RESUME_CHOICES,
        default=os.environ.get("WANDB_RESUME"),
        help=(
            "W&B resume policy for runs with --wandb-run-id. Defaults to "
            "'allow' when W&B is enabled, matching W&B's recommended explicit "
            "run-id resume pattern."
        ),
    )
    parser.add_argument(
        "--wandb-resume-from",
        default=os.environ.get("WANDB_RESUME_FROM"),
        help="Optional W&B resume_from value, for example '<run_id>?_step=<step>'.",
    )
    parser.add_argument(
        "--wandb-fork-from",
        default=os.environ.get("WANDB_FORK_FROM"),
        help="Optional W&B fork_from value, for example '<run_id>?_step=<step>'.",
    )
    parser.add_argument(
        "--wandb-run-group",
        default=os.environ.get("WANDB_RUN_GROUP"),
    )
    parser.add_argument(
        "--wandb-run-job-type",
        default=_env_first("WANDB_RUN_JOB_TYPE", "WANDB_JOB_TYPE"),
    )
    parser.add_argument(
        "--wandb-run-tags",
        default=_env_first("WANDB_RUN_TAGS", "WANDB_TAGS"),
        help="Comma-separated W&B run tags.",
    )
    parser.add_argument(
        "--wandb-run-notes",
        default=_env_first("WANDB_RUN_NOTES", "WANDB_NOTES"),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=WANDB_MODE_CHOICES,
        default=os.environ.get("WANDB_MODE"),
        help="Optional W&B mode such as 'online', 'offline', or 'disabled'.",
    )
    parser.add_argument(
        "--post-training-eval-command",
        default=os.environ.get("POST_TRAINING_EVAL_COMMAND", ""),
        help=(
            "Optional shell command to run after final artifact validation. "
            "The command receives SWEHERO_* environment variables pointing at "
            "the run spec, data manifest, final export, and validation report."
        ),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=_env_int("NPROC_PER_NODE", 8),
        help="GPU processes per node. Target pod is 8xH100.",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=_env_int("NNODES", 1),
        help="Number of torchrun nodes. Defaults to the current single-node pod.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=_env_int("NODE_RANK", 0),
        help="Rank of this node for multi-node torchrun launches.",
    )
    parser.add_argument(
        "--rdzv-backend",
        default=os.environ.get("RDZV_BACKEND", "c10d"),
        help="torchrun rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv-endpoint",
        default=os.environ.get("RDZV_ENDPOINT", "localhost:0"),
        help=(
            "torchrun rendezvous endpoint. Multi-node launches must provide a "
            "stable host:port reachable from every node."
        ),
    )
    parser.add_argument(
        "--rdzv-id",
        default=os.environ.get("RDZV_ID", ""),
        help="torchrun rendezvous id. Required when --nnodes > 1.",
    )
    parser.add_argument(
        "--torchrun-bin",
        default=os.environ.get("TORCHRUN_BIN", _default_torchrun_bin()),
    )
    parser.add_argument(
        "--log-rank",
        default=os.environ.get("LOG_RANK", "0"),
        help="Ranks TorchTitan should log. Passed through LOG_RANK.",
    )
    parser.add_argument(
        "--torchrun-log-rank-filter",
        default=os.environ.get("TORCHRUN_LOG_RANK_FILTER", "0"),
        help="Optional torchrun --local-ranks-filter value.",
    )
    parser.add_argument(
        "--cuda-device-max-connections",
        default=os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        help=(
            "Value exported to CUDA_DEVICE_MAX_CONNECTIONS for torchrun workers. "
            "The previous implicit default was 1."
        ),
    )
    parser.add_argument(
        "--torch-nccl-async-error-handling",
        default=os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
        help=(
            "Value exported to TORCH_NCCL_ASYNC_ERROR_HANDLING for torchrun "
            "workers. The previous implicit default was 1."
        ),
    )
    parser.add_argument(
        "--production-mode",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("PRODUCTION_MODE", False),
        help=(
            "Fail closed unless the launch uses the full reviewed "
            "direct-to-hero recipe. This rejects dry-run, synthetic smoke, "
            "subset, step-capped, and shortened-context settings."
        ),
    )
    parser.add_argument(
        "--production-acceptance-smoke",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("PRODUCTION_ACCEPTANCE_SMOKE", False),
        help=(
            "Explicit final-acceptance exception inside --production-mode. "
            "This keeps production provenance, real-data, checkpoint, export, "
            "validation, and durable W&B gates enabled while allowing a bounded "
            "tiny real dataset subset and step cap."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=_env_flag("DRY_RUN", False),
        help="Materialize data and print torchrun commands without launching.",
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        default=_env_flag("PREPARE_DATA_ONLY", False),
        help="Materialize bucket files and exit.",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        default=_env_flag("SKIP_DATA_PREP", False),
        help="Reuse existing manifest/bucket files under --out-dir/data.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=_env_flag("OVERWRITE_OUTPUT", False),
        help="Remove --out-dir before materializing a fresh run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=_env_flag("RESUME", False),
        help="Reuse an existing checkpoint folder in --out-dir/torchtitan.",
    )
    parser.add_argument(
        "--verify-hf-logits-parity",
        action="store_true",
        default=_env_flag("VERIFY_HF_LOGITS_PARITY", False),
        help=(
            "Before training, run the paper-aligned HF-vs-TorchTitan logits "
            "parity check for the Qwen2.5-Coder-7B-Instruct initial load."
        ),
    )
    args = parser.parse_args(argv)
    if args.dataset_revision:
        args.source_dataset_revision = args.dataset_revision
    args.workspace_root = _configured_workspace_root(args)
    return args


def _run_git(args: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd or Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_state_for_workspace(workspace_root: Path) -> dict[str, Any]:
    repo_root = _absolute_without_symlink_resolution(workspace_root)
    branch = _run_git(["branch", "--show-current"], cwd=repo_root)
    commit = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
    status_short = _run_git(["status", "--short"], cwd=repo_root)
    top_level = _run_git(["rev-parse", "--show-toplevel"], cwd=repo_root)
    available = commit is not None and status_short is not None
    return {
        "schema_version": GIT_STATE_SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "available": available,
        "top_level": top_level,
        "branch": branch,
        "commit": commit,
        "status_short": status_short,
        "dirty": bool(status_short) if status_short is not None else None,
    }


def _production_git_state_errors(args: argparse.Namespace) -> list[str]:
    if not args.production_mode:
        return []

    git_state = git_state_for_workspace(_configured_workspace_root(args))
    if not git_state["available"]:
        return [
            "--production-mode requires Git metadata to prove the launch code "
            f"is clean; could not read Git state under {git_state['repo_root']}. "
            "Install git in the launch environment and run from the repository."
        ]
    if git_state["dirty"]:
        status = str(git_state.get("status_short") or "").strip()
        preview = "\n".join(status.splitlines()[:20])
        extra = (
            ""
            if len(status.splitlines()) <= 20
            else f"\n... and {len(status.splitlines()) - 20} more dirty paths"
        )
        return [
            "--production-mode requires a clean Git worktree. Commit or stash "
            f"dirty changes before launching production:\n{preview}{extra}"
        ]
    return []


def _dataset_revision_info(dataset_id: str, revision: str | None) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id, revision=revision)
        card_data = getattr(info, "card_data", None)
        if card_data is None:
            serialized_card_data: dict[str, Any] = {}
        elif isinstance(card_data, Mapping):
            serialized_card_data = dict(card_data)
        elif hasattr(card_data, "to_dict"):
            serialized_card_data = card_data.to_dict()
        else:
            serialized_card_data = {"repr": repr(card_data)}
        return {
            "requested_revision": revision,
            "resolved_sha": getattr(info, "sha", None),
            "card_data": serialized_card_data,
        }
    except Exception as exc:
        return {
            "requested_revision": revision,
            "resolved_sha": None,
            "lookup_error": repr(exc),
        }


def _training_dataset_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"Expected a Parquet file, found {dataset_path}")
        return [dataset_path]

    data_dir = dataset_path / "data"
    search_dir = data_dir if data_dir.exists() else dataset_path
    files = sorted(search_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet training shards found under {dataset_path}. Expected "
            "a Hugging Face-style dataset directory with data/*.parquet."
        )
    return files


def _dataset_artifact_metadata(dataset_path: Path) -> dict[str, Any]:
    metadata_path = dataset_path / "metadata.json"
    selection_manifest_path = dataset_path / "selection_manifest.jsonl"
    data_files = _training_dataset_files(dataset_path)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            metadata = {"metadata_json_error": repr(exc)}

    return {
        "path": str(dataset_path),
        "realpath": str(dataset_path.resolve()),
        "metadata": metadata,
        "metadata_json": {
            "path": str(metadata_path),
            "exists": metadata_path.exists(),
            "bytes": metadata_path.stat().st_size if metadata_path.exists() else 0,
            "sha256": _hash_file(metadata_path),
        },
        "metadata_json_sha256": _hash_file(metadata_path),
        "selection_manifest": {
            "path": str(selection_manifest_path),
            "exists": selection_manifest_path.exists(),
            "bytes": selection_manifest_path.stat().st_size
            if selection_manifest_path.exists()
            else 0,
            "sha256": _hash_file(selection_manifest_path),
        },
        "selection_manifest_sha256": _hash_file(selection_manifest_path),
        "data_file_count": len(data_files),
        "data_files": [
            {
                "path": str(path),
                "relative_path": path.name
                if dataset_path.is_file()
                else path.relative_to(dataset_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in data_files
        ],
        "total_data_bytes": sum(path.stat().st_size for path in data_files),
    }


def build_source_dataset_command(args: argparse.Namespace) -> list[str]:
    repo_root = _configured_workspace_root(args)
    command = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_swehero_historical_one_rollout.py"),
        "--dataset-id",
        args.source_dataset_id,
        "--revision",
        args.source_dataset_revision,
        "--output-dir",
        str(args.dataset_path),
        "--rows-per-shard",
        str(args.source_dataset_rows_per_shard),
        "--batch-size",
        str(args.source_dataset_build_batch_size),
    ]
    if args.rebuild_source_dataset:
        command.append("--overwrite")
    return command


def ensure_training_dataset(args: argparse.Namespace) -> None:
    if args.rebuild_source_dataset:
        _ensure_safe_destructive_path("--dataset-path", args.dataset_path)

    if args.dataset_path.exists() and not args.rebuild_source_dataset:
        try:
            _training_dataset_files(args.dataset_path)
            return
        except FileNotFoundError as exc:
            if not args.build_dataset_if_missing:
                raise
            if args.dataset_path.is_file() or (
                args.dataset_path.is_dir() and any(args.dataset_path.iterdir())
            ):
                raise FileExistsError(
                    f"{args.dataset_path} exists but is not a valid "
                    "SWE-HERO Parquet dataset; pass --rebuild-source-dataset "
                    "to replace it after verifying the path."
                ) from exc

    if not (args.build_dataset_if_missing or args.rebuild_source_dataset):
        raise FileNotFoundError(
            f"{args.dataset_path} is missing. Either create it on the pod, or "
            "rerun with --build-dataset-if-missing."
        )

    command = build_source_dataset_command(args)
    print("Preparing SWE-Hero training dataset artifact:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=_configured_workspace_root(args))


def load_training_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    data_files = [str(path) for path in _training_dataset_files(args.dataset_path)]
    raw = load_dataset(
        "parquet",
        data_files={"train": data_files},
        split="train",
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return raw


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket_file_stats(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    bytes_read = 0
    records = 0
    with path.open("rb") as handle:
        for line in handle:
            bytes_read += len(line)
            digest.update(line)
            if line.strip():
                records += 1
    return {
        "bytes": bytes_read,
        "records": records,
        "sha256": digest.hexdigest(),
    }


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"json_error": repr(exc)}
    if isinstance(payload, dict):
        return payload
    return {"json_type": type(payload).__name__}


def _asset_file_kind(relative_path: str) -> str:
    name = Path(relative_path).name
    if name == "config.json":
        return "model_config"
    if name == "generation_config.json":
        return "generation_config"
    if name == "model.safetensors.index.json":
        return "safetensors_index"
    if relative_path.endswith(".safetensors"):
        return "safetensors_shard"
    if name.startswith("tokenizer") or name in {
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    }:
        return "tokenizer"
    return "auxiliary"


def _asset_file_inventory(hf_assets_path: Path) -> list[dict[str, Any]]:
    if not hf_assets_path.is_dir():
        raise FileNotFoundError(
            f"Hugging Face asset directory does not exist: {hf_assets_path}"
        )

    files = [
        path
        for path in hf_assets_path.rglob("*")
        if path.is_file()
    ]
    inventory = []
    files = sorted(
        files,
        key=lambda item: item.relative_to(hf_assets_path).as_posix(),
    )
    for path in files:
        relative_path = path.relative_to(hf_assets_path).as_posix()
        inventory.append(
            {
                "path": relative_path,
                "kind": _asset_file_kind(relative_path),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    return inventory


def _core_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "_name_or_path",
        "architectures",
        "auto_map",
        "bos_token_id",
        "eos_token_id",
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "model_type",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "rope_scaling",
        "rope_theta",
        "sliding_window",
        "tie_word_embeddings",
        "torch_dtype",
        "transformers_version",
        "use_sliding_window",
        "vocab_size",
    )
    return {
        key: config[key]
        for key in keys
        if key in config
    }


def _safetensors_index_summary(
    hf_assets_path: Path,
    inventory_by_path: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    index_path = hf_assets_path / "model.safetensors.index.json"
    index = _read_json_if_present(index_path)
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    metadata = index.get("metadata") if isinstance(index, Mapping) else None
    shard_paths = (
        sorted(set(str(path) for path in weight_map.values()))
        if isinstance(weight_map, Mapping)
        else []
    )
    shard_path_set = set(shard_paths)
    shard_files = []
    for shard_path in shard_paths:
        file_record = inventory_by_path.get(shard_path)
        shard_files.append(
            {
                "path": shard_path,
                "present": file_record is not None,
                "bytes": file_record.get("bytes") if file_record else None,
                "sha256": file_record.get("sha256") if file_record else None,
            }
        )

    return {
        "index_path": "model.safetensors.index.json" if index_path.exists() else None,
        "index_sha256": _hash_file(index_path),
        "metadata": metadata if isinstance(metadata, Mapping) else {},
        "weight_map_entries": len(weight_map) if isinstance(weight_map, Mapping) else 0,
        "shard_files": shard_files,
        "unindexed_safetensors_files": [
            record["path"]
            for record in inventory_by_path.values()
            if record.get("kind") == "safetensors_shard"
            and record["path"] not in shard_path_set
        ],
        "index_error": index.get("json_error") if isinstance(index, Mapping) else None,
    }


def _model_asset_provenance(
    *,
    model_id: str,
    model_revision: str,
    hf_assets_path: Path,
    tokenizer_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = _asset_file_inventory(hf_assets_path)
    inventory_by_path = {record["path"]: record for record in inventory}
    config = _read_json_if_present(hf_assets_path / "config.json")
    generation_config = _read_json_if_present(hf_assets_path / "generation_config.json")
    return {
        "schema_version": MODEL_ASSET_PROVENANCE_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "hf_assets_path": str(hf_assets_path),
        "hf_assets_realpath": str(hf_assets_path.resolve()),
        "file_count": len(inventory),
        "total_bytes": sum(int(record["bytes"]) for record in inventory),
        "files": inventory,
        "config": {
            "path": "config.json" if (hf_assets_path / "config.json").exists() else None,
            "sha256": _hash_file(hf_assets_path / "config.json"),
            "summary": _core_config_summary(config),
            "json_error": config.get("json_error"),
        },
        "generation_config": {
            "path": "generation_config.json"
            if (hf_assets_path / "generation_config.json").exists()
            else None,
            "sha256": _hash_file(hf_assets_path / "generation_config.json"),
            "summary": {
                key: generation_config[key]
                for key in ("bos_token_id", "eos_token_id", "pad_token_id")
                if key in generation_config
            },
            "json_error": generation_config.get("json_error"),
        },
        "safetensors": _safetensors_index_summary(hf_assets_path, inventory_by_path),
        "tokenizer": dict(tokenizer_metadata),
    }


def _tokenizer_metadata(tokenizer: Any, hf_assets_path: Path) -> dict[str, Any]:
    tokenizer_config_path = hf_assets_path / "tokenizer_config.json"
    config: dict[str, Any] = {}
    if tokenizer_config_path.exists():
        config = json.loads(tokenizer_config_path.read_text())
    chat_template = config.get("chat_template")
    return {
        "hf_assets_path": str(hf_assets_path),
        "tokenizer_json_sha256": _hash_file(hf_assets_path / "tokenizer.json"),
        "tokenizer_config_sha256": _hash_file(tokenizer_config_path),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()
        if isinstance(chat_template, str)
        else None,
        "bos_id": getattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None)),
        "eos_id": getattr(tokenizer, "eos_id", getattr(tokenizer, "eos_token_id", None)),
        "pad_id": infer_pad_token_id(tokenizer, hf_assets_path),
        "trace_serializer": "Qwen2.5 ChatML over OpenHands messages; assistant content/tool_calls trainable; tool observations masked",
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in ("torch", "torchtitan", "datasets", "tokenizers", "torchao"):
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", None)
        except Exception:
            versions[package_name] = None
    return versions


def _file_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": _hash_file(path),
    }


def _runtime_lockfile_metadata(
    *,
    repo_root: Path | None = None,
    python_executable: str | None = None,
) -> list[dict[str, Any]]:
    repo_root = repo_root or Path(__file__).resolve().parents[1]
    python_path = Path(python_executable or sys.executable).expanduser()
    venv_root = (
        python_path.parent.parent
        if python_path.parent.name == "bin"
        else None
    )
    lockfiles: list[tuple[str, Path]] = [
        ("pod_lock", repo_root / "requirements" / "torchtitan-pod-cu128.lock"),
        ("pod_requirements", repo_root / "requirements" / "torchtitan-pod-cu128.txt"),
        (
            "torchtitan_requirements",
            repo_root / "torchtitan" / ".ci" / "docker" / "requirements.txt",
        ),
    ]
    if venv_root is not None:
        lockfiles.append(
            (
                "venv_runtime_metadata",
                venv_root / "torchtitan-swehero-runtime.json",
            )
        )
    return [
        {"kind": kind, **_file_metadata(path)}
        for kind, path in lockfiles
    ]


def _runtime_environment_metadata() -> dict[str, str]:
    prefixes = ("CUDA_", "NCCL_", "TORCH_NCCL_")
    explicit_keys = {"CUDA_VISIBLE_DEVICES"}
    return {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key in explicit_keys or key.startswith(prefixes)
    }


def _metadata_text_tail(value: object, limit: int = 20_000) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-limit:]


def _run_metadata_command(
    command: list[str],
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "command": command,
            "available": False,
            "error": repr(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "available": True,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stdout": _metadata_text_tail(exc.stdout),
            "stderr": _metadata_text_tail(exc.stderr),
        }
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": _metadata_text_tail(completed.stdout),
        "stderr": _metadata_text_tail(completed.stderr),
    }


def _nvidia_smi_metadata() -> dict[str, Any]:
    query = _run_metadata_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query.get("returncode") == 0 and isinstance(query.get("stdout"), str):
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            gpu = {
                "index": parts[0],
                "name": parts[1],
                "uuid": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": parts[4],
            }
            if len(parts) >= 6:
                gpu["memory_free_mib"] = parts[5]
            gpus.append(gpu)

    banner = _run_metadata_command(["nvidia-smi"])
    cuda_version = None
    if banner.get("returncode") == 0 and isinstance(banner.get("stdout"), str):
        match = re.search(r"CUDA Version:\s*([0-9.]+)", banner["stdout"])
        if match:
            cuda_version = match.group(1)

    return {
        "query_gpu": query,
        "banner": banner,
        "gpus": gpus,
        "cuda_version_from_banner": cuda_version,
    }


def paper_alignment(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke_synthetic_buckets:
        dataset_scope = (
            "synthetic smoke records only; this mode exercises launcher "
            "bucket/CP paths and is not a training dataset"
        )
    else:
        dataset_scope = (
            "all materialized examples"
            if args.num_examples == 0
            else f"capped at {args.num_examples} examples for a smoke run"
        )
    return {
        "kept": {
            "base_model": args.model_id,
            "base_model_revision": args.model_revision,
            "dataset": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "source_dataset": args.source_dataset_id,
            "source_dataset_revision": args.source_dataset_revision,
            "epochs": args.num_train_epochs,
            "global_batch_size": args.global_batch_size,
            "lr_schedule": "cosine",
            "peak_lr": args.learning_rate,
            "min_lr": args.min_learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "context_length": PAPER_CONTEXT_LENGTH,
            "qwen_yarn_rope": expected_qwen_yarn_rope_config(),
            "loss_masking": "assistant content, assistant tool calls, and assistant turn terminators only",
            "swe_zero_stage": "skipped for direct-to-hero",
        },
        "paper_caveats": [
            "The paper reports direct-to-hero as a 32B ablation; this 7B run is a scale-study extension.",
            "The one-rollout public training artifact is generated from the closest public historical SWE-Hero revision and records the public-column filter limitation in metadata.json.",
        ],
        "run_safety": {
            "production_mode": args.production_mode,
            "production_acceptance_smoke": args.production_acceptance_smoke,
            "production_gate": (
                "enabled for final acceptance smoke: production provenance, "
                "real dataset, checkpoint/export/validation, and durable W&B "
                "gates are enforced; bounded subset and step-cap deviations "
                "are explicitly recorded"
                if args.production_mode and args.production_acceptance_smoke
                else
                "enabled: smoke, subset, step-capped, and shortened-context "
                "settings are rejected before launch"
                if args.production_mode
                else "disabled: smoke/prototype settings are allowed but recorded"
            ),
        },
        "intentional_engineering_deltas": [
            "TorchTitan distributed full-model SFT replaces the earlier local Transformers smoke script.",
            "FSDP uses BF16 mixed-precision parameters/reductions; FP8 is applied only to TorchTitan linear layers selected by its converter.",
            "Length buckets with per-bucket CP replace static 128k padding.",
            (
                f"Bucket curriculum is explicit: {args.bucket_curriculum}. The "
                "paper does not specify an intra-SWE-HERO length-bucket curriculum."
            ),
            "TorchTitan VarlenAttention currently does not support CP, so CP buckets use the supported SDPA/Flex attention path.",
            f"The tokenized bucket manifest covers {dataset_scope}.",
        ],
    }


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def allow_empty_rank_reuse(args: argparse.Namespace) -> bool:
    return (not args.production_mode) or bool(args.production_acceptance_smoke)


def _message_content_text(content: object) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if "text" in item:
                    parts.append(_stringify(item["text"]))
                elif item.get("type") == "text" and "content" in item:
                    parts.append(_stringify(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(_stringify(item))
        return "".join(parts)
    return _stringify(content)


def _tool_call_function(tool_call: object) -> Mapping[str, object]:
    if not isinstance(tool_call, Mapping):
        return {}
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        return function
    return tool_call


def _qwen_tool_call_text(tool_call: object) -> str:
    function = _tool_call_function(tool_call)
    name = _stringify(function.get("name") or "unknown")
    arguments = function.get("arguments")
    if arguments is None:
        arguments = {}
    return (
        "\n<tool_call>\n"
        + '{"name": '
        + json.dumps(name, ensure_ascii=False)
        + ', "arguments": '
        + json.dumps(arguments, ensure_ascii=False)
        + "}\n</tool_call>"
    )


def qwen_openhands_turn_segments(
    turn: object,
    *,
    previous_role: str | None = None,
    next_role: str | None = None,
) -> list[tuple[str, bool]]:
    """Render one OpenHands message with Qwen2.5-Coder's ChatML convention."""

    if not isinstance(turn, Mapping):
        return [(json.dumps(turn, ensure_ascii=False) + "\n", False)]

    role = _stringify(turn.get("role") or "unknown")
    content = _message_content_text(turn.get("content"))
    segments: list[tuple[str, bool]] = []

    if role == "assistant":
        segments.append(("<|im_start|>assistant", False))
        if content:
            segments.append(("\n" + content, True))
        tool_calls = turn.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                segments.append((_qwen_tool_call_text(tool_call), True))
        elif tool_calls:
            segments.append(("\n" + json.dumps(tool_calls, ensure_ascii=False), True))
        segments.append(("<|im_end|>\n", True))
        return segments

    if role == "tool":
        if previous_role != "tool":
            segments.append(("<|im_start|>user", False))
        segments.append(("\n<tool_response>\n", False))
        segments.append((content, False))
        segments.append(("\n</tool_response>", False))
        if next_role != "tool":
            segments.append(("<|im_end|>\n", False))
        return segments

    segments.append((f"<|im_start|>{role}\n", False))
    if content:
        segments.append((content, False))
    segments.append(("<|im_end|>\n", False))
    return segments


def qwen_openhands_segments(
    example: dict[str, object],
    *,
    include_model_patch: bool = False,
) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    trajectory = example.get("trajectory") or example.get("messages") or []
    if isinstance(trajectory, list):
        start_index = 0
        if trajectory and isinstance(trajectory[0], Mapping):
            first_role = _stringify(trajectory[0].get("role") or "unknown")
            if first_role == "system":
                segments.extend(qwen_openhands_turn_segments(trajectory[0]))
                start_index = 1
            else:
                segments.append(("<|im_start|>system\n", False))
                segments.append((QWEN_DEFAULT_SYSTEM_PROMPT, False))
                segments.append(("<|im_end|>\n", False))

        for index in range(start_index, len(trajectory)):
            previous_role = (
                _stringify(trajectory[index - 1].get("role"))
                if index > 0 and isinstance(trajectory[index - 1], Mapping)
                else None
            )
            next_role = (
                _stringify(trajectory[index + 1].get("role"))
                if index + 1 < len(trajectory)
                and isinstance(trajectory[index + 1], Mapping)
                else None
            )
            segments.extend(
                qwen_openhands_turn_segments(
                    trajectory[index],
                    previous_role=previous_role,
                    next_role=next_role,
                )
            )
    else:
        segments.append((_stringify(trajectory) + "\n", False))

    patch = example.get("model_patch")
    if include_model_patch and patch:
        segments.append(("<|im_start|>assistant\n", False))
        segments.append((_stringify(patch) + "\n", True))
        segments.append(("<|im_end|>\n", True))

    return segments


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_bos=False, add_eos=False))
    except TypeError:
        return list(tokenizer.encode(text, add_special_tokens=False))


def encode_swehero_example(
    tokenizer: Any,
    example: dict[str, object],
    *,
    max_length: int,
    min_trainable_tokens: int,
    include_model_patch: bool = False,
) -> dict[str, Any] | None:
    token_ids: list[int] = []
    labels: list[int] = []

    bos_id = getattr(tokenizer, "bos_id", getattr(tokenizer, "bos_token_id", None))
    if bos_id is not None:
        token_ids.append(int(bos_id))
        labels.append(IGNORE_INDEX)

    for text, is_trainable in qwen_openhands_segments(
        example, include_model_patch=include_model_patch
    ):
        ids = _tokenize_text(tokenizer, text)
        token_ids.extend(ids)
        labels.extend(ids if is_trainable else [IGNORE_INDEX] * len(ids))

    eos_id = getattr(tokenizer, "eos_id", getattr(tokenizer, "eos_token_id", None))
    if eos_id is not None:
        token_ids.append(int(eos_id))
        labels.append(int(eos_id) if labels and labels[-1] != IGNORE_INDEX else IGNORE_INDEX)

    if len(token_ids) > max_length + 1:
        raise LongExampleError(token_count=len(token_ids), max_length=max_length)
    if len(token_ids) < 2:
        return None

    shifted_input_ids = token_ids[:-1]
    shifted_labels = labels[1:]
    trainable_tokens = sum(label != IGNORE_INDEX for label in shifted_labels)
    if trainable_tokens < min_trainable_tokens:
        return None

    return {
        "input_ids": shifted_input_ids,
        "labels": shifted_labels,
        "length": len(shifted_input_ids),
        "trainable_tokens": trainable_tokens,
    }


def infer_pad_token_id(tokenizer: Any, hf_assets_path: Path) -> int:
    for attr in ("pad_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)

    tokenizer_config_path = hf_assets_path / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        config = json.loads(tokenizer_config_path.read_text())
        pad_token = config.get("pad_token") or config.get("eos_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content")
        if isinstance(pad_token, str):
            token_to_id = getattr(tokenizer, "token_to_id", None)
            if callable(token_to_id):
                pad_id = token_to_id(pad_token)
                if pad_id is not None:
                    return int(pad_id)

    eos_id = getattr(tokenizer, "eos_id", getattr(tokenizer, "eos_token_id", None))
    if eos_id is None:
        raise RuntimeError("Could not infer pad token id from tokenizer assets")
    return int(eos_id)


def _example_id(example: Mapping[str, object], fallback_index: int) -> str:
    for key in ("instance_id", "problem_statement_id", "task_id", "id"):
        value = example.get(key)
        if value:
            return str(value)
    return f"stream_row_{fallback_index}"


def _resolve_bucket_file_for_validation(
    path: Path,
    path_overrides: Mapping[str, Path] | None,
) -> Path:
    if path_overrides is None:
        return path
    return path_overrides.get(str(path), path)


def _id_list_record(source_ids: Iterable[str]) -> dict[str, Any]:
    values = list(source_ids)
    return {
        "count": len(values),
        "sha256": _sha256_json(values),
        "source_ids": values,
    }


def _numeric_summary(values: Iterable[int]) -> dict[str, int | None]:
    items = list(values)
    if not items:
        return {"min": None, "max": None, "sum": 0}
    return {"min": min(items), "max": max(items), "sum": sum(items)}


def _build_data_provenance(
    args: argparse.Namespace,
    *,
    buckets: tuple[int, ...],
    bucket_paths: Mapping[int, Path],
    bucket_file_integrity: Mapping[str, Mapping[str, Any]],
    bucket_counts: Mapping[int, int],
    dataset_artifact: Mapping[str, Any],
    source_dataset_revision: Mapping[str, Any],
    streamed_source_ids: list[str],
    included_source_ids: list[str],
    skipped_source_ids_by_reason: Mapping[str, list[str]],
    bucket_source_ids: Mapping[int, list[str]],
    bucket_lengths: Mapping[int, list[int]],
    bucket_trainable_tokens: Mapping[int, list[int]],
    bucket_length_histograms: Mapping[int, Counter[int]],
) -> dict[str, Any]:
    skipped_by_reason = {
        reason: _id_list_record(ids)
        for reason, ids in sorted(skipped_source_ids_by_reason.items())
    }
    return {
        "schema_version": DATA_PROVENANCE_SCHEMA_VERSION,
        "dataset": {
            "dataset_id": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "dataset_artifact": dict(dataset_artifact),
            "source_dataset_id": args.source_dataset_id,
            "source_dataset_revision": dict(source_dataset_revision),
        },
        "materialization": {
            "num_examples_requested": args.num_examples,
            "max_streamed_examples": args.max_streamed_examples,
            "smoke_synthetic_buckets": args.smoke_synthetic_buckets,
            "smoke_synthetic_examples_per_bucket": (
                args.smoke_synthetic_examples_per_bucket
            ),
            "bucket_curriculum": args.bucket_curriculum,
            "shuffle_buffer": args.shuffle_buffer,
            "seed": args.seed,
            "max_length": args.max_length,
            "long_example_policy": args.long_example_policy,
            "min_trainable_tokens": args.min_trainable_tokens,
            "include_model_patch": args.include_model_patch,
            "buckets": list(buckets),
        },
        "record_schema": {
            "format": "jsonl",
            "fields": ["input_ids", "labels", "length", "bucket", "source_id"],
            "label_ignore_index": IGNORE_INDEX,
            "source_id_priority": [
                "instance_id",
                "problem_statement_id",
                "task_id",
                "id",
                "stream_row_<n>",
            ],
        },
        "streamed": _id_list_record(streamed_source_ids),
        "included": _id_list_record(included_source_ids),
        "skipped": {
            "total": sum(len(ids) for ids in skipped_source_ids_by_reason.values()),
            "counts": {
                reason: len(record["source_ids"])
                for reason, record in skipped_by_reason.items()
            },
            "by_reason": skipped_by_reason,
        },
        "buckets": {
            str(bucket): {
                "sequence_length": bucket,
                "bucket_file": str(bucket_paths[bucket]),
                "record_count": int(bucket_counts[bucket]),
                "integrity": dict(bucket_file_integrity[str(bucket)]),
                "source_ids": _id_list_record(bucket_source_ids[bucket]),
                "length": _numeric_summary(bucket_lengths[bucket]),
                "trainable_tokens": _numeric_summary(bucket_trainable_tokens[bucket]),
                "length_histogram_rounded_to_1024": {
                    str(length): count
                    for length, count in sorted(bucket_length_histograms[bucket].items())
                },
            }
            for bucket in buckets
        },
    }


def _validate_id_list_record(record: Any, label: str) -> list[str]:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} must be an object")
    source_ids = record.get("source_ids")
    if not isinstance(source_ids, list) or not all(
        isinstance(source_id, str) for source_id in source_ids
    ):
        raise RuntimeError(f"{label}.source_ids must be a list of strings")
    if int(record.get("count", -1)) != len(source_ids):
        raise RuntimeError(
            f"{label}.count does not match source_ids length: "
            f"{record.get('count')!r} != {len(source_ids)}"
        )
    if record.get("sha256") != _sha256_json(source_ids):
        raise RuntimeError(f"{label}.sha256 does not match source_ids")
    return source_ids


def _validate_data_provenance(
    manifest: Mapping[str, Any],
    *,
    bucket_files: Mapping[str, Any],
    bucket_counts: Mapping[str, Any],
    bucket_integrity: Mapping[str, Any],
) -> None:
    data_provenance = manifest.get("data_provenance")
    if not isinstance(data_provenance, Mapping):
        raise RuntimeError(
            "Materialized data manifest is missing complete data_provenance"
        )
    if data_provenance.get("schema_version") != DATA_PROVENANCE_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported data provenance schema version: "
            f"{data_provenance.get('schema_version')!r}; expected "
            f"{DATA_PROVENANCE_SCHEMA_VERSION}"
        )

    materialization = data_provenance.get("materialization")
    if not isinstance(materialization, Mapping):
        raise RuntimeError("data_provenance.materialization must be an object")
    expected_materialization = {
        "max_length": manifest.get("max_length"),
        "long_example_policy": manifest.get("long_example_policy"),
        "include_model_patch": manifest.get("include_model_patch"),
        "buckets": manifest.get("buckets"),
    }
    for key, expected in expected_materialization.items():
        if materialization.get(key) != expected:
            raise RuntimeError(
                f"data_provenance.materialization.{key} does not match manifest: "
                f"{materialization.get(key)!r} != {expected!r}"
            )
    dataset = data_provenance.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeError("data_provenance.dataset must be an object")
    expected_dataset = {
        "dataset_id": manifest.get("dataset_id"),
        "dataset_path": manifest.get("dataset_path"),
        "dataset_artifact": manifest.get("dataset_artifact"),
        "source_dataset_id": manifest.get("source_dataset_id"),
        "source_dataset_revision": manifest.get("source_dataset_revision"),
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise RuntimeError(
                f"data_provenance.dataset.{key} does not match manifest"
            )

    streamed_source_ids = _validate_id_list_record(
        data_provenance.get("streamed"),
        "data_provenance.streamed",
    )
    if int(manifest.get("streamed_examples_scanned", -1)) != len(streamed_source_ids):
        raise RuntimeError(
            "data_provenance.streamed count does not match "
            f"streamed_examples_scanned: {len(streamed_source_ids)} != "
            f"{manifest.get('streamed_examples_scanned')!r}"
        )

    included_source_ids = _validate_id_list_record(
        data_provenance.get("included"),
        "data_provenance.included",
    )
    if int(manifest.get("num_usable_examples", -1)) != len(included_source_ids):
        raise RuntimeError(
            "data_provenance.included count does not match num_usable_examples: "
            f"{len(included_source_ids)} != {manifest.get('num_usable_examples')!r}"
        )

    skipped = data_provenance.get("skipped")
    if not isinstance(skipped, Mapping):
        raise RuntimeError("data_provenance.skipped must be an object")
    by_reason = skipped.get("by_reason")
    if not isinstance(by_reason, Mapping):
        raise RuntimeError("data_provenance.skipped.by_reason must be an object")
    skipped_counts: dict[str, int] = {}
    for reason, record in by_reason.items():
        skipped_counts[str(reason)] = len(
            _validate_id_list_record(
                record,
                f"data_provenance.skipped.by_reason.{reason}",
            )
        )
    if skipped_counts != {
        str(reason): int(count)
        for reason, count in manifest.get("skipped", {}).items()
    }:
        raise RuntimeError(
            "data_provenance skipped counts do not match manifest skipped counts: "
            f"{skipped_counts!r} != {manifest.get('skipped', {})!r}"
        )
    if int(skipped.get("total", -1)) != sum(skipped_counts.values()):
        raise RuntimeError(
            "data_provenance.skipped.total does not match skipped reason counts"
        )

    provenance_buckets = data_provenance.get("buckets")
    if not isinstance(provenance_buckets, Mapping):
        raise RuntimeError("data_provenance.buckets must be an object")
    if set(str(bucket) for bucket in provenance_buckets) != set(bucket_files):
        raise RuntimeError(
            "data_provenance.buckets does not match manifest bucket_files"
        )

    bucket_source_ids = []
    aggregate_length_histogram: Counter[str] = Counter()
    for bucket, bucket_file in bucket_files.items():
        bucket_record = provenance_buckets[str(bucket)]
        if not isinstance(bucket_record, Mapping):
            raise RuntimeError(f"data_provenance.buckets[{bucket!r}] must be an object")
        if bucket_record.get("bucket_file") != bucket_file:
            raise RuntimeError(
                f"data_provenance bucket {bucket} file does not match manifest"
            )
        if int(bucket_record.get("record_count", -1)) != int(bucket_counts[str(bucket)]):
            raise RuntimeError(
                f"data_provenance bucket {bucket} count does not match manifest"
            )
        if bucket_record.get("integrity") != bucket_integrity[str(bucket)]:
            raise RuntimeError(
                f"data_provenance bucket {bucket} integrity does not match manifest"
            )
        source_ids = _validate_id_list_record(
            bucket_record.get("source_ids"),
            f"data_provenance.buckets.{bucket}.source_ids",
        )
        if len(source_ids) != int(bucket_counts[str(bucket)]):
            raise RuntimeError(
                f"data_provenance bucket {bucket} source_id count does not match "
                "bucket count"
            )
        bucket_source_ids.extend(source_ids)
        histogram = bucket_record.get("length_histogram_rounded_to_1024")
        if not isinstance(histogram, Mapping):
            raise RuntimeError(
                f"data_provenance bucket {bucket} length histogram must be an object"
            )
        if sum(int(count) for count in histogram.values()) != int(
            bucket_counts[str(bucket)]
        ):
            raise RuntimeError(
                f"data_provenance bucket {bucket} length histogram does not sum "
                "to bucket count"
            )
        aggregate_length_histogram.update(
            {str(length): int(count) for length, count in histogram.items()}
        )

    if Counter(bucket_source_ids) != Counter(included_source_ids):
        raise RuntimeError(
            "data_provenance bucket source IDs do not match included source IDs"
        )
    expected_histogram = {
        str(length): int(count)
        for length, count in manifest.get(
            "length_histogram_rounded_to_1024", {}
        ).items()
    }
    if dict(aggregate_length_histogram) != expected_histogram:
        raise RuntimeError(
            "data_provenance bucket length histograms do not match manifest "
            "length_histogram_rounded_to_1024"
        )


def validate_materialized_data_manifest(
    manifest: Mapping[str, Any],
    *,
    path_overrides: Mapping[str, Path] | None = None,
) -> None:
    schema_version = manifest.get("materialized_data_schema_version")
    if schema_version != MATERIALIZED_DATA_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported materialized data manifest schema version: "
            f"{schema_version!r}; expected {MATERIALIZED_DATA_SCHEMA_VERSION}"
        )
    model_assets = manifest.get("model_assets")
    if not isinstance(model_assets, Mapping):
        raise RuntimeError(
            "Materialized data manifest is missing complete model_assets provenance"
        )
    if model_assets.get("schema_version") != MODEL_ASSET_PROVENANCE_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported model asset provenance schema version: "
            f"{model_assets.get('schema_version')!r}; expected "
            f"{MODEL_ASSET_PROVENANCE_SCHEMA_VERSION}"
        )
    model_revision = manifest.get("model_revision")
    if not isinstance(model_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}",
        model_revision,
    ):
        raise RuntimeError(
            "Materialized data manifest is missing exact model_revision"
        )
    if model_assets.get("model_revision") != model_revision:
        raise RuntimeError(
            "model_assets.model_revision does not match manifest.model_revision: "
            f"{model_assets.get('model_revision')!r} != {model_revision!r}"
        )
    model_asset_files = model_assets.get("files")
    if not isinstance(model_asset_files, list):
        raise RuntimeError("model_assets.files must contain the hashed asset inventory")
    if int(model_assets.get("file_count", -1)) != len(model_asset_files):
        raise RuntimeError(
            "model_assets.file_count does not match model_assets.files length: "
            f"{model_assets.get('file_count')!r} != {len(model_asset_files)}"
        )
    total_asset_bytes = 0
    for record in model_asset_files:
        if not isinstance(record, Mapping):
            raise RuntimeError("model_assets.files entries must be objects")
        for key in ("path", "kind", "bytes", "sha256"):
            if key not in record:
                raise RuntimeError(
                    f"model_assets.files entry is missing {key!r}: {record!r}"
                )
        total_asset_bytes += int(record["bytes"])
    if int(model_assets.get("total_bytes", -1)) != total_asset_bytes:
        raise RuntimeError(
            "model_assets.total_bytes does not match model_assets.files: "
            f"{model_assets.get('total_bytes')!r} != {total_asset_bytes}"
        )

    bucket_files = manifest.get("bucket_files")
    bucket_counts = manifest.get("bucket_counts")
    bucket_integrity = manifest.get("bucket_file_integrity")
    if not isinstance(bucket_files, Mapping):
        raise RuntimeError("Materialized data manifest is missing bucket_files")
    if not isinstance(bucket_counts, Mapping):
        raise RuntimeError("Materialized data manifest is missing bucket_counts")
    if not isinstance(bucket_integrity, Mapping):
        raise RuntimeError(
            "Materialized data manifest is missing bucket_file_integrity"
        )

    expected_buckets = set(str(bucket) for bucket in manifest.get("buckets", []))
    manifest_buckets = set(str(bucket) for bucket in bucket_files)
    count_buckets = set(str(bucket) for bucket in bucket_counts)
    integrity_buckets = set(str(bucket) for bucket in bucket_integrity)
    if expected_buckets and manifest_buckets != expected_buckets:
        raise RuntimeError(
            "Materialized data manifest bucket_files do not match buckets: "
            f"bucket_files={sorted(manifest_buckets)} buckets={sorted(expected_buckets)}"
        )
    if count_buckets != manifest_buckets:
        raise RuntimeError(
            "Materialized data manifest bucket_counts do not match bucket_files: "
            f"bucket_counts={sorted(count_buckets)} "
            f"bucket_files={sorted(manifest_buckets)}"
        )
    if integrity_buckets != manifest_buckets:
        raise RuntimeError(
            "Materialized data manifest bucket_file_integrity does not match "
            f"bucket_files: integrity={sorted(integrity_buckets)} "
            f"bucket_files={sorted(manifest_buckets)}"
        )
    _validate_data_provenance(
        manifest,
        bucket_files=bucket_files,
        bucket_counts=bucket_counts,
        bucket_integrity=bucket_integrity,
    )

    total_records = 0
    for bucket, raw_path in bucket_files.items():
        bucket_key = str(bucket)
        path = _resolve_bucket_file_for_validation(Path(raw_path), path_overrides)
        if not path.exists():
            raise RuntimeError(f"Materialized bucket file does not exist: {path}")

        stats = _bucket_file_stats(path)
        expected_integrity = bucket_integrity[bucket_key]
        if not isinstance(expected_integrity, Mapping):
            raise RuntimeError(
                f"bucket_file_integrity[{bucket_key!r}] must be an object"
            )
        expected_records = int(bucket_counts[bucket_key])
        total_records += expected_records

        if int(expected_integrity.get("records", -1)) != expected_records:
            raise RuntimeError(
                f"bucket_file_integrity[{bucket_key!r}].records does not match "
                f"bucket_counts: {expected_integrity.get('records')!r} != "
                f"{expected_records}"
            )
        if int(stats["records"]) != expected_records:
            raise RuntimeError(
                f"Materialized bucket {bucket_key} has {stats['records']} "
                f"record(s), expected {expected_records}"
            )
        if int(expected_integrity.get("bytes", -1)) != int(stats["bytes"]):
            raise RuntimeError(
                f"Materialized bucket {bucket_key} byte count mismatch: "
                f"{stats['bytes']} != {expected_integrity.get('bytes')!r}"
            )
        if expected_integrity.get("sha256") != stats["sha256"]:
            raise RuntimeError(
                f"Materialized bucket {bucket_key} sha256 mismatch: "
                f"{stats['sha256']} != {expected_integrity.get('sha256')!r}"
            )

    if int(manifest.get("num_usable_examples", -1)) != total_records:
        raise RuntimeError(
            "Materialized data manifest num_usable_examples does not match "
            f"bucket_counts total: {manifest.get('num_usable_examples')!r} != "
            f"{total_records}"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, path)


def _promote_materialized_data_dir(staging_data_dir: Path, final_data_dir: Path) -> None:
    backup_data_dir: Path | None = None
    if final_data_dir.exists():
        backup_data_dir = final_data_dir.with_name(
            f".data.previous-{os.getpid()}-{time.time_ns()}"
        )
        final_data_dir.replace(backup_data_dir)

    try:
        staging_data_dir.replace(final_data_dir)
    except Exception:
        if backup_data_dir is not None and backup_data_dir.exists():
            backup_data_dir.replace(final_data_dir)
        raise
    else:
        if backup_data_dir is not None:
            shutil.rmtree(backup_data_dir, ignore_errors=True)


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "data" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist; run without --skip-data-prep first"
        )
    manifest = json.loads(manifest_path.read_text())
    validate_materialized_data_manifest(manifest)
    return manifest


def _synthetic_smoke_dataset_artifact(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "path": str(args.dataset_path),
        "realpath": str(args.dataset_path.resolve()),
        "synthetic_smoke": True,
        "metadata": {
            "description": (
                "Synthetic tokenized records generated by "
                "--smoke-synthetic-buckets to exercise bucket/CP launcher paths."
            ),
            "swe_traces_consumed": False,
        },
        "metadata_json": {
            "path": None,
            "exists": False,
            "bytes": 0,
            "sha256": None,
        },
        "metadata_json_sha256": None,
        "selection_manifest": {
            "path": None,
            "exists": False,
            "bytes": 0,
            "sha256": None,
        },
        "selection_manifest_sha256": None,
        "data_file_count": 0,
        "data_files": [],
        "total_data_bytes": 0,
    }


def _synthetic_smoke_source_revision(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "requested_revision": args.source_dataset_revision,
        "resolved_sha": None,
        "synthetic_smoke": True,
        "lookup_error": None,
    }


def _synthetic_smoke_record(
    *,
    bucket: int,
    index: int,
    pad_token_id: int,
) -> dict[str, Any]:
    source_id = f"synthetic_smoke_bucket_{bucket}_{index}"
    return {
        "input_ids": [pad_token_id, 0],
        "labels": [IGNORE_INDEX, 0],
        "length": 2,
        "trainable_tokens": 1,
        "bucket": bucket,
        "source_id": source_id,
    }


def materialize_synthetic_smoke_buckets(args: argparse.Namespace) -> dict[str, Any]:
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    staging_data_dir = args.out_dir / f".data.tmp-{os.getpid()}-{time.time_ns()}"
    staging_data_dir.mkdir(parents=True)
    buckets = parse_bucket_list(args.buckets)

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(args.hf_assets_path))
    pad_token_id = infer_pad_token_id(tokenizer, args.hf_assets_path)
    tokenizer_metadata = _tokenizer_metadata(tokenizer, args.hf_assets_path)

    bucket_paths = {
        bucket: data_dir / f"bucket_{bucket}.jsonl" for bucket in buckets
    }
    staging_bucket_paths = {
        bucket: staging_data_dir / f"bucket_{bucket}.jsonl" for bucket in buckets
    }
    bucket_counts: Counter[int] = Counter()
    streamed_source_ids: list[str] = []
    included_source_ids: list[str] = []
    bucket_source_ids: dict[int, list[str]] = {bucket: [] for bucket in buckets}
    bucket_lengths: dict[int, list[int]] = {bucket: [] for bucket in buckets}
    bucket_trainable_tokens: dict[int, list[int]] = {bucket: [] for bucket in buckets}
    bucket_length_histograms: dict[int, Counter[int]] = {
        bucket: Counter() for bucket in buckets
    }
    length_histogram: Counter[int] = Counter()
    promoted = False

    try:
        for bucket in buckets:
            with staging_bucket_paths[bucket].open("w") as handle:
                for index in range(args.smoke_synthetic_examples_per_bucket):
                    record = _synthetic_smoke_record(
                        bucket=bucket,
                        index=index,
                        pad_token_id=pad_token_id,
                    )
                    source_id = str(record["source_id"])
                    handle.write(json.dumps(record) + "\n")
                    bucket_counts[bucket] += 1
                    streamed_source_ids.append(source_id)
                    included_source_ids.append(source_id)
                    bucket_source_ids[bucket].append(source_id)
                    bucket_lengths[bucket].append(int(record["length"]))
                    bucket_trainable_tokens[bucket].append(
                        int(record["trainable_tokens"])
                    )
                    rounded_length = int(math.ceil(int(record["length"]) / 1024) * 1024)
                    length_histogram[rounded_length] += 1
                    bucket_length_histograms[bucket][rounded_length] += 1

        bucket_file_integrity = {
            str(bucket): _bucket_file_stats(staging_bucket_paths[bucket])
            for bucket in buckets
        }
        dataset_artifact = _synthetic_smoke_dataset_artifact(args)
        source_dataset_revision = _synthetic_smoke_source_revision(args)
        data_provenance = _build_data_provenance(
            args,
            buckets=buckets,
            bucket_paths=bucket_paths,
            bucket_file_integrity=bucket_file_integrity,
            bucket_counts=bucket_counts,
            dataset_artifact=dataset_artifact,
            source_dataset_revision=source_dataset_revision,
            streamed_source_ids=streamed_source_ids,
            included_source_ids=included_source_ids,
            skipped_source_ids_by_reason={},
            bucket_source_ids=bucket_source_ids,
            bucket_lengths=bucket_lengths,
            bucket_trainable_tokens=bucket_trainable_tokens,
            bucket_length_histograms=bucket_length_histograms,
        )
        manifest = {
            "materialized_data_schema_version": MATERIALIZED_DATA_SCHEMA_VERSION,
            "created_at_unix": time.time(),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_id": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "dataset_artifact": dataset_artifact,
            "source_dataset_id": args.source_dataset_id,
            "source_dataset_revision": source_dataset_revision,
            "paper_alignment": paper_alignment(args),
            "model_assets": _model_asset_provenance(
                model_id=args.model_id,
                model_revision=args.model_revision,
                hf_assets_path=args.hf_assets_path,
                tokenizer_metadata=tokenizer_metadata,
            ),
            "data_provenance": data_provenance,
            "tokenizer": tokenizer_metadata,
            "pad_token_id": pad_token_id,
            "max_length": args.max_length,
            "long_example_policy": args.long_example_policy,
            "smoke_synthetic_buckets": True,
            "smoke_synthetic_examples_per_bucket": (
                args.smoke_synthetic_examples_per_bucket
            ),
            "bucket_curriculum": args.bucket_curriculum,
            "buckets": list(buckets),
            "bucket_files": {
                str(bucket): str(path) for bucket, path in bucket_paths.items()
            },
            "bucket_file_integrity": bucket_file_integrity,
            "bucket_counts": {str(bucket): bucket_counts[bucket] for bucket in buckets},
            "length_histogram_rounded_to_1024": {
                str(length): count for length, count in sorted(length_histogram.items())
            },
            "num_usable_examples": len(included_source_ids),
            "streamed_examples_scanned": len(streamed_source_ids),
            "skipped": {},
            "long_examples_sample": [],
            "include_model_patch": args.include_model_patch,
            "git": git_state_for_workspace(_configured_workspace_root(args)),
            "software_versions": _package_versions(),
        }
        path_overrides = {
            str(bucket_paths[bucket]): staging_bucket_paths[bucket]
            for bucket in buckets
        }
        validate_materialized_data_manifest(
            manifest,
            path_overrides=path_overrides,
        )
        _write_json_atomic(staging_data_dir / "manifest.json", manifest)
        _promote_materialized_data_dir(staging_data_dir, data_dir)
        promoted = True
        return _load_manifest(args.out_dir)
    finally:
        if not promoted:
            shutil.rmtree(staging_data_dir, ignore_errors=True)


def materialize_training_buckets(args: argparse.Namespace) -> dict[str, Any]:
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.out_dir / "data"
    staging_data_dir = args.out_dir / f".data.tmp-{os.getpid()}-{time.time_ns()}"
    staging_data_dir.mkdir(parents=True)
    buckets = parse_bucket_list(args.buckets)

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(args.hf_assets_path))
    pad_token_id = infer_pad_token_id(tokenizer, args.hf_assets_path)

    bucket_paths = {
        bucket: data_dir / f"bucket_{bucket}.jsonl" for bucket in buckets
    }
    staging_bucket_paths = {
        bucket: staging_data_dir / f"bucket_{bucket}.jsonl" for bucket in buckets
    }
    handles = {
        bucket: path.open("w") for bucket, path in staging_bucket_paths.items()
    }
    bucket_counts: Counter[int] = Counter()
    skipped: Counter[str] = Counter()
    streamed_source_ids: list[str] = []
    included_source_ids: list[str] = []
    skipped_source_ids_by_reason: dict[str, list[str]] = {}
    bucket_source_ids: dict[int, list[str]] = {bucket: [] for bucket in buckets}
    bucket_lengths: dict[int, list[int]] = {bucket: [] for bucket in buckets}
    bucket_trainable_tokens: dict[int, list[int]] = {bucket: [] for bucket in buckets}
    bucket_length_histograms: dict[int, Counter[int]] = {
        bucket: Counter() for bucket in buckets
    }
    long_examples_sample: list[dict[str, Any]] = []
    length_histogram: Counter[int] = Counter()
    streamed_examples = 0
    usable_examples = 0

    raw = load_training_dataset(args)
    promoted = False

    try:
        for example in raw:
            streamed_examples += 1
            source_id = _example_id(example, streamed_examples)
            streamed_source_ids.append(source_id)
            try:
                encoded = encode_swehero_example(
                    tokenizer,
                    example,
                    max_length=args.max_length,
                    min_trainable_tokens=args.min_trainable_tokens,
                    include_model_patch=args.include_model_patch,
                )
            except LongExampleError as exc:
                skipped["too_long_for_max_length"] += 1
                skipped_source_ids_by_reason.setdefault(
                    "too_long_for_max_length", []
                ).append(source_id)
                long_example = {
                    "source_id": source_id,
                    "token_count": exc.token_count,
                    "shifted_input_length": exc.shifted_input_length,
                    "max_length": exc.max_length,
                }
                if len(long_examples_sample) < 20:
                    long_examples_sample.append(long_example)
                if args.long_example_policy == "error":
                    raise RuntimeError(
                        "SWE-HERO example exceeds --max-length and would have "
                        f"been truncated by the old launcher: {long_example}. "
                        "Use --long-example-policy skip only for explicit smoke "
                        "runs or after accepting the dataset-scope change."
                    ) from exc
                continue
            if encoded is None:
                skipped["not_enough_trainable_tokens"] += 1
                skipped_source_ids_by_reason.setdefault(
                    "not_enough_trainable_tokens", []
                ).append(source_id)
            else:
                try:
                    bucket = choose_bucket(encoded["length"], buckets)
                except ValueError:
                    skipped["too_long_for_largest_bucket"] += 1
                    skipped_source_ids_by_reason.setdefault(
                        "too_long_for_largest_bucket", []
                    ).append(source_id)
                else:
                    record = {
                        **encoded,
                        "bucket": bucket,
                        "source_id": source_id,
                    }
                    handles[bucket].write(json.dumps(record) + "\n")
                    bucket_counts[bucket] += 1
                    included_source_ids.append(source_id)
                    bucket_source_ids[bucket].append(source_id)
                    bucket_lengths[bucket].append(int(encoded["length"]))
                    trainable_tokens = sum(
                        label != IGNORE_INDEX for label in encoded["labels"]
                    )
                    bucket_trainable_tokens[bucket].append(trainable_tokens)
                    rounded_length = int(math.ceil(encoded["length"] / 1024) * 1024)
                    length_histogram[rounded_length] += 1
                    bucket_length_histograms[bucket][rounded_length] += 1
                    usable_examples += 1

            if args.num_examples > 0 and usable_examples >= args.num_examples:
                break
            if (
                args.max_streamed_examples > 0
                and streamed_examples >= args.max_streamed_examples
            ):
                break
        for handle in handles.values():
            handle.close()
        handles = {}

        if usable_examples == 0:
            raise RuntimeError(
                "No usable SWE-HERO examples were materialized. Increase "
                "--max-streamed-examples, reduce --min-trainable-tokens, or inspect "
                "the dataset schema."
            )

        bucket_file_integrity = {
            str(bucket): _bucket_file_stats(staging_bucket_paths[bucket])
            for bucket in buckets
        }
        tokenizer_metadata = _tokenizer_metadata(tokenizer, args.hf_assets_path)
        dataset_artifact = _dataset_artifact_metadata(args.dataset_path)
        source_dataset_revision = _dataset_revision_info(
            args.source_dataset_id, args.source_dataset_revision
        )
        data_provenance = _build_data_provenance(
            args,
            buckets=buckets,
            bucket_paths=bucket_paths,
            bucket_file_integrity=bucket_file_integrity,
            bucket_counts=bucket_counts,
            dataset_artifact=dataset_artifact,
            source_dataset_revision=source_dataset_revision,
            streamed_source_ids=streamed_source_ids,
            included_source_ids=included_source_ids,
            skipped_source_ids_by_reason=skipped_source_ids_by_reason,
            bucket_source_ids=bucket_source_ids,
            bucket_lengths=bucket_lengths,
            bucket_trainable_tokens=bucket_trainable_tokens,
            bucket_length_histograms=bucket_length_histograms,
        )
        manifest = {
            "materialized_data_schema_version": MATERIALIZED_DATA_SCHEMA_VERSION,
            "created_at_unix": time.time(),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "dataset_id": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "dataset_artifact": dataset_artifact,
            "source_dataset_id": args.source_dataset_id,
            "source_dataset_revision": source_dataset_revision,
            "paper_alignment": paper_alignment(args),
            "model_assets": _model_asset_provenance(
                model_id=args.model_id,
                model_revision=args.model_revision,
                hf_assets_path=args.hf_assets_path,
                tokenizer_metadata=tokenizer_metadata,
            ),
            "data_provenance": data_provenance,
            "tokenizer": tokenizer_metadata,
            "pad_token_id": pad_token_id,
            "max_length": args.max_length,
            "long_example_policy": args.long_example_policy,
            "smoke_synthetic_buckets": False,
            "smoke_synthetic_examples_per_bucket": None,
            "bucket_curriculum": args.bucket_curriculum,
            "buckets": list(buckets),
            "bucket_files": {
                str(bucket): str(path) for bucket, path in bucket_paths.items()
            },
            "bucket_file_integrity": bucket_file_integrity,
            "bucket_counts": {str(bucket): bucket_counts[bucket] for bucket in buckets},
            "length_histogram_rounded_to_1024": {
                str(length): count for length, count in sorted(length_histogram.items())
            },
            "num_usable_examples": usable_examples,
            "streamed_examples_scanned": streamed_examples,
            "skipped": dict(skipped),
            "long_examples_sample": long_examples_sample,
            "include_model_patch": args.include_model_patch,
            "git": git_state_for_workspace(_configured_workspace_root(args)),
            "software_versions": _package_versions(),
        }
        path_overrides = {
            str(bucket_paths[bucket]): staging_bucket_paths[bucket]
            for bucket in buckets
        }
        validate_materialized_data_manifest(
            manifest,
            path_overrides=path_overrides,
        )
        _write_json_atomic(staging_data_dir / "manifest.json", manifest)
        _promote_materialized_data_dir(staging_data_dir, data_dir)
        promoted = True
        return _load_manifest(args.out_dir)
    finally:
        for handle in handles.values():
            handle.close()
        if not promoted:
            shutil.rmtree(staging_data_dir, ignore_errors=True)


def _bucket_files_from_manifest(manifest: Mapping[str, Any]) -> dict[int, Path]:
    return {
        int(bucket): Path(path)
        for bucket, path in manifest.get("bucket_files", {}).items()
    }


def _bucket_counts_from_manifest(manifest: Mapping[str, Any]) -> dict[int, int]:
    return {
        int(bucket): int(count)
        for bucket, count in manifest.get("bucket_counts", {}).items()
    }


def download_hf_assets_if_requested(args: argparse.Namespace) -> None:
    if not args.download_hf_assets:
        return
    repo_root = _configured_workspace_root(args)
    local_dir = args.hf_assets_path.parent
    command = [
        sys.executable,
        str(repo_root / "torchtitan" / "scripts" / "download_hf_assets.py"),
        "--repo_id",
        args.model_id,
        "--revision",
        args.model_revision,
        "--local_dir",
        str(local_dir),
        "--assets",
        "tokenizer",
        "config",
        "safetensors",
        "index",
    ]
    if args.hf_token:
        command.extend(["--hf_token", args.hf_token])
    subprocess.run(command, check=True)


def _resolve_executable(command: str) -> str | None:
    if not command:
        return None
    path = Path(command).expanduser()
    if path.parent != Path(".") or path.is_absolute():
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(command)


def _read_json_object_required(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing required {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Required {label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Required {label} must be a JSON object: {path}")
    return payload


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing required {label}: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Required {label} is empty: {path}")


def _safe_child_path(base_path: Path, relative_path: object, *, label: str) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"{label} contains an unsafe relative path: {relative_path!r}"
        )
    return base_path / relative


def _asset_path_from_relative(hf_assets_path: Path, relative_path: object) -> Path:
    return _safe_child_path(
        hf_assets_path,
        relative_path,
        label="Model asset provenance",
    )


def _safetensors_launch_summary(hf_assets_path: Path) -> dict[str, Any]:
    index_path = hf_assets_path / "model.safetensors.index.json"
    if index_path.exists():
        _require_nonempty_file(index_path, "safetensors index")
        index = _read_json_object_required(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise RuntimeError(
                f"Safetensors index has no weight_map entries: {index_path}"
            )
        shard_names = sorted(set(str(path) for path in weight_map.values()))
        total_bytes = 0
        for shard_name in shard_names:
            shard_path = _asset_path_from_relative(hf_assets_path, shard_name)
            _require_nonempty_file(shard_path, "safetensors shard")
            total_bytes += shard_path.stat().st_size
        return {
            "index_path": str(index_path),
            "weight_map_entries": len(weight_map),
            "shard_count": len(shard_names),
            "total_shard_bytes": total_bytes,
        }

    shards = sorted(hf_assets_path.rglob("*.safetensors"))
    if not shards:
        raise RuntimeError(
            f"No safetensors weights found under Hugging Face asset directory: "
            f"{hf_assets_path}"
        )
    total_bytes = 0
    for shard_path in shards:
        _require_nonempty_file(shard_path, "safetensors shard")
        total_bytes += shard_path.stat().st_size
    return {
        "index_path": None,
        "weight_map_entries": None,
        "shard_count": len(shards),
        "total_shard_bytes": total_bytes,
    }


def _validate_manifest_model_asset_preflight(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model_assets = manifest.get("model_assets")
    if not isinstance(model_assets, Mapping):
        raise RuntimeError("Materialized manifest is missing model_assets provenance")
    if model_assets.get("schema_version") != MODEL_ASSET_PROVENANCE_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported model_assets provenance schema version during preflight: "
            f"{model_assets.get('schema_version')!r}"
        )
    if model_assets.get("model_id") != args.model_id:
        raise RuntimeError(
            "Materialized model_assets.model_id does not match launch model_id: "
            f"{model_assets.get('model_id')!r} != {args.model_id!r}"
        )
    if model_assets.get("model_revision") != args.model_revision:
        raise RuntimeError(
            "Materialized model_assets.model_revision does not match launch "
            f"model_revision: {model_assets.get('model_revision')!r} != "
            f"{args.model_revision!r}"
        )

    recorded_realpath = model_assets.get("hf_assets_realpath")
    current_realpath = str(args.hf_assets_path.resolve())
    if recorded_realpath is not None and str(recorded_realpath) != current_realpath:
        raise RuntimeError(
            "Materialized model_assets.hf_assets_realpath does not match the "
            "current --hf-assets-path: "
            f"{recorded_realpath!r} != {current_realpath!r}"
        )
    recorded_path = model_assets.get("hf_assets_path")
    if recorded_realpath is None and recorded_path is not None:
        if str(Path(str(recorded_path)).resolve()) != current_realpath:
            raise RuntimeError(
                "Materialized model_assets.hf_assets_path does not match the "
                "current --hf-assets-path"
            )

    files = model_assets.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("model_assets.files must contain at least one asset file")

    checked_files = 0
    checked_bytes = 0
    checked_hashes = 0
    for record in files:
        if not isinstance(record, Mapping):
            raise RuntimeError("model_assets.files entries must be objects")
        asset_path = _asset_path_from_relative(args.hf_assets_path, record.get("path"))
        if not asset_path.is_file():
            raise RuntimeError(
                "Model asset from materialized provenance is missing on disk: "
                f"{asset_path}"
            )
        expected_bytes = int(record.get("bytes", -1))
        actual_bytes = asset_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                "Model asset byte size does not match model_assets provenance: "
                f"{asset_path} has {actual_bytes} byte(s), expected {expected_bytes}"
            )
        expected_sha256 = record.get("sha256")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}",
            expected_sha256,
        ):
            raise RuntimeError(
                "model_assets.files entry has invalid sha256 during preflight: "
                f"{record!r}"
            )
        actual_sha256 = _hash_file(asset_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Model asset sha256 does not match model_assets provenance: "
                f"{asset_path} has {actual_sha256}, expected {expected_sha256}"
            )
        checked_files += 1
        checked_bytes += actual_bytes
        checked_hashes += 1

    if int(model_assets.get("file_count", -1)) != checked_files:
        raise RuntimeError(
            "model_assets.file_count does not match checked file count during "
            f"preflight: {model_assets.get('file_count')!r} != {checked_files}"
        )
    if int(model_assets.get("total_bytes", -1)) != checked_bytes:
        raise RuntimeError(
            "model_assets.total_bytes does not match checked asset bytes during "
            f"preflight: {model_assets.get('total_bytes')!r} != {checked_bytes}"
        )

    safetensors = model_assets.get("safetensors")
    if not isinstance(safetensors, Mapping):
        raise RuntimeError("model_assets.safetensors must be present during preflight")
    for shard_record in safetensors.get("shard_files", []):
        if not isinstance(shard_record, Mapping):
            raise RuntimeError(
                "model_assets.safetensors.shard_files entries must be objects"
            )
        if not shard_record.get("present"):
            raise RuntimeError(
                "model_assets provenance recorded a missing safetensors shard: "
                f"{shard_record.get('path')!r}"
            )
        if int(shard_record.get("bytes") or 0) <= 0:
            raise RuntimeError(
                "model_assets provenance recorded an empty safetensors shard: "
                f"{shard_record.get('path')!r}"
            )

    return {
        "model_id": model_assets.get("model_id"),
        "model_revision": model_assets.get("model_revision"),
        "file_count": checked_files,
        "total_bytes": checked_bytes,
        "sha256_verified_files": checked_hashes,
    }


def validate_hf_asset_preflight(
    args: argparse.Namespace,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not args.hf_assets_path.is_dir():
        raise RuntimeError(
            f"Hugging Face asset directory does not exist: {args.hf_assets_path}"
        )

    for filename in REQUIRED_HF_ASSET_FILES:
        _require_nonempty_file(args.hf_assets_path / filename, filename)
    config = _read_json_object_required(
        args.hf_assets_path / "config.json",
        "config.json",
    )
    if config.get("model_type") != "qwen2":
        raise RuntimeError(
            "Hugging Face config.json is not a Qwen2-family model config: "
            f"model_type={config.get('model_type')!r}"
        )
    _read_json_object_required(
        args.hf_assets_path / "tokenizer_config.json",
        "tokenizer_config.json",
    )
    _read_json_object_required(
        args.hf_assets_path / "tokenizer.json",
        "tokenizer.json",
    )
    safetensors = _safetensors_launch_summary(args.hf_assets_path)
    manifest_assets = (
        _validate_manifest_model_asset_preflight(args, manifest)
        if manifest is not None
        else None
    )

    return {
        "model_revision": args.model_revision,
        "hf_assets_path": str(args.hf_assets_path),
        "required_files": list(REQUIRED_HF_ASSET_FILES),
        "config_model_type": config.get("model_type"),
        "safetensors": safetensors,
        "manifest_model_assets": manifest_assets,
    }


def _gib(value: float) -> int:
    return int(value * 1024 * 1024 * 1024)


def _mib(value: float) -> int:
    return int(value * 1024 * 1024)


def _disk_space_preflight(path: Path, *, min_free_gb: float) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    min_free_bytes = _gib(min_free_gb)
    if usage.free < min_free_bytes:
        raise RuntimeError(
            f"Disk preflight failed for {path}: free={usage.free} bytes, "
            f"required>={min_free_bytes} bytes"
        )
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "min_free_bytes": min_free_bytes,
    }


def _available_cpu_memory_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _cpu_memory_preflight(*, min_free_gb: float) -> dict[str, Any]:
    available_bytes = _available_cpu_memory_bytes()
    min_free_bytes = _gib(min_free_gb)
    if available_bytes is None:
        raise RuntimeError("CPU memory preflight could not determine available memory")
    if available_bytes < min_free_bytes:
        raise RuntimeError(
            f"CPU memory preflight failed: available={available_bytes} bytes, "
            f"required>={min_free_bytes} bytes"
        )
    return {
        "available_bytes": available_bytes,
        "min_free_bytes": min_free_bytes,
    }


def _gpu_memory_preflight(
    *,
    min_free_gb: float,
    required_gpus: int,
) -> dict[str, Any]:
    nvidia = _nvidia_smi_metadata()
    gpus = nvidia.get("gpus")
    if not isinstance(gpus, list) or len(gpus) < required_gpus:
        raise RuntimeError(
            "GPU memory preflight could not inspect enough GPUs: "
            f"found={0 if not isinstance(gpus, list) else len(gpus)}, "
            f"required={required_gpus}"
        )
    min_free_mib = min_free_gb * 1024
    checked = []
    for gpu in gpus[:required_gpus]:
        if not isinstance(gpu, Mapping):
            raise RuntimeError(f"GPU memory preflight got invalid GPU row: {gpu!r}")
        try:
            free_mib = float(gpu["memory_free_mib"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "GPU memory preflight requires nvidia-smi memory.free data"
            ) from exc
        if free_mib < min_free_mib:
            raise RuntimeError(
                "GPU memory preflight failed for GPU "
                f"{gpu.get('index')}: free={free_mib} MiB, "
                f"required>={min_free_mib} MiB"
            )
        checked.append(
            {
                "index": gpu.get("index"),
                "name": gpu.get("name"),
                "memory_free_mib": free_mib,
                "min_free_mib": min_free_mib,
            }
        )
    return {
        "required_gpus": required_gpus,
        "checked_gpus": checked,
        "nvidia_smi": nvidia,
    }


def _write_throughput_preflight(
    out_dir: Path,
    *,
    min_mb_s: float,
    probe_mb: int,
) -> dict[str, Any]:
    probe_bytes = _mib(float(probe_mb))
    chunk = b"\0" * _mib(1.0)
    probe = out_dir / f".write-throughput-preflight-{os.getpid()}-{time.time_ns()}"
    start = time.monotonic()
    try:
        with probe.open("wb") as handle:
            remaining = probe_bytes
            while remaining > 0:
                payload = chunk if remaining >= len(chunk) else chunk[:remaining]
                handle.write(payload)
                remaining -= len(payload)
            handle.flush()
            os.fsync(handle.fileno())
        elapsed = max(time.monotonic() - start, 1e-9)
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
    throughput_mb_s = (probe_bytes / (1024 * 1024)) / elapsed
    if throughput_mb_s < min_mb_s:
        raise RuntimeError(
            "Write-throughput preflight failed: "
            f"{throughput_mb_s:.2f} MiB/s < required {min_mb_s:.2f} MiB/s"
        )
    return {
        "path": str(out_dir),
        "probe_bytes": probe_bytes,
        "duration_seconds": elapsed,
        "throughput_mb_s": throughput_mb_s,
        "min_mb_s": min_mb_s,
    }


def validate_resource_preflights(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "disk": _disk_space_preflight(
            args.out_dir,
            min_free_gb=args.min_free_disk_gb,
        ),
        "cpu_memory": _cpu_memory_preflight(
            min_free_gb=args.min_free_cpu_memory_gb,
        ),
        "gpu_memory": _gpu_memory_preflight(
            min_free_gb=args.min_free_gpu_memory_gb,
            required_gpus=args.nproc_per_node,
        ),
        "write_throughput": _write_throughput_preflight(
            args.out_dir,
            min_mb_s=args.min_write_throughput_mb_s,
            probe_mb=args.write_throughput_probe_mb,
        ),
    }


def _cuda_launch_summary(
    torch_module: Any,
    *,
    nproc_per_node: int | None = None,
) -> dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    available = bool(cuda is not None and cuda.is_available())
    device_count = int(cuda.device_count()) if available else 0
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        device: dict[str, Any] = {"index": index}
        try:
            device["name"] = cuda.get_device_name(index)
        except Exception as exc:
            device["name_error"] = repr(exc)
        try:
            capability = cuda.get_device_capability(index)
            device["capability"] = list(capability)
        except Exception as exc:
            device["capability_error"] = repr(exc)
        devices.append(device)

    if nproc_per_node is not None:
        if nproc_per_node <= 0:
            raise RuntimeError(
                f"--nproc-per-node must be positive, got {nproc_per_node}"
            )
        if not available:
            raise RuntimeError(
                "CUDA is not available in the current Python environment; "
                "launch from the GPU pod venv."
            )
        if device_count < nproc_per_node:
            raise RuntimeError(
                f"Launch requires --nproc-per-node={nproc_per_node}, but only "
                f"{device_count} visible CUDA device(s) are available. Check "
                "CUDA_VISIBLE_DEVICES and the pod GPU allocation."
            )

    return {
        "available": available,
        "device_count": device_count,
        "required_nproc_per_node": nproc_per_node,
        "devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _torch_cuda_driver_version(torch_module: Any) -> Any:
    try:
        return torch_module._C._cuda_getDriverVersion()
    except Exception as exc:
        return {"error": repr(exc)}


def _torch_cudnn_version(torch_module: Any) -> Any:
    try:
        return torch_module.backends.cudnn.version()
    except Exception as exc:
        return {"error": repr(exc)}


def _torch_nccl_version(torch_module: Any) -> Any:
    try:
        return torch_module.cuda.nccl.version()
    except Exception as exc:
        return {"error": repr(exc)}


def validate_torchtitan_runtime(
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch.distributed.fsdp import DataParallelMeshDims
    except ImportError as exc:
        raise RuntimeError(
            "The current Python environment does not satisfy the vendored "
            "TorchTitan dependency contract. Create the canonical pod venv with "
            "scripts/setup_torchtitan_pod_venv.sh and launch through "
            "scripts/run_qwen_swehero_torchtitan_pod.sh."
        ) from exc

    try:
        import torchao
        from torchao.float8 import Float8LinearConfig

        Float8LinearConfig.from_recipe_name("rowwise")
    except Exception as exc:
        raise RuntimeError(
            "TorchAO float8 support is missing from the current Python "
            "environment. Rebuild the canonical pod venv with "
            "scripts/setup_torchtitan_pod_venv.sh."
        ) from exc

    try:
        from torchtitan.models.qwen2_5 import qwen2_5_configs

        qwen_config = qwen2_5_configs["coder7b"](attn_backend="sdpa")
        rope = qwen_config.rope
        qwen_yarn_rope = {
            "rope_type": rope.scaling,
            "max_position_embeddings": rope.max_seq_len,
            "original_max_position_embeddings": rope.original_seq_len,
            "factor": rope.rope_factor,
            "rope_theta": rope.theta,
            "beta_fast": rope.beta_fast,
            "beta_slow": rope.beta_slow,
            "backend": rope.backend,
        }
        expected_rope = expected_qwen_yarn_rope_config()
        mismatches = {
            key: {"expected": expected_rope[key], "actual": qwen_yarn_rope.get(key)}
            for key in expected_rope
            if qwen_yarn_rope.get(key) != expected_rope[key]
        }
        if mismatches:
            raise RuntimeError(
                "TorchTitan Qwen2.5-Coder-7B YaRN config does not match "
                f"the paper/Qwen 128k contract: {mismatches}"
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "Could not validate the TorchTitan Qwen2.5-Coder-7B YaRN config."
        ) from exc

    torchrun_bin = None
    cuda = _cuda_launch_summary(
        torch,
        nproc_per_node=args.nproc_per_node if args is not None else None,
    )
    if args is not None:
        torchrun_bin = _resolve_executable(args.torchrun_bin)
        if torchrun_bin is None:
            raise RuntimeError(
                f"torchrun executable is missing or not executable: "
                f"{args.torchrun_bin!r}"
            )

    return {
        "python": sys.executable,
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "node": platform.node(),
        },
        "package_versions": _package_versions(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_cuda_driver_version": _torch_cuda_driver_version(torch),
        "torch_cudnn_version": _torch_cudnn_version(torch),
        "torch_nccl_version": _torch_nccl_version(torch),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torchao": getattr(torchao, "__version__", None),
        "cuda": cuda,
        "torchrun_bin": {
            "requested": args.torchrun_bin if args is not None else None,
            "resolved": torchrun_bin,
        },
        "DataParallelMeshDims": repr(DataParallelMeshDims),
        "qwen_yarn_rope": qwen_yarn_rope,
    }


def write_runtime_metadata(
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        "schema_version": RUNTIME_METADATA_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "path": str(_runtime_metadata_path(args.out_dir)),
        "runtime": dict(runtime),
        "hardware": {
            "nvidia_smi": _nvidia_smi_metadata(),
        },
        "environment": _runtime_environment_metadata(),
        "workspace": workspace_root_metadata(args, include_cwd=True),
        "git": git_state_for_workspace(_configured_workspace_root(args)),
        "lockfiles": _runtime_lockfile_metadata(
            repo_root=_configured_workspace_root(args)
        ),
    }
    _write_json_atomic(_runtime_metadata_path(args.out_dir), metadata)
    return metadata


def validate_launch_preflight(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_torchrun = _resolve_executable(args.torchrun_bin)
    if resolved_torchrun is None:
        raise RuntimeError(
            f"torchrun executable is missing or not executable: {args.torchrun_bin!r}"
        )
    if plan.total_steps <= 0 or not plan.stages:
        raise RuntimeError("Launch plan has no training steps to run")

    resource_preflights = validate_resource_preflights(args)
    hf_assets = validate_hf_asset_preflight(args, manifest)
    checked_bucket_files = []
    for stage in plan.stages:
        data_parallel_degree = _stage_data_parallel_degree(args, stage)
        if stage.steps <= 0:
            raise RuntimeError(f"Launch stage has no steps: {stage}")
        if stage.example_count <= 0:
            raise RuntimeError(f"Launch stage has no examples: {stage}")
        if (
            args.production_mode
            and not args.production_acceptance_smoke
            and stage.example_count < data_parallel_degree
        ):
            raise RuntimeError(
                "Production bucket stage would leave data-parallel ranks empty: "
                f"bucket={stage.bucket}, examples={stage.example_count}, "
                f"data_parallel_degree={data_parallel_degree}. Refusing to rely "
                "on empty-rank data reuse for a production data run."
            )
        if not stage.bucket_file.is_file():
            raise RuntimeError(
                f"Launch bucket file does not exist: {stage.bucket_file}"
            )
        if not os.access(stage.bucket_file, os.R_OK):
            raise RuntimeError(
                f"Launch bucket file is not readable: {stage.bucket_file}"
            )
        bucket_bytes = stage.bucket_file.stat().st_size
        if bucket_bytes <= 0:
            raise RuntimeError(f"Launch bucket file is empty: {stage.bucket_file}")
        checked_bucket_files.append(
            {
                "bucket": stage.bucket,
                "path": str(stage.bucket_file),
                "bytes": bucket_bytes,
                "examples": stage.example_count,
                "steps": stage.steps,
                "data_parallel_degree": data_parallel_degree,
                "allow_empty_rank_reuse": allow_empty_rank_reuse(args),
            }
        )

    probe = (
        args.out_dir
        / f".launch-preflight-write-test-{os.getpid()}-{time.time_ns()}"
    )
    try:
        probe.write_text("ok\n")
    except Exception as exc:
        raise RuntimeError(
            f"Launch output directory is not writable: {args.out_dir}"
        ) from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass

    return {
        "torchrun_bin": {
            "requested": args.torchrun_bin,
            "resolved": resolved_torchrun,
        },
        "hf_assets": hf_assets,
        "resources": resource_preflights,
        "bucket_files": checked_bucket_files,
        "out_dir": str(args.out_dir),
    }


def _final_export_step_dir(out_dir: Path, step: int) -> Path:
    return _final_model_export_dir(out_dir) / f"step-{step}"


def _legacy_final_export_step_dir(out_dir: Path, step: int) -> Path:
    return _checkpoint_dir(out_dir) / f"step-{step}"


def _validate_dcp_checkpoint_step(step_dir: Path) -> dict[str, Any]:
    metadata_path = step_dir / ".metadata"
    _require_nonempty_file(metadata_path, "DCP checkpoint metadata")
    distcp_files = sorted(step_dir.glob("*.distcp"))
    if not distcp_files:
        raise RuntimeError(f"DCP checkpoint has no .distcp payload files: {step_dir}")

    payloads = []
    total_payload_bytes = 0
    rank_ids: set[int] = set()
    shard_ids: set[int] = set()
    for path in distcp_files:
        match = re.fullmatch(r"__(\d+)_(\d+)\.distcp", path.name)
        if not match:
            raise RuntimeError(
                "DCP checkpoint payload file name does not match the expected "
                f"'__<rank>_<shard>.distcp' pattern: {path}"
            )
        rank_id = int(match.group(1))
        shard_id = int(match.group(2))
        rank_ids.add(rank_id)
        shard_ids.add(shard_id)
        _require_nonempty_file(path, "DCP checkpoint payload")
        payload_bytes = path.stat().st_size
        total_payload_bytes += payload_bytes
        payloads.append(
            {
                "path": str(path),
                "bytes": payload_bytes,
                "rank": rank_id,
                "shard": shard_id,
            }
        )

    return {
        "step": _checkpoint_step(step_dir),
        "path": str(step_dir),
        "metadata_path": str(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": _hash_file(metadata_path),
        "payload_file_count": len(payloads),
        "payload_total_bytes": total_payload_bytes,
        "payload_rank_count": len(rank_ids),
        "payload_ranks": sorted(rank_ids),
        "payload_shards": sorted(shard_ids),
        "payload_files": payloads,
    }


def validate_first_step_checkpoint_report(
    args: argparse.Namespace,
) -> dict[str, Any]:
    report_path = _first_step_checkpoint_validation_path(args.out_dir)
    if not report_path.is_file():
        raise RuntimeError(
            "First-step checkpoint validation report is missing: "
            f"{report_path}. TorchTitan must validate step-1 DCP checkpoint "
            "contents before a production launch can continue."
        )
    report = _read_json_object_required(
        report_path,
        "first-step checkpoint validation report",
    )
    if report.get("schema_version") != FIRST_STEP_CHECKPOINT_VALIDATION_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported first-step checkpoint validation schema version in "
            f"{report_path}: {report.get('schema_version')!r}"
        )
    if report.get("step") != 1:
        raise RuntimeError(
            "First-step checkpoint validation report must describe step 1; "
            f"got {report.get('step')!r}"
        )
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(
            "First-step checkpoint validation report has no checkpoint object: "
            f"{report_path}"
        )
    if checkpoint.get("step") != 1:
        raise RuntimeError(
            "First-step checkpoint validation checkpoint payload must describe "
            f"step 1; got {checkpoint.get('step')!r}"
        )
    payload_file_count = checkpoint.get("payload_file_count")
    payload_total_bytes = checkpoint.get("payload_total_bytes")
    payload_rank_count = checkpoint.get("payload_rank_count")
    if (
        not isinstance(payload_file_count, int)
        or isinstance(payload_file_count, bool)
        or payload_file_count <= 0
    ):
        raise RuntimeError(
            "First-step checkpoint validation report has invalid "
            f"payload_file_count: {payload_file_count!r}"
        )
    if (
        not isinstance(payload_total_bytes, int)
        or isinstance(payload_total_bytes, bool)
        or payload_total_bytes <= 0
    ):
        raise RuntimeError(
            "First-step checkpoint validation report has invalid "
            f"payload_total_bytes: {payload_total_bytes!r}"
        )
    if (
        not isinstance(payload_rank_count, int)
        or isinstance(payload_rank_count, bool)
        or payload_rank_count <= 0
    ):
        raise RuntimeError(
            "First-step checkpoint validation report has invalid "
            f"payload_rank_count: {payload_rank_count!r}"
        )
    return dict(report)


def _select_final_export_step_dir(
    out_dir: Path,
    step: int,
    *,
    allow_legacy_export: bool,
) -> tuple[Path, str]:
    final_step_dir = _final_export_step_dir(out_dir, step)
    legacy_step_dir = _legacy_final_export_step_dir(out_dir, step)
    final_index = final_step_dir / "model.safetensors.index.json"
    legacy_index = legacy_step_dir / "model.safetensors.index.json"

    if final_index.is_file():
        if legacy_index.is_file():
            raise RuntimeError(
                "Final model export exists in both the separated final export "
                f"directory and the resumable checkpoint directory: {final_step_dir} "
                f"and {legacy_step_dir}"
            )
        return final_step_dir, "final_export"

    if allow_legacy_export and legacy_index.is_file():
        return legacy_step_dir, "legacy_checkpoint_export"

    raise RuntimeError(
        "Final model export is missing. Expected "
        f"{final_step_dir / 'model.safetensors.index.json'}"
    )


def _validate_final_export_step(
    out_dir: Path,
    step: int,
    *,
    allow_legacy_export: bool = False,
) -> dict[str, Any]:
    step_dir, layout = _select_final_export_step_dir(
        out_dir,
        step,
        allow_legacy_export=allow_legacy_export,
    )
    index_path = step_dir / "model.safetensors.index.json"
    _require_nonempty_file(index_path, "final model export safetensors index")
    index = _read_json_object_required(
        index_path,
        "final model export safetensors index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError(
            f"Final model export index has no weight_map entries: {index_path}"
        )
    metadata = index.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            f"Final model export index metadata must be an object: {index_path}"
        )
    metadata_total_size = metadata.get("total_size")
    if (
        not isinstance(metadata_total_size, int)
        or isinstance(metadata_total_size, bool)
        or metadata_total_size <= 0
    ):
        raise RuntimeError(
            "Final model export index metadata.total_size must be a positive "
            f"integer: {metadata_total_size!r}"
        )

    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise RuntimeError(
                "Final model export index contains a non-string or empty tensor "
                f"name: {tensor_name!r}"
            )
        if not isinstance(shard_name, str) or not shard_name:
            raise RuntimeError(
                "Final model export index contains a non-string or empty shard "
                f"path for tensor {tensor_name!r}: {shard_name!r}"
            )

    shard_names = sorted(set(weight_map.values()))
    referenced_shard_paths: set[str] = set()
    shards = []
    total_shard_bytes = 0
    for shard_name in shard_names:
        if not shard_name.endswith(".safetensors"):
            raise RuntimeError(
                "Final model export index references a non-safetensors shard: "
                f"{shard_name!r}"
            )
        shard_path = _safe_child_path(
            step_dir,
            shard_name,
            label="Final model export index",
        )
        _require_nonempty_file(shard_path, "final model export shard")
        relative_shard_path = shard_path.relative_to(step_dir).as_posix()
        referenced_shard_paths.add(relative_shard_path)
        shard_bytes = shard_path.stat().st_size
        total_shard_bytes += shard_bytes
        shards.append(
            {
                "path": relative_shard_path,
                "bytes": shard_bytes,
                "sha256": _hash_file(shard_path),
            }
        )
    top_level_safetensors = {
        path.relative_to(step_dir).as_posix()
        for path in step_dir.glob("*.safetensors")
        if path.is_file()
    }
    unindexed_shards = sorted(top_level_safetensors - referenced_shard_paths)
    if unindexed_shards:
        raise RuntimeError(
            "Final model export contains unindexed safetensors shard(s): "
            f"{unindexed_shards}"
        )
    if metadata_total_size > total_shard_bytes:
        raise RuntimeError(
            "Final model export index metadata.total_size exceeds total shard "
            f"bytes: {metadata_total_size} > {total_shard_bytes}"
        )

    return {
        "step": step,
        "layout": layout,
        "path": str(step_dir),
        "index_path": str(index_path),
        "index_sha256": _hash_file(index_path),
        "index_metadata": dict(metadata),
        "index_metadata_total_size": metadata_total_size,
        "weight_map_entries": len(weight_map),
        "shard_count": len(shards),
        "total_shard_bytes": total_shard_bytes,
        "shards": shards,
    }


def validate_final_artifacts(
    args: argparse.Namespace,
    plan: BucketPlan,
    *,
    allow_legacy_export: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    final_export = _validate_final_export_step(
        args.out_dir,
        plan.total_steps,
        allow_legacy_export=allow_legacy_export,
    )

    checkpoint_steps = _checkpoint_steps(_checkpoint_dir(args.out_dir))
    if checkpoint_steps and max(checkpoint_steps) > plan.total_steps:
        raise RuntimeError(
            "Latest resumable checkpoint step exceeds the final plan step: "
            f"{max(checkpoint_steps)} > {plan.total_steps}"
        )
    if plan.total_steps not in checkpoint_steps:
        raise RuntimeError(
            "Final resumable DCP checkpoint is missing for the plan total step: "
            f"expected {_checkpoint_dir(args.out_dir) / f'step-{plan.total_steps}'}"
        )

    dcp_checkpoints = [
        _validate_dcp_checkpoint_step(
            _checkpoint_dir(args.out_dir) / f"step-{step}"
        )
        for step in checkpoint_steps
    ]

    report = {
        "schema_version": FINAL_ARTIFACT_VALIDATION_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "plan_total_steps": plan.total_steps,
        "resumable_checkpoints": {
            "path": str(_checkpoint_dir(args.out_dir)),
            "steps": checkpoint_steps,
            "latest_step": max(checkpoint_steps) if checkpoint_steps else None,
            "checkpoints": dcp_checkpoints,
        },
        "final_export": final_export,
    }
    if write_report:
        _write_json_atomic(_final_artifact_validation_path(args.out_dir), report)
    return report


def build_hf_logits_parity_command(args: argparse.Namespace) -> list[str]:
    repo_root = _configured_workspace_root(args)
    return [
        sys.executable,
        str(repo_root / "scripts" / "qwen_swehero_logits_parity.py"),
        "--hf-model-id",
        args.model_id,
        "--hf-model-revision",
        args.model_revision,
        "--hf-assets-path",
        str(args.hf_assets_path),
        "--reference-model-path",
        str(args.hf_assets_path),
        "--reference-context",
        "paper-yarn-128k",
        "--json-out",
        str(args.out_dir / "hf_logits_parity.json"),
    ]


def verify_hf_logits_parity_if_requested(args: argparse.Namespace) -> None:
    if not args.verify_hf_logits_parity:
        return

    command = build_hf_logits_parity_command(args)
    print("Running HF logits parity check:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=_configured_workspace_root(args))


def _clean_optional_string(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_wandb_args(args: argparse.Namespace) -> None:
    for field in (
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_run_id",
        "wandb_resume",
        "wandb_resume_from",
        "wandb_fork_from",
        "wandb_run_group",
        "wandb_run_job_type",
        "wandb_run_tags",
        "wandb_run_notes",
        "wandb_mode",
    ):
        setattr(args, field, _clean_optional_string(getattr(args, field)))


def _validate_wandb_run_id(run_id: str) -> None:
    if len(run_id) > 64:
        raise ValueError("--wandb-run-id must be no longer than 64 characters")
    forbidden = sorted(WANDB_RUN_ID_FORBIDDEN_CHARS & set(run_id))
    if forbidden:
        raise ValueError(
            "--wandb-run-id contains characters W&B forbids: "
            + ", ".join(repr(char) for char in forbidden)
        )


def _generate_wandb_run_id() -> str:
    return f"swehero-{uuid.uuid4().hex}"


def _wandb_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if args.wandb_project:
        env["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        env["WANDB_RUN_NAME"] = args.wandb_run_name
        env["WANDB_NAME"] = args.wandb_run_name
    if not args.enable_wandb:
        return env
    if args.wandb_entity:
        env["WANDB_TEAM"] = args.wandb_entity
        env["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_run_id:
        env["WANDB_RUN_ID"] = args.wandb_run_id
    if args.wandb_resume:
        env["WANDB_RESUME"] = args.wandb_resume
    if args.wandb_resume_from:
        env["WANDB_RESUME_FROM"] = args.wandb_resume_from
    if args.wandb_fork_from:
        env["WANDB_FORK_FROM"] = args.wandb_fork_from
    if args.wandb_run_group:
        env["WANDB_RUN_GROUP"] = args.wandb_run_group
    if args.wandb_run_job_type:
        env["WANDB_RUN_JOB_TYPE"] = args.wandb_run_job_type
        env["WANDB_JOB_TYPE"] = args.wandb_run_job_type
    if args.wandb_run_tags:
        env["WANDB_RUN_TAGS"] = args.wandb_run_tags
        env["WANDB_TAGS"] = args.wandb_run_tags
    if args.wandb_run_notes:
        env["WANDB_RUN_NOTES"] = args.wandb_run_notes
        env["WANDB_NOTES"] = args.wandb_run_notes
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode
    return env


def _wandb_identity_contract(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": identity.get("enabled"),
        "project": identity.get("project"),
        "entity": identity.get("entity"),
        "run_name": identity.get("run_name"),
        "run_id": identity.get("run_id"),
        "resume": identity.get("resume"),
        "resume_from": identity.get("resume_from"),
        "fork_from": identity.get("fork_from"),
        "run_group": identity.get("run_group"),
        "run_job_type": identity.get("run_job_type"),
        "run_tags": identity.get("run_tags"),
        "run_notes": identity.get("run_notes"),
        "mode": identity.get("mode"),
    }


def _build_wandb_identity(
    args: argparse.Namespace,
    *,
    generated_run_id: bool,
    created_at_unix: float | None = None,
) -> dict[str, Any]:
    created_at = time.time() if created_at_unix is None else created_at_unix
    identity = {
        "schema_version": WANDB_IDENTITY_SCHEMA_VERSION,
        "created_at_unix": created_at,
        "updated_at_unix": time.time(),
        "enabled": bool(args.enable_wandb),
        "generated_run_id": generated_run_id,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "run_name": args.wandb_run_name,
        "run_id": args.wandb_run_id,
        "resume": args.wandb_resume,
        "resume_from": args.wandb_resume_from,
        "fork_from": args.wandb_fork_from,
        "run_group": args.wandb_run_group,
        "run_job_type": args.wandb_run_job_type,
        "run_tags": args.wandb_run_tags,
        "run_notes": args.wandb_run_notes,
        "mode": args.wandb_mode,
        "env": _wandb_env_overrides(args),
    }
    return identity


def _load_wandb_identity(path: Path) -> dict[str, Any]:
    identity = _read_json_object_required(path, "W&B identity")
    if identity.get("schema_version") != WANDB_IDENTITY_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported W&B identity schema version in "
            f"{path}: {identity.get('schema_version')!r}"
        )
    return identity


def _run_spec_wandb_args(out_dir: Path) -> dict[str, Any] | None:
    spec_path = _run_spec_path(out_dir)
    if not spec_path.is_file():
        return None
    spec = _read_json_object_required(spec_path, "immutable run spec")
    args = spec.get("args")
    return dict(args) if isinstance(args, Mapping) else None


def _apply_wandb_run_id_from_existing_record(
    args: argparse.Namespace,
    existing_identity: Mapping[str, Any] | None,
) -> None:
    if args.wandb_run_id:
        return
    if existing_identity and existing_identity.get("run_id"):
        args.wandb_run_id = str(existing_identity["run_id"])
        return
    run_spec_args = _run_spec_wandb_args(args.out_dir)
    if run_spec_args and run_spec_args.get("wandb_run_id"):
        args.wandb_run_id = str(run_spec_args["wandb_run_id"])


def resolve_wandb_identity(
    args: argparse.Namespace,
    *,
    resume_state: ResumeCheckpointState | None,
) -> dict[str, Any] | None:
    _normalize_wandb_args(args)
    if not args.enable_wandb:
        return None

    if args.wandb_resume_from and args.wandb_fork_from:
        raise ValueError(
            "--wandb-resume-from and --wandb-fork-from are mutually exclusive"
        )
    if args.wandb_resume and (args.wandb_resume_from or args.wandb_fork_from):
        raise ValueError(
            "--wandb-resume cannot be combined with --wandb-resume-from or "
            "--wandb-fork-from"
        )
    if args.wandb_resume is None and not (
        args.wandb_resume_from or args.wandb_fork_from
    ):
        args.wandb_resume = "allow"

    path = _wandb_identity_path(args.out_dir)
    existing_identity = _load_wandb_identity(path) if path.exists() else None
    _apply_wandb_run_id_from_existing_record(args, existing_identity)
    generated_run_id = False
    if not args.wandb_run_id:
        if resume_state is not None:
            raise RuntimeError(
                "--resume with --enable-wandb requires a persisted W&B run id "
                f"in {path} or {RUN_SPEC_FILENAME}, or an explicit --wandb-run-id."
            )
        args.wandb_run_id = _generate_wandb_run_id()
        generated_run_id = True

    _validate_wandb_run_id(args.wandb_run_id)
    identity = _build_wandb_identity(
        args,
        generated_run_id=generated_run_id,
        created_at_unix=existing_identity.get("created_at_unix")
        if isinstance(existing_identity, Mapping)
        else None,
    )
    if existing_identity is not None:
        existing_contract = _wandb_identity_contract(existing_identity)
        actual_contract = _wandb_identity_contract(identity)
        diffs = _contract_diffs(existing_contract, actual_contract)
        if diffs:
            preview = "\n".join(f"- {diff}" for diff in diffs[:20])
            extra = "" if len(diffs) <= 20 else f"\n... and {len(diffs) - 20} more"
            raise RuntimeError(
                "Current W&B identity does not match the existing run identity:\n"
                f"{preview}{extra}"
            )
        identity["generated_run_id"] = bool(existing_identity.get("generated_run_id"))

    _write_json_atomic(path, identity)
    return identity


def build_stage_env(
    args: argparse.Namespace,
    *,
    stage: BucketStage,
    total_steps: int,
    warmup_steps: int,
    pad_token_id: int,
    load_dataloader_state: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    for key in CONTROLLED_WANDB_ENV_KEYS:
        env.pop(key, None)
    repo_root = _configured_workspace_root(args)
    pythonpath_entries = [str(repo_root / "torchtitan"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_DEVICE_MAX_CONNECTIONS": str(args.cuda_device_max_connections),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": str(
                args.torch_nccl_async_error_handling
            ),
            "LOG_RANK": args.log_rank,
            "SWEHERO_WORKSPACE_ROOT": str(repo_root),
            "SWEHERO_MODEL_ID": args.model_id,
            "SWEHERO_MODEL_REVISION": args.model_revision,
            "SWEHERO_DATASET_ID": args.dataset_id,
            "SWEHERO_DATASET_PATH": str(args.dataset_path),
            "SWEHERO_BUCKET_FILE": str(stage.bucket_file),
            "SWEHERO_BUCKET_SEQ_LEN": str(stage.bucket),
            "SWEHERO_BUCKET_CP": str(stage.cp_degree),
            "SWEHERO_ALLOW_EMPTY_RANK_REUSE": "1"
            if allow_empty_rank_reuse(args)
            else "0",
            "SWEHERO_TOTAL_STEPS": str(total_steps),
            "SWEHERO_CUMULATIVE_STEPS": str(stage.cumulative_steps),
            "SWEHERO_WARMUP_STEPS": str(warmup_steps),
            "SWEHERO_HF_ASSETS_PATH": str(args.hf_assets_path),
            "SWEHERO_TORCHTITAN_DUMP_FOLDER": str(_torchtitan_dump_dir(args.out_dir)),
            "SWEHERO_FINAL_EXPORT_FOLDER": FINAL_MODEL_EXPORT_FOLDER,
            "SWEHERO_PAD_TOKEN_ID": str(pad_token_id),
            "SWEHERO_SEED": str(args.seed),
            "SWEHERO_GLOBAL_BATCH_SIZE": str(args.global_batch_size),
            "SWEHERO_LOCAL_BATCH_SIZE": str(args.local_batch_size),
            "SWEHERO_LEARNING_RATE": repr(args.learning_rate),
            "SWEHERO_MIN_LEARNING_RATE": repr(args.min_learning_rate),
            "SWEHERO_WEIGHT_DECAY": repr(args.weight_decay),
            "SWEHERO_MAX_GRAD_NORM": repr(args.max_grad_norm),
            "SWEHERO_OPTIMIZER_IMPL": args.optimizer_impl,
            "SWEHERO_ATTENTION_BACKEND": args.attention_backend,
            "SWEHERO_ENABLE_FP8": "1" if args.enable_fp8 else "0",
            "SWEHERO_FP8_RECIPE": args.fp8_recipe,
            "SWEHERO_COMPILE": "1" if args.compile else "0",
            "SWEHERO_TRAINING_DTYPE": args.training_dtype,
            "SWEHERO_MP_PARAM_DTYPE": args.mixed_precision_param_dtype,
            "SWEHERO_MP_REDUCE_DTYPE": args.mixed_precision_reduce_dtype,
            "SWEHERO_FSDP_RESHARD_AFTER_FORWARD": args.fsdp_reshard_after_forward,
            "SWEHERO_AC_MODE": args.activation_checkpoint_mode,
            "SWEHERO_CHUNKED_CE_CHUNKS": str(args.chunked_ce_chunks),
            "SWEHERO_DETECT_ANOMALY": "1" if args.detect_anomaly else "0",
            "SWEHERO_SAVE_FINAL_FULL_CHECKPOINT": "1",
            "SWEHERO_ENABLE_FIRST_STEP_CHECKPOINT": "1"
            if args.validate_first_step_checkpoint
            else "0",
            "SWEHERO_FIRST_STEP_CHECKPOINT_VALIDATION_REPORT": str(
                _first_step_checkpoint_validation_path(args.out_dir)
            ),
            "SWEHERO_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
            "SWEHERO_CHECKPOINT_ASYNC_MODE": args.checkpoint_async_mode,
            "SWEHERO_METRICS_LOG_FREQ": str(args.metrics_log_freq),
            "SWEHERO_ENABLE_PROFILER": "1" if args.enable_profiler else "0",
            "SWEHERO_PROFILER_TRACE_FOLDER": args.profiler_trace_folder,
            "SWEHERO_PROFILER_FREQ": str(args.profiler_freq),
            "SWEHERO_PROFILER_ACTIVE": str(args.profiler_active),
            "SWEHERO_PROFILER_WARMUP": str(args.profiler_warmup),
            "SWEHERO_ENABLE_MEMORY_SNAPSHOT": "1"
            if args.enable_memory_snapshot
            else "0",
            "SWEHERO_MEMORY_SNAPSHOT_FOLDER": args.memory_snapshot_folder,
            "SWEHERO_LOAD_DATALOADER_STATE": "1"
            if load_dataloader_state
            else "0",
            "SWEHERO_ENABLE_WANDB": "1" if args.enable_wandb else "0",
        }
    )
    if args.profiler_repeat is not None:
        env["SWEHERO_PROFILER_REPEAT"] = str(args.profiler_repeat)
    if args.profiler_skip_first is not None:
        env["SWEHERO_PROFILER_SKIP_FIRST"] = str(args.profiler_skip_first)
    if args.profiler_skip_first_wait is not None:
        env["SWEHERO_PROFILER_SKIP_FIRST_WAIT"] = str(
            args.profiler_skip_first_wait
        )
    env.update(_wandb_env_overrides(args))
    return env


def _distributed_launch_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "nnodes": args.nnodes,
        "node_rank": args.node_rank,
        "nproc_per_node": args.nproc_per_node,
        "world_size": args.nnodes * args.nproc_per_node,
        "rdzv_backend": args.rdzv_backend,
        "rdzv_endpoint": args.rdzv_endpoint,
        "rdzv_id": args.rdzv_id or None,
    }


def _stage_data_parallel_degree(
    args: argparse.Namespace,
    stage: BucketStage,
) -> int:
    world_size = args.nnodes * args.nproc_per_node
    if stage.cp_degree <= 0:
        raise RuntimeError(f"Stage CP degree must be positive: {stage}")
    if world_size <= 0:
        raise RuntimeError(f"Launch world size must be positive: {world_size}")
    if world_size % stage.cp_degree != 0:
        raise RuntimeError(
            f"Launch world size {world_size} is not divisible by "
            f"stage CP degree {stage.cp_degree}"
        )
    return world_size // stage.cp_degree


def build_torchrun_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--rdzv_backend",
        args.rdzv_backend,
        "--rdzv_endpoint",
        args.rdzv_endpoint,
        "--tee",
        "3",
    ]
    if args.nnodes != 1 or args.node_rank != 0:
        command.extend(["--nnodes", str(args.nnodes)])
        command.extend(["--node_rank", str(args.node_rank)])
    if args.rdzv_id:
        command.extend(["--rdzv_id", args.rdzv_id])
    if args.torchrun_log_rank_filter:
        command.extend(["--local-ranks-filter", args.torchrun_log_rank_filter])
    command.extend(
        [
            "-m",
            "torchtitan.train",
            "--module",
            "swehero",
            "--config",
            "qwen25_coder7b_direct_to_hero",
        ]
    )
    return command


def _stage_env_overrides(
    args: argparse.Namespace,
    *,
    stage: BucketStage,
    total_steps: int,
    warmup_steps: int,
    pad_token_id: int,
    load_dataloader_state: bool = False,
) -> dict[str, str]:
    env = build_stage_env(
        args,
        stage=stage,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        pad_token_id=pad_token_id,
        load_dataloader_state=load_dataloader_state,
    )
    return {
        key: env[key]
        for key in LAUNCH_STAGE_ENV_KEYS
        if key in env
    }


def _stage_launch_record(
    args: argparse.Namespace,
    stage: BucketStage,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
    *,
    load_dataloader_state: bool = False,
) -> dict[str, Any]:
    return {
        **asdict(stage),
        "bucket_file": str(stage.bucket_file),
        "torchrun_command": build_torchrun_command(args),
        "env_overrides": _stage_env_overrides(
            args,
            stage=stage,
            total_steps=plan.total_steps,
            warmup_steps=plan.warmup_steps,
            pad_token_id=int(manifest["pad_token_id"]),
            load_dataloader_state=load_dataloader_state,
        ),
    }


def build_run_spec(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "launcher": "scripts/qwen_swehero_train.py",
        "recipe": "qwen2.5-coder-7b-direct-to-hero-torchtitan",
        "paper_alignment": paper_alignment(args),
        "args": {
            field: _jsonable(getattr(args, field))
            for field in RUN_SPEC_ARG_FIELDS
        },
        "workspace": workspace_root_metadata(args),
        "git": git_state_for_workspace(_configured_workspace_root(args)),
        "paths": {
            "workspace_root": str(_configured_workspace_root(args)),
            "out_dir": str(args.out_dir),
            "launch_lock": str(_launch_lock_path(args.out_dir)),
            "data_manifest": str(args.out_dir / "data" / "manifest.json"),
            "launcher_plan": str(args.out_dir / "launcher_plan.json"),
            "resume_contract": str(_resume_contract_path(args.out_dir)),
            "wandb_identity": str(_wandb_identity_path(args.out_dir)),
            "torchtitan_dump": str(_torchtitan_dump_dir(args.out_dir)),
            "resumable_checkpoints": str(_checkpoint_dir(args.out_dir)),
            "first_step_checkpoint_validation": str(
                _first_step_checkpoint_validation_path(args.out_dir)
            ),
            "final_model_exports": str(_final_model_export_dir(args.out_dir)),
            "runtime_metadata": str(_runtime_metadata_path(args.out_dir)),
            "post_training_eval_status": str(
                _post_training_eval_status_path(args.out_dir)
            ),
        },
        "manifest": _resume_manifest_contract(manifest),
        "plan": {
            "bucket_curriculum": args.bucket_curriculum,
            "distributed": _distributed_launch_summary(args),
            "total_steps": plan.total_steps,
            "warmup_steps": plan.warmup_steps,
            "stages": [
                _stage_launch_record(args, stage, plan, manifest)
                for stage in plan.stages
            ],
        },
    }


def write_or_validate_run_spec(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
    *,
    require_existing: bool = False,
) -> bool:
    spec_path = _run_spec_path(args.out_dir)
    sha_path = _run_spec_sha256_path(args.out_dir)
    actual = build_run_spec(args, plan, manifest)

    if spec_path.exists():
        if not sha_path.exists():
            raise RuntimeError(
                f"Immutable run spec checksum is missing: {sha_path}"
            )
        existing_text = spec_path.read_text()
        expected_sha = sha_path.read_text().strip()
        actual_sha = _sha256_text(existing_text)
        if expected_sha != actual_sha:
            raise RuntimeError(
                f"Immutable run spec checksum mismatch for {spec_path}: "
                f"{actual_sha} != {expected_sha}"
            )
        expected = json.loads(existing_text)
        diffs = _contract_diffs(expected, actual)
        if diffs:
            preview = "\n".join(f"- {diff}" for diff in diffs[:20])
            extra = "" if len(diffs) <= 20 else f"\n... and {len(diffs) - 20} more"
            raise RuntimeError(
                "Current launch does not match the immutable run spec:\n"
                f"{preview}{extra}"
            )
        return False

    if require_existing:
        raise RuntimeError(
            f"--resume requires an immutable run spec at {spec_path}. "
            "Start a fresh run with the current launcher so future resumes can "
            "prove they match the original launch."
        )
    if sha_path.exists():
        raise RuntimeError(
            f"Run spec checksum exists without {spec_path}: {sha_path}"
        )

    spec_text = _canonical_json_text(actual)
    _write_text_atomic(spec_path, spec_text)
    _write_text_atomic(sha_path, _sha256_text(spec_text) + "\n")
    return True


def _send_signal_to_process_group(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        return


def _run_command_with_signal_forwarding(
    command: list[str],
    *,
    env: Mapping[str, str],
    cwd: Path,
    log_paths: Mapping[str, str] | None = None,
) -> None:
    stdout_handle = None
    stderr_handle = None
    if log_paths is not None:
        stdout_path = Path(str(log_paths["stdout"]))
        stderr_path = Path(str(log_paths["stderr"]))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("w")
        stderr_handle = stderr_path.open("w")

    try:
        process = subprocess.Popen(
            command,
            env=dict(env),
            cwd=cwd,
            start_new_session=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except Exception:
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        raise
    received_signal: dict[str, Any] = {}
    killed_after_grace = False

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal killed_after_grace
        now = time.monotonic()
        if not received_signal:
            received_signal["signum"] = int(signum)
            received_signal["received_at_monotonic"] = now
            _send_signal_to_process_group(process.pid, int(signum))
            return
        if not killed_after_grace and getattr(signal, "SIGKILL", None) is not None:
            killed_after_grace = True
            _send_signal_to_process_group(process.pid, int(signal.SIGKILL))

    previous_handlers: dict[int, Any] = {}
    try:
        for signum in TERMINATION_SIGNALS:
            previous_handlers[int(signum)] = signal.signal(signum, forward_signal)

        while True:
            returncode = process.poll()
            if returncode is not None:
                break
            if received_signal and not killed_after_grace:
                received_at = float(received_signal["received_at_monotonic"])
                if time.monotonic() - received_at >= SIGNAL_FORWARD_GRACE_SECONDS:
                    sigkill = getattr(signal, "SIGKILL", None)
                    if sigkill is not None:
                        killed_after_grace = True
                        _send_signal_to_process_group(process.pid, int(sigkill))
            time.sleep(0.1)
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()

    if received_signal:
        raise SignalTerminationError(
            signum=int(received_signal["signum"]),
            command=command,
            returncode=returncode,
            killed_after_grace=killed_after_grace,
        )
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def run_stage(
    args: argparse.Namespace,
    stage: BucketStage,
    plan: BucketPlan,
    pad_token_id: int,
    *,
    load_dataloader_state: bool = False,
    log_paths: Mapping[str, str] | None = None,
) -> None:
    env = build_stage_env(
        args,
        stage=stage,
        total_steps=plan.total_steps,
        warmup_steps=plan.warmup_steps,
        pad_token_id=pad_token_id,
        load_dataloader_state=load_dataloader_state,
    )
    command = build_torchrun_command(args)
    print(
        "Launching bucket "
        f"{stage.bucket} (CP={stage.cp_degree}, examples={stage.example_count}, "
        f"target_step={stage.cumulative_steps})"
    )
    _run_command_with_signal_forwarding(
        command,
        env=env,
        cwd=_configured_workspace_root(args),
        log_paths=log_paths,
    )


def _stage_status_id(index: int, stage: BucketStage) -> str:
    return f"stage-{index + 1:02d}-bucket-{stage.bucket}-step-{stage.cumulative_steps}"


def _stage_status_by_id(document: Mapping[str, Any], stage_id: str) -> dict[str, Any]:
    for stage_record in document.get("stages", []):
        if isinstance(stage_record, dict) and stage_record.get("id") == stage_id:
            return stage_record
    raise RuntimeError(f"Stage status record is missing stage id {stage_id!r}")


def _stage_status_counts(stages: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(stage.get("status")) for stage in stages)
    return dict(sorted(counts.items()))


def _stage_status_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    stages = [
        stage
        for stage in document.get("stages", [])
        if isinstance(stage, Mapping)
    ]
    counts = _stage_status_counts(stages)
    completed_statuses = {"succeeded", "completed_before_resume"}
    failed_stage_ids = [
        str(stage["id"])
        for stage in stages
        if stage.get("status") == "failed" and "id" in stage
    ]
    running_stage_ids = [
        str(stage["id"])
        for stage in stages
        if stage.get("status") == "running" and "id" in stage
    ]
    final_validation = document.get("final_artifact_validation")
    final_validation_status = (
        final_validation.get("status")
        if isinstance(final_validation, Mapping)
        else None
    )
    first_step_validation = document.get("first_step_checkpoint_validation")
    first_step_validation_status = (
        first_step_validation.get("status")
        if isinstance(first_step_validation, Mapping)
        else None
    )
    post_training_eval = document.get("post_training_eval")
    post_training_eval_status = (
        post_training_eval.get("status")
        if isinstance(post_training_eval, Mapping)
        else None
    )
    return {
        "stage_status_counts": counts,
        "completed_stage_count": sum(
            1 for stage in stages if stage.get("status") in completed_statuses
        ),
        "failed_stage_ids": failed_stage_ids,
        "running_stage_ids": running_stage_ids,
        "pending_stage_ids": [
            str(stage["id"])
            for stage in stages
            if stage.get("status") == "pending" and "id" in stage
        ],
        "failure_count": len(document.get("failures", [])),
        "first_step_checkpoint_validation_status": first_step_validation_status,
        "final_artifact_validation_status": final_validation_status,
        "post_training_eval_status": post_training_eval_status,
    }


def _write_stage_status_document(path: Path, document: dict[str, Any]) -> None:
    document["updated_at_unix"] = time.time()
    document["summary"] = _stage_status_summary(document)
    _write_json_atomic(path, document)


def _load_stage_status_document(path: Path) -> dict[str, Any]:
    document = _read_json_object_required(path, "stage status")
    if document.get("schema_version") != STAGE_STATUS_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported stage status schema version in "
            f"{path}: {document.get('schema_version')!r}"
        )
    if not isinstance(document.get("stages"), list):
        raise RuntimeError(f"Stage status has no stages list: {path}")
    if not isinstance(document.get("failures"), list):
        raise RuntimeError(f"Stage status has no failures list: {path}")
    return document


def _resume_state_status_record(
    resume_state: ResumeCheckpointState | None,
) -> dict[str, Any] | None:
    if resume_state is None:
        return None
    return {
        "checkpoint_dir": str(resume_state.checkpoint_dir),
        "final_export_dir": str(resume_state.final_export_dir),
        "latest_resumable_step": resume_state.latest_resumable_step,
        "latest_model_export_step": resume_state.latest_model_export_step,
        "latest_any_step": resume_state.latest_any_step,
    }


def _initial_stage_status(
    *,
    stage: BucketStage,
    stage_id: str,
    index: int,
    should_run: bool,
    load_dataloader_state: bool,
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "index": index,
        "bucket": stage.bucket,
        "cp_degree": stage.cp_degree,
        "example_count": stage.example_count,
        "steps": stage.steps,
        "cumulative_steps": stage.cumulative_steps,
        "bucket_file": str(stage.bucket_file),
        "load_dataloader_state": load_dataloader_state,
        "status": "pending" if should_run else "completed_before_resume",
        "started_at_unix": None,
        "finished_at_unix": None,
        "duration_seconds": None,
        "attempts": [],
        "failure": None,
    }


def _build_stage_status_document(
    args: argparse.Namespace,
    plan: BucketPlan,
    *,
    resume_state: ResumeCheckpointState | None,
    stages_to_run: Iterable[BucketStage],
    dataloader_resume_flags: Mapping[int, bool],
) -> dict[str, Any]:
    stages_to_run_by_step = {
        stage.cumulative_steps
        for stage in stages_to_run
    }
    stage_records = []
    for index, stage in enumerate(plan.stages):
        stage_id = _stage_status_id(index, stage)
        stage_records.append(
            _initial_stage_status(
                stage=stage,
                stage_id=stage_id,
                index=index,
                should_run=stage.cumulative_steps in stages_to_run_by_step,
                load_dataloader_state=dataloader_resume_flags.get(
                    stage.cumulative_steps, False
                ),
            )
        )

    now = time.time()
    return {
        "schema_version": STAGE_STATUS_SCHEMA_VERSION,
        "created_at_unix": now,
        "updated_at_unix": now,
        "paths": {
            "out_dir": str(args.out_dir),
            "run_spec": str(_run_spec_path(args.out_dir)),
            "launcher_plan": str(args.out_dir / "launcher_plan.json"),
            "wandb_identity": str(_wandb_identity_path(args.out_dir)),
            "runtime_metadata": str(_runtime_metadata_path(args.out_dir)),
            "first_step_checkpoint_validation": str(
                _first_step_checkpoint_validation_path(args.out_dir)
            ),
            "final_artifact_validation": str(
                _final_artifact_validation_path(args.out_dir)
            ),
            "post_training_eval_status": str(
                _post_training_eval_status_path(args.out_dir)
            ),
        },
        "wandb": {
            "enabled": bool(args.enable_wandb),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
            "run_id": args.wandb_run_id,
            "resume": args.wandb_resume,
            "resume_from": args.wandb_resume_from,
            "fork_from": args.wandb_fork_from,
            "mode": args.wandb_mode,
        },
        "launch": {
            "production_mode": bool(args.production_mode),
            "production_acceptance_smoke": bool(args.production_acceptance_smoke),
            "resume": bool(args.resume),
            "distributed": _distributed_launch_summary(args),
            "resume_state": _resume_state_status_record(resume_state),
            "stages_to_run": [
                _stage_status_id(index, stage)
                for index, stage in enumerate(plan.stages)
                if stage.cumulative_steps in stages_to_run_by_step
            ],
        },
        "plan": {
            "bucket_curriculum": args.bucket_curriculum,
            "total_steps": plan.total_steps,
            "warmup_steps": plan.warmup_steps,
        },
        "stages": stage_records,
        "first_step_checkpoint_validation": {
            "status": "pending"
            if args.validate_first_step_checkpoint and not resume_state
            else "disabled",
            "started_at_unix": None,
            "finished_at_unix": None,
            "duration_seconds": None,
            "report_path": str(_first_step_checkpoint_validation_path(args.out_dir)),
            "report_sha256": None,
            "summary": None,
            "failure": None,
        },
        "final_artifact_validation": {
            "status": "pending",
            "started_at_unix": None,
            "finished_at_unix": None,
            "duration_seconds": None,
            "report_path": str(_final_artifact_validation_path(args.out_dir)),
            "report_sha256": None,
            "summary": None,
            "failure": None,
        },
        "post_training_eval": {
            "status": "pending" if args.post_training_eval_command else "disabled",
            "command": args.post_training_eval_command or None,
            "started_at_unix": None,
            "finished_at_unix": None,
            "duration_seconds": None,
            "report_path": str(_post_training_eval_status_path(args.out_dir)),
            "report_sha256": None,
            "summary": None,
            "failure": None,
        },
        "failures": [],
    }


def _merge_existing_stage_status(
    new_document: dict[str, Any],
    existing_document: Mapping[str, Any],
) -> dict[str, Any]:
    recovered_at = time.time()
    recovered_failures: list[dict[str, Any]] = []

    def recover_stale_attempts(
        stage_id: str,
        attempts: list[Any],
    ) -> list[Any]:
        recovered = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                recovered.append(attempt)
                continue
            if attempt.get("status") != "running":
                recovered.append(attempt)
                continue
            failure = {
                "phase": "stage_recovery",
                "stage_id": stage_id,
                "created_at_unix": recovered_at,
                "exception_type": "StaleStageAttempt",
                "message": (
                    "Recovered stale running stage attempt while initializing "
                    "stage status; the previous launcher exited before "
                    "recording an attempt result."
                ),
                "attempt": attempt.get("attempt"),
            }
            attempt = dict(attempt)
            attempt["status"] = "stale_recovered"
            attempt["recovered_at_unix"] = recovered_at
            attempt["finished_at_unix"] = recovered_at
            started_at = attempt.get("started_at_unix")
            attempt["duration_seconds"] = (
                recovered_at - float(started_at)
                if isinstance(started_at, (int, float))
                else None
            )
            attempt["failure"] = failure
            recovered_failures.append(failure)
            recovered.append(attempt)
        return recovered

    existing_by_id = {
        str(stage.get("id")): stage
        for stage in existing_document.get("stages", [])
        if isinstance(stage, Mapping) and stage.get("id") is not None
    }
    for stage in new_document["stages"]:
        existing = existing_by_id.get(stage["id"])
        if not isinstance(existing, Mapping):
            continue
        existing_attempts = existing.get("attempts")
        stage["attempts"] = (
            recover_stale_attempts(stage["id"], list(existing_attempts))
            if isinstance(existing_attempts, list)
            else []
        )
        if stage["status"] == "completed_before_resume":
            stage["status"] = (
                "succeeded"
                if existing.get("status") == "succeeded"
                else "completed_before_resume"
            )
            stage["started_at_unix"] = existing.get("started_at_unix")
            stage["finished_at_unix"] = existing.get("finished_at_unix")
            stage["duration_seconds"] = existing.get("duration_seconds")
            stage["failure"] = existing.get("failure")
            if stage["status"] == "completed_before_resume":
                stage["finished_at_unix"] = None
                stage["duration_seconds"] = None
                stage["failure"] = None
        else:
            stage["started_at_unix"] = None
            stage["finished_at_unix"] = None
            stage["duration_seconds"] = None
            stage["failure"] = None

    new_document["created_at_unix"] = existing_document.get(
        "created_at_unix", new_document["created_at_unix"]
    )
    new_document["failures"] = [
        *list(existing_document.get("failures", [])),
        *recovered_failures,
    ]
    first_step_validation = existing_document.get("first_step_checkpoint_validation")
    if isinstance(first_step_validation, Mapping):
        new_document["first_step_checkpoint_validation"] = {
            **new_document["first_step_checkpoint_validation"],
            **dict(first_step_validation),
        }
        if new_document["first_step_checkpoint_validation"].get("status") == "running":
            new_document["first_step_checkpoint_validation"]["status"] = "pending"
            new_document["first_step_checkpoint_validation"]["finished_at_unix"] = None
            new_document["first_step_checkpoint_validation"]["duration_seconds"] = None
    final_validation = existing_document.get("final_artifact_validation")
    if isinstance(final_validation, Mapping):
        new_document["final_artifact_validation"] = {
            **new_document["final_artifact_validation"],
            **dict(final_validation),
        }
        if new_document["final_artifact_validation"].get("status") == "running":
            new_document["final_artifact_validation"]["status"] = "pending"
            new_document["final_artifact_validation"]["finished_at_unix"] = None
            new_document["final_artifact_validation"]["duration_seconds"] = None
    post_training_eval = existing_document.get("post_training_eval")
    if isinstance(post_training_eval, Mapping):
        new_document["post_training_eval"] = {
            **new_document["post_training_eval"],
            **dict(post_training_eval),
        }
        if new_document["post_training_eval"].get("status") == "running":
            new_document["post_training_eval"]["status"] = (
                "pending"
                if new_document["post_training_eval"].get("command")
                else "disabled"
            )
            new_document["post_training_eval"]["finished_at_unix"] = None
            new_document["post_training_eval"]["duration_seconds"] = None
    return new_document


def initialize_stage_status(
    args: argparse.Namespace,
    plan: BucketPlan,
    *,
    resume_state: ResumeCheckpointState | None,
    stages_to_run: Iterable[BucketStage],
    dataloader_resume_flags: Mapping[int, bool],
) -> dict[str, Any]:
    path = _stage_status_path(args.out_dir)
    document = _build_stage_status_document(
        args,
        plan,
        resume_state=resume_state,
        stages_to_run=stages_to_run,
        dataloader_resume_flags=dataloader_resume_flags,
    )
    if path.exists():
        document = _merge_existing_stage_status(
            document,
            _load_stage_status_document(path),
        )
    _write_stage_status_document(path, document)
    return document


def _exception_failure_record(
    exc: BaseException,
    *,
    phase: str,
    stage_id: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "phase": phase,
        "created_at_unix": time.time(),
        "exception_type": type(exc).__name__,
        "message": str(exc) or repr(exc),
    }
    if stage_id is not None:
        record["stage_id"] = stage_id
    if isinstance(exc, SignalTerminationError):
        record["terminated_by_signal"] = True
        record["signum"] = exc.signum
        record["signal_name"] = exc.signal_name
        record["returncode"] = exc.returncode
        record["command"] = _jsonable(exc.command)
        record["killed_after_grace"] = exc.killed_after_grace
    if isinstance(exc, subprocess.CalledProcessError):
        record["returncode"] = exc.returncode
        record["command"] = _jsonable(exc.cmd)
        if exc.stdout is not None:
            record["stdout"] = str(exc.stdout)
        if exc.stderr is not None:
            record["stderr"] = str(exc.stderr)
    return record


def _first_step_checkpoint_validation_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    checkpoint_mapping = checkpoint if isinstance(checkpoint, Mapping) else {}
    return {
        "step": report.get("step"),
        "checkpoint_path": checkpoint_mapping.get("path"),
        "metadata_sha256": checkpoint_mapping.get("metadata_sha256"),
        "payload_file_count": checkpoint_mapping.get("payload_file_count"),
        "payload_total_bytes": checkpoint_mapping.get("payload_total_bytes"),
        "payload_rank_count": checkpoint_mapping.get("payload_rank_count"),
        "payload_ranks": checkpoint_mapping.get("payload_ranks"),
    }


def _record_first_step_checkpoint_validation_started(args: argparse.Namespace) -> float:
    path = _stage_status_path(args.out_dir)
    if not path.exists():
        return time.time()
    document = _load_stage_status_document(path)
    record = document["first_step_checkpoint_validation"]
    started_at = time.time()
    record.update(
        {
            "status": "running",
            "started_at_unix": started_at,
            "finished_at_unix": None,
            "duration_seconds": None,
            "report_path": str(_first_step_checkpoint_validation_path(args.out_dir)),
            "report_sha256": None,
            "summary": None,
            "failure": None,
        }
    )
    _write_stage_status_document(path, document)
    return started_at


def _record_first_step_checkpoint_validation_finished(
    args: argparse.Namespace,
    *,
    started_at_unix: float,
    report: Mapping[str, Any] | None,
    failure: dict[str, Any] | None,
) -> None:
    path = _stage_status_path(args.out_dir)
    if not path.exists():
        return
    document = _load_stage_status_document(path)
    record = document["first_step_checkpoint_validation"]
    report_path = _first_step_checkpoint_validation_path(args.out_dir)
    finished_at = time.time()
    record.update(
        {
            "status": "failed" if failure is not None else "succeeded",
            "started_at_unix": started_at_unix,
            "finished_at_unix": finished_at,
            "duration_seconds": finished_at - started_at_unix,
            "report_path": str(report_path),
            "report_sha256": _hash_file(report_path) if report_path.is_file() else None,
            "summary": _first_step_checkpoint_validation_summary(report or {}),
            "failure": failure,
        }
    )
    _write_stage_status_document(path, document)


def validate_first_step_checkpoint_report_with_status(
    args: argparse.Namespace,
    *,
    stage_id: str | None = None,
) -> dict[str, Any]:
    started_at = _record_first_step_checkpoint_validation_started(args)
    try:
        report = validate_first_step_checkpoint_report(args)
    except BaseException as exc:
        failure = _exception_failure_record(
            exc,
            phase="first_step_checkpoint_validation",
            stage_id=stage_id,
        )
        _record_first_step_checkpoint_validation_finished(
            args,
            started_at_unix=started_at,
            report=None,
            failure=failure,
        )
        raise
    _record_first_step_checkpoint_validation_finished(
        args,
        started_at_unix=started_at,
        report=report,
        failure=None,
    )
    return report


def should_validate_first_step_checkpoint_after_stage(
    args: argparse.Namespace,
    stage: BucketStage,
    plan: BucketPlan,
) -> bool:
    return (
        bool(args.validate_first_step_checkpoint)
        and not args.resume
        and bool(plan.stages)
        and stage.cumulative_steps == plan.stages[0].cumulative_steps
    )


def _record_stage_started(
    args: argparse.Namespace,
    stage: BucketStage,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
    *,
    load_dataloader_state: bool,
) -> tuple[str, int]:
    path = _stage_status_path(args.out_dir)
    document = _load_stage_status_document(path)
    index = next(
        (
            candidate_index
            for candidate_index, candidate_stage in enumerate(plan.stages)
            if candidate_stage.cumulative_steps == stage.cumulative_steps
        ),
        None,
    )
    if index is None:
        raise RuntimeError(f"Stage is not present in the bucket plan: {stage}")
    stage_id = _stage_status_id(index, stage)
    stage_record = _stage_status_by_id(document, stage_id)
    now = time.time()
    attempts = stage_record.setdefault("attempts", [])
    attempt_number = len(attempts) + 1
    log_paths = _stage_attempt_log_paths(
        args.out_dir,
        stage_id=stage_id,
        attempt_number=attempt_number,
    )
    attempts.append(
        {
            "attempt": attempt_number,
            "status": "running",
            "started_at_unix": now,
            "finished_at_unix": None,
            "duration_seconds": None,
            "torchrun_command": build_torchrun_command(args),
            "load_dataloader_state": load_dataloader_state,
            "target_cumulative_steps": stage.cumulative_steps,
            "logs": log_paths,
            "env_overrides": _stage_env_overrides(
                args,
                stage=stage,
                total_steps=plan.total_steps,
                warmup_steps=plan.warmup_steps,
                pad_token_id=int(manifest["pad_token_id"]),
                load_dataloader_state=load_dataloader_state,
            ),
            "failure": None,
        }
    )
    stage_record["status"] = "running"
    stage_record["started_at_unix"] = now
    stage_record["finished_at_unix"] = None
    stage_record["duration_seconds"] = None
    stage_record["failure"] = None
    _write_stage_status_document(path, document)
    return stage_id, attempt_number


def _record_stage_finished(
    args: argparse.Namespace,
    *,
    stage_id: str,
    attempt_number: int,
    failure: dict[str, Any] | None,
) -> None:
    path = _stage_status_path(args.out_dir)
    document = _load_stage_status_document(path)
    stage_record = _stage_status_by_id(document, stage_id)
    attempts = stage_record.get("attempts")
    if not isinstance(attempts, list) or len(attempts) < attempt_number:
        raise RuntimeError(
            f"Stage status has no attempt {attempt_number} for {stage_id}"
        )
    attempt = attempts[attempt_number - 1]
    if not isinstance(attempt, dict):
        raise RuntimeError(
            f"Stage status attempt {attempt_number} for {stage_id} is invalid"
        )

    now = time.time()
    started_at = attempt.get("started_at_unix")
    duration = (
        now - float(started_at)
        if isinstance(started_at, (int, float))
        else None
    )
    status = "failed" if failure is not None else "succeeded"
    attempt["status"] = status
    attempt["finished_at_unix"] = now
    attempt["duration_seconds"] = duration
    attempt["failure"] = failure
    stage_record["status"] = status
    stage_record["finished_at_unix"] = now
    stage_record["duration_seconds"] = duration
    stage_record["failure"] = failure
    if failure is not None:
        document.setdefault("failures", []).append(failure)
    _write_stage_status_document(path, document)


def run_stage_with_status(
    args: argparse.Namespace,
    stage: BucketStage,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
    *,
    load_dataloader_state: bool = False,
) -> None:
    stage_id, attempt_number = _record_stage_started(
        args,
        stage,
        plan,
        manifest,
        load_dataloader_state=load_dataloader_state,
    )
    log_paths = _stage_attempt_log_paths(
        args.out_dir,
        stage_id=stage_id,
        attempt_number=attempt_number,
    )
    failure_phase = "stage"
    try:
        run_stage(
            args,
            stage,
            plan,
            int(manifest["pad_token_id"]),
            load_dataloader_state=load_dataloader_state,
            log_paths=log_paths,
        )
        if should_validate_first_step_checkpoint_after_stage(args, stage, plan):
            failure_phase = "first_step_checkpoint_validation"
            validate_first_step_checkpoint_report_with_status(
                args,
                stage_id=stage_id,
            )
    except BaseException as exc:
        failure = _exception_failure_record(
            exc,
            phase=failure_phase,
            stage_id=stage_id,
        )
        _record_stage_finished(
            args,
            stage_id=stage_id,
            attempt_number=attempt_number,
            failure=failure,
        )
        raise
    _record_stage_finished(
        args,
        stage_id=stage_id,
        attempt_number=attempt_number,
        failure=None,
    )


def record_launch_failure(
    args: argparse.Namespace,
    *,
    phase: str,
    exc: BaseException,
) -> None:
    path = _stage_status_path(args.out_dir)
    if not path.exists():
        return
    document = _load_stage_status_document(path)
    failure = _exception_failure_record(exc, phase=phase)
    document.setdefault("failures", []).append(failure)
    _write_stage_status_document(path, document)


def _final_validation_status_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    final_export = report.get("final_export")
    resumable = report.get("resumable_checkpoints")
    return {
        "plan_total_steps": report.get("plan_total_steps"),
        "resumable_checkpoint_steps": resumable.get("steps")
        if isinstance(resumable, Mapping)
        else None,
        "final_export": {
            key: final_export.get(key)
            for key in (
                "step",
                "layout",
                "weight_map_entries",
                "shard_count",
                "total_shard_bytes",
                "index_sha256",
            )
        }
        if isinstance(final_export, Mapping)
        else None,
    }


def _record_final_validation_started(args: argparse.Namespace) -> None:
    path = _stage_status_path(args.out_dir)
    document = _load_stage_status_document(path)
    record = document["final_artifact_validation"]
    now = time.time()
    record.update(
        {
            "status": "running",
            "started_at_unix": now,
            "finished_at_unix": None,
            "duration_seconds": None,
            "report_path": str(_final_artifact_validation_path(args.out_dir)),
            "report_sha256": None,
            "summary": None,
            "failure": None,
        }
    )
    _write_stage_status_document(path, document)


def _record_final_validation_finished(
    args: argparse.Namespace,
    *,
    report: Mapping[str, Any] | None,
    failure: dict[str, Any] | None,
) -> None:
    path = _stage_status_path(args.out_dir)
    document = _load_stage_status_document(path)
    record = document["final_artifact_validation"]
    now = time.time()
    started_at = record.get("started_at_unix")
    duration = (
        now - float(started_at)
        if isinstance(started_at, (int, float))
        else None
    )
    report_path = _final_artifact_validation_path(args.out_dir)
    record.update(
        {
            "status": "failed" if failure is not None else "succeeded",
            "finished_at_unix": now,
            "duration_seconds": duration,
            "report_path": str(report_path),
            "report_sha256": _hash_file(report_path) if report_path.is_file() else None,
            "summary": _final_validation_status_summary(report)
            if report is not None
            else None,
            "failure": failure,
        }
    )
    if failure is not None:
        document.setdefault("failures", []).append(failure)
    _write_stage_status_document(path, document)


def validate_final_artifacts_with_status(
    args: argparse.Namespace,
    plan: BucketPlan,
    *,
    allow_legacy_export: bool = False,
) -> dict[str, Any]:
    _record_final_validation_started(args)
    try:
        report = validate_final_artifacts(
            args,
            plan,
            allow_legacy_export=allow_legacy_export,
        )
    except BaseException as exc:
        failure = _exception_failure_record(exc, phase="final_artifact_validation")
        _record_final_validation_finished(args, report=None, failure=failure)
        raise
    _record_final_validation_finished(args, report=report, failure=None)
    return report


def _text_tail(text: str | None, limit: int = 20_000) -> str | None:
    if text is None:
        return None
    return text[-limit:]


def _post_training_eval_env(
    args: argparse.Namespace,
    plan: BucketPlan,
    final_validation: Mapping[str, Any],
) -> dict[str, str]:
    final_export = final_validation.get("final_export")
    final_export_path = ""
    final_export_step = ""
    if isinstance(final_export, Mapping):
        final_export_path = str(final_export.get("path") or "")
        final_export_step = str(final_export.get("step") or "")
    return {
        "SWEHERO_WORKSPACE_ROOT": str(_configured_workspace_root(args)),
        "SWEHERO_OUT_DIR": str(args.out_dir),
        "SWEHERO_RUN_SPEC": str(_run_spec_path(args.out_dir)),
        "SWEHERO_DATA_MANIFEST": str(args.out_dir / "data" / "manifest.json"),
        "SWEHERO_LAUNCHER_PLAN": str(args.out_dir / "launcher_plan.json"),
        "SWEHERO_STAGE_STATUS": str(_stage_status_path(args.out_dir)),
        "SWEHERO_FINAL_ARTIFACT_VALIDATION": str(
            _final_artifact_validation_path(args.out_dir)
        ),
        "SWEHERO_FINAL_EXPORT_PATH": final_export_path,
        "SWEHERO_FINAL_EXPORT_STEP": final_export_step,
        "SWEHERO_TOTAL_STEPS": str(plan.total_steps),
        "SWEHERO_POST_TRAINING_EVAL_STATUS": str(
            _post_training_eval_status_path(args.out_dir)
        ),
    }


def _post_training_eval_status_record(
    args: argparse.Namespace,
    plan: BucketPlan,
    final_validation: Mapping[str, Any],
    *,
    status: str,
    started_at_unix: float,
    finished_at_unix: float | None = None,
    returncode: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    duration = (
        finished_at_unix - started_at_unix
        if finished_at_unix is not None
        else None
    )
    env_overrides = _post_training_eval_env(args, plan, final_validation)
    return {
        "schema_version": POST_TRAINING_EVAL_STATUS_SCHEMA_VERSION,
        "status": status,
        "created_at_unix": started_at_unix,
        "updated_at_unix": finished_at_unix or started_at_unix,
        "started_at_unix": started_at_unix,
        "finished_at_unix": finished_at_unix,
        "duration_seconds": duration,
        "command": args.post_training_eval_command,
        "cwd": str(_configured_workspace_root(args)),
        "env_overrides": env_overrides,
        "returncode": returncode,
        "stdout_tail": _text_tail(stdout),
        "stderr_tail": _text_tail(stderr),
        "failure": failure,
    }


def _record_post_training_eval_started(
    args: argparse.Namespace,
    status_record: Mapping[str, Any],
) -> None:
    path = _stage_status_path(args.out_dir)
    if not path.exists():
        return
    document = _load_stage_status_document(path)
    record = document["post_training_eval"]
    record.update(
        {
            "status": "running",
            "command": args.post_training_eval_command,
            "started_at_unix": status_record.get("started_at_unix"),
            "finished_at_unix": None,
            "duration_seconds": None,
            "report_path": str(_post_training_eval_status_path(args.out_dir)),
            "report_sha256": None,
            "summary": None,
            "failure": None,
        }
    )
    _write_stage_status_document(path, document)


def _record_post_training_eval_finished(
    args: argparse.Namespace,
    *,
    status_record: Mapping[str, Any],
    failure: dict[str, Any] | None,
) -> None:
    path = _stage_status_path(args.out_dir)
    if not path.exists():
        return
    document = _load_stage_status_document(path)
    record = document["post_training_eval"]
    report_path = _post_training_eval_status_path(args.out_dir)
    record.update(
        {
            "status": status_record.get("status"),
            "command": args.post_training_eval_command,
            "started_at_unix": status_record.get("started_at_unix"),
            "finished_at_unix": status_record.get("finished_at_unix"),
            "duration_seconds": status_record.get("duration_seconds"),
            "report_path": str(report_path),
            "report_sha256": _hash_file(report_path) if report_path.is_file() else None,
            "summary": {
                "returncode": status_record.get("returncode"),
                "stdout_tail": status_record.get("stdout_tail"),
                "stderr_tail": status_record.get("stderr_tail"),
            },
            "failure": failure,
        }
    )
    if failure is not None:
        document.setdefault("failures", []).append(failure)
    _write_stage_status_document(path, document)


def run_post_training_eval(
    args: argparse.Namespace,
    plan: BucketPlan,
    final_validation: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not args.post_training_eval_command:
        return None

    started_at = time.time()
    status_record = _post_training_eval_status_record(
        args,
        plan,
        final_validation,
        status="running",
        started_at_unix=started_at,
    )
    _write_json_atomic(_post_training_eval_status_path(args.out_dir), status_record)
    _record_post_training_eval_started(args, status_record)

    env = os.environ.copy()
    env.update(_post_training_eval_env(args, plan, final_validation))
    completed = subprocess.run(
        args.post_training_eval_command,
        shell=True,
        cwd=_configured_workspace_root(args),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    finished_at = time.time()
    failure = None
    status = "succeeded" if completed.returncode == 0 else "failed"
    if completed.returncode != 0:
        failure = {
            "phase": "post_training_eval",
            "created_at_unix": finished_at,
            "exception_type": "CalledProcessError",
            "message": (
                "post-training eval command failed with return code "
                f"{completed.returncode}"
            ),
            "returncode": completed.returncode,
            "command": args.post_training_eval_command,
            "stdout": _text_tail(completed.stdout),
            "stderr": _text_tail(completed.stderr),
        }
    status_record = _post_training_eval_status_record(
        args,
        plan,
        final_validation,
        status=status,
        started_at_unix=started_at,
        finished_at_unix=finished_at,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        failure=failure,
    )
    _write_json_atomic(_post_training_eval_status_path(args.out_dir), status_record)
    _record_post_training_eval_finished(
        args,
        status_record=status_record,
        failure=failure,
    )
    if failure is not None:
        raise RuntimeError(failure["message"])
    return status_record


def _write_launcher_plan(
    args: argparse.Namespace,
    plan: BucketPlan,
    manifest: Mapping[str, Any],
    *,
    dataloader_resume_flags: Mapping[int, bool] | None = None,
) -> None:
    dataloader_resume_flags = dataloader_resume_flags or {}
    launcher_plan = {
        "stages": [
            _stage_launch_record(
                args,
                stage,
                plan,
                manifest,
                load_dataloader_state=dataloader_resume_flags.get(
                    stage.cumulative_steps, False
                ),
            )
            for stage in plan.stages
        ],
        "bucket_curriculum": args.bucket_curriculum,
        "distributed": _distributed_launch_summary(args),
        "workspace": workspace_root_metadata(args, include_cwd=True),
        "total_steps": plan.total_steps,
        "warmup_steps": plan.warmup_steps,
        "manifest": str(args.out_dir / "data" / "manifest.json"),
        "run_spec": str(_run_spec_path(args.out_dir)),
        "launch_lock": str(_launch_lock_path(args.out_dir)),
        "stage_status": str(_stage_status_path(args.out_dir)),
        "wandb_identity": str(_wandb_identity_path(args.out_dir)),
        "runtime_metadata": str(_runtime_metadata_path(args.out_dir)),
        "post_training_eval_status": str(_post_training_eval_status_path(args.out_dir)),
    }
    (args.out_dir / "launcher_plan.json").write_text(
        json.dumps(launcher_plan, indent=2)
    )


def _run_launch(
    args: argparse.Namespace,
    *,
    buckets: tuple[int, ...],
    bucket_cp: Mapping[int, int],
) -> None:
    resume_state = validate_resume_request(args)
    if resume_state is not None:
        args.skip_data_prep = True

    if args.overwrite_output and args.out_dir.exists():
        _ensure_safe_destructive_path("--out-dir", args.out_dir)
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = _checkpoint_dir(args.out_dir)
    final_export_dir = _final_model_export_dir(args.out_dir)
    if checkpoint_dir.exists() and not (args.resume or args.dry_run):
        raise RuntimeError(
            f"{checkpoint_dir} already exists. Pass --resume to continue or "
            "--overwrite-output to start fresh."
        )
    if final_export_dir.exists() and not (args.resume or args.dry_run):
        raise RuntimeError(
            f"{final_export_dir} already exists. Pass --resume to inspect the "
            "completed export or --overwrite-output to start fresh."
        )

    resolve_wandb_identity(args, resume_state=resume_state)

    download_hf_assets_if_requested(args)

    if not (args.dry_run or args.prepare_data_only):
        hf_asset_preflight = validate_hf_asset_preflight(args)
        print(json.dumps({"hf_asset_preflight": hf_asset_preflight}, indent=2))
        runtime = write_runtime_metadata(args, validate_torchtitan_runtime(args))
        print(json.dumps({"torchtitan_runtime": runtime}, indent=2))

    verify_hf_logits_parity_if_requested(args)

    if args.skip_data_prep:
        manifest = _load_manifest(args.out_dir)
    elif args.smoke_synthetic_buckets:
        manifest = materialize_synthetic_smoke_buckets(args)
    else:
        ensure_training_dataset(args)
        manifest = materialize_training_buckets(args)

    bucket_counts = _bucket_counts_from_manifest(manifest)
    bucket_files = _bucket_files_from_manifest(manifest)
    plan = build_bucket_plan(
        bucket_counts=bucket_counts,
        bucket_files=bucket_files,
        bucket_cp=bucket_cp,
        epochs=args.num_train_epochs,
        global_batch_size=args.global_batch_size,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        bucket_curriculum=args.bucket_curriculum,
    )
    write_or_validate_run_spec(
        args,
        plan,
        manifest,
        require_existing=resume_state is not None,
    )
    if resume_state is not None:
        validate_resume_contract(args, plan, manifest)
        validate_resume_progress(plan, resume_state)
    else:
        _write_resume_contract(args, plan, manifest)
    dataloader_resume_flags = dataloader_resume_flags_by_stage(plan, resume_state)
    stages_to_run = stages_to_run_for_resume(plan, resume_state)
    _write_launcher_plan(
        args,
        plan,
        manifest,
        dataloader_resume_flags=dataloader_resume_flags,
    )
    if not (args.dry_run or args.prepare_data_only):
        initialize_stage_status(
            args,
            plan,
            resume_state=resume_state,
            stages_to_run=stages_to_run,
            dataloader_resume_flags=dataloader_resume_flags,
        )
        try:
            launch_preflight = validate_launch_preflight(args, plan, manifest)
        except BaseException as exc:
            record_launch_failure(args, phase="launch_preflight", exc=exc)
            raise
        print(json.dumps({"launch_preflight": launch_preflight}, indent=2))

    print(json.dumps({"bucket_plan": [asdict(stage) for stage in plan.stages]}, default=str, indent=2))
    if args.prepare_data_only or args.dry_run:
        print(f"Wrote launcher plan to {args.out_dir / 'launcher_plan.json'}")
        return

    if resume_state is not None:
        print(
            "Resume state: "
            f"latest_full_checkpoint={resume_state.latest_resumable_step}, "
            f"latest_final_export={resume_state.latest_model_export_step}; "
            f"{len(stages_to_run)} bucket stage(s) remain."
        )
        if not stages_to_run:
            final_validation = validate_final_artifacts_with_status(
                args,
                plan,
                allow_legacy_export=True,
            )
            print(
                json.dumps(
                    {"final_artifact_validation": final_validation},
                    indent=2,
                )
            )
            eval_status = run_post_training_eval(args, plan, final_validation)
            if eval_status is not None:
                print(json.dumps({"post_training_eval": eval_status}, indent=2))
            print("No bucket stages remain; training is already complete for this plan.")
            return
    for stage in stages_to_run:
        run_stage_with_status(
            args,
            stage,
            plan,
            manifest,
            load_dataloader_state=dataloader_resume_flags.get(
                stage.cumulative_steps, False
            ),
        )
    final_validation = validate_final_artifacts_with_status(args, plan)
    print(json.dumps({"final_artifact_validation": final_validation}, indent=2))
    eval_status = run_post_training_eval(args, plan, final_validation)
    if eval_status is not None:
        print(json.dumps({"post_training_eval": eval_status}, indent=2))


def main(argv: list[str] | None = None) -> None:
    env_file = load_launch_env_file(argv)
    args = parse_args(argv, env_file_default=env_file)

    args.buckets = ",".join(str(b) for b in parse_bucket_list(args.buckets))
    buckets = parse_bucket_list(args.buckets)
    bucket_cp = parse_bucket_cp_map(args.bucket_cp)
    args.bucket_cp = _format_bucket_cp_map(bucket_cp)
    validate_launch_inputs(args, buckets=buckets, bucket_cp=bucket_cp)
    validate_bucket_config(
        buckets=buckets,
        bucket_cp=bucket_cp,
        nproc_per_node=args.nproc_per_node,
        attention_backend=args.attention_backend,
    )

    with launch_lock(args):
        _run_launch(args, buckets=buckets, bucket_cp=bucket_cp)


if __name__ == "__main__":
    main()
