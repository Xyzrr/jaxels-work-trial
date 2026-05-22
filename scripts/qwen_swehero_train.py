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
import re
import shutil
import subprocess
import sys
import time
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
TRAINING_DATASET_NAME = "swe-hero-openhands-trajectories-5b2ed21-one-rollout"
DATASET_ID = TRAINING_DATASET_NAME
SOURCE_DATASET_ID = one_rollout.DATASET_ID
SOURCE_DATASET_REVISION = one_rollout.HISTORICAL_REVISION
PAPER_CONTEXT_LENGTH = 131_072
QWEN_NATIVE_CONTEXT_LENGTH = 32_768
DEFAULT_OUT_DIR = Path("/workspace/qwen25-coder7b-swehero-torchtitan")
DEFAULT_HF_ASSETS_PATH = Path("/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct")
DEFAULT_DATASET_PATH = Path("/workspace/datasets") / TRAINING_DATASET_NAME
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
QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
QWEN_ROPE_THETA = 1_000_000.0
QWEN_YARN_BETA_FAST = 32.0
QWEN_YARN_BETA_SLOW = 1.0
MATERIALIZED_DATA_SCHEMA_VERSION = 1
MODEL_ASSET_PROVENANCE_SCHEMA_VERSION = 1
RUN_SPEC_SCHEMA_VERSION = 1
RUN_SPEC_FILENAME = "run_spec.json"
RUN_SPEC_SHA256_FILENAME = "run_spec.sha256"
RESUME_CONTRACT_SCHEMA_VERSION = 1
RESUME_CONTRACT_FILENAME = "resume_contract.json"
RESUME_ARG_FIELDS = (
    "model_id",
    "dataset_id",
    "dataset_path",
    "source_dataset_id",
    "source_dataset_revision",
    "hf_assets_path",
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
    "attention_backend",
    "enable_fp8",
    "fp8_recipe",
    "compile",
    "activation_checkpoint_mode",
    "chunked_ce_chunks",
    "nproc_per_node",
)
RUN_SPEC_ARG_FIELDS = (
    "model_id",
    "dataset_id",
    "dataset_path",
    "source_dataset_id",
    "source_dataset_revision",
    "build_dataset_if_missing",
    "source_dataset_rows_per_shard",
    "source_dataset_build_batch_size",
    "hf_assets_path",
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
    "attention_backend",
    "enable_fp8",
    "fp8_recipe",
    "compile",
    "activation_checkpoint_mode",
    "chunked_ce_chunks",
    "checkpoint_interval",
    "checkpoint_async_mode",
    "metrics_log_freq",
    "enable_wandb",
    "wandb_project",
    "wandb_run_name",
    "nproc_per_node",
    "torchrun_bin",
    "log_rank",
    "torchrun_log_rank_filter",
)
RESUME_STAGE_ENV_KEYS = (
    "SWEHERO_MODEL_ID",
    "SWEHERO_DATASET_ID",
    "SWEHERO_DATASET_PATH",
    "SWEHERO_BUCKET_FILE",
    "SWEHERO_BUCKET_SEQ_LEN",
    "SWEHERO_BUCKET_CP",
    "SWEHERO_TOTAL_STEPS",
    "SWEHERO_CUMULATIVE_STEPS",
    "SWEHERO_WARMUP_STEPS",
    "SWEHERO_HF_ASSETS_PATH",
    "SWEHERO_TORCHTITAN_DUMP_FOLDER",
    "SWEHERO_PAD_TOKEN_ID",
    "SWEHERO_SEED",
    "SWEHERO_GLOBAL_BATCH_SIZE",
    "SWEHERO_LOCAL_BATCH_SIZE",
    "SWEHERO_LEARNING_RATE",
    "SWEHERO_MIN_LEARNING_RATE",
    "SWEHERO_WEIGHT_DECAY",
    "SWEHERO_MAX_GRAD_NORM",
    "SWEHERO_ATTENTION_BACKEND",
    "SWEHERO_ENABLE_FP8",
    "SWEHERO_FP8_RECIPE",
    "SWEHERO_COMPILE",
    "SWEHERO_AC_MODE",
    "SWEHERO_CHUNKED_CE_CHUNKS",
)
LAUNCH_STAGE_ENV_KEYS = (
    *RESUME_STAGE_ENV_KEYS,
    "SWEHERO_CHECKPOINT_INTERVAL",
    "SWEHERO_CHECKPOINT_ASYNC_MODE",
    "SWEHERO_METRICS_LOG_FREQ",
    "SWEHERO_LOAD_DATALOADER_STATE",
    "SWEHERO_ENABLE_WANDB",
    "LOG_RANK",
    "WANDB_PROJECT",
    "WANDB_RUN_NAME",
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
    latest_resumable_step: int
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return default if raw is None else Path(raw)


def _default_torchrun_bin() -> str:
    candidate = Path(sys.executable).with_name("torchrun")
    if candidate.exists():
        return str(candidate)
    return "torchrun"


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
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(part) for part in raw]
    if not values:
        raise ValueError("at least one sequence bucket is required")
    if any(value <= 0 for value in values):
        raise ValueError(f"bucket sizes must be positive: {values}")
    values = sorted(set(values))
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
            parsed[int(bucket.strip())] = int(cp.strip())
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


def build_bucket_plan(
    *,
    bucket_counts: Mapping[int, int],
    bucket_files: Mapping[int, Path],
    bucket_cp: Mapping[int, int],
    epochs: float,
    global_batch_size: int,
    warmup_ratio: float,
    max_steps: int = 0,
) -> BucketPlan:
    if epochs <= 0 and max_steps <= 0:
        raise ValueError("epochs must be positive unless max_steps is set")
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")

    natural: list[tuple[int, int]] = []
    for bucket in sorted(bucket_counts):
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


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"step-(\d+)", path.name)
    return int(match.group(1)) if match else None


def _has_dcp_checkpoint_metadata(path: Path) -> bool:
    return (path / ".metadata").is_file()


def _has_any_checkpoint_metadata(path: Path) -> bool:
    return _has_dcp_checkpoint_metadata(path) or (
        path / "model.safetensors.index.json"
    ).is_file()


def _checkpoint_steps(
    checkpoint_dir: Path,
    *,
    include_model_exports: bool,
) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []

    steps = []
    for path in checkpoint_dir.iterdir():
        step = _checkpoint_step(path)
        if step is None:
            continue
        has_metadata = (
            _has_any_checkpoint_metadata(path)
            if include_model_exports
            else _has_dcp_checkpoint_metadata(path)
        )
        if has_metadata:
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

    checkpoint_dir = args.out_dir / "torchtitan" / "checkpoint"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"--resume requires an existing TorchTitan checkpoint directory: {checkpoint_dir}"
        )

    resumable_steps = _checkpoint_steps(checkpoint_dir, include_model_exports=False)
    all_steps = _checkpoint_steps(checkpoint_dir, include_model_exports=True)
    if not resumable_steps:
        latest = max(all_steps) if all_steps else None
        suffix = (
            f" Found only non-resumable model export checkpoint(s), latest step {latest}."
            if latest is not None
            else ""
        )
        raise RuntimeError(
            "--resume requires at least one full DCP checkpoint with optimizer, "
            f"scheduler, and train-state metadata under {checkpoint_dir}.{suffix}"
        )
    return ResumeCheckpointState(
        checkpoint_dir=checkpoint_dir,
        latest_resumable_step=max(resumable_steps),
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


def _canonical_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        "dataset_id": manifest.get("dataset_id"),
        "dataset_path": manifest.get("dataset_path"),
        "dataset_artifact": manifest.get("dataset_artifact"),
        "source_dataset_id": manifest.get("source_dataset_id"),
        "source_dataset_revision": manifest.get("source_dataset_revision"),
        "model_assets": manifest.get("model_assets"),
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
    if resume_state.latest_any_step > resume_state.latest_resumable_step:
        if resume_state.latest_any_step == plan.total_steps:
            return
        raise RuntimeError(
            "latest checkpoint is a model export without optimizer/train-state "
            f"metadata at step {resume_state.latest_any_step}, while latest full "
            f"checkpoint is step {resume_state.latest_resumable_step}. Refusing "
            "because TorchTitan would try to load the non-resumable export."
        )


def stages_to_run_for_resume(
    plan: BucketPlan,
    resume_state: ResumeCheckpointState | None,
) -> tuple[BucketStage, ...]:
    if resume_state is None:
        return plan.stages
    progress_step = resume_state.latest_resumable_step
    if resume_state.latest_any_step == plan.total_steps:
        progress_step = resume_state.latest_any_step
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
    if resume_state is None:
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
    parser = argparse.ArgumentParser(
        description="Materialize bucketed SWE-HERO training data and launch TorchTitan."
    )
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", MODEL_ID))
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
        default=os.environ.get("LONG_EXAMPLE_POLICY", "error"),
        help=(
            "How to handle examples whose shifted input length exceeds "
            "--max-length. The production default is error so over-context "
            "trajectories cannot be silently truncated."
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
        "--bucket-cp",
        default=os.environ.get(
            "SWEHERO_BUCKET_CP", _format_bucket_cp_map(DEFAULT_BUCKET_CP)
        ),
        help="Comma-separated '<bucket>:<context-parallel-degree>' entries.",
    )
    parser.add_argument(
        "--min-trainable-tokens",
        type=int,
        default=_env_int("MIN_TRAINABLE_TOKENS", 1),
        help="Drop examples with fewer trainable shifted labels than this.",
    )
    parser.add_argument(
        "--include-model-patch",
        action="store_true",
        default=_env_flag("INCLUDE_MODEL_PATCH", False),
        help="Also train on the model_patch field when present.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=_env_float("NUM_TRAIN_EPOCHS", 3.0),
        help="SFT epochs over the materialized subset. Paper uses up to 3.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int("MAX_STEPS", 0),
        help="Optional total optimizer-step cap across all bucket stages.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=_env_int("GLOBAL_BATCH_SIZE", 32),
        help="Paper global batch size is 32.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=_env_int("LOCAL_BATCH_SIZE", 1),
        help="TorchTitan local microbatch size per data-parallel rank.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_env_float("LEARNING_RATE", 1e-5),
        help="Peak AdamW learning rate. Paper uses 1e-5.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=_env_float("MIN_LEARNING_RATE", 1e-8),
        help="Cosine floor learning rate. Paper uses 1e-8.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=_env_float("WARMUP_RATIO", 0.1),
        help="Warmup ratio. Paper uses 0.1.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=_env_float("WEIGHT_DECAY", 0.0),
        help="AdamW weight decay. The paper does not report this.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=_env_float("MAX_GRAD_NORM", 1.0),
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
        "--metrics-log-freq",
        type=int,
        default=_env_int("METRICS_LOG_FREQ", 1),
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
        "--wandb-run-name",
        default=os.environ.get("WANDB_RUN_NAME", "qwen25-coder7b-swehero-tt"),
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=_env_int("NPROC_PER_NODE", 8),
        help="GPU processes per node. Target pod is 8xH100.",
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
    return args


def _run_git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


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
        "metadata": metadata,
        "metadata_json_sha256": _hash_file(metadata_path),
        "selection_manifest_sha256": _hash_file(selection_manifest_path),
        "data_files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in data_files
        ],
        "total_data_bytes": sum(path.stat().st_size for path in data_files),
    }


def build_source_dataset_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
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
    if args.dataset_path.exists() and not args.rebuild_source_dataset:
        try:
            _training_dataset_files(args.dataset_path)
            return
        except FileNotFoundError:
            if not args.build_dataset_if_missing:
                raise

    if not (args.build_dataset_if_missing or args.rebuild_source_dataset):
        raise FileNotFoundError(
            f"{args.dataset_path} is missing. Either create it on the pod, or "
            "rerun with --build-dataset-if-missing."
        )

    command = build_source_dataset_command(args)
    if (
        args.dataset_path.exists()
        and any(args.dataset_path.iterdir())
        and "--overwrite" not in command
    ):
        command.append("--overwrite")
    print("Preparing SWE-Hero training dataset artifact:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


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


def paper_alignment(args: argparse.Namespace) -> dict[str, Any]:
    dataset_scope = (
        "all materialized examples"
        if args.num_examples == 0
        else f"capped at {args.num_examples} examples for a smoke run"
    )
    return {
        "kept": {
            "base_model": args.model_id,
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
        "intentional_engineering_deltas": [
            "TorchTitan distributed full-model SFT replaces the earlier local Transformers smoke script.",
            "FSDP uses BF16 mixed-precision parameters/reductions; FP8 is applied only to TorchTitan linear layers selected by its converter.",
            "Length buckets with per-bucket CP replace static 128k padding.",
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
        '\n<tool_call>\n{"name": "'
        + name
        + '", "arguments": '
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
            else:
                try:
                    bucket = choose_bucket(encoded["length"], buckets)
                except ValueError:
                    skipped["too_long_for_largest_bucket"] += 1
                else:
                    record = {
                        **encoded,
                        "bucket": bucket,
                        "source_id": source_id,
                    }
                    handles[bucket].write(json.dumps(record) + "\n")
                    bucket_counts[bucket] += 1
                    rounded_length = int(math.ceil(encoded["length"] / 1024) * 1024)
                    length_histogram[rounded_length] += 1
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
        manifest = {
            "materialized_data_schema_version": MATERIALIZED_DATA_SCHEMA_VERSION,
            "created_at_unix": time.time(),
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "dataset_path": str(args.dataset_path),
            "dataset_artifact": _dataset_artifact_metadata(args.dataset_path),
            "source_dataset_id": args.source_dataset_id,
            "source_dataset_revision": _dataset_revision_info(
                args.source_dataset_id, args.source_dataset_revision
            ),
            "paper_alignment": paper_alignment(args),
            "model_assets": _model_asset_provenance(
                model_id=args.model_id,
                hf_assets_path=args.hf_assets_path,
                tokenizer_metadata=tokenizer_metadata,
            ),
            "tokenizer": tokenizer_metadata,
            "pad_token_id": pad_token_id,
            "max_length": args.max_length,
            "long_example_policy": args.long_example_policy,
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
            "git": {
                "branch": _run_git(["branch", "--show-current"]),
                "commit": _run_git(["rev-parse", "HEAD"]),
                "status_short": _run_git(["status", "--short"]),
            },
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
    repo_root = Path(__file__).resolve().parents[1]
    local_dir = args.hf_assets_path.parent
    command = [
        sys.executable,
        str(repo_root / "torchtitan" / "scripts" / "download_hf_assets.py"),
        "--repo_id",
        args.model_id,
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


def validate_torchtitan_runtime() -> dict[str, Any]:
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

    return {
        "python": sys.executable,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torchao": getattr(torchao, "__version__", None),
        "DataParallelMeshDims": repr(DataParallelMeshDims),
        "qwen_yarn_rope": qwen_yarn_rope,
    }


def build_hf_logits_parity_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return [
        sys.executable,
        str(repo_root / "scripts" / "qwen_swehero_logits_parity.py"),
        "--hf-model-id",
        args.model_id,
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
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


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
    repo_root = Path(__file__).resolve().parents[1]
    pythonpath_entries = [str(repo_root / "torchtitan"), str(repo_root)]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_DEVICE_MAX_CONNECTIONS": env.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": env.get(
                "TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"
            ),
            "LOG_RANK": args.log_rank,
            "SWEHERO_MODEL_ID": args.model_id,
            "SWEHERO_DATASET_ID": args.dataset_id,
            "SWEHERO_DATASET_PATH": str(args.dataset_path),
            "SWEHERO_BUCKET_FILE": str(stage.bucket_file),
            "SWEHERO_BUCKET_SEQ_LEN": str(stage.bucket),
            "SWEHERO_BUCKET_CP": str(stage.cp_degree),
            "SWEHERO_TOTAL_STEPS": str(total_steps),
            "SWEHERO_CUMULATIVE_STEPS": str(stage.cumulative_steps),
            "SWEHERO_WARMUP_STEPS": str(warmup_steps),
            "SWEHERO_HF_ASSETS_PATH": str(args.hf_assets_path),
            "SWEHERO_TORCHTITAN_DUMP_FOLDER": str(args.out_dir / "torchtitan"),
            "SWEHERO_PAD_TOKEN_ID": str(pad_token_id),
            "SWEHERO_SEED": str(args.seed),
            "SWEHERO_GLOBAL_BATCH_SIZE": str(args.global_batch_size),
            "SWEHERO_LOCAL_BATCH_SIZE": str(args.local_batch_size),
            "SWEHERO_LEARNING_RATE": repr(args.learning_rate),
            "SWEHERO_MIN_LEARNING_RATE": repr(args.min_learning_rate),
            "SWEHERO_WEIGHT_DECAY": repr(args.weight_decay),
            "SWEHERO_MAX_GRAD_NORM": repr(args.max_grad_norm),
            "SWEHERO_ATTENTION_BACKEND": args.attention_backend,
            "SWEHERO_ENABLE_FP8": "1" if args.enable_fp8 else "0",
            "SWEHERO_FP8_RECIPE": args.fp8_recipe,
            "SWEHERO_COMPILE": "1" if args.compile else "0",
            "SWEHERO_AC_MODE": args.activation_checkpoint_mode,
            "SWEHERO_CHUNKED_CE_CHUNKS": str(args.chunked_ce_chunks),
            "SWEHERO_CHECKPOINT_INTERVAL": str(args.checkpoint_interval),
            "SWEHERO_CHECKPOINT_ASYNC_MODE": args.checkpoint_async_mode,
            "SWEHERO_METRICS_LOG_FREQ": str(args.metrics_log_freq),
            "SWEHERO_LOAD_DATALOADER_STATE": "1"
            if load_dataloader_state
            else "0",
            "SWEHERO_ENABLE_WANDB": "1" if args.enable_wandb else "0",
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_RUN_NAME": args.wandb_run_name,
        }
    )
    return env


def build_torchrun_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.torchrun_bin,
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        "localhost:0",
        "--tee",
        "3",
    ]
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
        "paths": {
            "out_dir": str(args.out_dir),
            "data_manifest": str(args.out_dir / "data" / "manifest.json"),
            "launcher_plan": str(args.out_dir / "launcher_plan.json"),
            "resume_contract": str(_resume_contract_path(args.out_dir)),
            "torchtitan_dump": str(args.out_dir / "torchtitan"),
        },
        "manifest": _resume_manifest_contract(manifest),
        "plan": {
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


def run_stage(
    args: argparse.Namespace,
    stage: BucketStage,
    plan: BucketPlan,
    pad_token_id: int,
    *,
    load_dataloader_state: bool = False,
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
    subprocess.run(command, check=True, env=env, cwd=Path(__file__).resolve().parents[1])


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
        "total_steps": plan.total_steps,
        "warmup_steps": plan.warmup_steps,
        "manifest": str(args.out_dir / "data" / "manifest.json"),
        "run_spec": str(_run_spec_path(args.out_dir)),
    }
    (args.out_dir / "launcher_plan.json").write_text(
        json.dumps(launcher_plan, indent=2)
    )


def main(argv: list[str] | None = None) -> None:
    env_file = load_launch_env_file(argv)
    args = parse_args(argv, env_file_default=env_file)

    args.buckets = ",".join(str(b) for b in parse_bucket_list(args.buckets))
    buckets = parse_bucket_list(args.buckets)
    bucket_cp = parse_bucket_cp_map(args.bucket_cp)
    args.bucket_cp = _format_bucket_cp_map(bucket_cp)
    validate_bucket_config(
        buckets=buckets,
        bucket_cp=bucket_cp,
        nproc_per_node=args.nproc_per_node,
        attention_backend=args.attention_backend,
    )

    if args.max_length > PAPER_CONTEXT_LENGTH:
        raise ValueError(
            f"--max-length={args.max_length} exceeds paper context {PAPER_CONTEXT_LENGTH}"
        )
    if args.max_length > max(buckets):
        raise ValueError("--max-length cannot exceed the largest bucket")

    resume_state = validate_resume_request(args)
    if resume_state is not None:
        args.skip_data_prep = True

    if args.overwrite_output and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tt_dump = args.out_dir / "torchtitan"
    checkpoint_dir = tt_dump / "checkpoint"
    if checkpoint_dir.exists() and not (args.resume or args.dry_run):
        raise RuntimeError(
            f"{checkpoint_dir} already exists. Pass --resume to continue or "
            "--overwrite-output to start fresh."
        )

    download_hf_assets_if_requested(args)

    if not (args.dry_run or args.prepare_data_only):
        runtime = validate_torchtitan_runtime()
        print(json.dumps({"torchtitan_runtime": runtime}, indent=2))

    verify_hf_logits_parity_if_requested(args)

    if args.skip_data_prep:
        manifest = _load_manifest(args.out_dir)
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
    _write_launcher_plan(
        args,
        plan,
        manifest,
        dataloader_resume_flags=dataloader_resume_flags,
    )

    print(json.dumps({"bucket_plan": [asdict(stage) for stage in plan.stages]}, default=str, indent=2))
    if args.prepare_data_only or args.dry_run:
        print(f"Wrote launcher plan to {args.out_dir / 'launcher_plan.json'}")
        return

    pad_token_id = int(manifest["pad_token_id"])
    stages_to_run = stages_to_run_for_resume(plan, resume_state)
    if resume_state is not None:
        print(
            "Resuming from full checkpoint step "
            f"{resume_state.latest_resumable_step}; "
            f"{len(stages_to_run)} bucket stage(s) remain."
        )
        if not stages_to_run:
            print("No bucket stages remain; training is already complete for this plan.")
            return
    for stage in stages_to_run:
        run_stage(
            args,
            stage,
            plan,
            pad_token_id,
            load_dataloader_state=dataloader_resume_flags.get(
                stage.cumulative_steps, False
            ),
        )


if __name__ == "__main__":
    main()
