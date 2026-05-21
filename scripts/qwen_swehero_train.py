"""Launch the TorchTitan SWE-HERO direct-to-hero 7B training job.

This is the production entrypoint for the Qwen2.5-Coder-7B SWE-HERO
scale-study run. It intentionally keeps the paper-facing recipe visible:

* Qwen2.5-Coder-7B-Instruct initialized from the Hugging Face checkpoint;
* public ``nvidia/SWE-Hero-openhands-trajectories`` traces as the current
  canonical data source;
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

from scripts import qwen_swehero_smoke as smoke


IGNORE_INDEX = -100

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATASET_ID = "nvidia/SWE-Hero-openhands-trajectories"
PAPER_CONTEXT_LENGTH = 131_072
QWEN_NATIVE_CONTEXT_LENGTH = 32_768
DEFAULT_OUT_DIR = Path("/workspace/qwen25-coder7b-swehero-torchtitan")
DEFAULT_HF_ASSETS_PATH = Path("/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct")
DEFAULT_NUM_EXAMPLES = 64
DEFAULT_MAX_STREAMED_EXAMPLES = 1_024
DEFAULT_BUCKETS = (8_192, 16_384, 32_768, 65_536, PAPER_CONTEXT_LENGTH)
DEFAULT_BUCKET_CP = {
    8_192: 1,
    16_384: 1,
    32_768: 2,
    65_536: 4,
    PAPER_CONTEXT_LENGTH: 8,
}


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a bucketed SWE-HERO smoke subset and launch TorchTitan."
    )
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", MODEL_ID))
    parser.add_argument(
        "--dataset-id", default=os.environ.get("DATASET_ID", DATASET_ID)
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("DATASET_REVISION"),
        help="Optional Hugging Face dataset revision to materialize.",
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
        default=os.environ.get("ENV_FILE", smoke.ENV_FILE),
        help="Optional dotenv-style file loaded before work starts.",
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
        help="Usable examples to keep for the current smoke run.",
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
        default=os.environ.get("TORCHRUN_BIN", "torchrun"),
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
    return parser.parse_args(argv)


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


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "trace_serializer": "OpenHands role markers; assistant content/tool_calls trainable; tool observations masked",
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
    return {
        "kept": {
            "base_model": args.model_id,
            "dataset": args.dataset_id,
            "epochs": args.num_train_epochs,
            "global_batch_size": args.global_batch_size,
            "lr_schedule": "cosine",
            "peak_lr": args.learning_rate,
            "min_lr": args.min_learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "context_length": PAPER_CONTEXT_LENGTH,
            "loss_masking": "assistant content and assistant tool calls only",
            "swe_zero_stage": "skipped for direct-to-hero",
        },
        "paper_caveats": [
            "The paper reports direct-to-hero as a 32B ablation; this 7B run is a scale-study extension.",
            "The public Hugging Face SWE-HERO release is used as canonical even though its row count differs from the paper wording.",
        ],
        "intentional_engineering_deltas": [
            "TorchTitan distributed full-model SFT replaces the earlier local Transformers smoke script.",
            "FSDP uses BF16 mixed-precision parameters/reductions; FP8 is applied only to TorchTitan linear layers selected by its converter.",
            "Length buckets with per-bucket CP replace static 128k padding.",
            "TorchTitan VarlenAttention currently does not support CP, so CP buckets use the supported SDPA/Flex attention path.",
            f"This smoke run materializes {args.num_examples} examples while the filtered production dataset is being prepared.",
        ],
    }


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def qwen_openhands_turn_segments(turn: object) -> list[tuple[str, bool]]:
    if not isinstance(turn, dict):
        return [(json.dumps(turn, ensure_ascii=False) + "\n", False)]

    role = _stringify(turn.get("role") or "unknown")
    is_assistant = role == "assistant"
    segments: list[tuple[str, bool]] = [(f"<|{role}|>\n", False)]

    content = _stringify(turn.get("content"))
    if content:
        segments.append((content.rstrip("\n") + "\n", is_assistant))

    tool_calls = turn.get("tool_calls")
    if tool_calls:
        segments.append(("<|tool_calls|>\n", False))
        segments.append((json.dumps(tool_calls, ensure_ascii=False) + "\n", is_assistant))

    return segments


def qwen_openhands_segments(
    example: dict[str, object],
    *,
    include_model_patch: bool = False,
) -> list[tuple[str, bool]]:
    segments: list[tuple[str, bool]] = []
    trajectory = example.get("trajectory") or example.get("messages") or []
    if isinstance(trajectory, list):
        for turn in trajectory:
            segments.extend(qwen_openhands_turn_segments(turn))
    else:
        segments.append((_stringify(trajectory) + "\n", False))

    patch = example.get("model_patch")
    if include_model_patch and patch:
        segments.append(("<|assistant_final_patch|>\n", False))
        segments.append((_stringify(patch) + "\n", True))

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

    token_ids = token_ids[: max_length + 1]
    labels = labels[: max_length + 1]
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


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "data" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} does not exist; run without --skip-data-prep first"
        )
    return json.loads(manifest_path.read_text())


def materialize_smoke_subset(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    buckets = parse_bucket_list(args.buckets)

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(args.hf_assets_path))
    pad_token_id = infer_pad_token_id(tokenizer, args.hf_assets_path)

    bucket_paths = {
        bucket: data_dir / f"bucket_{bucket}.jsonl" for bucket in buckets
    }
    handles = {bucket: path.open("w") for bucket, path in bucket_paths.items()}
    bucket_counts: Counter[int] = Counter()
    skipped: Counter[str] = Counter()
    length_histogram: Counter[int] = Counter()
    streamed_examples = 0
    usable_examples = 0

    load_kwargs: dict[str, Any] = {"split": "train", "streaming": True}
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision

    raw = load_dataset(args.dataset_id, **load_kwargs)
    if args.shuffle_buffer > 0:
        raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    try:
        for example in raw:
            streamed_examples += 1
            encoded = encode_swehero_example(
                tokenizer,
                example,
                max_length=min(args.max_length, max(buckets)),
                min_trainable_tokens=args.min_trainable_tokens,
                include_model_patch=args.include_model_patch,
            )
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
                        "source_id": _example_id(example, streamed_examples),
                    }
                    handles[bucket].write(json.dumps(record) + "\n")
                    bucket_counts[bucket] += 1
                    rounded_length = int(math.ceil(encoded["length"] / 1024) * 1024)
                    length_histogram[rounded_length] += 1
                    usable_examples += 1

            if usable_examples >= args.num_examples:
                break
            if streamed_examples >= args.max_streamed_examples:
                break
    finally:
        for handle in handles.values():
            handle.close()

    if usable_examples == 0:
        raise RuntimeError(
            "No usable SWE-HERO examples were materialized. Increase "
            "--max-streamed-examples, reduce --min-trainable-tokens, or inspect "
            "the dataset schema."
        )

    manifest = {
        "created_at_unix": time.time(),
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_revision": _dataset_revision_info(
            args.dataset_id, args.dataset_revision
        ),
        "paper_alignment": paper_alignment(args),
        "tokenizer": _tokenizer_metadata(tokenizer, args.hf_assets_path),
        "pad_token_id": pad_token_id,
        "max_length": args.max_length,
        "buckets": list(buckets),
        "bucket_files": {
            str(bucket): str(path) for bucket, path in bucket_paths.items()
        },
        "bucket_counts": {str(bucket): bucket_counts[bucket] for bucket in buckets},
        "length_histogram_rounded_to_1024": {
            str(length): count for length, count in sorted(length_histogram.items())
        },
        "num_usable_examples": usable_examples,
        "streamed_examples_scanned": streamed_examples,
        "skipped": dict(skipped),
        "include_model_patch": args.include_model_patch,
        "git": {
            "branch": _run_git(["branch", "--show-current"]),
            "commit": _run_git(["rev-parse", "HEAD"]),
            "status_short": _run_git(["status", "--short"]),
        },
        "software_versions": _package_versions(),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


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


def build_stage_env(
    args: argparse.Namespace,
    *,
    stage: BucketStage,
    total_steps: int,
    warmup_steps: int,
    pad_token_id: int,
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


def run_stage(args: argparse.Namespace, stage: BucketStage, plan: BucketPlan, pad_token_id: int) -> None:
    env = build_stage_env(
        args,
        stage=stage,
        total_steps=plan.total_steps,
        warmup_steps=plan.warmup_steps,
        pad_token_id=pad_token_id,
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
) -> None:
    launcher_plan = {
        "stages": [
            {
                **asdict(stage),
                "bucket_file": str(stage.bucket_file),
                "torchrun_command": build_torchrun_command(args),
                "env_overrides": {
                    key: value
                    for key, value in build_stage_env(
                        args,
                        stage=stage,
                        total_steps=plan.total_steps,
                        warmup_steps=plan.warmup_steps,
                        pad_token_id=int(manifest["pad_token_id"]),
                    ).items()
                    if key.startswith("SWEHERO_")
                    or key in {"LOG_RANK", "WANDB_PROJECT", "WANDB_RUN_NAME"}
                },
            }
            for stage in plan.stages
        ],
        "total_steps": plan.total_steps,
        "warmup_steps": plan.warmup_steps,
        "manifest": str(args.out_dir / "data" / "manifest.json"),
    }
    (args.out_dir / "launcher_plan.json").write_text(
        json.dumps(launcher_plan, indent=2)
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    smoke.load_env_file(args.env_file)

    args.buckets = ",".join(str(b) for b in parse_bucket_list(args.buckets))
    buckets = parse_bucket_list(args.buckets)
    bucket_cp = parse_bucket_cp_map(args.bucket_cp)
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

    if args.skip_data_prep:
        manifest = _load_manifest(args.out_dir)
    else:
        manifest = materialize_smoke_subset(args)

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
    _write_launcher_plan(args, plan, manifest)

    print(json.dumps({"bucket_plan": [asdict(stage) for stage in plan.stages]}, default=str, indent=2))
    if args.prepare_data_only or args.dry_run:
        print(f"Wrote launcher plan to {args.out_dir / 'launcher_plan.json'}")
        return

    pad_token_id = int(manifest["pad_token_id"])
    for stage in plan.stages:
        run_stage(args, stage, plan, pad_token_id)


if __name__ == "__main__":
    main()
