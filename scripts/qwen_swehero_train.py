"""Run a short, paper-aligned SWE-HERO SFT job for Qwen2.5-Coder-7B.

This is the next step after ``qwen_swehero_smoke.py``: it performs real
optimizer updates, saves a reproducible run summary, and defaults to settings
that should fit a single 80GB H100 in roughly a few minutes when model weights
are already cached.

The script intentionally preserves the paper-facing choices that matter for a
direct-to-hero 7B scale study:

* Qwen2.5-Coder-Instruct base model;
* public ``nvidia/SWE-Hero-openhands-trajectories`` traces;
* three SFT epochs;
* effective global batch size 32 by gradient accumulation;
* cosine LR from 1e-5 toward 1e-8 with 0.1 warmup ratio;
* YaRN-capable 128k model context;
* assistant/tool-call-only loss masking, with tool observations masked out.

The runtime budget forces three material deviations from the paper recipe:

* LoRA adapters are trained by default instead of full-model SFT;
* only a small deterministic streamed subset is used;
* training examples are truncated to a short token window by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import qwen_swehero_smoke as smoke


DEFAULT_OUT_DIR = Path("/workspace/qwen25-coder7b-swehero-short-sft")
DEFAULT_TRAIN_MAX_LENGTH = 8_192
DEFAULT_NUM_EXAMPLES = 32
DEFAULT_MAX_STREAMED_EXAMPLES = 512
DEFAULT_LOGIT_CHUNK_SIZE = 1_024
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)


@dataclass(frozen=True)
class BatchConfig:
    global_batch_size: int
    per_device_train_batch_size: int
    world_size: int
    gradient_accumulation_steps: int
    effective_global_batch_size: int


@dataclass(frozen=True)
class TrainingPlan:
    num_unique_examples: int
    items_per_epoch: int
    steps_per_epoch: int
    total_optimizer_steps: int
    max_steps_override: int | None
    num_train_epochs: float
    batch: BatchConfig


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


def _optional_int(raw: str | None) -> int | None:
    if raw is None or raw == "":
        return None
    return int(raw)


def effective_batch_config(
    *,
    global_batch_size: int,
    per_device_train_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int | None,
) -> BatchConfig:
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = max(
            1,
            math.ceil(
                global_batch_size / (per_device_train_batch_size * world_size)
            ),
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    effective_global_batch_size = (
        per_device_train_batch_size * world_size * gradient_accumulation_steps
    )
    return BatchConfig(
        global_batch_size=global_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_global_batch_size=effective_global_batch_size,
    )


def build_training_plan(
    *,
    num_unique_examples: int,
    global_batch_size: int,
    per_device_train_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int | None,
    num_train_epochs: float,
    max_steps: int,
) -> TrainingPlan:
    if num_unique_examples <= 0:
        raise ValueError("num_unique_examples must be positive")
    if num_train_epochs <= 0 and max_steps <= 0:
        raise ValueError("num_train_epochs must be positive unless max_steps is set")

    batch = effective_batch_config(
        global_batch_size=global_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    items_per_epoch = max(
        batch.effective_global_batch_size,
        math.ceil(num_unique_examples / batch.effective_global_batch_size)
        * batch.effective_global_batch_size,
    )
    if max_steps > 0:
        items_per_epoch = max(
            items_per_epoch,
            batch.effective_global_batch_size * max_steps,
        )

    microbatches_per_epoch = math.ceil(
        items_per_epoch / (per_device_train_batch_size * world_size)
    )
    steps_per_epoch = math.ceil(
        microbatches_per_epoch / batch.gradient_accumulation_steps
    )
    total_optimizer_steps = (
        max_steps if max_steps > 0 else math.ceil(steps_per_epoch * num_train_epochs)
    )

    return TrainingPlan(
        num_unique_examples=num_unique_examples,
        items_per_epoch=items_per_epoch,
        steps_per_epoch=steps_per_epoch,
        total_optimizer_steps=total_optimizer_steps,
        max_steps_override=max_steps if max_steps > 0 else None,
        num_train_epochs=num_train_epochs,
        batch=batch,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a short SWE-HERO SFT job that keeps paper hyperparameters "
            "where practical under a single-H100 runtime budget."
        )
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", smoke.MODEL_ID),
        help="HF model id or local path for the Qwen2.5-Coder base model.",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.environ.get("DATASET_ID", smoke.DATASET_ID),
        help="HF dataset id or local dataset path for SWE-HERO traces.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=os.environ.get("DATASET_REVISION"),
        help="Optional HF dataset revision. Records the resolved SHA when available.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("OUT_DIR", str(DEFAULT_OUT_DIR))),
        help="Run output directory. Defaults under /workspace to stay out of git.",
    )
    parser.add_argument(
        "--env-file",
        default=os.environ.get("ENV_FILE", smoke.ENV_FILE),
        help="Optional dotenv-style file loaded before training.",
    )
    parser.add_argument(
        "--model-context-length",
        type=int,
        default=_env_int("MODEL_CONTEXT_LENGTH", smoke.PAPER_CONTEXT_LENGTH),
        help="Configured model context length. Defaults to the paper's 128k.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=_env_int("MAX_LENGTH", DEFAULT_TRAIN_MAX_LENGTH),
        help="Per-example training token window. Defaults shorter for <5min runs.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=_env_int("NUM_EXAMPLES", DEFAULT_NUM_EXAMPLES),
        help="Number of usable streamed training traces to keep.",
    )
    parser.add_argument(
        "--max-streamed-examples",
        type=int,
        default=_env_int("MAX_STREAMED_EXAMPLES", DEFAULT_MAX_STREAMED_EXAMPLES),
        help="Maximum raw streamed rows to scan while finding usable traces.",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=_env_int("SHUFFLE_BUFFER", 512),
        help="Streaming shuffle buffer. Set 0 to keep dataset order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("SEED", 17),
        help="Seed for dataset shuffling and Trainer.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=_env_float("NUM_TRAIN_EPOCHS", 3.0),
        help="SFT epochs over the retained subset. Paper uses up to 3.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=_env_int("MAX_STEPS", 0),
        help="Optional optimizer-step cap. 0 means derive from epochs.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=_env_int("GLOBAL_BATCH_SIZE", 32),
        help="Target global batch size. Paper uses 32.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=_env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1),
        help="Microbatch size per GPU.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=_env_int("WORLD_SIZE", 1),
        help="Number of data-parallel workers. Single-H100 default is 1.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=_optional_int(os.environ.get("GRADIENT_ACCUMULATION_STEPS")),
        help="Override accumulation. Defaults to ceil(global batch / microbatch).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_env_float("LEARNING_RATE", 1e-5),
        help="Peak learning rate. Paper uses 1e-5.",
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
        help="AdamW weight decay. The paper does not report this; default is 0.",
    )
    parser.add_argument(
        "--train-mode",
        choices=("lora", "full"),
        default=os.environ.get("TRAIN_MODE", "lora"),
        help="LoRA is the default single-H100 <5min compromise; full is paper-closer but heavy.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=_env_int("LORA_RANK", 16),
        help="LoRA adapter rank used when --train-mode=lora.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=_env_float("LORA_ALPHA", 32.0),
        help="LoRA alpha used when --train-mode=lora.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=_env_float("LORA_DROPOUT", 0.05),
        help="LoRA dropout used when --train-mode=lora.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=os.environ.get("LORA_TARGET_MODULES", DEFAULT_LORA_TARGET_MODULES),
        help="Comma-separated module names for LoRA injection.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        help="Transformers attention implementation, e.g. sdpa or flash_attention_2.",
    )
    parser.add_argument(
        "--logit-chunk-size",
        type=int,
        default=_env_int("LOGIT_CHUNK_SIZE", DEFAULT_LOGIT_CHUNK_SIZE),
        help="Chunk size for memory-bounded lm_head/cross-entropy.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=_env_int("LOGGING_STEPS", 1),
        help="Trainer logging interval.",
    )
    parser.add_argument(
        "--include-model-patch",
        action="store_true",
        default=_env_flag("INCLUDE_MODEL_PATCH", False),
        help="Also train on model_patch as a final assistant patch. Off by default.",
    )
    parser.add_argument(
        "--disable-yarn",
        action="store_true",
        default=not _env_flag("ENABLE_YARN", True),
        help="Disable YaRN context extension even when model context exceeds 32k.",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        default=not _env_flag("GRADIENT_CHECKPOINTING", True),
        help="Disable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=_env_flag("LOCAL_FILES_ONLY", False),
        help="Do not fetch model/tokenizer files from the network.",
    )
    parser.add_argument(
        "--dry-run-tokenize-only",
        action="store_true",
        default=_env_flag("DRY_RUN_TOKENIZE_ONLY", False),
        help="Load/tokenize data and write metadata without loading the model.",
    )
    parser.add_argument(
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("SAVE_FINAL", True),
        help="Save final adapter/model and tokenizer under the output directory.",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        default=_env_flag("ENABLE_WANDB", False),
        help=(
            "Enable Weights & Biases logging when WANDB_API_KEY is present. "
            "Disabled by default so logging permissions cannot block smoke training."
        ),
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", smoke.WANDB_ENTITY),
        help="Weights & Biases entity when WANDB_API_KEY is present.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", smoke.WANDB_PROJECT),
        help="Weights & Biases project when WANDB_API_KEY is present.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=os.environ.get(
            "WANDB_RUN_NAME", "qwen25-coder7b-swehero-short-sft"
        ),
        help="Weights & Biases run name when WANDB_API_KEY is present.",
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


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in ("torch", "transformers", "datasets", "peft"):
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", None)
        except Exception:
            versions[package_name] = None
    return versions


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest()
        if isinstance(chat_template, str)
        else None,
        "trace_serializer": "OpenHands role markers from qwen_swehero_smoke.example_segments",
    }


def _parameter_counts(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def _paper_alignment(args: argparse.Namespace) -> dict[str, Any]:
    kept = {
        "base_model_family": "Qwen2.5-Coder-Instruct",
        "dataset": "nvidia/SWE-Hero-openhands-trajectories compatible schema",
        "training_epochs": args.num_train_epochs,
        "global_batch_size": args.global_batch_size,
        "learning_rate_schedule": "cosine",
        "peak_learning_rate": args.learning_rate,
        "minimum_learning_rate": args.min_learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "model_context_length": args.model_context_length,
        "loss_masking": "assistant content/tool-call tokens only; tool observations masked",
        "swe_zero_stage": "skipped intentionally for direct-to-hero",
    }
    deviations = [
        "7B direct-to-hero is a scale-study extension; the paper's direct-to-hero ablation is reported for 32B.",
        f"trains {args.train_mode} parameters instead of confirmed full-model paper SFT",
        f"uses {args.num_examples} streamed traces instead of the full SWE-HERO corpus",
        f"uses {args.max_length} training tokens per example instead of 128k full trajectories",
        "skips SWE-bench Verified evaluation for the short training test",
    ]
    return {"kept": kept, "intentional_deviations": deviations}


def _load_encoded_examples(args: argparse.Namespace, tokenizer: Any) -> tuple[list[dict[str, list[int]]], int]:
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {
        "split": "train",
        "streaming": True,
    }
    if args.dataset_revision:
        load_kwargs["revision"] = args.dataset_revision

    raw = load_dataset(args.dataset_id, **load_kwargs)
    if args.shuffle_buffer > 0:
        raw = raw.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    encoded_examples = []
    streamed_examples = 0
    for example in raw:
        streamed_examples += 1
        encoded = smoke.encode_example(
            tokenizer,
            example,
            max_length=args.max_length,
            include_model_patch=args.include_model_patch,
        )
        if encoded is not None:
            encoded_examples.append(encoded)
        if len(encoded_examples) >= args.num_examples:
            break
        if streamed_examples >= args.max_streamed_examples:
            break

    if not encoded_examples:
        raise RuntimeError(
            "No usable training examples found. Increase --max-length, "
            "--max-streamed-examples, or disable shuffling if the retained "
            "rows contain no assistant/action tokens before truncation."
        )
    return encoded_examples, streamed_examples


def _build_optimizer(torch: Any, model: Any, args: argparse.Namespace) -> Any:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": args.weight_decay,
    }
    try:
        return torch.optim.AdamW(trainable_params, fused=True, **optimizer_kwargs)
    except TypeError:
        return torch.optim.AdamW(trainable_params, **optimizer_kwargs)


def _apply_lora(model: Any, args: argparse.Namespace) -> Any:
    if args.train_mode != "lora":
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "--train-mode=lora requires peft. Install peft in the training "
            "environment or rerun with --train-mode=full if memory permits."
        ) from exc

    target_modules = [
        module.strip()
        for module in args.lora_target_modules.split(",")
        if module.strip()
    ]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _base_causal_lm(model: Any) -> Any:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    smoke.load_env_file(args.env_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_wandb = args.enable_wandb and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        os.environ.setdefault("WANDB_RUN_NAME", args.wandb_run_name)
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    print(f"model={args.model_id}")
    print(f"dataset={args.dataset_id}")
    print(f"train_mode={args.train_mode}")
    print(f"model_context_length={args.model_context_length}")
    print(f"train_max_length={args.max_length}")
    print(f"wandb={'enabled' if use_wandb else 'disabled'}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(args.model_context_length, args.max_length)

    encoded_examples, streamed_examples = _load_encoded_examples(args, tokenizer)
    plan = build_training_plan(
        num_unique_examples=len(encoded_examples),
        global_batch_size=args.global_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        world_size=args.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
    )
    print(f"training_plan={json.dumps(asdict(plan), indent=2)}")

    class ShortSweHeroDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return plan.items_per_epoch

        def __getitem__(self, idx: int) -> dict[str, list[int]]:
            return encoded_examples[idx % len(encoded_examples)]

    def collate(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_batch_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full(
            (len(features), max_batch_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(features), max_batch_len), dtype=torch.long)
        labels = torch.full((len(features), max_batch_len), -100, dtype=torch.long)

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(
                feature["input_ids"], dtype=torch.long
            )
            attention_mask[row, :length] = 1
            labels[row, :length] = torch.tensor(feature["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    common_summary = {
        "model": args.model_id,
        "dataset": args.dataset_id,
        "dataset_revision": _dataset_revision_info(
            args.dataset_id, args.dataset_revision
        ),
        "paper_alignment": _paper_alignment(args),
        "tokenizer": _tokenizer_metadata(tokenizer),
        "num_unique_examples": len(encoded_examples),
        "streamed_examples_scanned": streamed_examples,
        "training_plan": asdict(plan),
        "max_length": args.max_length,
        "model_context_length": args.model_context_length,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "lr_scheduler": "cosine_with_min_lr",
        "loss_implementation": "chunked_causal_lm",
        "logit_chunk_size": args.logit_chunk_size,
        "git": {
            "branch": _run_git(["branch", "--show-current"])
            or os.environ.get("SOURCE_GIT_BRANCH"),
            "commit": _run_git(["rev-parse", "HEAD"])
            or os.environ.get("SOURCE_GIT_COMMIT"),
            "status_short": _run_git(["status", "--short"])
            or os.environ.get("SOURCE_GIT_STATUS"),
        },
        "software_versions": _package_versions(),
        "wandb": {
            "enabled": use_wandb,
            "entity": os.environ.get("WANDB_ENTITY") if use_wandb else None,
            "project": os.environ.get("WANDB_PROJECT") if use_wandb else None,
            "run_name": os.environ.get("WANDB_RUN_NAME") if use_wandb else None,
        },
    }

    if args.dry_run_tokenize_only:
        summary = {
            **common_summary,
            "dry_run_tokenize_only": True,
            "losses": [],
            "train_runtime": None,
        }
        (args.out_dir / "train_result.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    smoke.maybe_enable_yarn(
        config,
        max_length=args.model_context_length,
        enable_yarn=not args.disable_yarn,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model.to("cuda")
    if not args.no_gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    model = _apply_lora(model, args)
    if not args.no_gradient_checkpointing and hasattr(
        model, "enable_input_require_grads"
    ):
        model.enable_input_require_grads()

    optimizer = _build_optimizer(torch, model, args)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        smoke.build_cosine_with_min_lr_lambda(
            plan.total_optimizer_steps,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_ratio=args.warmup_ratio,
        ),
    )

    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "trainer"),
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=plan.batch.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=["wandb"] if use_wandb else [],
        run_name=args.wandb_run_name if use_wandb else None,
        do_train=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )

    class ChunkedCausalLMTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs: bool = False,
            num_items_in_batch=None,
        ):
            labels = inputs.pop("labels")
            causal_lm = _base_causal_lm(model)
            if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
                outputs = model(**inputs, labels=labels, use_cache=False)
                return (outputs.loss, outputs) if return_outputs else outputs.loss

            outputs = causal_lm.model(**inputs, use_cache=False)
            hidden_states = outputs.last_hidden_state
            shifted_hidden_states = hidden_states[:, :-1, :].reshape(
                -1, hidden_states.shape[-1]
            )
            shifted_labels = labels[:, 1:].reshape(-1)

            loss_sum = shifted_hidden_states.new_zeros(())
            token_count = shifted_hidden_states.new_zeros(())
            for start in range(
                0, shifted_hidden_states.shape[0], args.logit_chunk_size
            ):
                end = min(
                    start + args.logit_chunk_size,
                    shifted_hidden_states.shape[0],
                )
                label_chunk = shifted_labels[start:end]
                valid_tokens = label_chunk.ne(-100).sum()
                if valid_tokens.item() == 0:
                    continue

                logits = causal_lm.lm_head(shifted_hidden_states[start:end])
                loss_sum = loss_sum + torch.nn.functional.cross_entropy(
                    logits.float(),
                    label_chunk,
                    ignore_index=-100,
                    reduction="sum",
                )
                token_count = token_count + valid_tokens

            loss = loss_sum / token_count.clamp_min(1)
            if return_outputs:
                return loss, outputs
            return loss

    trainer = ChunkedCausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=ShortSweHeroDataset(),
        data_collator=collate,
        optimizers=(optimizer, lr_scheduler),
    )
    started_at = time.time()
    result = trainer.train()
    wall_time = time.time() - started_at

    if args.save_final:
        save_dir = args.out_dir / ("adapter" if args.train_mode == "lora" else "model")
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(args.out_dir / "tokenizer")

    losses = [
        entry["loss"]
        for entry in trainer.state.log_history
        if "loss" in entry and isinstance(entry["loss"], float)
    ]
    cuda_summary = None
    if torch.cuda.is_available():
        cuda_summary = {
            "device_name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }

    summary = {
        **common_summary,
        "dry_run_tokenize_only": False,
        "train_mode": args.train_mode,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [
                module.strip()
                for module in args.lora_target_modules.split(",")
                if module.strip()
            ],
        }
        if args.train_mode == "lora"
        else None,
        **_parameter_counts(model),
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "attn_implementation": args.attn_implementation,
        "losses": losses,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "train_runtime": result.metrics.get("train_runtime"),
        "wall_time_seconds": wall_time,
        "cuda": cuda_summary,
        "save_final": args.save_final,
    }
    (args.out_dir / "train_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
