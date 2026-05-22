# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.qwen2_5 import model_registry
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.trainer import Trainer

from .dataloader import SweHeroDataLoader


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, repr(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _checkpoint_exclude_from_loading() -> list[str]:
    if _env_bool("SWEHERO_LOAD_DATALOADER_STATE", False):
        return []
    return ["dataloader"]


def _fp8_converters(
    *,
    enabled: bool,
    compile_enabled: bool,
    recipe_name: str,
) -> list[ModelConfigConverter.Config] | None:
    if not enabled:
        return None

    from torchtitan.components.quantization.float8 import Float8LinearConverter

    return [
        Float8LinearConverter.Config(
            recipe_name=recipe_name,  # type: ignore[arg-type]
            model_compile_enabled=compile_enabled,
            filter_fqns=["lm_head", "auto_filter_small_kn"],
        )
    ]


def qwen25_coder7b_direct_to_hero() -> Trainer.Config:
    seq_len = _env_int("SWEHERO_BUCKET_SEQ_LEN", 8192)
    cp_degree = _env_int("SWEHERO_BUCKET_CP", 1)
    cumulative_steps = _env_int("SWEHERO_CUMULATIVE_STEPS", 1)
    total_steps = _env_int("SWEHERO_TOTAL_STEPS", cumulative_steps)
    warmup_steps = _env_int("SWEHERO_WARMUP_STEPS", 0)
    learning_rate = _env_float("SWEHERO_LEARNING_RATE", 1e-5)
    min_learning_rate = _env_float("SWEHERO_MIN_LEARNING_RATE", 1e-8)
    compile_enabled = _env_bool("SWEHERO_COMPILE", True)
    enable_wandb = _env_bool("SWEHERO_ENABLE_WANDB", False)
    is_final_stage = cumulative_steps >= total_steps

    converters = _fp8_converters(
        enabled=_env_bool("SWEHERO_ENABLE_FP8", True),
        compile_enabled=compile_enabled,
        recipe_name=_env("SWEHERO_FP8_RECIPE", "rowwise"),
    )

    return Trainer.Config(
        loss=ChunkedCELoss.Config(
            num_chunks=_env_int("SWEHERO_CHUNKED_CE_CHUNKS", 8),
        ),
        hf_assets_path=_env(
            "SWEHERO_HF_ASSETS_PATH",
            "/workspace/assets/hf/Qwen2.5-Coder-7B-Instruct",
        ),
        dump_folder=_env(
            "SWEHERO_TORCHTITAN_DUMP_FOLDER",
            "/workspace/qwen25-coder7b-swehero-torchtitan/torchtitan",
        ),
        model_spec=model_registry(
            "coder7b",
            attn_backend=_env("SWEHERO_ATTENTION_BACKEND", "sdpa"),
            converters=converters,
        ),
        dataloader=SweHeroDataLoader.Config(
            dataset_path=_env("SWEHERO_BUCKET_FILE", ""),
            pad_token_id=_env_int("SWEHERO_PAD_TOKEN_ID", 151_643),
            seed=_env_int("SWEHERO_SEED", 17),
            shuffle=True,
            infinite=True,
            pin_memory=True,
        ),
        optimizer=OptimizersContainer.Config(
            name="AdamW",
            lr=learning_rate,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=_env_float("SWEHERO_WEIGHT_DECAY", 0.0),
            implementation=_env(  # type: ignore[arg-type]
                "SWEHERO_OPTIMIZER_IMPL", "foreach"
            ),
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_type="cosine",
            min_lr_factor=min_learning_rate / learning_rate,
        ),
        training=TrainingConfig(
            local_batch_size=_env_int("SWEHERO_LOCAL_BATCH_SIZE", 1),
            global_batch_size=_env_int("SWEHERO_GLOBAL_BATCH_SIZE", 32),
            seq_len=seq_len,
            steps=cumulative_steps,
            max_norm=_env_float("SWEHERO_MAX_GRAD_NORM", 1.0),
            dtype=_env("SWEHERO_TRAINING_DTYPE", "float32"),  # type: ignore[arg-type]
            mixed_precision_param=_env(  # type: ignore[arg-type]
                "SWEHERO_MP_PARAM_DTYPE", "bfloat16"
            ),
            mixed_precision_reduce=_env(  # type: ignore[arg-type]
                "SWEHERO_MP_REDUCE_DTYPE", "bfloat16"
            ),
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward=_env(  # type: ignore[arg-type]
                "SWEHERO_FSDP_RESHARD_AFTER_FORWARD", "never"
            ),
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=cp_degree,
            context_parallel_load_balancer="headtail",
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            interval=_env_int("SWEHERO_CHECKPOINT_INTERVAL", 25),
            final_model_export_folder=_env(
                "SWEHERO_FINAL_EXPORT_FOLDER",
                "final_export",
            ),
            initial_load_in_hf=True,
            initial_load_model_only=True,
            last_save_model_only=is_final_stage,
            last_save_in_hf=is_final_stage,
            export_dtype="bfloat16",
            async_mode=_env("SWEHERO_CHECKPOINT_ASYNC_MODE", "async"),  # type: ignore[arg-type]
            exclude_from_loading=_checkpoint_exclude_from_loading(),
            keep_latest_k=3,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode=_env("SWEHERO_AC_MODE", "full"),  # type: ignore[arg-type]
        ),
        compile=CompileConfig(
            enable=compile_enabled,
            components=["model", "loss"],
        ),
        metrics=MetricsProcessor.Config(
            log_freq=_env_int("SWEHERO_METRICS_LOG_FREQ", 1),
            enable_wandb=enable_wandb,
        ),
        debug=DebugConfig(
            seed=_env_int("SWEHERO_SEED", 17),
            detect_anomaly=_env_bool("SWEHERO_DETECT_ANOMALY", False),
            print_config=True,
            save_config_file="config.json",
        ),
    )
