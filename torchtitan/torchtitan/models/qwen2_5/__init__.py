# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan model registry for Qwen2.5-Coder dense decoder models.

Qwen2.5-Coder is architecturally close enough to the vendored Qwen3 dense
decoder that we reuse the Qwen3 module and parallelization code, but build the
exact Qwen2.5-Coder-7B dimensions and checkpoint mapping here.
"""

from collections.abc import Callable
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RoPE, TransformerBlock
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.qwen3.model import Qwen3Model, Qwen3TransformerBlock
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .state_dict_adapter import Qwen25StateDictAdapter

__all__ = [
    "QWEN25_CODER_7B_CONTEXT",
    "model_registry",
    "qwen2_5_configs",
]


QWEN25_CODER_7B_CONTEXT = 131_072
QWEN25_NATIVE_CONTEXT = 32_768
_EPS = 1e-6

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_SKIP_INIT = {"weight": skip_param_init}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    scale = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=scale, a=-3 * scale, b=3 * scale),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _qwen25_norm(dim: int) -> RMSNorm.Config:
    return RMSNorm.Config(normalized_shape=dim, eps=_EPS, param_init=_NORM_INIT)


def _build_qwen25_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
    attn_backend: str,
) -> list[TransformerBlock.Config]:
    inner_attention, mask_type = get_attention_config(attn_backend)
    layers = []
    for layer_id in range(n_layers):
        attention = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_depth_init(layer_id),
            inner_attention=inner_attention,
            mask_type=mask_type,
            rope_backend="cos_sin",
            qk_norm=None,
        )
        assert isinstance(attention.qkv_linear, QKVLinear.Config)
        attention.qkv_linear.wq.bias = True
        attention.qkv_linear.wkv.bias = True
        layers.append(
            Qwen3TransformerBlock.Config(
                attention_norm=_qwen25_norm(dim),
                ffn_norm=_qwen25_norm(dim),
                attention=attention,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )
        )
    return layers


def _coder7b(attn_backend: str) -> Qwen3Model.Config:
    dim = 3_584
    head_dim = 128
    vocab_size = 152_064
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen25_norm(dim),
        enable_weight_tying=False,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_SKIP_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=QWEN25_CODER_7B_CONTEXT,
            theta=1_000_000.0,
            backend="cos_sin",
            scaling="yarn",
            rope_factor=QWEN25_CODER_7B_CONTEXT / QWEN25_NATIVE_CONTEXT,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=QWEN25_NATIVE_CONTEXT,
        ),
        layers=_build_qwen25_layers(
            n_layers=28,
            dim=dim,
            n_heads=28,
            n_kv_heads=4,
            head_dim=head_dim,
            hidden_dim=18_944,
            attn_backend=attn_backend,
        ),
    )


qwen2_5_configs = {
    "coder7b": _coder7b,
}


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    config = qwen2_5_configs[flavor](attn_backend=attn_backend)
    if converters is not None:
        validate_converter_order(converters)
        for converter in converters:
            converter.build().convert(config)
    return ModelSpec(
        name="qwen2_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen25StateDictAdapter,
    )
