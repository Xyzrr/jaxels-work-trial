# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen2.5 Hugging Face <-> TorchTitan state-dict mapping."""

from __future__ import annotations

import re
from typing import Any

from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.protocols.state_dict_adapter import StateDictAdapter


class Qwen25StateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: Qwen3Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.from_hf_map: dict[str, str | None] = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.qkv_linear.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.qkv_linear.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.qkv_linear.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict: dict[str, Any] = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key)
                if layer_num is None:
                    continue
                hf_state_dict[to_hf_map[abstract_key].format(layer_num.group(0))] = value
                continue

            if key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                new_abstract_key = self.from_hf_map[abstract_key]
                if new_abstract_key is None:
                    continue
                layer_num = re.search(r"\d+", key)
                if layer_num is None:
                    continue
                state_dict[new_abstract_key.format(layer_num.group(0))] = value
                continue

            if key not in self.from_hf_map:
                continue
            new_key = self.from_hf_map[key]
            if new_key is not None:
                state_dict[new_key] = value

        return state_dict
