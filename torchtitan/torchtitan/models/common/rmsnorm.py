# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.protocols.module import Module


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm.

    Uses diamond inheritance (nn.RMSNorm + Module) so that:
    - The module hierarchy stays flat (no extra wrapper layer).
    - nn.RMSNorm state_dict semantics are reused as-is.
    - The Module protocol is satisfied and ``build()`` is inherited from
      ``Configurable.Config``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dtype = input.dtype
        hidden_states = input.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        if self.weight is not None:
            # FSDP2 may release the all-gathered parameter storage before
            # MulBackward reads saved tensors. The clone is differentiable,
            # keeps a tiny local copy alive for backward, and still routes
            # gradients to the original RMSNorm weight.
            hidden_states = hidden_states * self.weight.clone()
        return hidden_states
