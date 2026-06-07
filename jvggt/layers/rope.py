# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# Implementation of 2D Rotary Position Embeddings (RoPE).

import jittor as jt
from jittor import nn
from typing import Dict, Tuple

from jvggt.ops import F


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], jt.Var] = {}

    def __call__(self, batch_size: int, height: int, width: int, device=None) -> jt.Var:
        if (height, width) not in self.position_cache:
            y_coords = jt.arange(height)
            x_coords = jt.arange(width)
            yy, xx = jt.meshgrid(y_coords, x_coords)
            positions = jt.stack([yy.reshape(-1), xx.reshape(-1)], dim=1)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[jt.Var, jt.Var]] = {}

    def _compute_frequency_components(self, dim: int, seq_len: int, dtype) -> Tuple[jt.Var, jt.Var]:
        cache_key = (dim, seq_len, str(dtype))
        if cache_key not in self.frequency_cache:
            exponents = jt.arange(0, dim, 2).astype("float32") / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)
            positions = jt.arange(seq_len).astype(inv_freq.dtype)
            angles = jt.einsum("i,j->ij", positions, inv_freq)
            angles = angles.astype(dtype)
            angles = jt.cat((angles, angles), dim=-1)
            cos_components = angles.cos().astype(dtype)
            sin_components = angles.sin().astype(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: jt.Var) -> jt.Var:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return jt.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: jt.Var, positions: jt.Var, cos_comp: jt.Var, sin_comp: jt.Var
    ) -> jt.Var:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def execute(self, tokens: jt.Var, positions: jt.Var) -> jt.Var:
        assert tokens.shape[-1] % 2 == 0, "Feature dimension must be even"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)"

        feature_dim = tokens.shape[-1] // 2
        max_position = int(positions.max().item()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.dtype)

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return jt.cat((vertical_features, horizontal_features), dim=-1)
