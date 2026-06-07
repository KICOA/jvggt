# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jittor as jt
from jittor import nn
def position_grid_to_embed(pos_grid: jt.Var, embed_dim: int, omega_0: float = 100) -> jt.Var:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = jt.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: jt.Var, omega_0: float = 100) -> jt.Var:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = jt.arange(embed_dim // 2, dtype="float64")
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega

    pos = pos.reshape(-1)
    out = jt.einsum("m,d->md", pos, omega)

    emb_sin = jt.sin(out)  # (M, D/2)
    emb_cos = jt.cos(out)  # (M, D/2)

    emb = jt.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# Inspired by https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype=None, device=None
) -> jt.Var:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (jt.dtype, optional): Data type of the resulting tensor.
        device (jt.device, optional): Device on which the tensor is created.

    Returns:
        jt.Var: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Jittor linspace(start, end, steps) has no dtype= kwarg (PyTorch does).
    x_coords = jt.linspace(left_x, right_x, steps=width)
    y_coords = jt.linspace(top_y, bottom_y, steps=height)
    if dtype is not None:
        x_coords = x_coords.astype(dtype)
        y_coords = y_coords.astype(dtype)

    uu, vv = jt.meshgrid(x_coords, y_coords)
    uu, vv = uu.transpose(), vv.transpose()
    uv_grid = jt.stack((uu, vv), dim=-1)

    return uv_grid
