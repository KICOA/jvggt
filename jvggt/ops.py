# Copyright (c) Meta Platforms, Inc. and affiliates.
# Jittor compatibility helpers for VGGT inference.

from __future__ import annotations

import jittor as jt
from jittor import nn


class _Functional:
    """Subset of torch.nn.functional implemented with Jittor."""

    @staticmethod
    def silu(x):
        return x * nn.sigmoid(x)

    @staticmethod
    def relu(x):
        return nn.relu(x)

    @staticmethod
    def embedding(input, weight):
        # Same order as PyTorch F.embedding(input, weight).
        return nn.embedding(input, weight)

    @staticmethod
    def scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_training=False):
        scale = q.shape[-1] ** -0.5
        attn = jt.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jt.nn.softmax(attn, dim=-1)
        if dropout_p > 0.0 and is_training:
            attn = nn.dropout(attn, p=dropout_p, is_train=True)
        return jt.matmul(attn, v)

    @staticmethod
    def interpolate(
        x,
        size=None,
        scale_factor=None,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    ):
        if size is None and scale_factor is not None:
            if isinstance(scale_factor, (tuple, list)):
                size = (int(x.shape[-2] * scale_factor[0]), int(x.shape[-1] * scale_factor[1]))
            else:
                size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
        if antialias and mode == "bicubic":
            pass
        return nn.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    @staticmethod
    def pad(x, pad, mode="constant", value=0.0):
        return nn.pad(x, pad, mode=mode, value=value)

    @staticmethod
    def one_hot(tensor, num_classes):
        shape = list(tensor.shape) + [num_classes]
        out = jt.zeros(shape, dtype="float32")
        idx = tensor.unsqueeze(-1)
        out.scatter_(-1, idx, 1.0)
        return out


F = _Functional()


class ReLU(nn.Module):
    """Drop-in for nn.ReLU; Jittor relu has no inplace kwarg."""

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()

    def execute(self, x):
        return nn.relu(x)


def sign(x):
    return jt.where(x > 0, 1.0, jt.where(x < 0, -1.0, 0.0))


def nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0):
    """numpy.nan_to_num compatible helper (Jittor has no jt.nan_to_num)."""
    finite = jt.isfinite(x)
    nan_v = jt.array(nan, dtype="float32")
    pos_v = jt.array(posinf, dtype="float32")
    neg_v = jt.array(neginf, dtype="float32")
    out = jt.where(finite, x, nan_v)
    # Jittor: no bitwise ~ on Var; use (1 - finite) and * for logical ops.
    not_finite = 1 - finite
    not_nan = x == x
    is_pos_inf = not_finite * not_nan * (x > 0)
    is_neg_inf = not_finite * not_nan * (x <= 0)
    out = jt.where(is_pos_inf, pos_v, out)
    out = jt.where(is_neg_inf, neg_v, out)
    return out


def expm1(x):
    return x.exp() - 1.0


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Approximate timm trunc_normal_ init using numpy."""
    import numpy as np

    size = tensor.shape
    data = np.random.randn(*size).astype("float32") * std + mean
    data = np.clip(data, a * std + mean, b * std + mean)
    if hasattr(tensor, "assign"):
        tensor.assign(data)
    else:
        return jt.array(data)


def normal_(param, mean=0.0, std=1.0) -> None:
    """In-place normal init (PyTorch nn.init.normal_ compatible)."""
    if hasattr(param, "assign"):
        param.assign(jt.init.gauss(param.shape, "float32", mean, std))
    else:
        raise TypeError(f"Cannot initialize {type(param)}")


def zeros_(param) -> None:
    """In-place zero init (PyTorch nn.init.zeros_ compatible)."""
    if hasattr(param, "assign"):
        param.assign(jt.zeros(param.shape, "float32"))
    else:
        raise TypeError(f"Cannot initialize {type(param)}")
