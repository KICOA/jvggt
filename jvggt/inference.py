# Unified Jittor inference helpers for demo scripts.

from __future__ import annotations

from typing import Any, Dict

import jittor as jt
import numpy as np

from jvggt.models.vggt import VGGT
from jvggt.utils.pose_enc import pose_encoding_to_extri_intri
from jvggt.weight_loader import load_vggt_pretrained


def setup_jittor(use_cuda: bool = True) -> None:
    jt.flags.use_cuda = bool(use_cuda)


def create_vggt_model(weights: str | None = None) -> VGGT:
    """Create VGGT and load weights. Default: repo-root ``model.pt`` if present."""
    setup_jittor(use_cuda=jt.has_cuda)
    model = VGGT()
    load_vggt_pretrained(model, weights)
    return model


def run_vggt_forward(
    model: VGGT,
    images: jt.Var,
    frames_chunk_size: int = 4,
    run_depth: bool = True,
    run_point: bool = True,
) -> Dict[str, Any]:
    """
    Run VGGT forward (same entry as ``vggt.models.VGGT.forward``).

    For 8GB GPUs use ``run_depth=False`` (``--skip_depth`` in demo) and ``frames_chunk_size=1``.    """
    with jt.no_grad():
        predictions = model(
            images,
            frames_chunk_size=frames_chunk_size,
            run_depth=run_depth,
            run_point=run_point,
        )
    jt.gc()
    return predictions


def _sanitize_world_points_numpy(world_points: np.ndarray, max_abs: float = 10000.0) -> np.ndarray:
    """Zero out non-finite / overflowed coordinates from unstable heads."""
    out = np.array(world_points, copy=True)
    bad = ~np.isfinite(out) | (np.abs(out) > max_abs)
    if np.any(bad):
        out[bad] = 0.0
    return out


def _default_fov_rad(image_hw: tuple[int, int]) -> float:
    H, W = image_hw
    focal = max(W, H) * 0.9
    return float(2.0 * np.arctan((max(H, W) / 2.0) / focal))


def _sanitize_pose_enc_numpy(pose_enc: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Normalize quaternion and repair degenerate FoV before intrinsics decode."""
    out = np.array(pose_enc, copy=True, dtype=np.float32)
    default_fov = _default_fov_rad(image_hw)

    q = np.nan_to_num(out[..., 3:7], nan=0.0, posinf=0.0, neginf=0.0)
    q = np.clip(q, -1e4, 1e4)
    qn = np.linalg.norm(q, axis=-1, keepdims=True)
    identity = np.zeros_like(q)
    identity[..., 3] = 1.0
    out[..., 3:7] = np.where(qn < 1e-6, identity, q / np.maximum(qn, 1e-8))

    fov = np.nan_to_num(out[..., 7:9], nan=default_fov, posinf=default_fov, neginf=default_fov)
    fov = np.clip(fov, 0.4, 2.0)
    out[..., 7:9] = fov
    return out


def _sanitize_intrinsics_numpy(intrinsic: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    """Ensure fx/fy are finite and in a plausible pixel range after pose_enc decode."""
    H, W = image_hw
    default_fx = float(W) * 0.9
    default_fy = float(H) * 0.9
    min_f = float(min(W, H)) * 0.2
    max_f = float(max(W, H)) * 3.0
    out = np.array(intrinsic, copy=True)
    if out.ndim == 2:
        mats = out[None]
    else:
        mats = out.reshape(-1, 3, 3)
    for K in mats:
        if not np.isfinite(K[0, 0]) or K[0, 0] < min_f or K[0, 0] > max_f:
            K[0, 0] = default_fx
        if not np.isfinite(K[1, 1]) or K[1, 1] < min_f or K[1, 1] > max_f:
            K[1, 1] = default_fy
        if not np.isfinite(K[0, 2]):
            K[0, 2] = W / 2.0
        if not np.isfinite(K[1, 2]):
            K[1, 2] = H / 2.0
        K[2, 2] = 1.0
    return out.reshape(intrinsic.shape)


def predictions_to_numpy(
    predictions: Dict[str, Any],
    image_hw: tuple,
    squeeze_batch: bool = True,
) -> Dict[str, np.ndarray]:
    """Convert Jittor predictions to numpy and add extrinsic/intrinsic."""
    predictions = dict(predictions)
    pose_enc = predictions["pose_enc"]
    if isinstance(pose_enc, jt.Var):
        jt.sync_all()
        pose_enc_np = _sanitize_pose_enc_numpy(pose_enc.numpy(), image_hw)
    else:
        pose_enc_np = _sanitize_pose_enc_numpy(np.asarray(pose_enc), image_hw)
    predictions["pose_enc"] = pose_enc_np
    extrinsic, intrinsic = pose_encoding_to_extri_intri(jt.array(pose_enc_np), image_hw)
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    priority = (
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
    )
    keys = [k for k in priority if k in predictions]
    keys.extend(k for k in predictions if k not in keys and not k.endswith("_list"))

    out = {}
    for key in keys:
        val = predictions[key]
        if isinstance(val, jt.Var):
            jt.sync_all()
            arr = np.ascontiguousarray(val.numpy(), dtype=np.float32)
        elif isinstance(val, np.ndarray):
            arr = np.ascontiguousarray(val, dtype=np.float32)
        else:
            out[key] = val
            continue
        if squeeze_batch and arr.ndim > 0 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        out[key] = arr

    if "intrinsic" in out and isinstance(out["intrinsic"], np.ndarray):
        out["intrinsic"] = _sanitize_intrinsics_numpy(out["intrinsic"], image_hw)
    if "world_points" in out and isinstance(out["world_points"], np.ndarray):
        out["world_points"] = _sanitize_world_points_numpy(out["world_points"])
    return out


def predictions_to_numpy_like_vggt_demo(
    predictions: Dict[str, Any],
    image_hw: tuple,
) -> Dict[str, np.ndarray]:
    """Same postprocess as ``demo_viser.py``: pose_enc -> cameras, squeeze batch."""
    return predictions_to_numpy(predictions, image_hw=image_hw, squeeze_batch=True)


def run_vggt_colmap(
    model: VGGT,
    images: jt.Var,
    resolution: int = 518,
) -> tuple:
    """Colmap demo path: aggregator + camera_head + depth_head."""
    from jvggt.ops import F

    assert len(images.shape) == 4 and images.shape[1] == 3
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with jt.no_grad():
        images = images.unsqueeze(0)
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    return (
        extrinsic.squeeze(0).numpy(),
        intrinsic.squeeze(0).numpy(),
        depth_map.squeeze(0).numpy(),
        depth_conf.squeeze(0).numpy(),
    )
