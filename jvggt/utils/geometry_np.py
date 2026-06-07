# NumPy-only geometry helpers for jvggt viser (no torch).

from __future__ import annotations

import numpy as np

_FOCAL_EPS = 1e-3
_DEPTH_MAX = 1e4


def _safe_focal_lengths(intrinsic: np.ndarray) -> tuple[float, float, float, float]:
    fu = float(intrinsic[0, 0])
    fv = float(intrinsic[1, 1])
    cu = float(intrinsic[0, 2])
    cv = float(intrinsic[1, 2])
    if not np.isfinite(fu) or abs(fu) < _FOCAL_EPS:
        fu = _FOCAL_EPS
    if not np.isfinite(fv) or abs(fv) < _FOCAL_EPS:
        fv = _FOCAL_EPS
    return fu, fv, cu, cv


def closed_form_inverse_se3(se3: np.ndarray, R=None, T=None) -> np.ndarray:
    """Inverse of batched SE3 matrices (N, 3, 4) or (N, 4, 4)."""
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be of shape (N,4,4) or (N,3,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    R_transposed = np.transpose(R, (0, 2, 1))
    top_right = -np.matmul(R_transposed, T)
    inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth_map = np.asarray(depth_map, dtype=np.float64)
    depth_map = np.where(np.isfinite(depth_map), depth_map, 0.0)
    depth_map = np.clip(depth_map, 0.0, _DEPTH_MAX)

    H, W = depth_map.shape
    fu, fv, cu, cv = _safe_focal_lengths(intrinsic)

    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1)
    return np.nan_to_num(cam_coords, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps: float = 1e-8,
):
    point_mask = depth_map > eps
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3].astype(np.float64)
    t_cam_to_world = cam_to_world_extrinsic[:3, 3].astype(np.float64)
    world_coords_points = cam_coords_points.astype(np.float64) @ R_cam_to_world.T + t_cam_to_world
    world_coords_points = np.nan_to_num(world_coords_points, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return world_coords_points, cam_coords_points, point_mask


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
) -> np.ndarray:
    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1),
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points_list.append(cur_world_points)
    return np.stack(world_points_list, axis=0)
