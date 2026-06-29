# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import jittor as jt
from jittor import nn

from jvggt.models.aggregator import Aggregator
from jvggt.heads.camera_head import CameraHead
from jvggt.heads.dpt_head import DPTHead


def _materialize_var(x: jt.Var) -> jt.Var:
    """Copy GPU tensor to CPU-backed Var to free VRAM."""
    return jt.array(x.numpy())


def _materialize_token_list(tokens_list: list) -> list:
    """Move cached aggregator outputs to CPU to free VRAM before heads."""
    out = []
    for t in tokens_list:
        if t is None:
            out.append(None)
        elif isinstance(t, jt.Var):
            out.append(_materialize_var(t))
        else:
            out.append(t)
    return out


class VGGT(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=False,
    ):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
            if enable_depth
            else None
        )

    def execute(
        self,
        images: jt.Var,
        query_points: jt.Var = None,
        frames_chunk_size: int = 4,
        run_depth: bool = True,
        run_point: bool = True,
    ):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        aggregated_tokens_list = _materialize_token_list(aggregated_tokens_list)
        jt.sync_all()
        jt.gc()

        predictions = {}

        with jt.no_grad():
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = _materialize_var(pose_enc_list[-1])
                jt.sync_all()
                jt.gc()

            if run_depth and self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=frames_chunk_size,
                )
                predictions["depth"] = _materialize_var(depth)
                predictions["depth_conf"] = _materialize_var(depth_conf)
                jt.gc()

            if run_point and self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    frames_chunk_size=frames_chunk_size,
                )
                predictions["world_points"] = _materialize_var(pts3d)
                predictions["world_points_conf"] = _materialize_var(pts3d_conf)
                jt.gc()

        predictions["images"] = images
        return predictions

    forward = execute
