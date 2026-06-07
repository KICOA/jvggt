#!/usr/bin/env python3
"""Sanity-check weight loading and one forward pass (world_points stats)."""
from __future__ import annotations

import argparse
import glob
import os

from jvggt.jittor_env import configure_jittor_compiler

configure_jittor_compiler()

import jittor as jt
import numpy as np

from jvggt.inference import create_vggt_model, predictions_to_numpy, run_vggt_forward
from jvggt.utils.load_fn import load_and_preprocess_images
from jvggt.weight_loader import load_state_dict, resolve_weights_path, assign_state_dict
from jvggt.models.vggt import VGGT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--image_folder", default="examples/kitchen/images/")
    args = parser.parse_args()

    path = resolve_weights_path(args.weights)
    sd = load_state_dict(path)
    model = VGGT()
    assign_state_dict(model, sd)
    model.eval()

    w = model.state_dict()["point_head.scratch.output_conv2.2.weight"]
    w_np = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
    print(f"point_head output conv weight: mean={w_np.mean():.6f} std={w_np.std():.6f}")

    ckpt_key = "point_head.scratch.output_conv2.2.weight"
    if ckpt_key in sd:
        c = sd[ckpt_key]
        print(f"checkpoint same key: mean={c.mean():.6f} std={c.std():.6f}")

    images_paths = sorted(glob.glob(os.path.join(args.image_folder, "*")))[:1]
    images = load_and_preprocess_images(images_paths)
    print(f"images {images.shape}")

    with jt.no_grad():
        preds = run_vggt_forward(model, images, frames_chunk_size=1, run_depth=True, run_point=True)
    out = predictions_to_numpy(preds, image_hw=images.shape[-2:])

    wp = out["world_points"]
    wc = out["world_points_conf"]
    print(f"world_points shape={wp.shape} max|.|={np.abs(wp).max():.6f} min={wp.min():.6f} max={wp.max():.6f}")
    print(f"world_points_conf min={wc.min():.6f} max={wc.max():.6f} mean={wc.mean():.6f}")
    pe = out["pose_enc"]
    print(f"pose_enc: {pe}")


if __name__ == "__main__":
    main()
