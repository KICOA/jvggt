# Copyright (c) Meta Platforms, Inc. and affiliates.
# Jittor inference demo for VGGT (jvggt). Training code is unchanged.

"""
Run VGGT inference with the Jittor backend (jvggt) and visualize with viser.

Aligned with ``demo_viser.py`` (PyTorch vggt):
  - sorted glob under ``--image_folder``
  - ``load_and_preprocess_images`` -> ``model(images)`` -> pose_enc -> extrinsic/intrinsic
  - viser on numpy predictions (batch dim squeezed)

8GB GPU: use ``--low_vram`` (1 view) or ``--skip_depth`` to save VRAM.
15GB GPU (2 views): use ``--twoview`` or ``--max_images 2 --skip_point``.
Default runs depth + world_points (same heads as PyTorch ``demo_viser.py``).

Example:
    python demo_jvggt.py --image_folder examples/kitchen/images/
    python demo_jvggt.py --low_vram
    python demo_jvggt.py --twoview --save_preview outputs/preview
    python demo_jvggt.py --max_images 0 --vggt_compat   # full vggt logic, high VRAM
"""

from __future__ import annotations

import argparse
import os
import time

# Must run before any ``import jittor`` (Windows: use system cl.exe, not bundled msvc.zip).
from jvggt.jittor_env import configure_jittor_compiler

configure_jittor_compiler()

import numpy as np

from jvggt.inference import (
    create_vggt_model,
    predictions_to_numpy_like_vggt_demo,
    run_vggt_forward,
    setup_jittor,
)
from jvggt.utils.load_fn import list_images_from_folder, load_and_preprocess_images, select_image_paths
from jvggt.weight_loader import resolve_weights_path


def print_prediction_summary(predictions: dict) -> None:
    print("Inference outputs (numpy):")
    for key, val in sorted(predictions.items()):
        if key.endswith("_list"):
            continue
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {type(val).__name__}")

    if "intrinsic" in predictions and isinstance(predictions["intrinsic"], np.ndarray):
        K = predictions["intrinsic"]
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        print(f"  camera fx/fy: fx={np.asarray(fx).round(2)}, fy={np.asarray(fy).round(2)}")
        if np.any(fx < 50) or np.any(fy < 50) or np.any(fx > 5000) or np.any(fy > 5000):
            print("  WARNING: intrinsics look abnormal (sanitized defaults may be in use)")
    if "pose_enc" in predictions and isinstance(predictions["pose_enc"], np.ndarray):
        fov = predictions["pose_enc"][..., 7:9]
        print(f"  pose_enc fov (h,w): {np.asarray(fov).round(6)}")
    if "depth" in predictions and isinstance(predictions["depth"], np.ndarray):
        d = predictions["depth"]
        print(f"  depth: shape={d.shape}, min={float(np.nanmin(d)):.4f}, max={float(np.nanmax(d)):.4f}")
    if "world_points" in predictions and isinstance(predictions["world_points"], np.ndarray):
        wp = predictions["world_points"]
        finite = np.isfinite(wp).all(axis=-1).mean() * 100.0
        wp_max = float(np.max(np.abs(wp[np.isfinite(wp)]))) if np.isfinite(wp).any() else 0.0
        print(f"  world_points finite: {finite:.1f}% of pixels, max|coord|={wp_max:.4f}")
        if wp_max < 1e-3:
            print("  WARNING: world_points are near zero — check weight loading (tools/check_forward.py)")


def save_prediction_previews(predictions: dict, out_dir: str) -> None:
    """Save per-frame PNGs: input | depth | point confidence (Kaggle / headless friendly)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    images = predictions.get("images")
    if images is None:
        raise ValueError("predictions['images'] missing — cannot save previews")

    n_frames = int(images.shape[0])
    for i in range(n_frames):
        img = np.transpose(images[i], (1, 2, 0))
        img = np.clip(np.nan_to_num(img, nan=0.0), 0.0, 1.0)

        panels = [("Input", img, None)]
        if "depth" in predictions:
            panels.append(("Depth", predictions["depth"][i].squeeze(), "turbo"))
        if "depth_conf" in predictions:
            panels.append(("Depth conf", predictions["depth_conf"][i].squeeze(), "viridis"))
        if "world_points_conf" in predictions:
            panels.append(("Point conf", predictions["world_points_conf"][i].squeeze(), "viridis"))

        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, data, cmap) in zip(axes, panels):
            if data.ndim == 3:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"frame_{i:02d}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    npz_path = os.path.join(out_dir, "predictions.npz")
    np.savez(npz_path, **{k: v for k, v in predictions.items() if isinstance(v, np.ndarray)})
    print(f"  saved {npz_path}")


parser = argparse.ArgumentParser(description="VGGT Jittor (jvggt) demo with viser visualization")
parser.add_argument(
    "--image_folder",
    type=str,
    default="examples/kitchen/images/",
    help="Path to folder containing images (default: examples/kitchen/images/)",
)
parser.add_argument(
    "--weights",
    type=str,
    default=None,
    help="Checkpoint: local model.pt or URL. Default: ./model.pt in repo root if it exists",
)
parser.add_argument(
    "--use_depth_unproject",
    action="store_true",
    help="Viser: depth+camera unproject (demo_viser default). Use --use_point_map for world_points instead",
)
parser.add_argument(
    "--use_point_map",
    action="store_true",
    help="Viser: use world_points instead of depth unproject",
)
parser.add_argument("--background_mode", action="store_true", help="Run viser server in a background thread")
parser.add_argument("--port", type=int, default=8080, help="Viser server port")
parser.add_argument(
    "--conf_threshold",
    type=float,
    default=25.0,
    help="Initial percentile of low-confidence points to filter out in viser",
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation (requires onnxruntime)")
parser.add_argument(
    "--inference_only",
    action="store_true",
    help="Run inference and print tensor shapes; skip viser",
)
parser.add_argument("--cpu", action="store_true", help="Force Jittor CPU (no CUDA)")
parser.add_argument(
    "--max_images",
    type=int,
    default=8,
    help="Max views to load (default: 8, kitchen example). 0 = all in folder (vggt demo_viser)",
)
parser.add_argument(
    "--frames_chunk_size",
    type=int,
    default=None,
    help="DPT frames per chunk. Default: 1 (8GB) or 4 with --vggt_compat",
)
parser.add_argument(
    "--skip_depth",
    action="store_true",
    help="Skip depth head (~save 1GB VRAM). Default: run depth head",
)
parser.add_argument(
    "--skip_point",
    action="store_true",
    help="Skip world_points head (~save 1GB VRAM). Useful with 2+ views on 15GB GPUs",
)
parser.add_argument(
    "--twoview",
    action="store_true",
    help="2-view preset: max_images=2, frames_chunk_size=1, skip_point (keep depth)",
)
parser.add_argument(
    "--vggt_compat",
    action="store_true",
    help="Match PyTorch demo_viser: frames_chunk_size=4, depth-unproject vis default",
)
parser.add_argument(
    "--low_vram",
    action="store_true",
    help="8GB shortcut: max_images=1, frames_chunk_size=1 (depth still on unless --skip_depth)",
)
parser.add_argument(
    "--save_preview",
    type=str,
    default=None,
    metavar="DIR",
    help="Save input/depth/conf PNGs (+ predictions.npz) to DIR; skips viser (good for Kaggle)",
)


def main() -> None:
    args = parser.parse_args()

    if args.twoview:
        args.max_images = 2
        args.skip_point = True
        if args.frames_chunk_size is None:
            args.frames_chunk_size = 1
        print("Two-view preset: max_images=2, frames_chunk_size=1, skip_point (depth on)")

    if args.low_vram:
        args.max_images = 1
        if args.frames_chunk_size is None:
            args.frames_chunk_size = 1
        print("Low VRAM: max_images=1, frames_chunk_size=1")

    if args.vggt_compat:
        if args.frames_chunk_size is None:
            args.frames_chunk_size = 4
        print("vggt_compat: frames_chunk_size=4 (same logic as demo_viser.py)")

    if args.frames_chunk_size is None:
        args.frames_chunk_size = 1

    setup_jittor(use_cuda=not args.cpu)
    import jittor as jt

    print(f"Jittor CUDA: {jt.has_cuda and not args.cpu}")

    image_folder = os.path.normpath(args.image_folder.strip().strip("、,，。；;"))
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Image folder not found: {image_folder}\n"
            "Use --image_folder to point at examples/kitchen/images/ or your own images."
        )

    all_names = list_images_from_folder(image_folder)
    if len(all_names) == 0:
        raise ValueError(f"No images found in {image_folder}")

    print(f"Loading images from {image_folder} ...")
    print(f"Found {len(all_names)} images in folder")

    image_names = select_image_paths(all_names, args.max_images)
    if len(image_names) < len(all_names):
        print(f"Using {len(image_names)} of {len(all_names)} views (--max_images {args.max_images}; 0 = all)")

    for i, name in enumerate(image_names):
        print(f"  [{i}] {os.path.basename(name)}")

    weights_path = resolve_weights_path(args.weights)
    print(f"Loading VGGT-1B weights into jvggt from:\n  {weights_path}")
    t0 = time.perf_counter()
    model = create_vggt_model(weights=args.weights)
    print(f"Model ready in {time.perf_counter() - t0:.1f}s")

    images = load_and_preprocess_images(image_names)
    print(f"Preprocessed images: shape={images.shape}, dtype={images.dtype}")

    jt.sync_all()
    input_images_np = np.ascontiguousarray(images.numpy(), dtype=np.float32)

    run_depth = not args.skip_depth
    run_point = not args.skip_point
    if not run_depth:
        print("Skipping depth head (--skip_depth). Outputs will not include depth / depth_conf.")
    if not run_point:
        print("Skipping point head (--skip_point). Outputs will not include world_points.")

    print("Running inference ...")
    t1 = time.perf_counter()
    raw_predictions = run_vggt_forward(
        model,
        images,
        frames_chunk_size=args.frames_chunk_size,
        run_depth=run_depth,
        run_point=run_point,
    )
    predictions = predictions_to_numpy_like_vggt_demo(raw_predictions, image_hw=images.shape[-2:])
    predictions["images"] = input_images_np
    print(f"Inference done in {time.perf_counter() - t1:.1f}s")

    print_prediction_summary(predictions)

    if args.save_preview:
        print(f"Saving preview images to {args.save_preview} ...")
        save_prediction_previews(predictions, args.save_preview)

    if args.inference_only or args.save_preview:
        if args.inference_only:
            print("Done (--inference_only, skipping viser).")
        else:
            print("Done (--save_preview, skipping viser).")
        return

    # demo_viser default: depth unproject; --use_point_map uses world_points instead
    use_point_map = args.use_point_map

    if use_point_map:
        print("Viser: using predicted world point map")
    else:
        print("Viser: unprojecting depth maps with predicted cameras (demo_viser default)")

    if args.mask_sky:
        print("Viser: sky masking enabled")

    print("Starting viser visualization ...")
    from jvggt.viser_wrapper import viser_wrapper

    viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=image_folder,
    )


if __name__ == "__main__":
    main()
