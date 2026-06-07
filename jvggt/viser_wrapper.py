# Viser visualization for jvggt predictions (no torch required).

from __future__ import annotations

import glob
import os
import threading
import time
from typing import List, Optional

from pathlib import Path

import cv2
import numpy as np
import viser
import viser.transforms as viser_tf
from tqdm.auto import tqdm

from jvggt.utils.geometry_np import closed_form_inverse_se3, unproject_depth_map_to_point_map

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    if onnxruntime is None:
        raise ImportError("mask_sky requires onnxruntime: pip install onnxruntime")

    from visual_util import download_file_from_url, segment_sky

    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            "skyseg.onnx",
        )

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for image_path in tqdm(image_files[:S]):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    sky_mask_array = np.array(sky_mask_list)
    conf = conf * (sky_mask_array > 0.1).astype(np.float32)
    print("Sky segmentation applied successfully")
    return conf


# Reject overflowed coordinates (VGGT scenes are typically well below this scale).
_MAX_POINT_ABS = 10000.0


def _images_to_uint8_rgb(flat_rgb: np.ndarray) -> np.ndarray:
    """(N, 3) RGB in [0,1] -> uint8."""
    rgb = np.nan_to_num(flat_rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _image_chw_to_uint8_hwc(img_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) -> (H, W, 3) uint8 for viser frustum."""
    hwc = np.transpose(img_chw, (1, 2, 0))
    return _images_to_uint8_rgb(hwc.reshape(-1, 3)).reshape(hwc.shape[0], hwc.shape[1], 3)


def _valid_points_mask(points: np.ndarray, conf_flat: np.ndarray, max_abs: float = _MAX_POINT_ABS) -> np.ndarray:
    return (
        np.isfinite(points).all(axis=1)
        & np.isfinite(conf_flat)
        & (np.abs(points) <= max_abs).all(axis=1)
    )


def _nanmean_axis0(arr: np.ndarray) -> np.ndarray:
    mask = np.isfinite(arr).all(axis=1)
    if not np.any(mask):
        return np.zeros(3, dtype=np.float64)
    return np.mean(arr[mask], axis=0)


def _finite_conf_percentile(conf: np.ndarray, percentile: float) -> float:
    conf_finite = conf[np.isfinite(conf)]
    if conf_finite.size == 0:
        return 0.0
    return float(np.percentile(conf_finite, percentile))


def _auto_point_size(points: np.ndarray) -> float:
    mask = np.isfinite(points).all(axis=1)
    if not np.any(mask):
        return 0.02
    pts = points[mask]
    extent = float(np.max(np.ptp(pts, axis=0)))
    if extent <= 0 or not np.isfinite(extent):
        return 0.02
    return float(np.clip(extent * 0.003, 0.008, 0.08))


def _focus_camera_on_points(client: viser.ClientHandle, visible: np.ndarray) -> None:
    if visible.size == 0:
        client.camera.position = (0.0, 0.0, 3.0)
        client.camera.look_at = (0.0, 0.0, 0.0)
        return
    center = np.mean(visible, axis=0)
    radius = float(np.max(np.linalg.norm(visible - center, axis=1)))
    if not np.isfinite(radius) or radius < 1e-6:
        radius = 1.0
    client.camera.position = (
        float(center[0]),
        float(center[1]),
        float(center[2] + radius * 2.5),
    )
    client.camera.look_at = (float(center[0]), float(center[1]), float(center[2]))


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 25.0,
    use_point_map: bool = True,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
):
    print(f"Starting viser server on port {port}")

    client_index = Path(viser.__file__).resolve().parent / "client" / "build" / "index.html"
    if not client_index.is_file():
        raise FileNotFoundError(
            f"Viser web client missing: {client_index}\n"
            "Reinstall: pip install --force-reinstall viser"
        )

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    print(f"\n>>> Open in browser: http://127.0.0.1:{port}")
    print(">>> Keep this terminal running. Press Ctrl+C to stop.\n")

    images = np.asarray(pred_dict["images"], dtype=np.float32)
    if images.ndim == 5:
        images = np.squeeze(images, axis=0)

    world_points_map = np.asarray(pred_dict["world_points"], dtype=np.float32)
    if world_points_map.ndim == 5:
        world_points_map = np.squeeze(world_points_map, axis=0)

    conf_map = np.asarray(pred_dict["world_points_conf"], dtype=np.float32)
    if conf_map.ndim == 4:
        conf_map = np.squeeze(conf_map, axis=0)
    if conf_map.ndim == 4 and conf_map.shape[-1] == 1:
        conf_map = conf_map[..., 0]

    if not use_point_map:
        if "depth" not in pred_dict or "depth_conf" not in pred_dict:
            print(
                "WARNING: depth outputs missing (depth head was skipped). "
                "Using world_points instead. Re-run with --with_depth for depth unproject."
            )
            use_point_map = True
        else:
            depth_map = np.asarray(pred_dict["depth"], dtype=np.float32)
            if depth_map.ndim == 5:
                depth_map = np.squeeze(depth_map, axis=0)

            depth_conf = np.asarray(pred_dict["depth_conf"], dtype=np.float32)
            if depth_conf.ndim == 4:
                depth_conf = np.squeeze(depth_conf, axis=0)
            if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
                depth_conf = depth_conf[..., 0]

    extrinsics_cam = np.asarray(pred_dict["extrinsic"], dtype=np.float64)
    if extrinsics_cam.ndim == 4 and extrinsics_cam.shape[0] == 1:
        extrinsics_cam = np.squeeze(extrinsics_cam, axis=0)

    intrinsics_cam = np.asarray(pred_dict["intrinsic"], dtype=np.float64)
    if intrinsics_cam.ndim == 4 and intrinsics_cam.shape[0] == 1:
        intrinsics_cam = np.squeeze(intrinsics_cam, axis=0)

    if not use_point_map:
        print(f"Camera intrinsics fy: {intrinsics_cam[:, 1, 1]}")
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        valid_ratio = float(np.isfinite(world_points).all(axis=-1).mean())
        if valid_ratio < 0.9:
            print(
                f"WARNING: depth unproject only {valid_ratio:.1%} valid; "
                "falling back to model world_points."
            )
            world_points = world_points_map
            conf = conf_map
        else:
            conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    colors = images.transpose(0, 2, 3, 1)
    S, H, W, _ = world_points.shape

    points = world_points.reshape(-1, 3).astype(np.float64)
    colors_flat = _images_to_uint8_rgb(colors.reshape(-1, 3))
    conf_flat = conf.reshape(-1).astype(np.float64)

    finite_mask = _valid_points_mask(points, conf_flat)
    n_finite = int(np.sum(finite_mask))
    n_overflow = int(np.sum(np.isfinite(points).all(axis=1) & ~finite_mask))
    if n_overflow > 0:
        print(
            f"WARNING: filtered {n_overflow} points with |coord|>{_MAX_POINT_ABS} "
            "(numerical overflow; try --low_vram or fewer --max_images)"
        )
    if n_finite == 0:
        print(
            "WARNING: no finite 3D points to display. "
            "Try default world_points mode, or check model outputs."
        )
    else:
        pts_valid = points[finite_mask]
        print(
            f"Point cloud: {n_finite}/{len(points)} finite points, "
            f"range [{pts_valid.min(axis=0)}, {pts_valid.max(axis=0)}]"
        )

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :].astype(np.float64)

    scene_center = _nanmean_axis0(points)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    point_size = _auto_point_size(points_centered[finite_mask] if n_finite else points_centered)

    frame_indices = np.repeat(np.arange(S), H * W)

    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )
    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )
    gui_reset_camera = server.gui.add_button("Reset Camera to Point Cloud")

    init_threshold_val = _finite_conf_percentile(conf_flat[finite_mask], init_conf_threshold)
    init_conf_mask = finite_mask & (conf_flat >= init_threshold_val)
    if not np.any(init_conf_mask):
        print("WARNING: confidence filter removed all points; showing all finite points.")
        init_conf_mask = finite_mask

    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask].astype(np.float32),
        colors=colors_flat[init_conf_mask],
        point_size=point_size,
        point_shape="circle",
        precision="float32",
    )
    print(f"Viser: showing {int(np.sum(init_conf_mask))} points (size={point_size:.4f})")
    visible_points = points_centered[init_conf_mask]

    def reset_all_cameras() -> None:
        for client in server.get_clients().values():
            _focus_camera_on_points(client, visible_points)

    @gui_reset_camera.on_click
    def _(_) -> None:
        reset_all_cameras()

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(S)):
            cam2world_3x4 = cam_to_world[img_id]
            if not np.isfinite(cam2world_3x4).all():
                continue
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = _image_chw_to_uint8_hwc(images_[img_id])
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        current_percentage = gui_points_conf.value
        threshold_val = _finite_conf_percentile(conf_flat[finite_mask], current_percentage)
        conf_mask = finite_mask & (conf_flat >= threshold_val)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        if not np.any(combined_mask):
            combined_mask = finite_mask & frame_mask
        nonlocal visible_points
        visible_points = points_centered[combined_mask]
        point_cloud.points = visible_points.astype(np.float32)
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    visualize_frames(cam_to_world, images)

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        _focus_camera_on_points(client, visible_points)

    print("Viser server running. Press Ctrl+C to stop.")
    print("Tip: if the view is empty, click 'Reset Camera to Point Cloud' in the left panel.")
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server
