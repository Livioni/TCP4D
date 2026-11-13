"""Single-arm and dual-arm Stage1 viewer adapted from the 4RC sliding-window viewer."""
from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image

from tcp4d.inference import matrix_to_rpy

ARM_COLORS = ((255, 96, 64), (45, 180, 255))


def prepare_frame_point_clouds(predictions, colors, episode, *, max_points_per_frame, seed=0):
    """Unproject cropped metric depth into each frame's OpenCV camera coordinates.

    TCP predictions already use these coordinates. Episode calibration avoids
    mixing window-local ray-pose coordinates with camera-frame TCP positions.
    """
    if max_points_per_frame < 0:
        raise ValueError("--max-points cannot be negative")
    depth = predictions["depth"][0].detach().float().cpu().numpy()
    confidence = predictions["depth_conf"][0].detach().float().cpu().numpy()
    left, right, top, bottom = episode.padding
    height, width = episode.source_height, episode.source_width
    expected = (len(colors), height + top + bottom, width + left + right)
    if depth.shape != expected or confidence.shape != expected:
        raise ValueError(f"Expected depth/confidence shape {expected}, got {depth.shape}/{confidence.shape}")
    yy, xx = np.mgrid[:height, :width]
    pixels = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3)
    intrinsics = np.asarray(episode.intrinsics)
    if intrinsics.shape == (3, 3):
        intrinsics = np.broadcast_to(intrinsics, (len(colors), 3, 3))
    clouds = []
    for slot, color in enumerate(colors):
        rays = (pixels @ np.linalg.inv(intrinsics[slot]).T).astype(np.float32)
        if color.shape != (height, width, 3):
            raise ValueError(f"Unexpected RGB shape: {color.shape}")
        frame_depth = depth[slot, top:top + height, left:left + width].reshape(-1)
        frame_confidence = confidence[slot, top:top + height, left:left + width].reshape(-1)
        valid = np.isfinite(frame_depth) & (frame_depth > 0) & np.isfinite(frame_confidence)
        indices = np.flatnonzero(valid)
        if not len(indices):
            raise ValueError(f"Frame slot {slot} has no valid geometry")
        if max_points_per_frame and len(indices) > max_points_per_frame:
            indices = np.random.default_rng(seed + slot).choice(indices, max_points_per_frame, replace=False)
        order = np.argsort(frame_confidence[indices], kind="stable")[::-1]
        indices = indices[order]
        clouds.append({
            "points": rays[indices] * frame_depth[indices, None],
            "colors": color.reshape(-1, 3)[indices].astype(np.uint8, copy=False),
            "confidence": frame_confidence[indices].astype(np.float32, copy=False),
        })
    return clouds


def _filtered_cloud(
    frame_cloud: dict[str, np.ndarray],
    confidence_percentile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not 0.0 <= confidence_percentile <= 100.0:
        raise ValueError("Confidence percentile must be in [0,100]")
    total = len(frame_cloud["points"])
    keep_count = max(
        1, int(math.ceil(total * (100.0 - confidence_percentile) / 100.0))
    )
    threshold = float(frame_cloud["confidence"][keep_count - 1])
    return (
        frame_cloud["points"][:keep_count],
        frame_cloud["colors"][:keep_count],
        threshold,
    )


def _format_frame_panel(
    tcp: dict[str, np.ndarray],
    *,
    frame_slot: int,
    frame_index: int,
    image_path: Path,
    num_frames: int,
    confidence_percentile: float,
    visible_points: int,
    confidence_threshold: float,
    arm_names: tuple[str, ...] = ("tcp",),
) -> str:
    rpy = matrix_to_rpy(tcp["rotation"][frame_slot])
    lines = [
        f"# Frame {frame_slot + 1}/{num_frames}",
        f"- **Episode frame:** `{frame_index}`",
        f"- **Image:** `{image_path.name}`",
        (
            f"- **Geometry:** `{visible_points:,}` points, top "
            f"`{100.0 - confidence_percentile:.1f}%` "
            f"(confidence ≥ `{confidence_threshold:.4f}`)"
        ),
    ]
    for arm_index, arm_name in enumerate(arm_names):
        xyz_text = ", ".join(
            f"{value:+.4f}" for value in tcp["position"][frame_slot, arm_index]
        )
        rpy_text = ", ".join(f"{value:+.4f}" for value in rpy[arm_index])
        lines.extend(
            (
                "## TCP" if arm_name == "tcp" else f"## {arm_name.title()} TCP",
                f"- **xyz (m):** `{xyz_text}`",
                f"- **rpy (rad):** `{rpy_text}`",
                (
                    f"- **gripper:** "
                    f"`{tcp['gripper'][frame_slot, arm_index]:.4f}`"
                ),
                (
                    f"- **confidence:** "
                    f"`{tcp['confidence'][frame_slot, arm_index]:.4f}`"
                ),
            )
        )
    return "\n".join(lines)


def compute_default_third_camera_view(
    extrinsic_w2c: np.ndarray,
    scene_center: np.ndarray,
    scene_extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the viewer just behind an OpenCV camera, looking along its +Z axis."""
    extrinsic = np.asarray(extrinsic_w2c, dtype=np.float64)
    center = np.asarray(scene_center, dtype=np.float64)
    if extrinsic.shape not in ((3, 4), (4, 4)):
        raise ValueError(
            f"Expected a [3,4] or [4,4] world-to-camera matrix, got {extrinsic.shape}"
        )
    if center.shape != (3,):
        raise ValueError(f"Expected a scene center [3], got {center.shape}")
    if not np.isfinite(extrinsic).all() or not np.isfinite(center).all():
        raise ValueError("Camera extrinsic and scene center must be finite")
    if not math.isfinite(scene_extent) or scene_extent <= 0:
        raise ValueError("Scene extent must be finite and positive")

    rotation_w2c = extrinsic[:3, :3]
    translation_w2c = extrinsic[:3, 3]
    rotation_c2w = rotation_w2c.T
    camera_center = -rotation_c2w @ translation_w2c
    forward = rotation_c2w[:, 2]
    up = -rotation_c2w[:, 1]
    forward_norm = np.linalg.norm(forward)
    up_norm = np.linalg.norm(up)
    if forward_norm < 1e-8 or up_norm < 1e-8:
        raise ValueError("Camera extrinsic has a degenerate rotation")
    forward /= forward_norm
    up /= up_norm

    backoff = float(np.clip(scene_extent * 0.15, 0.05, 0.25))
    target_distance = float(np.dot(center - camera_center, forward))
    target_distance = max(target_distance, scene_extent * 0.5, 0.1)
    viewer_position = camera_center - forward * backoff
    look_at = camera_center + forward * target_distance
    return (
        viewer_position.astype(np.float32),
        look_at.astype(np.float32),
        up.astype(np.float32),
    )


def start_visualization(
    frame_clouds: list[dict[str, np.ndarray]],
    tcp: dict[str, np.ndarray],
    frame_indices: list[int],
    image_paths: list[Path],
    *,
    host: str,
    port: int,
    point_size: float,
    confidence_percentile: float,
    reference_extrinsic_w2c: np.ndarray | None = None,
    frame_rate: float = 5.0,
    arm_names: tuple[str, ...] = ("tcp",),
):
    try:
        import viser
        import viser.transforms as tf
    except ImportError as error:
        raise ImportError(
            "viser is required for visualization; run: pip install -e '.[visualization]'"
        ) from error

    from tcp4d.viser_video_export import EpisodeVideoExport

    scene_lock = threading.RLock()
    num_frames = len(frame_clouds)
    if not (
        num_frames
        == len(frame_indices)
        == len(image_paths)
        == tcp["position"].shape[0]
        == tcp["rotation"].shape[0]
        == tcp["gripper"].shape[0]
        == tcp["confidence"].shape[0]
    ):
        raise ValueError("Geometry, TCP, image, and frame-index lengths do not match")

    if len(arm_names) not in (1, 2) or tcp["position"].shape[1:] != (len(arm_names), 3):
        raise ValueError("TCP arm names and prediction shapes do not match")

    bounds_samples: list[np.ndarray] = []
    for frame_cloud in frame_clouds:
        high_confidence_count = max(1, int(len(frame_cloud["points"]) * 0.95))
        high_confidence_points = frame_cloud["points"][:high_confidence_count]
        sample_step = max(1, len(high_confidence_points) // 10_000)
        bounds_samples.append(high_confidence_points[::sample_step])
    bounds_samples.append(tcp["position"].reshape(-1, 3))
    bounds_points = np.concatenate(bounds_samples, axis=0)
    bounds_min = np.percentile(bounds_points, 1.0, axis=0)
    bounds_max = np.percentile(bounds_points, 99.0, axis=0)
    center = (bounds_min + bounds_max) * 0.5
    extent = max(float(np.max(bounds_max - bounds_min)), 0.1)
    marker_scale = float(np.clip(extent * 0.08, 0.025, 0.12))
    tcp_axes_length = marker_scale * 0.55
    axes_radius = marker_scale * 0.018
    origin_radius = marker_scale * 0.05

    default_camera_position = center + extent * np.array(
        [0.0, 0.0, -1.2], dtype=np.float32
    )
    default_camera_look_at = center
    default_camera_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if reference_extrinsic_w2c is not None:
        try:
            (
                default_camera_position,
                default_camera_look_at,
                default_camera_up,
            ) = compute_default_third_camera_view(
                reference_extrinsic_w2c, center, extent
            )
        except ValueError as error:
            print(
                "Warning: could not initialize behind the third camera; "
                f"using the scene fallback view: {error}"
            )

    server = viser.ViserServer(host=host, port=port)
    server.gui.set_panel_label("TCP4D Stage1: Geometry + TCP")
    server.gui.configure_theme(
        control_layout="floating", control_width="large", show_logo=False
    )
    server.scene.set_up_direction(tuple(default_camera_up.tolist()))
    world_axes = server.scene.world_axes
    world_axes.axes_length = tcp_axes_length * 2.0
    world_axes.axes_radius = axes_radius
    world_axes.origin_radius = origin_radius
    world_axes.visible = True

    with server.gui.add_folder("Playback", expand_by_default=True):
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
        gui_previous = server.gui.add_button("Previous")
        gui_next = server.gui.add_button("Next")
        gui_playing = server.gui.add_checkbox("Play", False)
        gui_fps = server.gui.add_slider(
            "FPS", min=0.25, max=max(30.0, frame_rate), step=0.25, initial_value=frame_rate
        )

    with server.gui.add_folder("Geometry", expand_by_default=True):
        gui_confidence = server.gui.add_slider(
            "Confidence percentile",
            min=0.0,
            max=99.0,
            step=0.5,
            initial_value=float(confidence_percentile),
        )
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=1e-5,
            max=0.01,
            step=1e-5,
            initial_value=point_size,
        )
        gui_show_points = server.gui.add_checkbox("Show point cloud", True)

    with server.gui.add_folder("TCP", expand_by_default=True):
        gui_show_tcp = server.gui.add_checkbox("Show TCP spheres + axes", True)
        gui_show_labels = server.gui.add_checkbox("Show TCP labels", True)

    initial_points, _, initial_threshold = _filtered_cloud(
        frame_clouds[0], confidence_percentile
    )
    gui_info = server.gui.add_markdown(
        _format_frame_panel(
            tcp,
            frame_slot=0,
            frame_index=frame_indices[0],
            image_path=image_paths[0],
            num_frames=num_frames,
            confidence_percentile=confidence_percentile,
            visible_points=len(initial_points),
            confidence_threshold=initial_threshold,
            arm_names=arm_names,
        )
    )

    with Image.open(image_paths[0]) as image:
        gui_rgb = server.gui.add_image(np.asarray(image.convert("RGB")), label="RGB frame")

    frame_handles: list[dict[str, Any]] = []
    for frame_slot, frame_cloud in enumerate(frame_clouds):
        filtered_points, filtered_colors, _ = _filtered_cloud(
            frame_cloud, confidence_percentile
        )
        root_node = server.scene.add_frame(
            f"/frames/t{frame_slot}", show_axes=False, visible=frame_slot == 0
        )
        point_node = server.scene.add_point_cloud(
            f"/frames/t{frame_slot}/geometry",
            points=filtered_points,
            colors=filtered_colors,
            point_size=point_size,
            point_shape="rounded",
            visible=frame_slot == 0,
        )

        tcp_frames = []
        tcp_spheres = []
        tcp_labels = []
        for arm_index, (arm_name, arm_color) in enumerate(
            zip(arm_names, ARM_COLORS)
        ):
            quaternion = tf.SO3.from_matrix(
                tcp["rotation"][frame_slot, arm_index]
            ).wxyz
            tcp_frame = server.scene.add_frame(
                f"/frames/t{frame_slot}/tcp/{arm_name}",
                wxyz=quaternion,
                position=tcp["position"][frame_slot, arm_index],
                axes_length=tcp_axes_length,
                axes_radius=axes_radius,
                origin_radius=origin_radius,
                origin_color=arm_color,
                visible=frame_slot == 0,
            )
            tcp_sphere = server.scene.add_icosphere(
                f"/frames/t{frame_slot}/tcp/{arm_name}/xyz",
                radius=marker_scale * 0.11,
                color=arm_color,
                position=(0.0, 0.0, 0.0),
                visible=frame_slot == 0,
            )
            tcp_label = server.scene.add_label(
                f"/frames/t{frame_slot}/tcp/{arm_name}/label",
                text=(
                    f"{arm_name.upper() if arm_name == 'tcp' else arm_name.title()}: "
                    f"gripper={tcp['gripper'][frame_slot, arm_index]:.3f}"
                ),
                position=(0.0, 0.0, marker_scale * 1.25),
                anchor="bottom-center",
                visible=frame_slot == 0,
            )
            tcp_frames.append(tcp_frame)
            tcp_spheres.append(tcp_sphere)
            tcp_labels.append(tcp_label)

        frame_handles.append(
            {
                "root": root_node,
                "points": point_node,
                "tcp_frames": tcp_frames,
                "tcp_spheres": tcp_spheres,
                "tcp_labels": tcp_labels,
                "filter_percentile": float(confidence_percentile),
            }
        )

    def _apply_confidence_filter(frame_slot: int) -> tuple[int, float]:
        frame_handle = frame_handles[frame_slot]
        percentile = float(gui_confidence.value)
        points, colors, threshold = _filtered_cloud(
            frame_clouds[frame_slot], percentile
        )
        if frame_handle["filter_percentile"] != percentile:
            frame_handle["points"].points = points
            frame_handle["points"].colors = colors
            frame_handle["filter_percentile"] = percentile
        return len(points), threshold

    def _update_panel(frame_slot: int, visible_points: int, threshold: float) -> None:
        with Image.open(image_paths[frame_slot]) as image:
            gui_rgb.image = np.asarray(image.convert("RGB"))
        gui_info.content = _format_frame_panel(
            tcp,
            frame_slot=frame_slot,
            frame_index=frame_indices[frame_slot],
            image_path=image_paths[frame_slot],
            num_frames=num_frames,
            confidence_percentile=float(gui_confidence.value),
            visible_points=visible_points,
            confidence_threshold=threshold,
            arm_names=arm_names,
        )

    def _update_visibility(current: int | None = None) -> None:
        if current is None:
            current = int(gui_frame.value)
        with server.atomic():
            for frame_slot, handles in enumerate(frame_handles):
                active = frame_slot == current
                handles["root"].visible = active
                handles["points"].visible = active and bool(gui_show_points.value)
                for tcp_frame, tcp_sphere, tcp_label in zip(
                    handles["tcp_frames"],
                    handles["tcp_spheres"],
                    handles["tcp_labels"],
                ):
                    tcp_frame.visible = active and bool(gui_show_tcp.value)
                    tcp_sphere.visible = active and bool(gui_show_tcp.value)
                    tcp_label.visible = (
                        active
                        and bool(gui_show_tcp.value)
                        and bool(gui_show_labels.value)
                    )
        server.flush()

    def _render_frame(slot: int) -> None:
        visible_points, threshold = _apply_confidence_filter(slot)
        _update_panel(slot, visible_points, threshold)
        _update_visibility(slot)

    def _viewer_update(callback):
        def guarded(event):
            with scene_lock:
                if not video_export.busy.is_set():
                    callback(event)
        return guarded

    @gui_previous.on_click
    @_viewer_update
    def _previous_frame(_) -> None:
        gui_frame.value = (int(gui_frame.value) - 1) % num_frames

    @gui_next.on_click
    @_viewer_update
    def _next_frame(_) -> None:
        gui_frame.value = (int(gui_frame.value) + 1) % num_frames

    @gui_playing.on_update
    @_viewer_update
    def _toggle_playing(_) -> None:
        playing = bool(gui_playing.value)
        gui_frame.disabled = playing
        gui_previous.disabled = playing
        gui_next.disabled = playing

    @gui_frame.on_update
    @_viewer_update
    def _select_frame(_) -> None:
        current = int(gui_frame.value)
        visible_points, threshold = _apply_confidence_filter(current)
        _update_panel(current, visible_points, threshold)
        _update_visibility()

    @gui_confidence.on_update
    @_viewer_update
    def _change_confidence(_) -> None:
        current = int(gui_frame.value)
        visible_points, threshold = _apply_confidence_filter(current)
        _update_panel(current, visible_points, threshold)
        server.flush()

    @gui_point_size.on_update
    @_viewer_update
    def _change_point_size(_) -> None:
        with server.atomic():
            for handles in frame_handles:
                handles["points"].point_size = float(gui_point_size.value)
        server.flush()

    @gui_show_points.on_update
    @_viewer_update
    def _toggle_points(_) -> None:
        _update_visibility()

    @gui_show_tcp.on_update
    @_viewer_update
    def _toggle_tcp(_) -> None:
        _update_visibility()

    @gui_show_labels.on_update
    @_viewer_update
    def _toggle_labels(_) -> None:
        _update_visibility()

    @server.on_client_connect
    def _set_initial_camera(client: viser.ClientHandle) -> None:
        with client.atomic():
            client.camera.position = tuple(default_camera_position.tolist())
            client.camera.look_at = tuple(default_camera_look_at.tolist())
            client.camera.up_direction = tuple(default_camera_up.tolist())
        client.flush()

    playback_stop_event = threading.Event()
    setattr(server, "_tcp_playback_stop_event", playback_stop_event)

    video_export = EpisodeVideoExport(
        server, frame_count=num_frames, frame=gui_frame, fps=gui_fps,
        controls=[gui_frame, gui_previous, gui_next, gui_playing, gui_fps,
                  gui_confidence, gui_point_size, gui_show_points, gui_show_tcp, gui_show_labels],
        render_frame=_render_frame, lock=scene_lock, stop_event=playback_stop_event,
        filename="stage1_episode.mp4",
    )

    def _playback_loop() -> None:
        while not playback_stop_event.is_set():
            with scene_lock:
                if not video_export.busy.is_set() and bool(gui_playing.value):
                    gui_frame.value = (int(gui_frame.value) + 1) % num_frames
            playback_stop_event.wait(timeout=1.0 / float(gui_fps.value))

    playback_thread = threading.Thread(target=_playback_loop, daemon=True)
    playback_thread.start()
    _update_visibility()
    return server


def stop_visualization(server: Any | None) -> None:
    """Stop a TCP viser server and its playback worker."""
    if server is None:
        return
    stop_event = getattr(server, "_tcp_playback_stop_event", None)
    if stop_event is not None:
        stop_event.set()
    server.stop()
