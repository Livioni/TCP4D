"""Causal Stage2 window playback adapted from the 4RC inference viewer."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
from PIL import Image, ImageOps

WIDTH, HEIGHT = 320, 240
PADDING = (1, 1, 6, 6)
ARMS = ("left", "right")
ARM_COLORS = ((255, 100, 55), (45, 175, 255))


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()


def cache_window(episode, reconstruction, record, start, end, history_padding):
    """Keep predicted history and anchor geometry; never read future labels."""
    def array(key):
        return reconstruction[key][0].detach().float().cpu().numpy()
    position = array("tcp_position")[history_padding:]
    rotation = array("tcp_rotation")[history_padding:]
    position, rotation = transform_history(position, rotation, episode.extrinsics[start:end])
    view = dict(record, start=start, end=end,
                frame_indices=episode.frame_indices[start:end].copy(),
                history_position=position, history_rotation=rotation,
                future_frame_indices=int(episode.frame_indices[end-1])+np.asarray(record["future_step_indices"]))
    depth = array("depth")[-1]
    confidence = array("depth_conf")[-1]
    geometry = {
        "depth": depth[PADDING[2]:PADDING[2]+HEIGHT, PADDING[0]:PADDING[0]+WIDTH].copy(),
        "confidence": confidence[PADDING[2]:PADDING[2]+HEIGHT, PADDING[0]:PADDING[0]+WIDTH].copy(),
    }
    return view, geometry


def transform_history(position: np.ndarray, rotation: np.ndarray, extrinsics: np.ndarray):
    relative = extrinsics[-1:] @ np.linalg.inv(extrinsics)
    positions = np.einsum("kij,kaj->kai", relative[:, :3, :3], position) + relative[:, None, :3, 3]
    rotations = relative[:, None, :3, :3] @ rotation
    return positions.astype(np.float32), rotations.astype(np.float32)


def make_point_cloud(depth: np.ndarray, confidence: np.ndarray, rgb: np.ndarray,
                     intrinsics: np.ndarray, frame_w2c: np.ndarray, anchor_w2c: np.ndarray,
                     *, confidence_percentile: float = 2.5, max_points: int = 100_000,
                     max_depth: float | None = 3.0, seed: int = 42):
    if not 0 <= confidence_percentile <= 99 or max_points < 0:
        raise ValueError("Invalid point cloud filtering options")
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    rays = np.stack((columns, rows, np.ones_like(rows)), axis=-1) @ np.linalg.inv(intrinsics).T
    points = rays * depth[..., None]
    transform = anchor_w2c @ np.linalg.inv(frame_w2c)
    points = points @ transform[:3, :3].T + transform[:3, 3]
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & np.isfinite(points).all(-1)
    if max_depth is not None:
        valid &= depth <= max_depth
    if valid.any():
        valid &= confidence >= np.percentile(confidence[valid], confidence_percentile)
    indices = np.flatnonzero(valid)
    if max_points and len(indices) > max_points:
        indices = np.random.default_rng(seed).choice(indices, max_points, replace=False)
    return points.reshape(-1, 3)[indices].astype(np.float32), rgb.reshape(-1, 3)[indices]


def build_playback_frames(result: dict) -> list[tuple[int, int]]:
    """Play each causal prediction at its anchor; retain legacy result playback."""
    windows = result["windows"]
    geometry = result.get("_geometry", [])
    if not windows or len(geometry) != len(windows):
        raise ValueError("Playback requires in-memory geometry for every completed window")
    if result.get("window_schedule") == "causal":
        return [(index, record["end"] - record["start"] - 1) for index, record in enumerate(windows)]
    slots = {}
    for index, record in enumerate(windows):
        for frame in range(record["start"], record["end"]):
            slots.setdefault(frame, (index, frame - record["start"]))
    frames = sorted(slots)
    if frames != list(range(len(frames))):
        raise ValueError("Playback windows must cover contiguous episode frames starting at zero")
    return [slots[frame] for frame in frames]


class Stage2Viewer:
    """Play episode frames in order using in-memory window geometry."""

    def __init__(self, result: dict, episode: object, *, host="127.0.0.1", port=8020,
                 point_size=0.003, fps=5.0, confidence_percentile=2.5, max_points=100_000, max_depth=3.0):
        import viser
        import viser.transforms as tf

        self.tf = tf
        self.result, self.episode = result, episode
        self.max_points, self.max_depth = max_points, max_depth
        self.frame_slots = build_playback_frames(result)
        if not math.isfinite(fps) or not 0.25 <= fps <= 30:
            raise ValueError("FPS must be between 0.25 and 30")
        if not math.isfinite(point_size) or point_size <= 0:
            raise ValueError("Point size must be finite and positive")
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("-y")
        self.server.gui.set_panel_label("TCP4D Stage2: Window TCP Trajectories")
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._cloud_handle = None
        self._nodes = []
        gui = self.server.gui
        with gui.add_folder("Playback", expand_by_default=True):
            self.frame = gui.add_slider("Window", min=0, max=max(1, len(self.frame_slots) - 1), step=1, initial_value=0,
                                        disabled=len(self.frame_slots) == 1)
            self.previous = gui.add_button("Previous")
            self.next = gui.add_button("Next")
            self.play = gui.add_checkbox("Play", False)
            self.fps = gui.add_slider("FPS", min=0.25, max=30, step=0.25, initial_value=fps)
        with gui.add_folder("Geometry", expand_by_default=True):
            self.point_size = gui.add_slider("Point size", min=min(1e-5, point_size), max=max(0.01, point_size),
                                             step=1e-5, initial_value=point_size)
            self.confidence = gui.add_slider("Confidence percentile", min=0, max=99, step=0.5,
                                             initial_value=confidence_percentile)
            self.show_cloud = gui.add_checkbox("Point cloud", True)
        with gui.add_folder("TCP", expand_by_default=True):
            self.future = gui.add_slider("Future steps", min=1, max=result["prediction_horizon"], step=1,
                                         initial_value=result["prediction_horizon"])
            self.show_history = gui.add_checkbox("Recovered TCP (observed frames)", True)
            self.show_prediction = gui.add_checkbox("Predicted future", True)
            self.show_axes = gui.add_checkbox("TCP orientation axes", True)
        gui.add_markdown(
            f"**Planning:** Each inference predicts {result['prediction_horizon']} action steps; "
            f"replan every {result['window_stride']} steps.\n\n"
            f"**Instruction:** {episode.instruction}\n\n"
            "Left: orange; right: blue. Thin: recovered history; thick: predicted future.")
        self.info = gui.add_markdown("")
        self.rgb_handle = gui.add_image(load_rgb(episode.image_paths[0]), label="Historical RGB")
        for control in (self.frame, self.future, self.show_cloud, self.show_history,
                        self.show_prediction, self.show_axes, self.confidence):
            control.on_update(lambda _: self.refresh())
        self.previous.on_click(lambda _: self.step_frame(-1))
        self.next.on_click(lambda _: self.step_frame(1))
        self.play.on_update(lambda _: self._toggle_playback())
        self.point_size.on_update(lambda _: self._update_point_size())
        self.refresh()

        @self.server.on_client_connect
        def initialize_camera(client):
            client.camera.position = (0.0, -0.1, -0.5)
            client.camera.look_at = (0.0, 0.0, 1.0)
            client.camera.up_direction = (0.0, -1.0, 0.0)

        from tcp4d.viser_video_export import EpisodeVideoExport

        self.video_export = EpisodeVideoExport(
            self.server, frame_count=len(self.frame_slots), frame=self.frame, fps=self.fps,
            controls=[self.frame, self.previous, self.next, self.play, self.fps,
                      self.point_size, self.confidence, self.show_cloud, self.future,
                      self.show_history, self.show_prediction, self.show_axes],
            render_frame=self.refresh, lock=self._lock, stop_event=self._closed,
            filename="stage2_episode.mp4",
        )
        self._thread = threading.Thread(target=self._playback, daemon=True)
        self._thread.start()

    def step_frame(self, direction=1):
        with self._lock:
            if not self.video_export.busy.is_set():
                self.frame.value = (int(self.frame.value) + direction) % len(self.frame_slots)

    def _toggle_playback(self):
        self.frame.disabled = bool(self.play.value) or len(self.frame_slots) == 1
        self.previous.disabled = self.next.disabled = bool(self.play.value)

    def _update_point_size(self):
        with self._lock:
            if self._cloud_handle is not None:
                self._cloud_handle.point_size = float(self.point_size.value)
        self.server.flush()

    def _playback(self):
        deadline = time.monotonic()
        while not self._closed.wait(0.01):
            now = time.monotonic()
            if not self.play.value:
                deadline = now
            elif now >= deadline:
                self.step_frame()
                deadline = now + 1.0 / float(self.fps.value)

    def _trajectory(self, name, positions, colors, *, dashed=False, width=3):
        positions = np.asarray(positions, dtype=np.float32)
        for arm, color in enumerate(colors):
            p = positions[:, arm]
            finite = np.isfinite(p).all(-1)
            if finite.any():
                self._nodes.append(self.server.scene.add_point_cloud(f"/{name}/{arm}/points", points=p[finite],
                    colors=np.broadcast_to(np.array(color, dtype=np.uint8), (int(finite.sum()), 3)).copy(), point_size=0.006))
            segments = np.stack((p[:-1], p[1:]), axis=1)
            segments = segments[finite[:-1] & finite[1:]]
            if dashed and len(segments):
                delta = segments[:, 1] - segments[:, 0]
                segments = np.concatenate([np.stack((segments[:, 0] + delta * lo, segments[:, 0] + delta * hi), 1)
                                           for lo, hi in ((0, 0.2), (0.4, 0.6), (0.8, 1))])
            if len(segments):
                self._nodes.append(self.server.scene.add_line_segments(f"/{name}/{arm}/lines", points=segments,
                    colors=np.broadcast_to(np.array(color, dtype=np.uint8), segments.shape).copy(), line_width=width))

    def _axes(self, name, positions, rotations):
        for arm in range(2):
            if np.isfinite(positions[arm]).all() and np.isfinite(rotations[arm]).all():
                self._nodes.append(self.server.scene.add_frame(f"/{name}/{arm}", position=positions[arm],
                    wxyz=self.tf.SO3.from_matrix(rotations[arm]).wxyz, axes_length=0.035, axes_radius=0.001))

    def refresh(self, frame_slot=None):
        with self._lock, self.server.atomic():
            if self._closed.is_set():
                return
            index, local = self.frame_slots[int(self.frame.value) if frame_slot is None else frame_slot]
            record = self.result["windows"][index]
            geometry = self.result["_geometry"][index]
            for node in self._nodes:
                node.remove()
            self._nodes.clear()
            self._cloud_handle = None
            frame = record["start"] + local
            rgb = load_rgb(self.episode.image_paths[frame])
            self.rgb_handle.image = rgb
            if self.show_cloud.value:
                points, colors = make_point_cloud(geometry["depth"], geometry["confidence"],
                    rgb, self.episode.intrinsics, self.episode.extrinsics[frame], self.episode.extrinsics[record["end"] - 1],
                    confidence_percentile=self.confidence.value, max_points=self.max_points, max_depth=self.max_depth)
                self._cloud_handle = self.server.scene.add_point_cloud("/cloud", points=points, colors=colors,
                    point_size=float(self.point_size.value), point_shape="rounded")
                self._nodes.append(self._cloud_handle)
            hp = np.asarray(record["history_position"], dtype=np.float32)
            hr = np.asarray(record["history_rotation"], dtype=np.float32)
            ap = np.asarray(record["action_position"], dtype=np.float32)
            ar = np.asarray(record["action_rotation"], dtype=np.float32)
            n = self.future.value
            if self.show_history.value:
                self._trajectory("history", hp, ARM_COLORS, width=1)
                if self.show_axes.value:
                    self._axes("history_axes", hp[local], hr[local])
            if self.show_prediction.value and record["success"]:
                self._trajectory("prediction", np.concatenate((hp[-1:], ap[:n])), ARM_COLORS, width=4)
                if self.show_axes.value:
                    self._axes("future_axes", ap[n - 1], ar[n - 1])
            grip = np.asarray(record["action_gripper"])[n - 1]
            states = ", ".join(f"{arm}: {'open' if value == 1 else 'closed' if value == 0 else 'invalid'}" for arm, value in zip(ARMS, grip))
            self.info.content = (f"**Frame {self.episode.frame_indices[frame]}** · window {index} · history {record['frame_indices'][0]}–{record['anchor_frame']} "
                f"· future frames {record['future_frame_indices'][0]}–{record['future_frame_indices'][n - 1]}\n\nSuccess: **{record['success']}** · {states}\n\n"
                "Coordinates: the current window's last observation camera. "
                "Thin paths: recovered observed TCPs. Thick paths: predicted future TCPs.")

    def close(self):
        self._closed.set()
        self._thread.join(timeout=2)
        # Wait for any ongoing refresh before releasing the previous run's cache.
        with self._lock:
            self.result = {}
            self._nodes.clear()
            self._cloud_handle = None
        self.server.stop()
