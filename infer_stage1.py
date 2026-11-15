"""TCP4D Stage1 tracking for DROID, AUTOLAB UR5, and RoboTwin with optional interactive visualization."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import dataclass, field
import gc
import html
import json
import math
from pathlib import Path
import threading
import time
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tcp4d.droid_checkpoint import weight_file
from tcp4d.inference import (_frame_index, _remove_training_prefix, collect_rgb_paths,
    resolve_device, resolve_dtype, matrix_to_rpy, build_sliding_windows, write_json_atomic)
SOURCE_WIDTH, SOURCE_HEIGHT = 320, 180
PADDED_WIDTH, PADDED_HEIGHT = 322, 182
FRAME_RATE = 15
ROBOTWIN_TCP_DIRS = {"third_views": "TCP_third", "head_view": "TCP_head"}
ARM_COLORS = ((255, 96, 64), (45, 180, 255))

@dataclass
class Episode:
    path: Path
    camera: str
    image_paths: list[Path]
    intrinsics: np.ndarray | None
    dataset: str = field(default="droid", kw_only=True)
    frame_rate: float = field(default=FRAME_RATE, kw_only=True)
    timestamps: np.ndarray | None = field(default=None, kw_only=True)
    metadata: dict = field(default_factory=dict, kw_only=True)
    source_width: int = field(default=SOURCE_WIDTH, kw_only=True)
    source_height: int = field(default=SOURCE_HEIGHT, kw_only=True)
    padding: tuple[int, int, int, int] = field(default=(1, 1, 1, 1), kw_only=True)

    arm_names: tuple[str, ...] = field(default=("tcp",), kw_only=True)

    @property
    def num_arms(self) -> int:
        return len(self.arm_names)

    def time_seconds(self, index: int) -> float:
        return float(self.timestamps[index]) if self.timestamps is not None else index / self.frame_rate

    def clip(self, start_frame: int, max_frames: int = 0) -> list[Path]:
        if not 0 <= start_frame < len(self.image_paths):
            raise ValueError(f"Start frame {start_frame} outside episode [0,{len(self.image_paths) - 1}]")
        if max_frames < 0 or max_frames == 1:
            raise ValueError("--max-frames must be 0 (all remaining frames) or at least 2")
        paths = self.image_paths[start_frame:]
        if max_frames:
            paths = paths[:max_frames]
        if len(paths) < 2:
            raise ValueError(f"Camera {self.camera}: fewer than 2 frames remain from frame {start_frame}")
        return paths


def load_episode(path: Path, camera: str | None = None) -> Episode:
    path = path.expanduser().resolve()
    image_root = path / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Expected an episode with images/: {path}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    rgb_only = metadata.get('format_version') == 'tcp4d_rgb_v1'
    if str(metadata.get("format_version", "")).startswith("robotwin"):
        dataset = "robotwin"
    elif str(metadata.get("format", "")).startswith("berkeley_autolab_ur5"):
        dataset = "autolab_ur5"
    else:
        dataset = "droid"
    cameras = sorted(p.name for p in image_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not cameras:
        raise ValueError(f"No camera directories found in {image_root}")
    camera = camera or ("third_views" if dataset == "robotwin" else cameras[0])
    if camera not in cameras:
        raise ValueError(f"Unknown camera {camera!r}; available cameras: {', '.join(cameras)}")
    if dataset == "robotwin" and camera not in ROBOTWIN_TCP_DIRS:
        raise ValueError("RoboTwin Stage1 requires --camera third_views or head_view")
    rate = metadata.get("fps", metadata.get("frequency_hz", FRAME_RATE))
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Episode fps/frequency_hz must be finite and positive")
    count_key = "num_frames" if dataset == "robotwin" else "frame_count"
    count = metadata.get(count_key)
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"Expected integer {count_key} >= 2 in {path / 'metadata.json'}")
    paths = collect_rgb_paths(image_root / camera, max_frames=0, start_frame=0)
    if len(paths) != count:
        raise ValueError(f"Camera {camera}: metadata declares {count} frames, found {len(paths)}")
    if [_frame_index(p) for p in paths] != list(range(count)):
        raise ValueError("Episode RGB frame indices must be contiguous and start at zero")
    with Image.open(paths[0]) as image:
        width, height = image.size
        allowed_sizes = {(320, 240)} if dataset == "robotwin" else {(320, 180), (320, 240)}
        if not rgb_only and (width, height) not in allowed_sizes:
            raise ValueError(f"Unexpected native RGB size for {dataset}: {image.size}: {paths[0]}")
    width_key, height_key = ("image_width", "image_height") if dataset == "robotwin" else ("width", "height")
    if metadata.get(width_key, width) != width or metadata.get(height_key, height) != height:
        raise ValueError("Episode metadata image size disagrees with RGB")
    # Centered reflect padding preserves the native image and fills a patch-14 grid.
    pad_width, pad_height = (-width) % 14, (-height) % 14
    padding = (pad_width // 2, pad_width - pad_width // 2,
               pad_height // 2, pad_height - pad_height // 2)
    time_path = path / "timestamps.npy"
    timestamps = None
    if time_path.is_file():
        timestamps = np.load(time_path, allow_pickle=False).astype(np.float64)
        if (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
                or timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0)):
            raise ValueError(f"Expected {count} finite, nonnegative, increasing timestamps: {time_path}")
    k_path = path / ("intrinsics" if dataset == "robotwin" else "intrinsic") / f"{camera}.npy"
    k = None if rgb_only else np.load(k_path, allow_pickle=False).astype(np.float32)
    if k is not None and (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError(f"Expected finite pinhole intrinsic matrix [3,3]: {k_path}")
    return Episode(path, camera, paths, k, dataset=dataset, frame_rate=float(rate),
                   timestamps=timestamps, metadata=metadata,
                   source_width=width, source_height=height, padding=padding,
                   arm_names=("left", "right") if dataset == "robotwin" else ("tcp",))


def validate_query(point: Any, source: str = "TCP query", *, image_width=SOURCE_WIDTH,
                   image_height=SOURCE_HEIGHT, num_arms=1) -> np.ndarray:
    point = np.asarray(point, dtype=np.float32)
    if point.shape == (num_arms * 2,):
        point = point.reshape(num_arms, 2)
    if point.shape != (num_arms, 2) or not np.isfinite(point).all():
        raise ValueError(f"{source}: expected {num_arms} finite (x,y) point(s), got {point.tolist()}")
    inside = ((point[:, 0] >= 0) & (point[:, 0] < image_width)
              & (point[:, 1] >= 0) & (point[:, 1] < image_height))
    if not inside.all():
        raise ValueError(f"{source}: TCP pixels {point.tolist()} outside {image_width}x{image_height} image")
    return point


def project_query(position: Any, intrinsics: np.ndarray, source: str, *,
                  image_width=SOURCE_WIDTH, image_height=SOURCE_HEIGHT, num_arms=1) -> np.ndarray:
    xyz = np.asarray(position, dtype=np.float32)
    if xyz.shape != (num_arms, 3) or not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 0):
        raise ValueError(f"{source}: expected finite camera TCP [{num_arms},3] with z > 0, got {xyz.tolist()}")
    homogeneous = xyz @ intrinsics.T
    return validate_query(homogeneous[:, :2] / homogeneous[:, 2:3], source,
                          image_width=image_width, image_height=image_height, num_arms=num_arms)


def load_views(paths: list[Path], *, image_width=SOURCE_WIDTH, image_height=SOURCE_HEIGHT, padding=(1, 1, 1, 1)) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    views, colors = [], []
    for path in paths:
        with Image.open(path) as image:
            if image.size != (image_width, image_height):
                raise ValueError(f"Expected native {image_width}x{image_height} RGB, got {image.size}: {path}")
            rgb = np.array(image.convert("RGB"))
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(127.5).sub_(1)
        tensor = F.pad(tensor, padding, mode="reflect")
        views.append({"img": tensor[None], "true_shape": torch.tensor([[image_height + padding[2] + padding[3], image_width + padding[0] + padding[1]]]),
                      "idx": _frame_index(path), "instance": str(path)})
        colors.append(rgb)
    return views, colors


def restore_inference_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Strict restoration, allowing only omitted copies of genuinely tied weights."""
    state = dict(state)
    for name in ("position_mean", "position_std"):
        value = state.get(f"tcp_track_head.{name}")
        expected_shape = (model.num_arms, 3)
        if value is None or value.shape != expected_shape or not torch.isfinite(value).all():
            raise ValueError(f"TCP4D checkpoint requires finite tcp_track_head.{name} {expected_shape}")
        if name == "position_std" and torch.any(value <= 0):
            raise ValueError("TCP position_std must be positive")
    groups: dict[tuple, list[str]] = {}
    for name, tensor in model.state_dict().items():
        identity = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tensor.stride())
        groups.setdefault(identity, []).append(name)
    for names in groups.values():
        source = next((name for name in names if name in state), None)
        if source is not None:
            for name in names:
                state.setdefault(name, state[source])
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Incompatible TCP4D Stage1 checkpoint: {error}") from error


def load_model(checkpoint: Path, device: torch.device):
    from tcp4d.models.tcp4d.tcp4d import TCP4D

    path = weight_file(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        saved = load_file(str(path), device="cpu")
    else:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved = saved.get("state_dict", saved)
    state = {}
    for name, value in saved.items():
        normalized = _remove_training_prefix(name)
        if normalized in state:
            raise ValueError(f"Duplicate checkpoint parameter: {normalized}")
        state[normalized] = value
    del saved
    arm = state.get("tcp_visual_query_encoder.arm_embedding")
    offsets = state.get("tcp_visual_query_encoder.offset_embedding")
    if arm is None or arm.ndim != 2 or arm.shape[0] not in (1, 2):
        raise ValueError("Expected a trained one-arm or two-arm TCP4D Stage1 checkpoint")
    if offsets is None or offsets.ndim != 2:
        raise ValueError("Checkpoint is missing TCP visual query offset embeddings")
    window = math.isqrt(offsets.shape[0])
    if window < 1 or window % 2 != 1 or window * window != offsets.shape[0]:
        raise ValueError("Invalid TCP query neighborhood in checkpoint")
    model = TCP4D(num_arms=arm.shape[0], tcp_query_window_size=window)
    restore_inference_weights(model, state)
    del state
    gc.collect()
    return model.to(device).eval()

def ground_truth_paths(episode):
    if episode.dataset == "robotwin":
        directory = episode.path / ROBOTWIN_TCP_DIRS[episode.camera]
        return [directory / f"{arm}_state.npy" for arm in episode.arm_names]
    return [episode.path / "TCP" / episode.camera / "state.npy"]


def initial_query(episode, frame, point, from_gt):
    bounds = {"image_width": episode.source_width, "image_height": episode.source_height,
              "num_arms": episode.num_arms}
    if not from_gt:
        return validate_query(point, **bounds), "manual pixels" if episode.num_arms == 2 else "manual pixel"
    # Read only the chosen observation; never use future TCP labels for queries.
    count = len(episode.image_paths)
    positions = []
    for path in ground_truth_paths(episode):
        states = np.load(path, mmap_mode="r", allow_pickle=False)
        if states.shape not in {(count, 7), (count, 1, 7)}:
            raise ValueError(f"Expected TCP states [T,7] or [T,1,7], got {states.shape}: {path}")
        positions.append(np.asarray(states[frame]).reshape(7)[:3])
    return project_query(np.stack(positions), episode.intrinsics, "initial GT", **bounds), "initial GT"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Trained Stage1 weights for DROID, UR5, or RoboTwin; config and normalization JSON files are not required.',
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Single episode directory with RGB and metadata.json; prepared tcp4d_rgb_v1 inputs require no calibration.',
    )
    parser.add_argument(
        '--camera', '--view',
        help='Camera ID or name; defaults to third_views for RoboTwin and the first sorted camera otherwise.',
    )
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument(
        '--tcp-query-point',
        nargs=2,
        type=float,
        metavar=('X', 'Y'),
        help='Initial TCP pixel coordinates X Y in the original, unpadded image (px); mutually exclusive with --query-from-gt.',
    )
    query.add_argument(
        '--tcp-query-points',
        nargs=4,
        type=float,
        metavar=('LX', 'LY', 'RX', 'RY'),
        help='Initial RoboTwin TCP pixels in the unpadded image, ordered left arm then right arm.',
    )
    query.add_argument(
        '--query-from-gt',
        action='store_true',
        help='Initialize by projecting only the starting frame GT TCP; subsequent windows use predicted TCP positions without reading future GT.',
    )
    query.add_argument(
        '--interactive',
        action='store_true',
        help='Open the English interface to select the TCP pixel(s) and run inference with a 3D viewer.',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='Zero-based frame index at which to start inference (default: 0).',
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='Maximum number of frames to process from the starting frame; 0 uses all remaining frames, otherwise must be at least 2 (default: 0).',
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=9,
        help='Number of frames per sliding window, from 2 to 18; adjacent windows share one frame (default: 9).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/stage1_predictions.json'),
        help='Output JSON path for predictions, including per-frame TCP, gripper, coordinate frame, units, and timing information.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Inference device, such as cuda, cuda:0, or cpu; auto prefers an available GPU (default: auto).',
    )
    parser.add_argument(
        '--dtype',
        choices=('auto', 'float32', 'float16', 'bfloat16'),
        default='auto',
        help='Inference precision; auto prefers bf16 on CUDA, falls back to fp16, and uses float32 on CPU.',
    )
    parser.add_argument('--visualize', action='store_true', help='Open the 3D geometry and TCP viewer after inference.')
    parser.add_argument('--geometry-output', type=Path, help='Save predicted camera intrinsics, world-to-camera poses and per-frame metric depth to this directory.')
    parser.add_argument('--confidence-percentile', type=float, default=2.5, help='Filter out this percentage of low-confidence points (0-99; default: 2.5).')
    parser.add_argument('--max-points', type=int, default=100_000, help='Maximum cached points per frame; 0 keeps all points (default: 100000).')
    parser.add_argument('--point-size', type=float, default=0.003, help='Initial viewer point size (default: 0.003).')
    parser.add_argument('--host', default='127.0.0.1', help='Viser server bind address (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8020, help='Viser server port (default: 8020).')
    parser.add_argument('--ui-host', default='127.0.0.1', help='Interactive interface bind address (default: 127.0.0.1).')
    parser.add_argument('--ui-port', type=int, default=7860, help='Interactive interface port (default: 7860).')
    return parser


def validate_args(args):
    if not 2 <= args.window_size <= 18:
        raise ValueError('--window-size must be between 2 and 18')
    if not math.isfinite(args.confidence_percentile) or not 0 <= args.confidence_percentile <= 99:
        raise ValueError('--confidence-percentile must be in [0,99]')
    if args.max_points < 0:
        raise ValueError('--max-points cannot be negative')
    if not math.isfinite(args.point_size) or not 1e-5 <= args.point_size <= 0.01:
        raise ValueError('--point-size must be in [0.00001,0.01]')
    if not 1 <= args.port <= 65535 or not 1 <= args.ui_port <= 65535:
        raise ValueError('--port and --ui-port must be in [1,65535]')
    if args.interactive and args.port == args.ui_port:
        raise ValueError('--port and --ui-port must differ in interactive mode')


def run_inference(args, episode, paths, point, source, model, device, dtype,
                  *, keep_geometry=False, progress_callback=None):
    """Run shared single-arm or dual-arm windows for CLI and interactive inference."""
    if model.num_arms != episode.num_arms:
        raise ValueError(f"Checkpoint has {model.num_arms} arm(s), but {episode.dataset} requires {episode.num_arms}")
    if keep_geometry:
        from tcp4d.stage1_visualization import prepare_frame_point_clouds
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    clouds = [] if keep_geometry else None
    geometry_output = getattr(args, 'geometry_output', None)
    camera_records, intrinsic_records = [], []
    if geometry_output is not None:
        geometry_output = Path(geometry_output)
        (geometry_output / 'depth').mkdir(parents=True, exist_ok=True)
    windows = build_sliding_windows(len(paths), args.window_size)
    for wi, (start, end) in enumerate(windows):
        message = f'Inferring window {wi + 1}/{len(windows)}: frames {_frame_index(paths[start])}-{_frame_index(paths[end - 1])}'
        if progress_callback:
            progress_callback(wi / len(windows), message)
        else:
            print(message, flush=True)
        views, colors = load_views(paths[start:end], image_width=episode.source_width,
                                   image_height=episode.source_height, padding=episode.padding)
        views = [{**v, 'img': v['img'].to(device)} for v in views]
        context = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
        with torch.inference_mode(), context:
            pred = model(
                views, force_no_output_conversion=True, inference_track=False,
                decode_camera=episode.dataset != "robotwin", decode_motion=False, decode_tcp=True,
                tcp_query_points=torch.from_numpy(
                    point + np.asarray([episode.padding[0], episode.padding[2]], dtype=np.float32)
                )[None].to(device),
                return_aux_pyramid=False, ref_view_strategy='first',
            )
        def array(key):
            return pred[key][0].detach().float().cpu().numpy()
        position, rotation = array('tcp_position'), array('tcp_rotation')
        gripper = pred['tcp_gripper_logit'][0].detach().float().sigmoid().cpu().numpy()
        confidence = array('tcp_confidence')
        for value in [position, rotation, gripper, confidence]:
            if not np.isfinite(value).all():
                raise ValueError('Non-finite TCP prediction')
        camera_intrinsics = camera_extrinsics = None
        if geometry_output is not None or episode.intrinsics is None:
            from tcp4d.stage1_geometry import decode_camera_geometry
            camera_intrinsics, camera_extrinsics = decode_camera_geometry(pred, episode)
        if keep_geometry:
            from dataclasses import replace
            geometry_episode = (replace(episode, intrinsics=camera_intrinsics)
                                if episode.intrinsics is None else episode)
            window_clouds = prepare_frame_point_clouds(
                pred, colors, geometry_episode, max_points_per_frame=args.max_points, seed=start,
            )
            # Keep the previous window's prediction for the shared boundary frame.
            clouds.extend(window_clouds[int(wi > 0):])
        for slot in range(int(wi > 0), end - start):
            index = _frame_index(paths[start + slot])
            if geometry_output is not None:
                left, _, top, _ = episode.padding
                depth = pred['depth'][0, slot].detach().float().cpu().numpy()
                depth = depth[top:top + episode.source_height, left:left + episode.source_width]
                if depth.shape != (episode.source_height, episode.source_width):
                    raise ValueError('Predicted depth size disagrees with RGB')
                depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
                if not np.any(depth > 0):
                    raise ValueError(f'Frame {index} has no valid predicted depth')
                np.save(geometry_output / 'depth' / f'{index:06d}.npy', depth)
                intrinsic_records.append(camera_intrinsics[slot])
                camera_records.append(camera_extrinsics[slot])
            record = {'frame_index': index, 'time_seconds': episode.time_seconds(index)}
            arms = {}
            for arm_index, arm_name in enumerate(episode.arm_names):
                arms[arm_name] = {
                    'xyz_m': position[slot, arm_index].tolist(),
                    'rotation_matrix': rotation[slot, arm_index].tolist(),
                    'rpy_rad': matrix_to_rpy(rotation[slot, arm_index]).tolist(),
                    'gripper_open': float(gripper[slot, arm_index]),
                    'confidence': float(confidence[slot, arm_index]),
                }
            # Retain the existing flat JSON schema for single-arm consumers.
            record.update(arms['tcp'] if episode.num_arms == 1 else {'arms': arms})
            records.append(record)
        if end < len(paths):
            query_intrinsics = episode.intrinsics if episode.intrinsics is not None else camera_intrinsics[-1]
            point = project_query(position[-1], query_intrinsics, 'previous-window prediction',
                                  image_width=episode.source_width, image_height=episode.source_height,
                                  num_arms=episode.num_arms)
        del pred, views, colors
    result = {
        'schema_version': 1, 'stage': 1, 'dataset': episode.dataset, 'camera': episode.camera,
        'frame_rate_hz': episode.frame_rate,
        'source_size_wh': [episode.source_width, episode.source_height],
        'padding_lrtb': list(episode.padding),
        'coordinate_frame': 'OpenCV camera (+x right, +y down, +z forward)',
        'position_unit': 'meter', 'rotation_unit': 'radian',
        'gripper': 'continuous [0,1], 1=open',
        'initial_query_source': source, 'window_query_source': 'previous prediction',
        'num_frames': len(records), 'episode_num_frames': len(episode.image_paths),
        'complete': args.start_frame == 0 and len(records) == len(episode.image_paths),
        'frames': records,
        'peak_cuda_memory_mib': torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None,
    }
    if episode.num_arms == 2:
        result.update(num_arms=2, arm_names=list(episode.arm_names))
    if geometry_output is not None:
        np.save(geometry_output / 'intrinsics.npy', np.asarray(intrinsic_records, dtype=np.float32))
        np.save(geometry_output / 'extrinsics.npy', np.asarray(camera_records, dtype=np.float32))
        result['geometry'] = dict(source='TCP4D', directory=str(geometry_output.resolve()),
                                  depth_unit='meter', extrinsics_transform='world_to_camera')
    write_json_atomic(result, args.output)
    if progress_callback:
        progress_callback(1.0, 'Sliding-window inference complete')
    print(f'Saved {len(records)} frames to {args.output}', flush=True)
    return result, clouds


def start_viewer(args, episode, paths, result, clouds):
    from tcp4d.stage1_visualization import start_visualization
    records = result['frames']
    per_arm = [[r] if episode.num_arms == 1 else [r['arms'][name] for name in episode.arm_names]
               for r in records]
    tcp = {key: np.asarray([[arm[field] for arm in frame] for frame in per_arm], dtype=np.float32)
           for key, field in [('position', 'xyz_m'), ('rotation', 'rotation_matrix'),
                              ('gripper', 'gripper_open'), ('confidence', 'confidence')]}
    return start_visualization(
        clouds, tcp, [r['frame_index'] for r in records], paths,
        host=args.host, port=args.port, point_size=args.point_size,
        confidence_percentile=args.confidence_percentile,
        reference_extrinsic_w2c=np.eye(4), frame_rate=episode.frame_rate,
        arm_names=episode.arm_names,
    )


def render_query_overlay(image, points, arm_names=("tcp",)):
    """Mark one TCP or the left/right TCPs without resizing the source image."""
    if not points:
        return image.copy()
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    radius = 6
    for index, (x, y) in enumerate(points):
        color = ARM_COLORS[index]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        draw.line((x - radius - 2, y, x + radius + 2, y), fill=color, width=2)
        draw.line((x, y - radius - 2, x, y + radius + 2), fill=color, width=2)
        label = 'TCP' if arm_names[index] == 'tcp' else arm_names[index][0].upper()
        draw.text((min(x + 9, image.shape[1] - 24), max(y - 15, 0)), label, fill=color)
    return np.asarray(canvas)


def start_interactive_page(args, episode, paths, device, dtype):
    try:
        import gradio as gr
    except ImportError as error:
        raise ImportError("gradio is required for --interactive; run: pip install -e '.[visualization]'") from error
    from tcp4d.stage1_visualization import stop_visualization

    with Image.open(paths[0]) as image:
        base_image = np.asarray(image.convert('RGB')).copy()
    runtime = {'model': None, 'viser': None}
    runtime_lock = threading.Lock()
    dual_arm = episode.num_arms == 2
    prompt = ('Click the left-arm TCP, then the right-arm TCP in the starting frame.' if dual_arm
              else 'Click the TCP in the starting frame, or use its ground-truth projection.')

    def selection(points, source):
        ready = len(points) == episode.num_arms
        if ready:
            status = 'Both TCPs selected. Ready to run inference.' if dual_arm else f'TCP selected at ({points[0][0]:.1f}, {points[0][1]:.1f}). Ready to run inference.'
        elif points:
            status = 'Left TCP selected. Now click the right-arm TCP.'
        else:
            status = prompt
        return render_query_overlay(base_image, points, episode.arm_names), points, source, status, gr.update(interactive=ready)

    def select_point(points, event: gr.SelectData):
        point = validate_query(event.index, 'interactive click', image_width=episode.source_width,
                               image_height=episode.source_height)
        updated = point.tolist()
        if dual_arm:
            updated = [*points, *updated] if len(points) < episode.num_arms else updated
        return selection(updated, 'interactive starting-frame clicks')
    select_point.__annotations__['event'] = gr.SelectData

    def reset():
        return selection([], 'interactive starting-frame click')

    def use_ground_truth():
        try:
            point, source = initial_query(episode, args.start_frame, None, True)
        except (OSError, ValueError) as error:
            raise gr.Error(f'Cannot load starting-frame GT TCP: {error}') from error
        return selection(point.tolist(), source)

    def run(points, source, progress=gr.Progress()):
        point = validate_query(points, source, image_width=episode.source_width,
                               image_height=episode.source_height, num_arms=episode.num_arms)
        with runtime_lock:
            stop_visualization(runtime['viser'])
            runtime['viser'] = None
            progress(0.02, desc='Loading the TCP4D Stage1 model')
            if runtime['model'] is None:
                runtime['model'] = load_model(args.checkpoint, device)
            result, clouds = run_inference(
                args, episode, paths, point, source, runtime['model'], device, dtype,
                keep_geometry=True,
                progress_callback=lambda fraction, message: progress(0.08 + 0.78 * fraction, desc=message),
            )
            progress(0.90, desc='Starting the 3D viewer')
            runtime['viser'] = start_viewer(args, episode, paths, result, clouds)
            display_host = '127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host
            url = f'http://{display_host}:{runtime["viser"].get_port()}'
            safe_url = html.escape(url, quote=True)
            viewer_html = (
                f'<a href="{safe_url}" target="_blank" rel="noopener">Open the 3D viewer in a new tab</a>'
                f'<iframe src="{safe_url}" title="TCP4D Stage1 Viewer" '
                'style="width:100%;height:720px;border:1px solid #ddd;border-radius:8px" allowfullscreen></iframe>'
            )
            progress(1.0, desc='Complete')
            return result, f'Saved {result["num_frames"]} frames to `{args.output}`. [Open the viewer]({url}).', viewer_html

    with gr.Blocks(title='TCP4D Stage1 Inference') as demo:
        gr.Markdown(
            ('# Dual-Arm TCP4D Inference\n' if dual_arm else '# Single-Arm TCP4D Inference\n') +
            f'Dataset: `{episode.dataset}` | Camera: `{episode.camera}` | '
            f'Starting frame: `{args.start_frame}` | Frames: `{len(paths)}`\n\n'
            f'Select {episode.num_arms} TCP pixel(s), then run sliding-window inference. '
            'The 3D viewer shows geometry and TCP poses in the current OpenCV camera frame.'
        )
        selected_points = gr.State([])
        query_source = gr.State('interactive starting-frame click')
        with gr.Row():
            with gr.Column():
                query_image = gr.Image(value=base_image, label='Starting frame: click left TCP, then right TCP' if dual_arm else 'Starting frame: click the TCP',
                                       type='numpy', interactive=False, format='png', buttons=[])
                selection_status = gr.Markdown(prompt)
                with gr.Row():
                    gt_button = gr.Button('Use starting-frame GT TCP', visible=all(path.is_file() for path in ground_truth_paths(episode)))
                    reset_button = gr.Button('Reset selection')
                    infer_button = gr.Button('Run inference', variant='primary', interactive=False)
            with gr.Column():
                inference_status = gr.Markdown('Inference has not started.')
                result_json = gr.JSON(label='TCP predictions (JSON)')
        gr.Markdown('## 3D Geometry and TCP Playback')
        viewer = gr.HTML('<p>Run inference to display frame-by-frame geometry and TCP poses.</p>')
        outputs = (query_image, selected_points, query_source, selection_status, infer_button)
        query_image.select(select_point, inputs=selected_points, outputs=outputs, queue=False)
        gt_button.click(use_ground_truth, outputs=outputs, queue=False)
        reset_button.click(reset, outputs=outputs, queue=False)
        infer_button.click(run, inputs=(selected_points, query_source),
                           outputs=(result_json, inference_status, viewer), concurrency_limit=1)
    display_host = '127.0.0.1' if args.ui_host in {'0.0.0.0', '::'} else args.ui_host
    print(f'Interactive TCP selection: http://{display_host}:{args.ui_port}', flush=True)
    try:
        demo.queue(default_concurrency_limit=1).launch(server_name=args.ui_host,
                                                     server_port=args.ui_port, show_error=True, footer_links=[])
    finally:
        stop_visualization(runtime['viser'])
        runtime['model'] = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.visualize or args.interactive:
        import importlib.util
        required = ['viser', 'cv2'] + (['gradio'] if args.interactive else [])
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            parser.error(f"Missing visualization dependencies: {', '.join(missing)}; run: pip install -e '.[visualization]'")
    episode = load_episode(args.input, args.camera)
    paths = episode.clip(args.start_frame, args.max_frames)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.interactive:
        start_interactive_page(args, episode, paths, device, dtype)
        return
    manual_points = args.tcp_query_points if args.tcp_query_points is not None else args.tcp_query_point
    point, source = initial_query(episode, args.start_frame, manual_points, args.query_from_gt)
    model = load_model(args.checkpoint, device)
    try:
        result, clouds = run_inference(args, episode, paths, point, source, model, device, dtype,
                                       keep_geometry=args.visualize)
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if args.visualize:
        from tcp4d.stage1_visualization import stop_visualization
        server = start_viewer(args, episode, paths, result, clouds)
        display_host = '127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host
        print(f'Visualization: http://{display_host}:{server.get_port()}', flush=True)
        print('Press Ctrl+C to stop.', flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_visualization(server)


if __name__ == '__main__':
    main()
