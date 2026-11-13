"""Causal RoboTwin V2 action generation with interactive per-window TCP trajectory playback."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import dataclass, replace
import json
import gc
import math
from pathlib import Path
import re
import threading
import time
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from tcp4d.inference import write_json_atomic, json_values
WIDTH, HEIGHT = 320, 240
PADDING = (1,1,6,6)
ARMS = ("left","right")
TCP_DIRS = {"third_views":"TCP_third","head_view":"TCP_head"}

@dataclass
class EpisodeInputs:
    path: Path
    view: str
    image_paths: list[Path]
    frame_indices: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray  # Selected frames, homogeneous OpenCV world-to-camera.
    frequency_hz: float
    instruction: str


def homogeneous(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-2:] == (4, 4):
        return value.copy()
    if value.shape[-2:] != (3, 4):
        raise ValueError("Camera extrinsics must end in [3,4] or [4,4]")
    result = np.zeros((*value.shape[:-2], 4, 4), dtype=np.float32)
    result[..., :3, :] = value
    result[..., 3, 3] = 1
    return result


def load_episode(input_path: Path, view: str, instruction: str | None = None, *, allow_empty_instruction=False) -> EpisodeInputs:
    path = input_path.expanduser().resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    rate = metadata.get("frequency_hz")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("metadata.json requires a finite positive frequency_hz")
    if instruction is None:
        texts = metadata.get("instructions", [])
        instruction = next((s.strip() for s in texts if isinstance(s, str) and s.strip()), "") if isinstance(texts, list) else ""
    if not instruction.strip() and not allow_empty_instruction:
        raise ValueError("Provide --instruction or a non-empty metadata instructions list")
    indexed = {}
    for image_path in (path / "images" / view).iterdir():
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not image_path.is_file():
            continue
        match = re.search(r"(\d+)$", image_path.stem)
        if match is None:
            raise ValueError(f"RGB filename needs a trailing frame number: {image_path}")
        index = int(match.group(1))
        if index in indexed:
            raise ValueError(f"Duplicate RGB frame number {index}")
        indexed[index] = image_path
    indices = np.array(sorted(indexed), dtype=np.int64)
    if not len(indices) or np.any(np.diff(indices) != 1):
        raise ValueError("Episode RGB frames must be non-empty and contiguous")
    intrinsics = np.load(path / "intrinsics" / f"{view}.npy", allow_pickle=False).astype(np.float32)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or np.linalg.det(intrinsics) == 0:
        raise ValueError("Expected finite, invertible intrinsics [3,3]")
    extrinsics = homogeneous(np.load(path / "extrinsics" / f"{view}.npy", allow_pickle=False))
    if extrinsics.ndim != 3 or len(extrinsics) <= indices[-1]:
        raise ValueError("Camera extrinsics do not cover all RGB frame indices")
    extrinsics = extrinsics[indices]
    if not np.isfinite(extrinsics).all() or np.any(np.abs(np.linalg.det(extrinsics)) < 1e-8):
        raise ValueError("Invalid camera extrinsics")
    return EpisodeInputs(path, view, [indexed[i] for i in indices], indices, intrinsics,
                         extrinsics, float(rate), instruction.strip())


def build_windows(num_frames: int, history_frames: int = 4, stride: int = 20) -> list[tuple[int, int]]:
    """Schedule causal observations at exact replanning intervals: 0, stride, ... ."""
    if history_frames < 2:
        raise ValueError("history_frames must be at least 2")
    if stride < 1:
        raise ValueError("Replanning interval must be positive")
    if num_frames < 1:
        raise ValueError("Need at least one RGB frame")
    return [(max(0, anchor-history_frames+1), anchor+1)
            for anchor in range(0, num_frames, stride)]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (WIDTH, HEIGHT):
            raise ValueError(f"Expected {WIDTH}x{HEIGHT} RGB, got {image.size}: {path}")
        return np.asarray(image).copy()


def prepare_images(paths: list[Path], intrinsics: np.ndarray, query_points: np.ndarray):
    """Return [1,K,3,252,322] images and padded-pixel calibration/queries."""
    rgb = np.stack([load_rgb(path) for path in paths])
    images = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255
    images = F.pad(images, PADDING, mode="reflect").mul_(2).sub_(1).unsqueeze(0)
    padded_k = torch.tensor(intrinsics, dtype=torch.float32).clone()
    padded_k[0, 2] += PADDING[0]
    padded_k[1, 2] += PADDING[2]
    queries = torch.from_numpy(validate_query_points(query_points)).unsqueeze(0)
    queries = queries + queries.new_tensor([PADDING[0], PADDING[2]])
    return images, padded_k.expand(1, len(paths), 3, 3).clone(), queries


def prepare_wrist_images(episode: EpisodeInputs, anchor_offset: int):
    """Use the actual source frame number, including non-zero episode starts."""
    from tcp4d.datasets.utils.wrist import load_wrist_images
    return load_wrist_images(episode.path, episode.frame_indices[anchor_offset]).unsqueeze(0)


def validate_query_points(points: Any) -> np.ndarray:
    value = np.asarray(points, dtype=np.float32)
    if value.shape != (2, 2) or not np.isfinite(value).all():
        raise ValueError("Select two finite pixel points, in left/right TCP order")
    if np.any(value < 0) or np.any(value[:, 0] >= WIDTH) or np.any(value[:, 1] >= HEIGHT):
        raise ValueError("TCP query points fall outside the original RGB image")
    return value.copy()


def project_queries(positions: np.ndarray, intrinsics: np.ndarray, *, allow_outside: bool = False) -> np.ndarray:
    xyz = np.asarray(positions, dtype=np.float32)
    if xyz.shape != (2, 3) or not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 1e-6):
        raise ValueError("Cannot propagate TCP queries: non-finite or behind-camera TCP")
    pixels = xyz @ intrinsics.T
    if np.any(np.abs(pixels[:, 2]) <= 1e-6):
        raise ValueError("Cannot project TCP queries: zero projection denominator")
    uv = pixels[:, :2] / pixels[:, 2:3]
    if not np.isfinite(uv).all():
        raise ValueError("Cannot project TCP queries: non-finite projection")
    return uv.copy() if allow_outside else validate_query_points(uv)


def resolve_device_dtype(device_name: str, dtype_name: str) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type not in {"cuda", "cpu"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Device unavailable or unsupported: {device}")
    if dtype_name == "auto":
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, dtype_name)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference requires float32")
    return device, dtype


def resolve_t5_resource(checkpoint: Path, configured: str, override: str | None = None):
    if override is not None:
        return override
    if Path(configured).is_absolute() and not Path(configured).exists():
        for candidate in [checkpoint / 't5-base', checkpoint.parent / 't5-base']:
            if (candidate / 'config.json').is_file():
                return str(candidate)
        print('The checkpoint T5 path is unavailable; using google-t5/t5-base. Override with --t5-model if needed.', flush=True)
        return 'google-t5/t5-base'
    return configured


def load_stage2_policy(checkpoint: Path, device: torch.device, *, t5_model: str | None = None,
                       overrides: dict | None = None):
    from tcp4d.models.tcp4d.tcp4d import TCP4D
    from tcp4d.models.tcp4d.tcp4d_action import TCP4DPolicy
    from tcp4d.stage2_checkpoint import inspect_stage2_checkpoint, restore_stage2_weights

    path, config = inspect_stage2_checkpoint(checkpoint, overrides)
    tcp4d = TCP4D(tcp_query_window_size=config["tcp_query_window_size"])
    policy = TCP4DPolicy(
        tcp4d, t5_model=resolve_t5_resource(path.parent, config["t5_model"], t5_model),
        text_max_length=config["text_max_length"], dim=config["action_dim"],
        depth=config["action_depth"], heads=config["action_heads"],
        condition_dim=config["condition_dim"], output_dim=config["action_output_dim"],
        dropout=config["action_dropout"], global_token_grid=config["global_token_grid"],
        gradient_checkpointing=config["action_gradient_checkpointing"],
        prediction_horizon=config["prediction_horizon"], wrist_pretrained=False,
        wrist_encoder_config=config["wrist_encoder_config"],
    )
    policy.decode_geometry = config["decode_geometry"]
    restore_stage2_weights(policy,path)
    return policy.eval().requires_grad_(False).to(device), config

def initial_query(episode, points, from_gt):
    if not from_gt: return validate_query_points(np.asarray(points).reshape(2,2))
    directory=episode.path/TCP_DIRS[episode.view]
    frame=int(episode.frame_indices[0])
    positions=np.stack([np.asarray(np.load(directory/f"{arm}_state.npy",mmap_mode="r")[frame,:3]) for arm in ARMS])
    return project_queries(positions,episode.intrinsics)

def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Stage2 model.safetensors file or checkpoint directory; config and normalization JSON files are optional.',
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Directory for a single RoboTwin episode containing main-view RGB, RGB from both wrists, calibration, actual robot_state.npy, and metadata.',
    )
    parser.add_argument(
        '--view',
        choices=tuple(TCP_DIRS),
        default='third_views',
        help='Main camera view used to locate RGB images, camera calibration, and initialization GT (default: third_views).',
    )
    parser.add_argument(
        '--instruction',
        help='Task language instruction; defaults to the first non-empty text in metadata.instructions.',
    )
    query=parser.add_mutually_exclusive_group(required=True)
    query.add_argument(
        '--tcp-query-points',
        nargs=4,
        type=float,
        metavar=('LX', 'LY', 'RX', 'RY'),
        help='Initial TCP pixel coordinates LX LY RX RY in the original, unpadded image (px), ordered left arm then right arm.',
    )
    query.add_argument(
        '--query-from-gt',
        action='store_true',
        help='Initialize using only the left/right GT TCP positions from the first observed frame; mutually exclusive with --tcp-query-points.',
    )
    query.add_argument('--interactive', action='store_true',
                       help='Open the English interface for left/right TCP selection and window trajectory playback.')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/stage2_predictions.json'),
        help='Output JSON path for future actions per window, including coordinate units, completion status, and window truncation information.',
    )
    parser.add_argument(
        '--window-stride', '--replan-interval',
        type=int,
        default=20,
        help='Replan every N observed frames/action steps; each inference predicts 32 action steps by default (default interval: 20).',
    )
    parser.add_argument(
        '--max-windows',
        type=int,
        help='Maximum number of inference windows for debugging; defaults to the full episode, with truncated results marked incomplete.',
    )
    parser.add_argument(
        '--sampling-steps',
        type=int,
        help='Number of Euler sampling steps for flow matching; defaults to the checkpoint configuration, typically 4.',
    )
    parser.add_argument(
        '--t5-model',
        help='Local directory or Hugging Face ID for the T5-base model/tokenizer; overrides the resource path in the checkpoint.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for action sampling (default: 42).',
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
    parser.add_argument('--visualize', action='store_true', help='Open the window trajectory viewer after CLI inference.')
    parser.add_argument('--host', default='127.0.0.1', help='Viser bind address (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8021, help='Viser window viewer port (default: 8021).')
    parser.add_argument('--ui-host', default='127.0.0.1', help='Interactive interface bind address (default: 127.0.0.1).')
    parser.add_argument('--ui-port', type=int, default=7861, help='Interactive interface port (default: 7861).')
    parser.add_argument('--point-size', type=float, default=0.003, help='Viewer point size (default: 0.003).')
    parser.add_argument('--fps', type=float, default=5.0, help='Initial window playback speed, from 0.25 to 30 FPS.')
    parser.add_argument('--confidence-percentile', type=float, default=2.5, help='Hide this percentage of low-confidence scene points (0-99).')
    parser.add_argument('--max-points', type=int, default=100_000, help='Maximum displayed points per window; 0 keeps all points.')
    parser.add_argument('--max-depth', type=float, default=3.0, help='Viewer depth limit in meters; 0 disables the limit.')
    parser.add_argument('--history-frames', type=int, help='Observed frames per window (default: optional config value or 4).')
    parser.add_argument('--prediction-horizon', type=int, help='Future TCP steps per window (default: optional config value or 32).')
    parser.add_argument('--action-heads', type=int, help='Attention head count for a custom checkpoint (default: optional config value or 32).')
    parser.add_argument('--global-token-grid', nargs=2, type=int, metavar=('H','W'), help='Global visual token grid for a custom checkpoint (default: 14 19).')
    return parser


def validate_args(args):
    if args.history_frames is not None and args.history_frames < 2:
        raise ValueError('--history-frames must be at least 2')
    if args.prediction_horizon is not None and not 1 <= args.prediction_horizon <= 1024:
        raise ValueError('--prediction-horizon must be in [1,1024]')
    if args.action_heads is not None and args.action_heads < 1:
        raise ValueError('--action-heads must be positive')
    if args.global_token_grid is not None and min(args.global_token_grid) < 1:
        raise ValueError('--global-token-grid must have two positive dimensions')
    if args.window_stride < 1:
        raise ValueError('--window-stride/--replan-interval must be positive')
    if args.max_windows is not None and args.max_windows < 1:
        raise ValueError('--max-windows must be positive')
    if args.sampling_steps is not None and args.sampling_steps < 1:
        raise ValueError('--sampling-steps must be positive')
    if not 1 <= args.port <= 65535 or not 1 <= args.ui_port <= 65535:
        raise ValueError('--port and --ui-port must be in [1,65535]')
    if args.interactive and args.port == args.ui_port:
        raise ValueError('--port and --ui-port must differ')
    if not math.isfinite(args.point_size) or args.point_size <= 0:
        raise ValueError('--point-size must be finite and positive')
    if not math.isfinite(args.fps) or not .25 <= args.fps <= 30:
        raise ValueError('--fps must be between 0.25 and 30')
    if not math.isfinite(args.confidence_percentile) or not 0 <= args.confidence_percentile <= 99:
        raise ValueError('--confidence-percentile must be in [0,99]')
    if args.max_points < 0 or not math.isfinite(args.max_depth) or args.max_depth < 0:
        raise ValueError('--max-points and --max-depth must be nonnegative')


def checkpoint_overrides(args):
    return {key:getattr(args,key,None) for key in
            ['history_frames','prediction_horizon','action_heads','global_token_grid']
            if getattr(args,key,None) is not None}


def planning_description(prediction_horizon, replan_interval):
    return f'Each inference predicts {prediction_horizon} action steps; replan every {replan_interval} steps.'


def propagate_replan_queries(episode, previous, next_start):
    """Initialize the next observed history without reading future TCP labels.

    Overlapping windows reuse recovered TCPs. Disjoint histories use the
    previous plan's action at the next history's first frame, transformed from
    the previous anchor camera into that newly observed frame's camera.
    """
    if next_start < previous['end']:
        positions = previous['history_position'][next_start-previous['start']]
        source = 'previous recovered TCP'
    else:
        step = next_start-(previous['end']-1)
        if not 1 <= step <= len(previous['action_position']):
            raise ValueError('The previous action horizon does not reach the next observation window')
        positions = previous['action_position'][step-1]
        transform = episode.extrinsics[next_start] @ np.linalg.inv(episode.extrinsics[previous['end']-1])
        positions = positions @ transform[:3,:3].T + transform[:3,3]
        source = f'previous predicted action step {step}'
    projected = project_queries(positions,episode.intrinsics,allow_outside=True)
    queries = np.clip(projected,[0,0],[WIDTH-1,HEIGHT-1]).astype(np.float32)
    return queries, bool(np.any(queries != projected)), source


def run_inference(args, episode, queries, policy, config, device, dtype, *,
                  query_source='manual pixels', keep_geometry=False, progress_callback=None):
    if args.window_stride > config['prediction_horizon']:
        raise ValueError('Replanning interval cannot exceed the action horizon; reduce --window-stride or increase --prediction-horizon')
    previous_geometry = policy.decode_geometry
    try:
        if keep_geometry:
            policy.decode_geometry = True
        return _run_windows(args, episode, queries, policy, config, device, dtype,
                            query_source=query_source, keep_geometry=keep_geometry,
                            progress_callback=progress_callback)
    finally:
        policy.decode_geometry = previous_geometry


def _run_windows(args, episode, queries, policy, config, device, dtype, *,
                 query_source, keep_geometry, progress_callback):
    from tcp4d.state_normalization import read_robot_states,validate_robot_states
    states=read_robot_states(episode.path)
    validate_robot_states(states)
    if len(states)<=episode.frame_indices[-1]: raise ValueError("robot_state.npy does not cover RGB frames")
    windows=build_windows(len(episode.image_paths),config["history_frames"],args.window_stride)
    print(planning_description(config['prediction_horizon'],args.window_stride),flush=True)
    scheduled=windows[:args.max_windows] if args.max_windows else windows
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    result={"schema_version":1,"stage":2,"dataset":"robotwin","num_arms":2,
        "coordinate_frame":"last observation OpenCV camera (+x right, +y down, +z forward)",
        "position_unit":"meter","rotation":"rotation_matrix","gripper":"1=open, 0=closed; score >=0 means open",
        "history_frames":config["history_frames"],"prediction_horizon":config["prediction_horizon"],
        "window_stride":args.window_stride,
        "instruction":episode.instruction,"initial_query_source":query_source,
        "planned_windows":len(windows),"complete":False,"windows":[]}
    playback = {"window_schedule": "causal", "window_stride": args.window_stride,
                "prediction_horizon": config["prediction_horizon"], "windows": [], "_geometry": []} if keep_geometry else None
    previous = None
    current_query_source = query_source
    try:
        for wi,(start,end) in enumerate(scheduled):
            if previous is not None and start != previous['start']:
                queries, clipped, current_query_source = propagate_replan_queries(episode,previous,start)
                result['windows'][-1]['next_query_clipped'] = clipped
            message = f"Window {wi + 1}/{len(scheduled)}: anchor frame {episode.frame_indices[end-1]}"
            if progress_callback:
                progress_callback(wi / len(scheduled), message)
            else:
                print(message, flush=True)
            # Early windows repeat the oldest available RGB; all current wrist
            # images and joint states come from this window's observed anchor.
            offsets=[start]*max(0,config["history_frames"]-(end-start))+list(range(start,end))
            images,intrinsics,points=prepare_images([episode.image_paths[i] for i in offsets],episode.intrinsics,queries)
            images,intrinsics,points=[v.to(device) for v in (images,intrinsics,points)]
            state=torch.from_numpy(np.asarray(states[episode.frame_indices[end-1]]).copy())[None].to(device)
            wrists=prepare_wrist_images(episode,end-1).to(device)
            context=torch.autocast(device_type=device.type,dtype=dtype) if dtype!=torch.float32 else contextlib.nullcontext()
            with torch.inference_mode(),context:
                reconstruction,features=policy.reconstruct(images,points)
                condition=policy.make_condition(images,intrinsics,[episode.instruction],reconstruction,features,
                    current_robot_state=state,wrist_images=wrists)
                actions=policy.sample_condition(condition,steps=args.sampling_steps or config["sampling_steps"],
                    generator=torch.Generator(device=device).manual_seed(args.seed+wi))
            if not bool(actions["success"][0]): raise ValueError(f"Non-finite action prediction at window {wi}")
            record={"window_index":wi,"anchor_frame":int(episode.frame_indices[end-1]),
                "input_frame_indices":episode.frame_indices[offsets],"query_points_px":queries.copy(),
                "query_source":current_query_source,
                **{key:value[0] for key,value in actions.items()}}
            result["windows"].append(json_values(record))
            if keep_geometry:
                from tcp4d.stage2_visualization import cache_window
                view, geometry = cache_window(episode, reconstruction, result["windows"][-1],
                                               start, end, max(0, config["history_frames"]-(end-start)))
                playback["windows"].append(view)
                playback["_geometry"].append(geometry)
            if wi+1 < len(scheduled):
                padding = max(0,config['history_frames']-(end-start))
                previous = {
                    'start':start, 'end':end,
                    'history_position':reconstruction['tcp_position'][0,padding:].detach().float().cpu().numpy().copy(),
                    'action_position':actions['action_position'][0].detach().float().cpu().numpy().copy(),
                }
        result["complete"]=len(scheduled)==len(windows)
        if not result["complete"]: result["stop_reason"]="max_windows limit"
    except Exception as error:
        result["error"]=f"{type(error).__name__}: {error}"
        raise
    finally:
        result["peak_cuda_memory_mib"]=torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else None
        write_json_atomic(result,args.output)
    print(f"Saved {len(result['windows'])} windows to {args.output}")
    if progress_callback:
        progress_callback(1.0, 'Window inference complete')
    return result, playback


def create_viewer(args, playback, episode, *, host=None):
    from tcp4d.stage2_visualization import Stage2Viewer
    return Stage2Viewer(playback, episode, host=args.host if host is None else host, port=args.port,
                        point_size=args.point_size, fps=args.fps,
                        confidence_percentile=args.confidence_percentile, max_points=args.max_points,
                        max_depth=args.max_depth or None)


def draw_queries(rgb, points):
    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    for index, (x, y) in enumerate(points):
        color = ((255, 100, 55), (45, 175, 255))[index]
        draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        draw.text((min(x+7, WIDTH-35), max(y-8, 0)),ARMS[index],fill=color)
    return np.asarray(image)


def start_interactive_page(args, episode, device, dtype):
    import gradio as gr
    from starlette.middleware import Middleware
    from tcp4d.viser_proxy import ViserProxy
    rgb = load_rgb(episode.image_paths[0])
    runtime = {'policy': None, 'config': None, 'viewer': None}
    runtime_lock = threading.Lock()

    def viewer_port():
        viewer = runtime['viewer']
        return viewer.server.get_port() if viewer is not None else None

    def selection(points, source):
        ready = len(points) == 2
        status = ('Both TCPs selected. Ready to predict window trajectories.' if ready else
                  'Left TCP selected. Now click the right-arm TCP.' if points else
                  'Click the left-arm TCP, then the right-arm TCP in the first frame.')
        return draw_queries(rgb, points), points, source, status, gr.update(interactive=ready)

    def select(points, event: gr.SelectData):
        click = np.asarray(event.index, dtype=np.float32)
        if click.shape != (2,) or not np.isfinite(click).all() or not (0 <= click[0] < WIDTH and 0 <= click[1] < HEIGHT):
            raise gr.Error('Select a pixel inside the original 320x240 image.')
        updated = [] if len(points) == 2 else list(points)
        updated.append(click.tolist())
        return selection(updated, 'interactive first-frame clicks')
    select.__annotations__['event'] = gr.SelectData

    def use_truth():
        try:
            points = initial_query(episode, None, True)
        except (OSError, ValueError) as error:
            raise gr.Error(f'Cannot project the first-frame GT TCPs: {error}') from error
        return selection(points.tolist(), 'initial GT')

    def reset():
        return selection([], 'interactive first-frame clicks')

    def run(points, source, instruction, progress=gr.Progress()):
        queries = validate_query_points(points)
        if not instruction.strip():
            raise gr.Error('Enter a non-empty task instruction.')
        active_episode = replace(episode, instruction=instruction.strip())
        with runtime_lock:
            if runtime['viewer'] is not None:
                runtime['viewer'].close()
                runtime['viewer'] = None
            progress(.02, desc='Loading the TCP4D Stage2 policy')
            if runtime['policy'] is None:
                runtime['policy'], runtime['config'] = load_stage2_policy(args.checkpoint, device, t5_model=args.t5_model, overrides=checkpoint_overrides(args))
            result, playback = run_inference(
                args, active_episode, queries, runtime['policy'], runtime['config'], device, dtype,
                query_source=source, keep_geometry=True,
                progress_callback=lambda fraction, message: progress(.08+.78*fraction,desc=message))
            progress(.92, desc='Starting per-window TCP trajectory playback')
            runtime['viewer'] = create_viewer(args, playback, active_episode, host='127.0.0.1')
            viewer_html = ('<iframe src="stage2-viewer/" title="TCP4D Stage2 Window Trajectories" '
                           'style="width:100%;height:740px;border:1px solid #ddd;border-radius:8px" '
                           'allowfullscreen></iframe>')
            progress(1,desc='Complete')
            summary = planning_description(result['prediction_horizon'],args.window_stride)
            return f'Saved {len(result["windows"])} windows to `{args.output}`. {summary} View the trajectories below.', viewer_html, str(args.output)

    with gr.Blocks(title='TCP4D Stage2 Window Inference') as demo:
        gr.Markdown('# Stage2 · Per-Window TCP Trajectories\n'
                    'Select the left TCP, then the right TCP in the first observed frame. '
                    'Each window shows recovered history and predicted future trajectories in its anchor camera frame. '
                    f'Each inference predicts {args.prediction_horizon or 32} action steps by default; '
                    f'replan every {args.window_stride} steps.')
        points, source = gr.State([]), gr.State('interactive first-frame clicks')
        with gr.Row():
            with gr.Column():
                image = gr.Image(rgb,type='numpy',interactive=False,format='png',buttons=[],
                                 label='First frame: click left TCP, then right TCP')
                selection_status = gr.Markdown('Select the left-arm TCP first.')
                instruction = gr.Textbox(value=episode.instruction,label='Task instruction',lines=2)
                with gr.Row():
                    has_truth = all((episode.path/TCP_DIRS[episode.view]/f'{arm}_state.npy').is_file() for arm in ARMS)
                    truth_button = gr.Button('Use first-frame GT TCPs',visible=has_truth)
                    reset_button = gr.Button('Reset selection')
                run_button = gr.Button('Predict window trajectories',variant='primary',interactive=False)
            with gr.Column():
                status = gr.Markdown('Inference has not started.')
                output_file = gr.File(label='Window predictions (JSON)',interactive=False)
        viewer = gr.HTML('<p>Run inference to display geometry and TCP trajectories for each window.</p>')
        outputs = [image,points,source,selection_status,run_button]
        image.select(select,points,outputs,queue=False)
        truth_button.click(use_truth,outputs=outputs,queue=False)
        reset_button.click(reset,outputs=outputs,queue=False)
        run_button.click(run,[points,source,instruction],[status,viewer,output_file],concurrency_limit=1)
    host = '127.0.0.1' if args.ui_host in {'0.0.0.0','::'} else args.ui_host
    print(f'Interactive Stage2: http://{host}:{args.ui_port}',flush=True)
    try:
        demo.queue(default_concurrency_limit=1).launch(server_name=args.ui_host,server_port=args.ui_port,
                                                     show_error=True,footer_links=[],
                                                     app_kwargs={'middleware': [
                                                         Middleware(ViserProxy, get_port=viewer_port)]})
    finally:
        if runtime['viewer'] is not None:
            runtime['viewer'].close()
        runtime['policy'] = None
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.interactive or args.visualize:
        import importlib.util
        required = ['viser','cv2'] + (['gradio'] if args.interactive else [])
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            parser.error(f"Missing visualization dependencies: {', '.join(missing)}; run: pip install -e '.[visualization]'")
    episode = load_episode(args.input,args.view,args.instruction,allow_empty_instruction=args.interactive)
    device,dtype = resolve_device_dtype(args.device,args.dtype)
    if args.interactive:
        start_interactive_page(args,episode,device,dtype)
        return
    queries = initial_query(episode,args.tcp_query_points,args.query_from_gt)
    policy,config = load_stage2_policy(args.checkpoint,device,t5_model=args.t5_model,overrides=checkpoint_overrides(args))
    try:
        result,playback = run_inference(args,episode,queries,policy,config,device,dtype,
                                       query_source='initial GT' if args.query_from_gt else 'manual pixels',
                                       keep_geometry=args.visualize)
    finally:
        del policy
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if args.visualize:
        viewer = create_viewer(args,playback,episode)
        host = '127.0.0.1' if args.host in {'0.0.0.0','::'} else args.host
        print(f'Window trajectories: http://{host}:{viewer.server.get_port()}',flush=True)
        print('Press Ctrl+C to stop.',flush=True)
        try:
            while True:
                time.sleep(.5)
        except KeyboardInterrupt:
            pass
        finally:
            viewer.close()


if __name__ == '__main__':
    main()
