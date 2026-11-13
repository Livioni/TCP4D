"""Shared headless inference helpers."""
from __future__ import annotations
import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch
TRAIN_MAX_FRAMES = 18

def _remove_training_prefix(key: str) -> str:
    for prefix in ("model.", "module."):
        if key.startswith(prefix):
            return _remove_training_prefix(key[len(prefix) :])
    return key


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested}) but is not available")
    return device


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        if device.type != "cuda":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError(f"{requested} inference is only supported on CUDA by this script")
    return dtype


def _frame_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(
            f"Cannot infer the RoboTwin frame index from image name {path.name!r}; "
            "pass --tcp-query-points or --tcp-query-points-file explicitly"
        )
    return int(match.group(1))


def matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Inverse of RoboTwin's Rz(yaw) @ Ry(pitch) @ Rx(roll) convention."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices [...,3,3], got {rotation.shape}")
    horizontal = np.sqrt(rotation[..., 0, 0] ** 2 + rotation[..., 1, 0] ** 2)
    singular = horizontal < 1e-7

    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = np.arctan2(-rotation[..., 2, 0], horizontal)
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    singular_roll = np.arctan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    roll = np.where(singular, singular_roll, roll)
    yaw = np.where(singular, 0.0, yaw)
    return np.stack((roll, pitch, yaw), axis=-1).astype(np.float32)


def build_sliding_windows(
    num_frames: int, window_size: int
) -> list[tuple[int, int]]:
    """Return half-open windows with exactly one shared boundary frame."""
    if num_frames < 2:
        raise ValueError("Sliding-window TCP inference requires at least 2 frames")
    if not 2 <= window_size <= TRAIN_MAX_FRAMES:
        raise ValueError(
            f"--window-size must be between 2 and {TRAIN_MAX_FRAMES}"
        )

    windows: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + window_size, num_frames)
        if end - start < 2:
            raise RuntimeError("Internal window construction produced one frame")
        windows.append((start, end))
        if end == num_frames:
            break
        start = end - 1
    return windows


def write_json_atomic(result: dict[str, Any], output_path: Path) -> Path:
    """Replace the destination only after a complete JSON file is durable."""
    destination = output_path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                result,
                temporary,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination.resolve()

def collect_rgb_paths(input_dir, max_frames=0, *, start_frame=0):
    paths = sorted((p for p in Path(input_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}), key=_frame_index)
    paths = paths[start_frame:]
    return paths[:max_frames] if max_frames else paths

def json_values(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_values(v) for key, v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_values(v) for v in value]
    return value
