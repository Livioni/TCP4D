#!/usr/bin/env python3
"""Export rendered frames paired with the single target IK trajectory."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TCP_COLUMNS = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper_open"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_paths(frames_dir: Path, count: int) -> list[Path]:
    paths = sorted(Path(frames_dir).glob("*.png"))
    stems = [path.stem for path in paths]
    if any(not stem.isdecimal() for stem in stems):
        raise ValueError(f"{frames_dir}: expected numerically named PNG frames")
    if [int(stem) for stem in stems] != list(range(count)):
        raise ValueError(
            f"{frames_dir}: expected frames 0..{count - 1}, "
            f"found {len(paths)} ({stems[:3]} ... {stems[-3:]})"
        )
    return paths


def _tcp_array(prediction_json: Path, side: str, count: int) -> np.ndarray:
    document = _load_json(prediction_json)
    frames = document.get("frames")
    if not isinstance(frames, list) or len(frames) != count:
        raise ValueError(
            f"{prediction_json}: expected {count} frame entries, "
            f"found {len(frames) if isinstance(frames, list) else 'no'} frames"
        )
    values = []
    for index, frame in enumerate(frames):
        item = frame.get(side)
        if item is None:
            raise ValueError(f"{prediction_json}: frame {index} has no {side} TCP")
        values.append(list(item["xyz_m"]) + list(item["rpy_rad"]) + [float(item["gripper_open"])])
    return np.asarray(values, dtype=np.float32)


def export_arm_dataset(
    *,
    frames_dir: Path,
    ik_dir: Path,
    converted: Path,
    arm: str,
    dataset: str,
    task: str,
    episode_name: str,
    prediction_json: Path,
    output_root: Path,
    mask_method: str = "sam3",
) -> Path:
    frames_dir = Path(frames_dir)
    ik_dir = Path(ik_dir)
    converted = Path(converted)
    output_root = Path(output_root)

    ik_metadata = _load_json(ik_dir / "metadata.json")
    if ik_metadata.get("status") != "success":
        raise ValueError(f"{ik_dir}: IK status is {ik_metadata.get('status')!r}, not success")
    count = int(ik_metadata["frame_count"])
    if count <= 0:
        raise ValueError(f"{ik_dir}: invalid frame_count {count}")

    side = "left"
    if set(ik_metadata.get("arms") or {}) != {side}:
        raise ValueError(f"{ik_dir}: expected single-arm metadata; rerun Retarget")
    arm_metadata = (ik_metadata.get("arms") or {}).get(side)
    if not arm_metadata:
        raise ValueError(f"{ik_dir}: metadata has no arms.{side}")

    state_full = np.load(ik_dir / "robot_state.npy", allow_pickle=False)
    if state_full.ndim != 2 or state_full.shape[0] != count:
        raise ValueError(
            f"{ik_dir}: robot_state has {state_full.shape[0]} frames, expected {count}"
        )
    columns = [int(column) for column in arm_metadata.get("joint_columns", [])]
    gripper_column = int(arm_metadata.get("gripper_column", -1))
    joint_names = list(arm_metadata.get("active_joint_names", []))
    if len(columns) != len(joint_names):
        raise ValueError(
            f"{ik_dir}: {len(columns)} joint columns but {len(joint_names)} joint names"
        )
    if not columns or any(column < 0 or column >= state_full.shape[1] for column in columns):
        raise ValueError(f"{ik_dir}: invalid joint_columns {columns}")
    if gripper_column < 0 or gripper_column >= state_full.shape[1]:
        raise ValueError(f"{ik_dir}: invalid gripper_column {gripper_column}")

    states = np.concatenate(
        [state_full[:, columns], state_full[:, [gripper_column]]], axis=1
    ).astype(np.float32)
    actions = states.copy()
    if len(actions) > 1:
        actions[:-1] = states[1:]

    converted_metadata = _load_json(converted / "metadata.json")
    if int(converted_metadata.get("num_frames", -1)) != count:
        raise ValueError(
            f"{converted}: num_frames={converted_metadata.get('num_frames')} "
            f"does not match IK frame_count={count}"
        )
    intrinsics = np.load(converted / "intrinsics/third_views.npy", allow_pickle=False)
    extrinsics = np.load(converted / "extrinsics/third_views.npy", allow_pickle=False)
    if intrinsics.shape not in ((3, 3), (count, 3, 3)):
        raise ValueError(f"{converted}: expected intrinsics [3,3] or [T,3,3], got {intrinsics.shape}")
    extrinsics_constant = extrinsics.shape == (4, 4)
    if extrinsics_constant:
        extrinsics = np.broadcast_to(extrinsics, (count, 4, 4)).copy()
    if extrinsics.shape != (count, 4, 4):
        raise ValueError(
            f"{converted}: expected extrinsics [4,4] or [{count},4,4], got {extrinsics.shape}"
        )

    paths = _frame_paths(frames_dir, count)
    height = int(converted_metadata["image_height"])
    width = int(converted_metadata["image_width"])
    with Image.open(paths[0]) as image:
        if image.size != (width, height):
            raise ValueError(
                f"{paths[0]}: expected {width}x{height}, got {image.size[0]}x{image.size[1]}"
            )

    tcp = _tcp_array(prediction_json, side, count)
    timestamps = np.asarray(
        [float(frame.get("time_seconds", index / max(1.0, float(converted_metadata["frequency_hz"]))))
         for index, frame in enumerate(_load_json(prediction_json)["frames"])],
        dtype=np.float32,
    )
    if timestamps.shape != (count,):
        raise ValueError(f"{prediction_json}: expected {count} timestamps")

    destination = (output_root / dataset / (task or "_") / episode_name
                   / mask_method / arm)
    if destination.exists():
        shutil.rmtree(destination)
    image_dir = destination / "images"
    image_dir.mkdir(parents=True)

    for path in paths:
        shutil.copy2(path, image_dir / path.name)

    np.save(destination / "states.npy", states)
    np.save(destination / "actions.npy", actions)
    np.save(destination / "tcp.npy", tcp)
    np.save(destination / "timestamps.npy", timestamps)
    np.save(destination / "intrinsics.npy", intrinsics.astype(np.float32))
    np.save(destination / "extrinsics.npy", extrinsics.astype(np.float32))

    metadata = {
        "format": "droid_augmented_pairs_v1",
        "dataset": dataset,
        "task": task,
        "episode": episode_name,
        "arm": arm,
        "mask_method": mask_method,
        "embodiment": ik_metadata.get("embodiment"),
        "render_side": side,
        "num_frames": count,
        "frame_rate_hz": float(converted_metadata["frequency_hz"]),
        "image_height": height,
        "image_width": width,
        "rgb": {"dir": "images", "pattern": "%06d.png", "format": "PNG",
                "channel_order": "RGB"},
        "state": {
            "file": "states.npy",
            "shape": list(states.shape),
            "columns": joint_names + ["gripper_open"],
            "joint_columns": list(range(len(joint_names))),
            "gripper_column": len(joint_names),
            "units": ["rad"] * len(joint_names) + ["open_fraction_0_1"],
        },
        "action": {
            "file": "actions.npy",
            "shape": list(actions.shape),
            "convention": "action[t] = state[t+1]; the last frame repeats the final state",
        },
        "tcp": {
            "file": "tcp.npy",
            "columns": TCP_COLUMNS,
            "coordinate_frame": "OpenCV camera (+x right, +y down, +z forward)",
            "source": "TCP4D Stage1 prediction",
        },
        "timestamps": {"file": "timestamps.npy", "unit": "second"},
        "intrinsics": {"file": "intrinsics.npy", "shape": list(intrinsics.shape),
                       "source": converted_metadata.get('extrinsics', {}).get('source', 'input calibration')},
        "extrinsics": {"file": "extrinsics.npy", "shape": [count, 4, 4],
                       "transform": "world_to_camera",
                       "per_frame_constant": extrinsics_constant},
        "source_episode": converted_metadata.get("source_file"),
        "converted_episode": str(converted),
        "prediction_json": str(prediction_json),
        "ik_dir": str(ik_dir),
        "augmented_frames_source": str(frames_dir),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination
