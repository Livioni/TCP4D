"""Validated camera frames and target IK results."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from . import urdf
from .ik._kinematics import gripper_positions

CAMERA = 'third_views'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(result, dict):
        raise ValueError(f'{path}: expected an object')
    return result


def validate_transform(value, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{label}: expected finite 4x4 transforms')
    rotation = value[..., :3, :3]
    if (not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError(f'{label}: invalid rigid transform')
    return value


def frame_paths(directory, count):
    paths = list(directory.glob('*.png'))
    if len(paths) != count or any(not p.stem.isdecimal() for p in paths):
        raise ValueError(f'{directory}: expected {count} numerically named PNG frames')
    paths.sort(key=lambda p: int(p.stem))
    if [int(p.stem) for p in paths] != list(range(count)):
        raise ValueError(f'{directory}: frame numbers must be contiguous from zero')
    return paths


@dataclass
class Episode:
    path: Path
    metadata: dict
    rgb_paths: list[Path]
    depth_paths: list[Path] | None
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    height: int
    width: int

    @property
    def count(self):
        return len(self.rgb_paths)

    def read_frame(self, index):
        with Image.open(self.rgb_paths[index]) as image:
            rgb = np.array(image.convert('RGB'))
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f'{self.rgb_paths[index]}: image size disagrees with metadata')
        depth = None
        if self.depth_paths is not None:
            depth = np.load(self.depth_paths[index], allow_pickle=False)
            if (depth.shape != (self.height, self.width) or depth.dtype.kind != 'f'
                    or not np.isfinite(depth).all() or np.any(depth < 0)):
                raise ValueError(f'{self.depth_paths[index]}: expected finite nonnegative metric depth')
        return rgb, depth


@dataclass
class Robot:
    name: str
    urdf_path: Path
    config: dict
    state: np.ndarray
    arms: dict
    locks: dict
    model: urdf.RobotModel

    @property
    def slug(self):
        return self.name.lower().replace('-', '_')

    def values(self, side, index, include_locks=True):
        arm = self.arms[side]
        values = dict(self.locks.get(side, {})) if include_locks else {}
        values.update(zip(arm['active_joint_names'], self.state[index, arm['joint_columns']]))
        values.update(gripper_positions(self.model, self.config, 0,
                                       self.state[index, arm['gripper_column']]))
        return values


def validate_robot(robot, count, binary_gripper):
    state = robot.state
    if state.ndim != 2 or len(state) != count or not np.isfinite(state).all():
        raise ValueError('Robot state must be a finite [T,D] array matching the episode')
    if set(robot.arms) != {'left'}:
        raise ValueError('Expected single-arm metadata in the left slot; rerun Retarget')
    used = []
    for side, arm in robot.arms.items():
        names, columns = arm['active_joint_names'], arm['joint_columns']
        if len(names) != len(columns) or len(names) != len(set(names)):
            raise ValueError(f'{side}: invalid active joint mapping')
        cols = [*columns, arm['gripper_column']]
        if any(not isinstance(c, int) or c < 0 or c >= state.shape[1] for c in cols):
            raise ValueError(f'{side}: state column out of bounds')
        used.extend(cols)
        for name in names:
            if name not in robot.model.joints_by_name:
                raise ValueError(f'{side}: unknown joint {name}')
            joint = robot.model.joints_by_name[name]
            q = state[:, columns[names.index(name)]]
            if ((joint.lower is not None and np.any(q < joint.lower - 1e-4))
                    or (joint.upper is not None and np.any(q > joint.upper + 1e-4))):
                raise ValueError(f'{side}: {name} exceeds URDF position limits')
        root = np.asarray(arm['world_from_root'])
        if root.shape != (4, 4):
            raise ValueError(f'{side}: world_from_root must be 4x4')
        validate_transform(root, f'{side} root')
        g = state[:, arm['gripper_column']]
        if np.any((g < 0) | (g > 1)) or (binary_gripper and not np.isin(g, [0, 1]).all()):
            raise ValueError(f'{side}: invalid normalized gripper states')
        values = robot.values(side, 0)
        if not set(values).issubset(robot.model.joints_by_name) or not np.isfinite(list(values.values())).all():
            raise ValueError(f'{side}: invalid locked/gripper joint mapping')
    if sorted(used) != list(range(state.shape[1])):
        raise ValueError('Arm mappings must cover every state column exactly once')
    return robot


def load_target(ik_dir, episode, embodiments_root):
    ik_dir = Path(ik_dir).expanduser().resolve()
    meta = load_json(ik_dir / 'metadata.json')
    if (meta.get('format') != 'robotwin_closed_tcp_trajectory_v3'
            or meta.get('schema_version') != 3 or meta.get('status') != 'success'
            or meta['output'].get('state_file') != 'robot_state.npy'
            or meta['output'].get('role') != 'solution'):
        raise ValueError('Expected a successful v3 IK trajectory')
    if meta['frame_count'] != episode.count or meta.get('camera') != CAMERA:
        raise ValueError('IK frame count or camera does not match this episode')
    if meta['inputs'].get('extrinsics_semantics') != 'world_to_camera':
        raise ValueError('IK extrinsics must use world_to_camera')
    if sha256(episode.path / 'extrinsics' / f'{CAMERA}.npy') != meta['inputs']['extrinsics_sha256']:
        raise ValueError('Camera extrinsics differ from those used for IK')
    # Fixed cameras can have identical hashes across episodes. Verify the local
    # prediction too when available, without depending on stale absolute paths.
    prediction = episode.path / Path(meta['inputs'].get('prediction_json', 'tcp_episode.json')).name
    expected = meta['inputs'].get('prediction_sha256')
    if expected and prediction.is_file() and sha256(prediction) != expected:
        raise ValueError('Prediction hash differs: IK may belong to a different episode')
    name = meta['embodiment']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Invalid embodiment name')
    config = meta['robot_config_snapshot']
    local = Path(embodiments_root).expanduser().resolve() / name / config['urdf_path']
    recorded = Path(meta['inputs']['urdf']).expanduser()
    candidates = [local.resolve(), recorded.resolve()]
    path = next((p for p in candidates if p.is_file() and sha256(p) == meta['inputs']['urdf_sha256']), None)
    if path is None:
        raise ValueError(f'No matching URDF for {name}; missing resource or URDF hash changed')
    state = np.load(ik_dir / 'robot_state.npy', allow_pickle=False)
    if list(state.shape) != meta['output']['shape']:
        raise ValueError('IK state shape disagrees with metadata')
    locks = {}
    runtime = meta.get('runtime_robot_cfg', {})
    for side in meta['arms']:
        cfg = runtime.get(side, runtime)
        locks[side] = cfg.get('kinematics', {}).get('lock_joints') or {}
    robot = Robot(name, path, config, state, meta['arms'], locks, urdf.load_robot_model(str(path)))
    return validate_robot(robot, episode.count, True)

