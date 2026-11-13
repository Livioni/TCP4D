"""URDF parsing and rigid transforms used by IK and rendering."""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class RobotModel:
    root_link: str
    joints_by_name: dict[str, Joint]
    joints_by_child: dict[str, Joint]


def _vector(
    element: ET.Element | None, attribute: str, default: str
) -> tuple[float, float, float]:
    values = (element.get(attribute, default) if element is not None else default).split()
    if len(values) != 3:
        raise ValueError(f"Expected three values in {attribute!r}, got {values}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


@lru_cache(maxsize=None)
def load_robot_model(urdf_path: str) -> RobotModel:
    """Parse only the URDF fields needed for forward kinematics."""
    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    links = {element.get("name") for element in root.findall("link")}
    links.discard(None)
    joints_by_name: dict[str, Joint] = {}
    joints_by_child: dict[str, Joint] = {}

    for element in root.findall("joint"):
        name = element.get("name")
        joint_type = element.get("type")
        parent_element = element.find("parent")
        child_element = element.find("child")
        if not name or not joint_type or parent_element is None or child_element is None:
            raise ValueError(f"Malformed joint in {path}")
        parent = parent_element.get("link")
        child = child_element.get("link")
        if not parent or not child:
            raise ValueError(f"Joint {name} has no parent or child link")

        limit = element.find("limit")
        joint = Joint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin_xyz=_vector(element.find("origin"), "xyz", "0 0 0"),
            origin_rpy=_vector(element.find("origin"), "rpy", "0 0 0"),
            axis=_vector(element.find("axis"), "xyz", "1 0 0"),
            lower=(
                float(limit.get("lower"))
                if limit is not None and limit.get("lower")
                else None
            ),
            upper=(
                float(limit.get("upper"))
                if limit is not None and limit.get("upper")
                else None
            ),
        )
        joints_by_name[name] = joint
        if child in joints_by_child:
            raise ValueError(f"URDF link {child} has more than one parent joint")
        joints_by_child[child] = joint

    root_links = sorted(links - set(joints_by_child))
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link, found {root_links}")
    return RobotModel(root_links[0], joints_by_name, joints_by_child)


def rotation_matrix_from_rpy(rpy: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_matrix_from_axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("A movable URDF joint has a zero-length axis")
    x, y, z = direction / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    complement = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def joint_chain(model: RobotModel, target_link: str) -> list[Joint]:
    chain: list[Joint] = []
    current = target_link
    visited: set[str] = set()
    while current != model.root_link:
        if current in visited:
            raise ValueError(f"Cycle in URDF chain at {current}")
        visited.add(current)
        joint = model.joints_by_child.get(current)
        if joint is None:
            raise ValueError(f"{target_link} is not connected to URDF root {model.root_link}")
        chain.append(joint)
        current = joint.parent
    return list(reversed(chain))


def homogeneous_extrinsics(path: Path, frame_count: int) -> np.ndarray:
    extrinsics = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if extrinsics.shape == (3, 4) or extrinsics.shape == (4, 4):
        extrinsics = np.broadcast_to(extrinsics, (frame_count, *extrinsics.shape)).copy()
    if extrinsics.shape == (frame_count, 3, 4):
        bottom = np.broadcast_to([0.0, 0.0, 0.0, 1.0], (frame_count, 1, 4))
        extrinsics = np.concatenate((extrinsics, bottom), axis=1)
    if extrinsics.shape != (frame_count, 4, 4):
        raise ValueError(
            f"{path} has shape {extrinsics.shape}; expected [T,3,4] or "
            f"[T,4,4] for T={frame_count}"
        )
    if not np.all(np.isfinite(extrinsics)):
        raise ValueError(f"{path} contains non-finite values")
    if not np.allclose(extrinsics[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{path} has invalid homogeneous bottom rows")
    return extrinsics

