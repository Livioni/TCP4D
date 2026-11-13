#!/usr/bin/env python3
"""Retarget the DROID Robotiq TCP frame to each target robot."""

from __future__ import annotations

import math
import sys

import numpy as np


from aug_pipeline.droid.ik import _common as common
from aug_pipeline.droid.ik._kinematics import fk_link

from aug_pipeline.config import ASSETS as SLUGS

ROBOTIQ_TO_TARGET = np.array([[0.0, 0.0, -1.0],
                              [0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0]])


def extract_flags(argv):
    """Pull the wrapper-only flags out of argv, leaving normal solver args."""
    clean, yaws, align, jaw_flip, identity, tcp_rotation = [], [], False, False, False, True
    tcp_jaw_flip = False
    roll_deg = 0.0
    rotation_tolerance_rad = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--rotation-tolerance-rad":
            index += 1
            rotation_tolerance_rad = float(argv[index]); index += 1; continue
        if arg == "--base-yaw-deg":
            index += 1
            values = []
            while index < len(argv) and not argv[index].startswith("--"):
                values.append(float(argv[index]))
                index += 1
            yaws = values
            continue
        if arg == "--align-gripper-geometry":
            align = True; index += 1; continue
        if arg == "--align-gripper-jaw-flip":
            jaw_flip = True; index += 1; continue
        if arg == "--tcp-jaw-flip":
            tcp_jaw_flip = True; index += 1; continue
        if arg == "--tcp-roll-deg":
            index += 1
            roll_deg = float(argv[index]); index += 1; continue
        if arg == "--identity-tcp-rotation":
            identity = True; index += 1; continue
        if arg == "--no-tcp-frame-rotation":
            tcp_rotation = False; index += 1; continue
        clean.append(arg); index += 1
    if rotation_tolerance_rad is not None:
        if not math.isfinite(rotation_tolerance_rad) or rotation_tolerance_rad <= 0:
            raise SystemExit("--rotation-tolerance-rad must be positive and finite")
        clean += ["--rotation-tolerance-deg",
                  repr(math.degrees(rotation_tolerance_rad))]
    return (clean, yaws, align, jaw_flip, identity, tcp_rotation, tcp_jaw_flip,
            roll_deg)


def apply_yaw(robot, yaws):
    if not yaws:
        return
    angles = np.deg2rad(np.asarray(yaws, dtype=float))
    if angles.size == 1:
        angles = {side: angles[0] for side in robot.world_from_root}
    else:
        raise SystemExit("--base-yaw-deg takes one number (degrees)")
    for side, angle in angles.items():
        if not angle:
            continue
        rotation = np.eye(4)
        rotation[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0],
                            [np.sin(angle), np.cos(angle), 0.0],
                            [0.0, 0.0, 1.0]]
        robot.world_from_root[side] = np.asarray(robot.world_from_root[side], dtype=float) @ rotation


def apply_alignment(robot, jaw_flip):
    fa = np.array([1.0, 0.0, 0.0])
    ja = np.array([0.0, -1.0 if jaw_flip else 1.0, 0.0])
    for side, geometry in robot.geometries.items():
        spec = robot.config["gripper_name"][0]
        model = robot.model
        values = {j.name: 0.0 for j in model.joints_by_name.values()
                  if j.joint_type != "fixed"}
        values.update({spec["base"]: 0.03, spec["mimic"][0][0]: 0.03})
        links = [model.joints_by_name[j].child
                 for j in [spec["base"]] + [m[0] for m in spec["mimic"]]]
        ee = fk_link(model, geometry.ee_link, values)
        points = np.stack([fk_link(model, link, values)[:3, 3] for link in links])
        f = ee[:3, :3].T @ (points.mean(axis=0) - ee[:3, 3]); f /= np.linalg.norm(f)
        j = ee[:3, :3].T @ (points[0] - points[1]); j /= np.linalg.norm(j)
        j = j - (j @ f) * f; j /= np.linalg.norm(j)
        basis = np.column_stack([f, j, np.cross(f, j)])
        source = np.column_stack([fa, ja, np.cross(fa, ja)])
        rotation = basis @ source.T
        old = np.asarray(geometry.link6_from_tcp, dtype=float).copy()
        new = np.eye(4); new[:3, :3] = rotation; new[:3, 3] = old[:3, 3]
        object.__setattr__(geometry, "link6_from_tcp", new)
        print(f"{side}: gripper-geometry alignment applied", flush=True)


def apply_identity(robot):
    for geometry in robot.geometries.values():
        old = np.asarray(geometry.link6_from_tcp, dtype=float).copy()
        new = np.eye(4); new[:3, 3] = old[:3, 3]
        object.__setattr__(geometry, "link6_from_tcp", new)
    print("identity TCP rotation applied", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    slug = sys.argv[1]
    if slug not in SLUGS:
        raise SystemExit(f"unknown slug {slug!r}; expected one of {sorted(SLUGS)}")
    (clean, yaws, align, jaw_flip, identity, tcp_rotation, tcp_jaw_flip,
     roll_deg) = extract_flags(sys.argv[2:])

    original_load = common.load_embodiment

    def load_with_compat(name, embodiments_root):
        robot = original_load(name, embodiments_root)
        apply_yaw(robot, yaws)
        if align:
            apply_alignment(robot, jaw_flip)
        if identity:
            apply_identity(robot)
        return robot

    common.load_embodiment = load_with_compat

    if tcp_rotation:
        correction = np.eye(4)
        correction[:3, :3] = ROBOTIQ_TO_TARGET
        if tcp_jaw_flip:
            # 180 deg about the target approach (+X): an equivalent grasp for a
            # parallel gripper, selecting the other wrist branch.
            correction[:3, :3] = correction[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            print("TCP jaw flip applied (180 deg about the approach axis)", flush=True)
        if roll_deg:
            angle = np.deg2rad(roll_deg)
            c, s = np.cos(angle), np.sin(angle)
            correction[:3, :3] = correction[:3, :3] @ np.array(
                [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            print(f"TCP roll {roll_deg:+.1f} deg about the approach applied",
                  flush=True)
        original_build = common.build_targets

        def build_with_correction(robot, prediction, extrinsics):
            for side in prediction.camera_tcp:
                prediction.camera_tcp[side] = prediction.camera_tcp[side] @ correction
            return original_build(robot, prediction, extrinsics)

        common.build_targets = build_with_correction
        print("DROID TCP-frame correction applied (Robotiq +Z approach -> target +X)",
              flush=True)

    common.main(SLUGS[slug], clean)


if __name__ == "__main__":
    main()
