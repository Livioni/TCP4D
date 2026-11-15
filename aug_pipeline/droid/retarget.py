"""Robot placements, strict IK execution and diagnostics."""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

STD_UR5 = [0, 650, -650]
STD_FRANKA = [0, 650, -750]
SPECS = {
    "ur5_wsg": {"offset": STD_UR5, "yaw": 0, "align": False, "tcp_jaw_flip": True},
    "franka_panda": {"offset": STD_FRANKA, "yaw": -90, "align": True, "tcp_jaw_flip": False},
    "jaco": {"offset": [0, 450, -750], "yaw": -90, "align": False,
             "tcp_jaw_flip": False, "tcp_roll_deg": 45},
    # validated with solve_droid.py on AUTOLab__Mon_Nov_27 (403/403, 0 drops)
    "sawyer": {"offset": [0, 450, -750], "yaw": -90, "align": False,
               "tcp_jaw_flip": False},
    "kinova_gen3": {"offset": [0, 650, -750], "yaw": -90, "align": False,
                    "tcp_jaw_flip": True},
}

def solver_flags(spec):
    flags = ['--base-yaw-deg', str(spec['yaw'])]
    for key, flag in [('align', '--align-gripper-geometry'),
                      ('align_jaw_flip', '--align-gripper-jaw-flip'),
                      ('tcp_jaw_flip', '--tcp-jaw-flip')]:
        if spec.get(key):
            flags.append(flag)
    if spec.get('tcp_roll_deg'):
        flags.extend(['--tcp-roll-deg', str(spec['tcp_roll_deg'])])
    return flags


def solve(episode, arm, spec, prediction_json):
    target = episode / 'TCP_prediction_ik' / arm
    command = [sys.executable, '-m', 'aug_pipeline.droid.solve', arm, str(episode),
        '--prediction-json', str(prediction_json),
        '--embodiments-root', os.environ['EMBODIMENTS_ROOT'],
        '--base-offset-mm', *map(str, spec['offset']),
        '--ik-seeds', '256', '--max-dropped-frames', '0',
        '--position-tolerance-mm', '5', '--rotation-tolerance-rad', '0.05',
        '--device', 'cuda:0', '--output-dir', str(target), '--overwrite', *solver_flags(spec)]
    result = subprocess.run(command)
    return result.returncode == 0 and (target / 'robot_state.npy').is_file()


def search_base(episode, arm, prediction_json, stride=12):
    results = episode / 'base_search' / f'{arm}_screen/search_results.json'
    results.unlink(missing_ok=True)
    subprocess.run([sys.executable, '-m', 'aug_pipeline.droid.search', str(episode), arm,
                    '--stride', str(stride), '--prediction-json', str(prediction_json)])
    return json.loads(results.read_text()).get('best') if results.is_file() else None


def ik_metrics(episode: Path, arm: str, ik_name: str | None = None) -> dict:
    """Drops / jump frames / max step for the single target arm."""
    ik_dir = episode / "TCP_prediction_ik" / (ik_name or arm)
    metadata_path = ik_dir / "metadata.json"
    state_path = ik_dir / "robot_state.npy"
    if not metadata_path.is_file() or not state_path.is_file():
        return {}
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "success" or set(metadata.get("arms", {})) != {"left"}:
        return {}
    state = np.load(state_path, allow_pickle=False)
    frames = int(metadata.get("frame_count", len(state)))
    if frames <= 0:
        return {}
    sides = {}
    for side in metadata["arms"]:
        columns = [int(c) for c in
                   (metadata.get("arms", {}).get(side, {}) or {}).get("joint_columns", [])
                   if 0 <= int(c) < state.shape[1]]
        if not columns:
            continue
        steps = np.abs(np.diff(state[:, columns], axis=0)).max(axis=1)
        steps = np.nan_to_num(steps, nan=1e9, posinf=1e9)
        sides[side] = {
            "drops": len(metadata.get("dropped_frames", {}).get(side, [])),
            "jumps": int((steps > 0.5).sum()) if steps.size else 0,
            "max_step": float(steps.max()) if steps.size else 0.0,
        }
    if not sides:
        return {}
    chosen = "left"
    current = sides[chosen]
    return {"frames": frames, "side": chosen, "sides": sides,
            "drops": current["drops"], "jumps": current["jumps"],
            "max_step": current["max_step"],
            "legal": max(0.0, (frames - current["drops"]) / frames)}


def failure_reasons(episode: Path, arm: str, ik_name: str | None = None) -> dict:
    """Aggregate per-side failure reasons from diagnostics.json."""
    path = episode / "TCP_prediction_ik" / (ik_name or arm) / "diagnostics.json"
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    counts = {}
    for frame in document.get("frames", []) or []:
        for side in ("left",):
            data = frame.get(side) or {}
            values = list(data.get("pre_repairfailure_reasons")
                          or data.get("failure_reasons") or [])
            if data.get("constraint_feasible") is False and "model_constraints" not in values:
                values.append("model_constraints")
            for value in values:
                key = f"{side}/{value}"
                counts[key] = counts.get(key, 0) + 1
    return counts

