#!/usr/bin/env python3
"""Screen target-base translations before a full IK retry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from .retarget import SPECS, solver_flags


def subset_frames(episode: Path, arm: str, stride: int, prediction_json: Path) -> list[int]:
    diagnostics = episode / "TCP_prediction_ik" / arm / "diagnostics.json"
    frames: list[int] = []
    metadata_path = episode / "TCP_prediction_ik" / arm / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for side in ("left",):
            frames.extend(int(v) for v in metadata.get("dropped_frames", {}).get(side, []))
    if diagnostics.is_file():
        document = json.loads(diagnostics.read_text(encoding="utf-8"))
        for item in document.get("frames", []):
            if not item.get("left", {}).get("success"):
                frames.append(int(item["frame_index"]))
    prediction = json.loads(Path(prediction_json).read_text(encoding="utf-8"))
    count = len(prediction["frames"])
    sampled = list(range(0, count, stride))
    chosen = sorted(set(frames[:60] + sampled))
    if len(chosen) > 80:
        chosen = sorted(set(frames[:40] + sampled[:40]))
    return chosen or sampled


def make_screen_episode(episode: Path, arm: str, stride: int,
                        prediction_json: Path) -> Path:
    indices = subset_frames(episode, arm, stride, prediction_json)
    source = json.loads(Path(prediction_json).read_text(encoding="utf-8"))
    extrinsics = np.load(episode / "extrinsics" / "third_views.npy", allow_pickle=False)
    if extrinsics.ndim == 2:
        extrinsics = np.broadcast_to(extrinsics, (len(source["frames"]), *extrinsics.shape))
    out = episode / "base_search" / f"{arm}_screen"
    (out / "extrinsics").mkdir(parents=True, exist_ok=True)
    (out / "TCP_prediction_ik").mkdir(parents=True, exist_ok=True)
    np.save(out / "extrinsics" / "third_views.npy", extrinsics[indices].astype(np.float32))
    frames = []
    for new_index, old_index in enumerate(indices):
        item = dict(source["frames"][old_index])
        item["frame_index"] = new_index
        item["time_seconds"] = float(source["frames"][old_index].get("time_seconds", old_index / source["frame_rate_hz"]))
        frames.append(item)
    document = dict(source)
    document["num_frames"] = len(frames)
    document["frames"] = frames
    (out / "tcp_episode.json").write_text(json.dumps(document), encoding="utf-8")
    metadata = {
        "format_version": "robotwin_4d_v1", "embodiment": "franka_robotiq",
        "num_frames": len(frames), "image_height": 180, "image_width": 320,
        "frequency_hz": source.get("frame_rate_hz", 15),
        "extrinsics": {"transform": "world_to_camera"},
        "depth": {"unit": "millimeter", "invalid_value": 0},
    }
    (out / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (out / "subset_frames.json").write_text(json.dumps(indices), encoding="utf-8")
    return out


def run_candidate(screen: Path, arm: str, offset: list[float], yaw: float,
                  seeds: int, device: str, joint_step: float) -> dict:
    tag = "off_" + "_".join(f"{v:+.0f}" for v in offset) + f"_yaw_{yaw:+.0f}"
    out = screen / "TCP_prediction_ik" / tag
    command = [
        sys.executable, "-m", "aug_pipeline.droid.solve", arm, str(screen),
        "--prediction-json", str(screen / "tcp_episode.json"),
        "--embodiments-root", os.environ["EMBODIMENTS_ROOT"],
        "--base-offset-mm", *[str(v) for v in offset],
        "--ik-seeds", str(seeds), "--max-joint-step-rad", str(joint_step),
        "--max-dropped-frames", "0",
        "--position-tolerance-mm", "5", "--rotation-tolerance-rad", "0.05",
        "--device", device, "--output-dir", str(out), "--overwrite",
        *solver_flags({**SPECS[arm], "yaw": yaw}),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    success = completed.returncode == 0 and (out / "robot_state.npy").is_file()
    counts = {}
    diagnostics = out / "diagnostics.json"
    if diagnostics.is_file():
        document = json.loads(diagnostics.read_text(encoding="utf-8"))
        for side in ("left",):
            frames = document.get("frames", [])
            counts[side] = sum(1 for f in frames if f.get(side, {}).get("success"))
        counts["frames"] = len(document.get("frames", []))
    return {"tag": tag, "offset": offset, "yaw": yaw, "success": success,
            "counts": counts, "tail": completed.stdout.strip().splitlines()[-3:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path)
    parser.add_argument("arm", choices=sorted(SPECS))
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--screen-joint-step-rad", type=float, default=6.283185307179586,
                        help="step limit used only for the strided screen "
                             "(disabled so non-consecutive frames are not rejected)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--range", type=float, default=100.0)
    parser.add_argument("--step", type=float, default=50.0)
    parser.add_argument("--prediction-json", type=Path, default=None,
                        help="TCP prediction JSON (default: <episode>/tcp_episode.json)")
    args = parser.parse_args()
    episode = args.episode.expanduser().resolve()
    prediction_json = (args.prediction_json or (episode / "tcp_episode.json")
                       ).expanduser().resolve()
    screen = make_screen_episode(episode, args.arm, args.stride, prediction_json)
    subset = json.loads((screen / "subset_frames.json").read_text())
    base = list(SPECS[args.arm]["offset"])
    base_yaw = SPECS[args.arm]["yaw"]
    values = sorted((float(v) for v in
                     np.arange(-args.range, args.range + 1e-9, args.step)), key=abs)

    results = []
    best = None
    # Coordinate search: dx, then dy, then dz.
    for axis in (0, 1, 2):
        candidates = []
        for value in values:
            offset = list(base)
            offset[axis] += float(value)
            result = run_candidate(screen, args.arm, offset, base_yaw, args.seeds,
                                   args.device, args.screen_joint_step_rad)
            result["axis"] = axis
            candidates.append(result)
            print(f"[axis {axis}] {result['tag']}: {'OK' if result['success'] else 'fail'} {result['counts']}", flush=True)
        chosen = next((r for r in candidates if r["success"]), None)
        chosen = chosen or max(candidates, key=lambda r: sum(r["counts"].values()))
        base = list(chosen["offset"])
        results.extend(candidates)
    # The coordinate-system orientation is fixed: only translate the base.
    # The best candidate above already uses the arm's fixed yaw.
    best = next((r for r in results if r["success"]), None)
    ranked = sorted(results, key=lambda r: (not r["success"], -sum(r["counts"].values())))
    report = {"arm": args.arm, "subset_frames": subset, "base": base,
              "best": best or ranked[0], "results": results}
    (screen / "search_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"best": report["best"], "subset": subset}, indent=2))
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())
