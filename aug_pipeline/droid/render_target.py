"""Map the single target arm to its visible grasp aperture."""
from __future__ import annotations
import numpy as np
from .robot import load_target
from .gripper import remap_command

SOURCE_FULL_GAP_M = 0.085

def ramp_column(values: np.ndarray, ramp: int) -> np.ndarray:
    """Replace each one-frame step in a command column with a linear ramp."""
    out = values.astype(np.float64).copy()
    if ramp <= 1:
        return out
    for index in [i for i in range(1, len(values)) if values[i] != values[i - 1]]:
        start = max(0, index - ramp // 2)
        end = min(len(values), start + ramp)
        out[start:end] = np.linspace(values[index - 1], values[index], end - start)
    return out


def render_target(ik_dir, episode, embodiments_root):
    """Load the IK result, then replace ONLY the in-memory render aperture."""
    target = load_target(ik_dir, episode, embodiments_root)
    openness = np.load(episode.path / 'gripper_openness.npy', allow_pickle=False)
    if openness.shape != (episode.count,) or not np.isfinite(openness).all():
        raise ValueError('Gripper openness must cover every frame')
    width_m = None
    for side, arm in target.arms.items():
        column = arm["gripper_column"]
        binary = target.state[:, column].astype(np.float64)
        closed = binary < 0.5
        if not closed.any():
            continue
        if width_m is None:
            # The median of the stable closed interval estimates the grasp
            # opening (= object width); the threshold-edge frames would bias a
            # high percentile toward "still open".
            width_m = float(np.clip(np.median(openness[closed])
                                    * SOURCE_FULL_GAP_M, 0.005, 0.080))
            print(f"render aperture: object width {width_m * 1000:.1f} mm "
                  f"for {target.name} (labels stay binary)")
        render = np.array([remap_command(target.name, float(value), width_m)
                           for value in binary], dtype=np.float64)
        target.state[:, column] = ramp_column(render, 6)
    return target
