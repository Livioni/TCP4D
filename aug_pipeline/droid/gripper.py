"""Map binary gripper commands to visible object-width apertures."""
import json
from pathlib import Path
import numpy as np

TABLES = json.loads(Path(__file__).with_name('gripper.json').read_text())
_FALLBACK_TABLE = {
    'state_inverted': False, 'finger_thickness_m': 0.0,
    'max_gap_m': 0.1, 'gap_table': [[0.0, 0.0], [1.0, 0.1]],
}

def interpolate(table: list, x: float) -> float:
    xs = np.array([p[0] for p in table])
    ys = np.array([p[1] for p in table])
    return float(np.interp(x, xs, ys))


def invert(table: list, gap: float) -> float:
    xs = np.array([p[0] for p in table])
    ys = np.array([p[1] for p in table])
    return float(np.interp(gap, ys, xs))


def _table(robot: str) -> dict:
    return TABLES.get(robot) or _FALLBACK_TABLE


def state_to_gap(robot: str, state_column: float) -> float:
    """Recorded state column -> raw fingertip gap (metres)."""
    entry = _table(robot)
    column = 1.0 - state_column if entry["state_inverted"] else state_column
    return interpolate(entry["gap_table"], column)


def remap_command(robot: str, state_column: float, object_width_m: float) -> float:
    """State column -> config-space command, with the object width as a floor."""
    entry = _table(robot)
    gap = state_to_gap(robot, state_column)
    ensured = max(gap, object_width_m + entry["finger_thickness_m"])
    ensured = min(ensured, entry["max_gap_m"])
    return invert(entry["gap_table"], ensured)

