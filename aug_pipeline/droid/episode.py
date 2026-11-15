"""Load RGB frames and TCP4D-predicted camera geometry."""
from pathlib import Path

import numpy as np

from .robot import CAMERA, Episode, frame_paths, load_json, validate_transform


def load_episode(path, with_depth=True):
    path = Path(path).expanduser().resolve()
    meta = load_json(path / "metadata.json")
    if meta.get("format_version") != "tcp4d_rgb_v1":
        raise ValueError("Expected a prepared RGB episode")
    if meta.get("extrinsics", {}).get("transform") != "world_to_camera":
        raise ValueError("Episode extrinsics must be world_to_camera in OpenCV coordinates")
    count = int(meta["num_frames"])
    height, width = int(meta["image_height"]), int(meta["image_width"])
    if min(count, height, width) <= 0:
        raise ValueError("Frame count and image dimensions must be positive")
    rgb = frame_paths(path / "images" / CAMERA, count)
    depth = None
    if with_depth:
        spec = meta.get("depth", {})
        if spec.get('format') != 'NPY' or spec.get('unit') != 'meter':
            raise ValueError('Expected TCP4D metric depth arrays')
        depth = sorted((path / 'depths' / CAMERA).glob('*.npy'))
        if [p.stem for p in depth] != [p.stem for p in rgb]:
            raise ValueError('Predicted depth must cover every RGB frame')
    k = np.load(path / "intrinsics" / f"{CAMERA}.npy", allow_pickle=False).astype(float)
    if k.shape == (3, 3):
        k = np.broadcast_to(k, (count, 3, 3))
    if (k.shape != (count, 3, 3) or not np.isfinite(k).all()
            or np.any(k[:, (0, 1), (0, 1)] <= 0)
            or not np.allclose(k[:, 2], [0, 0, 1]) or not np.allclose(k[:, 1, 0], 0)):
        raise ValueError("Expected finite pinhole intrinsics [3,3] or [T,3,3]")
    e = np.load(path / "extrinsics" / f"{CAMERA}.npy", allow_pickle=False).astype(float)
    if e.shape in ((3, 4), (4, 4)):
        e = np.broadcast_to(e, (count, *e.shape))
    if e.shape == (count, 3, 4):
        bottom = np.broadcast_to([0, 0, 0, 1], (count, 1, 4))
        e = np.concatenate((e, bottom), axis=1)
    if e.shape != (count, 4, 4):
        raise ValueError("Expected extrinsics [3,4], [4,4], [T,3,4] or [T,4,4]")
    return Episode(path, meta, rgb, depth, k,
                   validate_transform(e, "extrinsics"), height, width)
