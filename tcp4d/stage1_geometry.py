"""Decode the camera head trained with absolute metric DROID supervision."""
from __future__ import annotations

import numpy as np
import torch

from tcp4d.models.tcp4d.utils.transform import pose_encoding_to_extri_intri


def decode_camera_geometry(predictions, episode):
    left, right, top, bottom = episode.padding
    height = episode.source_height + top + bottom
    width = episode.source_width + left + right
    encoding = predictions['pose_enc'].float()
    if not torch.isfinite(encoding).all() or torch.any(encoding[..., 3:7].norm(dim=-1) < 1e-8):
        raise ValueError('TCP4D returned an invalid camera pose')
    # The absolute-camera loss supervises translation/quaternion as camera-to-world.
    camera_to_world, intrinsics = pose_encoding_to_extri_intri(encoding, (height, width))
    bottom_row = camera_to_world.new_tensor([0, 0, 0, 1]).expand(*camera_to_world.shape[:-2], 1, 4)
    extrinsics = torch.linalg.inv(torch.cat((camera_to_world, bottom_row), dim=-2))
    k = intrinsics[0].detach().cpu().numpy().copy()
    e = extrinsics[0].detach().cpu().numpy()
    # Camera decoding operates on reflect-padded pixels; consumers use RGB pixels.
    k[:, 0, 2] -= left
    k[:, 1, 2] -= top
    if (not np.isfinite(k).all() or not np.isfinite(e).all()
            or np.any(k[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError('TCP4D returned invalid camera geometry')
    return k, e
