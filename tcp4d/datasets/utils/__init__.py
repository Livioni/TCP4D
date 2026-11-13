"""Ground-truth preprocessing utilities shared by TCP4D datasets."""

from .geometry import (
    compute_gt_ray_map,
    geometry_to_first_camera,
    normalize_geometry_to_first_camera,
    resize_ray_valid_mask,
)

__all__ = [
    "compute_gt_ray_map",
    "geometry_to_first_camera",
    "normalize_geometry_to_first_camera",
    "resize_ray_valid_mask",
]
