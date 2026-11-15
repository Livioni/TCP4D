"""Shared scheduling and metric geometry preprocessing."""
from __future__ import annotations
import math
from typing import Any
import torch
from torch.optim.lr_scheduler import LambdaLR
from tcp4d.datasets.utils import compute_gt_ray_map, geometry_to_first_camera, resize_ray_valid_mask
from tcp4d.checkpoint import save_checkpoint

def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min_factor: float,
) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return eta_min_factor + (1.0 - eta_min_factor) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, scale)


def distributed_scheduler_steps(
    warmup_steps: int,
    total_steps: int,
    *,
    num_processes: int,
    split_batches: bool,
) -> tuple[int, int]:
    """Match AcceleratedScheduler's internal stepping convention."""
    multiplier = 1 if split_batches else num_processes
    return warmup_steps * multiplier, total_steps * multiplier


def training_total_steps(
    steps_per_epoch: int,
    num_train_epochs: int | None,
    max_train_steps: int | None,
) -> int:
    """Resolve the cumulative optimizer-step target, allowing an unlimited epoch loop."""
    for name, value in (("num_train_epochs", num_train_epochs), ("max_train_steps", max_train_steps)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer or None, got {value!r}")
    if num_train_epochs is None and max_train_steps is None:
        raise ValueError("max_train_steps is required when num_train_epochs is None")
    if steps_per_epoch <= 0:
        raise ValueError("Training dataloader must provide at least one optimizer step per epoch")
    if max_train_steps is not None:
        return max_train_steps
    assert num_train_epochs is not None
    return steps_per_epoch * num_train_epochs


def align_resumed_scheduler(
    scheduler: LambdaLR,
    global_step: int,
    *,
    num_processes: int,
    split_batches: bool,
) -> list[float]:
    """Set the resumed LR before the first optimizer update, using cumulative steps.

    LambdaLR restores counters but not its Python lambda closure. Recompute with
    the current schedule and process count rather than trusting a stale saved LR.
    The original global warmup remains part of the curve, not a new resume warmup.
    """
    if global_step < 0 or num_processes < 1:
        raise ValueError("Resume step must be nonnegative and num_processes positive")
    multiplier = 1 if split_batches else num_processes
    scheduler_step = global_step * multiplier
    if not (len(scheduler.base_lrs) == len(scheduler.lr_lambdas) == len(scheduler.optimizer.param_groups)):
        raise ValueError("Scheduler and optimizer parameter groups do not match")
    learning_rates = [
        base_lr * scale(scheduler_step)
        for base_lr, scale in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    scheduler.last_epoch = scheduler_step
    scheduler._step_count = scheduler_step + 1
    for group, base_lr, lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs, learning_rates):
        group["initial_lr"] = base_lr
        if isinstance(group["lr"], torch.Tensor):
            group["lr"].fill_(lr)
        else:
            group["lr"] = lr
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]
    return [float(lr) for lr in learning_rates]


def prepare_geometry_batch(
    batch: dict[str, Any],
    predictions: dict,
    *,
    normalize: bool = False,
) -> dict[str, Any]:
    depth, extrinsics, scale = geometry_to_first_camera(
        batch["depth"],
        batch["valid_mask"],
        batch["intrinsics"],
        batch["extrinsics"],
        normalize=normalize,
    )
    batch["depth"] = depth
    batch["extrinsics"] = extrinsics
    batch["scale"] = scale
    ray_height, ray_width = predictions["ray"].shape[-3:-1]
    image_height, image_width = batch["images"].shape[-2:]
    batch["ray_map"] = compute_gt_ray_map(
        extrinsics,
        batch["intrinsics"],
        ray_height,
        ray_width,
        image_height,
        image_width,
    )
    batch["ray_valid_mask"] = resize_ray_valid_mask(
        batch["original_mask"], ray_height, ray_width
    )
    return batch

def record_metrics(output_dir, global_step, metrics, accelerator):
    """Portable scalar logs with no external tracker dependency."""
    if not accelerator.is_main_process:
        return
    import json
    from pathlib import Path
    values = {"global_step": global_step, **metrics}
    if accelerator.device.type == "cuda":
        values["peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(accelerator.device) / 2**20
    with (Path(output_dir) / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(values, allow_nan=False) + "\n")
