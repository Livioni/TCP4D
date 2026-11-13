from __future__ import annotations
import copy
import logging
import sys
import time
from typing import Any
import numpy as np
from ._io import POSITION_TOLERANCE_METERS, ROTATION_TOLERANCE_RADIANS, RANDOM_SEED
from ._kinematics import fk_tcp_batch, joint_bounds, nearest_valid_branch
from ._trajectory import validate_inputs, motion_metrics, smoothness_cost, select_paths, diverse_candidates, interpolate_candidate_gaps

# Joint limits are inclusive mechanical stops; float32 payload rounding at an
# exact stop (e.g. float32(2.9) > float64(2.9) by 1e-7) must not read as a
# violation. Same convention as robotwin_ik._initial_state.
JOINT_BOUND_EPSILON = 1e-6

class CuroboExtensionLogFilter(logging.Filter):
    """Clarify CuRobo's unconditional JIT messages without hiding build failures."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for suffix in (
            " not found, JIT compiling...",
            " not found, jit compiling...",
            " binary not found, jit compiling...",
        ):
            if message.endswith(suffix):
                name = message[:-len(suffix)]
                if name in {
                    "kinematics_fused_cu", "geom_cu", "lbfgs_step_cu",
                    "line_search_cu", "tensor_step_cu",
                }:
                    record.msg = (
                        f"{name}: loading via PyTorch JIT cache "
                        "(builds only if needed)"
                    )
                    record.args = ()
        return True

class CuroboRuntime:
    def __init__(self, device: str) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "IK requires CUDA-enabled PyTorch and compatible CuRobo in the "
                "current Python environment; the environment need not be named RoboTwin. "
                f"Python: {sys.executable}; import error: {error}"
            ) from error

        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError(f"--device must be a CUDA device, got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU")
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {index} is invalid for {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(index)
        logger = logging.getLogger("curobo")
        extension_filter = CuroboExtensionLogFilter()
        logger.addFilter(extension_filter)
        try:
            import curobo
            from curobo.types.base import TensorDeviceType
            from curobo.types.math import Pose
            from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        except ImportError as error:
            raise RuntimeError(
                "IK requires compatible CuRobo (curobo.wrap.reacher.ik_solver API) "
                "and CUDA-enabled PyTorch in the current Python environment; "
                "the environment need not be named RoboTwin. "
                f"Python: {sys.executable}; import error: {error}"
            ) from error
        finally:
            logger.removeFilter(extension_filter)
        self.torch = torch
        self.curobo = curobo
        self.Pose = Pose
        self.IKSolver = IKSolver
        self.IKSolverConfig = IKSolverConfig
        self.device = torch.device(f"cuda:{index}")
        self.tensor_args = TensorDeviceType(device=self.device, dtype=torch.float32)

    def create_solver(self, robot_cfg: dict[str, Any], num_seeds: int,
                      position_tolerance: float = POSITION_TOLERANCE_METERS,
                      rotation_tolerance: float = ROTATION_TOLERANCE_RADIANS) -> Any:
        config = self.IKSolverConfig.load_from_robot_config(
            copy.deepcopy(robot_cfg),
            world_model=None,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            position_threshold=position_tolerance,
            rotation_threshold=rotation_tolerance,
            use_cuda_graph=True,
            self_collision_check=True,
            self_collision_opt=True,
            sync_cuda_time=True,
            store_debug=False,
            regularization=True,
            seed=RANDOM_SEED,
        )
        return self.IKSolver(config)

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def pose_from_matrix(self, matrix: np.ndarray) -> Any:
        tensor = self.torch.as_tensor(
            matrix, device=self.device, dtype=self.torch.float32
        )
        return self.Pose.from_matrix(tensor)


def pose_residuals(model, geometry, q, targets):
    actual = fk_tcp_batch(model, geometry, q)
    position = np.linalg.norm(actual[:, :3, 3] - targets[:, :3, 3], axis=-1)
    relative = np.swapaxes(targets[:, :3, :3], -1, -2) @ actual[:, :3, :3]
    rotation = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) * .5, -1, 1))
    return position, rotation


def _tensor(runtime, values):
    return runtime.torch.as_tensor(values, device=runtime.device, dtype=runtime.torch.float32).contiguous()


def _locked_mask(result, locks):
    shape = tuple(result.success.shape)
    mask = np.ones(shape, dtype=bool)
    if not locks:
        return mask
    names = list(result.js_solution.joint_names or [])
    full = result.js_solution.position.detach().cpu().numpy()
    for name, value in locks.items():
        if name not in names or full.shape[:-1] != shape:
            return np.zeros(shape, dtype=bool)
        mask &= np.isclose(full[..., names.index(name)], value, atol=1e-5, rtol=0)
    return mask


def _valid_configs(runtime, validator, q, canonical_names):
    order = [canonical_names.index(name) for name in validator.joint_names]
    return validator.check_valid(_tensor(runtime, q[:, order])).detach().cpu().numpy().reshape(-1).astype(bool)


def evaluate_trajectory(runtime, validator, geometry, model, q, targets, timestamps, step_limit,
                        step_exempt=(), position_tolerance=POSITION_TOLERANCE_METERS,
                        rotation_tolerance=ROTATION_TOLERANCE_RADIANS):
    """Validate the exact float32 payload; do not rely on optimizer success flags."""
    q = np.asarray(q, dtype=np.float32).astype(float)
    if not np.isfinite(q).all():
        raise ValueError('Trajectory contains nonfinite values')
    exempt = set(int(i) for i in step_exempt)
    lower, upper = joint_bounds(model, geometry.canonical_joint_names)
    positions, rotations = pose_residuals(model, geometry, q, targets)
    feasible = _valid_configs(runtime, validator, q, geometry.canonical_joint_names)
    bounds = np.all((q >= lower - JOINT_BOUND_EPSILON) & (q <= upper + JOINT_BOUND_EPSILON), axis=1)
    steps, velocity, acceleration = motion_metrics(q, timestamps)
    frames = []
    violation = 0.
    for i in range(len(q)):
        step = float(np.max(np.abs(steps[i-1]))) if i else None
        step_exempted = bool(i and i in exempt and step > step_limit)
        reasons = []
        if positions[i] > position_tolerance:
            reasons.append('tcp_position_tolerance')
        if rotations[i] > rotation_tolerance:
            reasons.append('tcp_rotation_tolerance')
        if not bounds[i]:
            reasons.append('joint_limits')
        if not feasible[i]:
            reasons.append('model_constraints')
        if i and step > step_limit and not step_exempted:
            reasons.append('joint_step_limit')
        violation += max(positions[i] / position_tolerance - 1, 0) ** 2
        violation += max(rotations[i] / rotation_tolerance - 1, 0) ** 2
        violation += float(not bounds[i]) + float(not feasible[i])
        if i and not step_exempted:
            violation += max(step / step_limit - 1, 0) ** 2
        frames.append({
            'success': not reasons, 'failure_reasons': reasons,
            'position_residual_m': float(positions[i]),
            'rotation_residual_rad': float(rotations[i]),
            'rotation_residual_deg': float(np.degrees(rotations[i])),
            'joint_limits_valid': bool(bounds[i]), 'constraint_feasible': bool(feasible[i]),
            'joint_step_max_abs_rad': step,
            'joint_step_exempt_due_to_dropped_repair': step_exempted,
            'joint_step_l2_rad': float(np.linalg.norm(steps[i-1])) if i else None,
            'joint_velocity_rad_s': velocity[i-1].tolist() if i else None,
            'joint_acceleration_rad_s2': acceleration[i-1].tolist() if 0 < i < len(q)-1 else None,
        })
    return q, frames, float(violation)


def _wrap_out_of_bounds_frames(model, names, q, initial):
    """Radian-equivalent winding fix for out-of-bounds revolute frames."""
    lower, upper = joint_bounds(model, names)
    out = np.asarray(q, dtype=float).copy()
    for i, values in enumerate(out):
        if np.all((values >= lower - JOINT_BOUND_EPSILON) & (values <= upper + JOINT_BOUND_EPSILON)):
            continue
        wrapped = nearest_valid_branch(model, names, values, initial)
        if np.all((wrapped >= lower - JOINT_BOUND_EPSILON) & (wrapped <= upper + JOINT_BOUND_EPSILON)):
            out[i] = wrapped
    return out


def _interpolate_dropped_frames(model, names, q, dropped):
    """Replace dropped frames by joint-space interpolation between valid neighbours."""
    lower, upper = joint_bounds(model, names)
    out = np.asarray(q, dtype=float).copy()
    dropped_set = set(dropped)
    valid = [i for i in range(len(out)) if i not in dropped_set]
    if not valid:
        return np.clip(out, lower + JOINT_BOUND_EPSILON, upper - JOINT_BOUND_EPSILON)
    for i in dropped:
        left = next((j for j in reversed(valid) if j < i), None)
        right = next((j for j in valid if j > i), None)
        if left is None:
            out[i] = out[right]
        elif right is None:
            out[i] = out[left]
        else:
            t = (i - left) / (right - left)
            out[i] = (1.0 - t) * out[left] + t * out[right]
        out[i] = np.clip(out[i], lower + JOINT_BOUND_EPSILON, upper - JOINT_BOUND_EPSILON)
    return out


def _equivalent_candidates(model, names, values, initial):
    """Retain raw and legal equivalent windings, instead of wrapping edge distances."""
    if not len(values):
        return values
    lower, upper = joint_bounds(model, names)
    nearest = np.asarray([nearest_valid_branch(model, names, q, initial) for q in values])
    variants = [values, nearest]
    for j, name in enumerate(names):
        if model.joints_by_name[name].joint_type not in ('revolute', 'continuous'):
            continue
        for sign in (-1, 1):
            shifted = nearest.copy()
            shifted[:, j] += sign * 2 * np.pi
            variants.append(shifted)
    combined = np.concatenate(variants)
    return combined[np.all((combined >= lower) & (combined <= upper), axis=1)]


def _collect_candidates(runtime, solver, validator, geometry, model, targets, ee_targets,
                        initial, layers, approximate, seeds, propagation, locks,
                        position_tolerance=POSITION_TOLERANCE_METERS,
                        rotation_tolerance=ROTATION_TOLERANCE_RADIANS):
    """Fixed-shape batched IK; adjacent candidate sets are read from a frozen pass."""
    count = len(targets)
    batch = min(32, count)
    names = geometry.canonical_joint_names
    order = [names.index(name) for name in solver.joint_names]
    inverse = [list(solver.joint_names).index(name) for name in names]
    snapshot = [layer.copy() for layer in layers]
    started = time.perf_counter()
    reported_success = 0
    for start in range(0, count, batch):
        indices = np.minimum(np.arange(start, start+batch), count-1)
        seed_list = []
        width = min(seeds - 1, 33) if propagation else 1
        for t in indices:
            pool = [initial[None]]
            if propagation:
                # Both directions influence the next batch pass; no previous output exists.
                for neighbor in (max(0, t-1), min(count-1, t+1), t):
                    pool.append(diverse_candidates(snapshot[neighbor], initial, maximum=max(1, (width-1)//3)))
            pool = np.concatenate(pool)[:width]
            seed_list.append(np.concatenate([pool, np.tile(initial, (width-len(pool), 1))]))
        result = solver.solve_batch(runtime.pose_from_matrix(ee_targets[indices]),
                                    seed_config=_tensor(runtime, np.asarray(seed_list)[..., order]),
                                    return_seeds=seeds, num_seeds=seeds, use_nn_seed=False)
        runtime.synchronize()
        values = result.solution.detach().cpu().numpy()[..., inverse]
        success = result.success.detach().cpu().numpy().astype(bool)
        # Lock values come from the runtime model and are verified in the full IK result.
        success &= _locked_mask(result, locks)
        for local, t in enumerate(indices[:min(batch, count-start)]):
            finite = np.isfinite(values[local]).all(axis=1)
            raw = values[local][finite]
            if not len(raw):
                continue
            pos, rot = pose_residuals(model, geometry, raw, np.repeat(targets[t][None], len(raw), axis=0))
            errors = (pos / position_tolerance) ** 2 + (rot / rotation_tolerance) ** 2
            best = int(np.argmin(errors))
            if approximate[t] is None or errors[best] < approximate[t][0]:
                approximate[t] = (float(errors[best]), raw[best].copy())
            qualified = success[local][finite] & (pos <= position_tolerance) & (rot <= rotation_tolerance)
            reported_success += int(qualified.sum())
            extended = _equivalent_candidates(model, names, raw[qualified], initial)
            merged = diverse_candidates(np.concatenate([layers[t], extended]), initial)
            if len(merged):
                valid = _valid_configs(runtime, validator, merged, names)
                lower, upper = joint_bounds(model, names)
                valid &= np.all((merged >= lower) & (merged <= upper), axis=1)
                # Equivalent turns can incur floating-point FK differences; recheck them.
                p, r = pose_residuals(model, geometry, merged, np.repeat(targets[t][None], len(merged), axis=0))
                layers[t] = merged[valid & (p <= position_tolerance) & (r <= rotation_tolerance)]
        if start == 0 or start+batch >= count or (start//batch) % 4 == 0:
            print(f'  {geometry.side}: candidates {min(start+batch, count)}/{count}, seeds={seeds}, neighbors={propagation}', flush=True)
    return {'seeds': seeds, 'neighbor_propagation': propagation,
            'qualified_ik_candidates': reported_success, 'candidate_counts': [len(x) for x in layers],
            'wall_time_seconds': time.perf_counter()-started}


def solve_arm(runtime, runtime_config, geometry, model, footprint_targets, link6_targets,
              q_initial, timestamps, ik_seeds=64, max_joint_step_rad=.5, max_dropped_frames=0,
              position_tolerance=POSITION_TOLERANCE_METERS,
              rotation_tolerance=ROTATION_TOLERANCE_RADIANS, hold_static_arm_m=0.0,
              reference_q=None, posture_weight=0.0, branch_constraint=None):
    """Search IK branches and optimize all TCP frames jointly, returning no held frames."""
    from ._trajectory_opt import refine_trajectory
    if isinstance(max_dropped_frames, bool) or int(max_dropped_frames) != max_dropped_frames or max_dropped_frames < 0:
        raise ValueError('max_dropped_frames must be a non-negative integer')
    max_dropped_frames = int(max_dropped_frames)
    if not (np.isfinite(hold_static_arm_m) and hold_static_arm_m >= 0):
        raise ValueError('hold_static_arm_m must be finite and non-negative')
    if not (np.isfinite(position_tolerance) and position_tolerance > 0
            and np.isfinite(rotation_tolerance) and rotation_tolerance > 0):
        raise ValueError('position/rotation tolerances must be finite and positive')
    validate_inputs(q_initial, footprint_targets, timestamps, ik_seeds, max_joint_step_rad)
    validate_inputs(q_initial, link6_targets, timestamps, ik_seeds, max_joint_step_rad)
    names = geometry.canonical_joint_names
    if len(q_initial) != len(names):
        raise ValueError('Initial state does not match active joints')
    count = len(timestamps)
    initial = np.asarray(q_initial, dtype=float)
    if reference_q is None:
        reference = initial
    else:
        reference = np.asarray(reference_q, dtype=float)
        if reference.shape != initial.shape or not np.isfinite(reference).all():
            raise ValueError('reference_q must be a finite vector matching the active joints')
    if not (np.isfinite(posture_weight) and posture_weight >= 0):
        raise ValueError('posture_weight must be finite and non-negative')
    posture_weight = float(posture_weight)
    if branch_constraint is not None:
        branch_constraint = dict(branch_constraint)
        frame = int(branch_constraint['frame'])
        joint = int(branch_constraint['joint'])
        low = float(branch_constraint['min'])
        high = float(branch_constraint['max'])
        if not (0 <= frame < count and 0 <= joint < len(names) and low < high):
            raise ValueError('invalid branch_constraint (frame/joint/min/max)')
        branch_constraint.update(frame=frame, joint=joint, min=low, max=high)

    def _branch_layers(source):
        if branch_constraint is None:
            return source
        filtered = list(source)
        values = filtered[branch_constraint['frame']]
        if len(values):
            keep = ((values[:, branch_constraint['joint']] >= branch_constraint['min'])
                    & (values[:, branch_constraint['joint']] <= branch_constraint['max']))
            filtered[branch_constraint['frame']] = values[keep]
        return filtered

    timestamps = np.asarray(timestamps, dtype=float)
    validator = runtime.create_solver(runtime_config.robot_cfg, 1)
    if set(validator.joint_names) != set(names) or len(validator.joint_names) != len(names):
        raise ValueError('CuRobo active joints do not match the target arm')
    layers = [np.empty((0, len(names))) for _ in range(count)]
    approximate = [None] * count
    search, paths = [], []
    seeds = ik_seeds
    while True:
        solver = runtime.create_solver(runtime_config.robot_cfg, seeds, position_tolerance, rotation_tolerance)
        for propagation in (False, True, True):
            search.append(_collect_candidates(runtime, solver, validator, geometry, model,
                                             footprint_targets, link6_targets, initial, layers,
                                             approximate, seeds, propagation, runtime_config.lock_joints,
                                             position_tolerance, rotation_tolerance))
            if branch_constraint is not None:
                frame = branch_constraint['frame']
                joint = branch_constraint['joint']
                values = layers[frame]
                if len(values):
                    column = values[:, joint]
                    in_range = int(((column >= branch_constraint['min'])
                                    & (column <= branch_constraint['max'])).sum())
                    print(f'  {geometry.side}: branch frame {frame} candidates={len(values)} '
                          f'in_range={in_range} joint[{column.min():.2f},{column.max():.2f}]', flush=True)
                else:
                    print(f'  {geometry.side}: branch frame {frame} has no candidates', flush=True)
            paths = select_paths(_branch_layers(layers), timestamps, initial, max_joint_step_rad,
                                 reference=reference, posture_weight=posture_weight)
            if paths and propagation:
                break
        del solver
        if paths or seeds >= 256:
            break
        seeds = min(seeds*2, 256)
    graph_connected = bool(paths)
    if not paths:
        if all(item is None for item in approximate):
            raise RuntimeError('IK returned no finite candidate for the entire trajectory')
        guide = interpolate_candidate_gaps(layers, timestamps, initial, max_joint_step_rad)
        lower, upper = joint_bounds(model, names)
        relaxed_layers, penalties = [], []
        for t, layer in enumerate(layers):
            if len(layer):
                relaxed_layers.append(layer)
                penalties.append(np.zeros(len(layer)))
            else:
                values = [guide[t]]
                if approximate[t] is not None:
                    raw = nearest_valid_branch(model, names, approximate[t][1], initial)
                    # Bounds projection is only for an optimization INITIALIZATION.
                    # Final states are never clipped and always undergo independent FK.
                    values.append(np.clip(raw, lower+1e-6, upper-1e-6))
                values = diverse_candidates(values, initial)
                p, r = pose_residuals(model, geometry, values, np.repeat(footprint_targets[t][None], len(values), axis=0))
                node_cost = np.maximum(p/position_tolerance-1, 0)**2
                node_cost += np.maximum(r/rotation_tolerance-1, 0)**2
                node_cost += ~_valid_configs(runtime, validator, values, names)
                relaxed_layers.append(values)
                penalties.append(1e4 * node_cost)
        paths = select_paths(_branch_layers(relaxed_layers), timestamps, initial, max_joint_step_rad,
                             count=3, relaxed=True, node_penalties=penalties,
                             reference=reference, posture_weight=posture_weight)
        if not any(np.array_equal(guide, path) for path in paths):
            paths.append(guide)
    if not paths:
        raise RuntimeError('No finite whole-trajectory initialization could be generated')
    def evaluate(q, step_exempt=()):
        return evaluate_trajectory(runtime, validator, geometry, model, q, footprint_targets,
                                   timestamps, max_joint_step_rad, step_exempt,
                                   position_tolerance, rotation_tolerance)

    def branch_violation(values):
        if branch_constraint is None:
            return 0.0
        value = float(np.asarray(values)[branch_constraint['frame'], branch_constraint['joint']])
        return (max(0.0, branch_constraint['min'] - value)
                + max(0.0, value - branch_constraint['max']))

    best = None
    best_rank = None
    optimization = []
    for index, path in enumerate(paths):
        print(f'  {geometry.side}: joint optimization {index+1}/{len(paths)} ({count} frames)', flush=True)
        q, frames, violation, report = refine_trajectory(
            runtime, runtime_config, geometry, model, path, footprint_targets, timestamps,
            initial, max_joint_step_rad, evaluate, position_tolerance, rotation_tolerance,
            reference=reference, posture_weight=posture_weight,
            branch_constraint=branch_constraint)
        branch_error = branch_violation(q)
        rank = (violation > 0 or branch_error > 0, branch_error > 0,
                violation, branch_error,
                smoothness_cost(q, timestamps, initial, reference, posture_weight))
        if best_rank is None or rank < best_rank:
            best, best_rank = (q, frames, violation), rank
        optimization.append(report)
    q, frames, violation = best
    # Final validation occurs after all optimization and after output quantization.
    # Out-of-bounds frames that are 2*pi-equivalent to a legal winding are wrapped
    # first: the physical configuration is unchanged, only the representation.
    q = _wrap_out_of_bounds_frames(model, names, q, initial)
    q, frames, violation = evaluate(q)
    static_held = False
    if hold_static_arm_m and len(frames):
        roots = np.asarray(footprint_targets)[:, :3, 3]
        spread = float(np.max(np.linalg.norm(roots - roots[0], axis=1)))
        if spread <= hold_static_arm_m and frames[0]['success']:
            # A target hand that does not move has a stationary solution: hold the
            # validated frame-0 configuration instead of letting the optimizer hop
            # between mirrored near-limit branches for a constant pose.
            q = np.tile(np.asarray(q[0], dtype=float), (len(q), 1))
            q, frames, violation = evaluate(q)
            static_held = True
            for frame in frames:
                frame['static_target_hold'] = True
            print(f'  {geometry.side}: target hand static (spread {spread*1000:.1f} mm <= '
                  f'{hold_static_arm_m*1000:.1f} mm); holding the frame-0 configuration', flush=True)
    dropped = [i for i, frame in enumerate(frames) if not frame['success']]
    if dropped and len(dropped) <= max_dropped_frames:
        reasons_before = {i: list(frames[i]['failure_reasons']) for i in dropped}
        q = _interpolate_dropped_frames(model, names, q, dropped)
        # The step entering the first valid frame after a repaired run belongs to
        # the repair (it would be a mid-run step without the gap), not to that frame.
        exempt = sorted({i for i in dropped} | {i+1 for i in dropped if i+1 < count})
        q, frames, violation = evaluate(q, step_exempt=exempt)
        for i in dropped:
            frames[i]['dropped'] = True
            frames[i]['success'] = True
            frames[i]['pre_repair_failure_reasons'] = reasons_before[i]
        print(f'  {geometry.side}: dropped {len(dropped)} frame(s) '
              f'(<= --max-dropped-frames {max_dropped_frames}); joints interpolated '
              f'between valid neighbours: {dropped}', flush=True)
    else:
        dropped = []
    if branch_constraint is not None:
        value = float(q[branch_constraint['frame'], branch_constraint['joint']])
        low, high = branch_constraint['min'], branch_constraint['max']
        if not (low - 1e-6 <= value <= high + 1e-6):
            raise RuntimeError(
                f'branch constraint not satisfied on {geometry.side}: frame '
                f"{branch_constraint['frame']} joint {branch_constraint['joint']}="
                f'{value:.3f} outside [{low:.3f}, {high:.3f}]')
    for i, frame in enumerate(frames):
        frame['candidate_count'] = len(layers[i])
    successes = sum(f['success'] for f in frames)
    summary = {'success': successes, 'failure': count-successes, 'dropped': len(dropped),
               'dropped_frames': dropped, 'static_target_hold': static_held,
               'trajectory_success': successes == count,
               'graph_connected': graph_connected,
               'search': search, 'optimization': optimization,
               'constraint_violation': violation, 'smoothness_cost': smoothness_cost(q, timestamps, initial)}
    return q.astype(np.float32), frames, summary
