"""Solve and validate a complete target-arm trajectory."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import numpy as np
from ._embodiments import DEFAULT_EMBODIMENTS_ROOT, load_embodiment, sha256, slug
from ._io import OUTPUT_FILENAMES, RANDOM_SEED, load_prediction, load_extrinsics, positive_seed_count, atomic_write_json, save_trajectory_output, package_version
from ._solver import CuroboRuntime, solve_arm
from ._initial_state import load_initial_state


def parse_args(name, argv=None):
    parser = argparse.ArgumentParser(description=f'Whole-trajectory IK for {name}')
    parser.add_argument('episode', type=Path)
    parser.add_argument('--prediction-json', type=Path)
    parser.add_argument('--embodiments-root', type=Path, default=DEFAULT_EMBODIMENTS_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ik-seeds', type=positive_seed_count, default=256)
    parser.add_argument('--max-joint-step-rad', type=float, default=.5)
    parser.add_argument('--max-dropped-frames', type=int, default=0)
    parser.add_argument('--position-tolerance-mm', type=float, default=5.)
    parser.add_argument('--rotation-tolerance-deg', type=float, default=np.rad2deg(.05))
    parser.add_argument('--base-offset-mm', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if not np.isfinite(args.max_joint_step_rad) or args.max_joint_step_rad <= 0:
        parser.error('--max-joint-step-rad must be finite and positive')
    if not np.isfinite(args.base_offset_mm).all():
        parser.error('--base-offset-mm must be finite')
    if args.max_dropped_frames < 0:
        parser.error('--max-dropped-frames must be non-negative')
    if not np.isfinite(args.position_tolerance_mm) or args.position_tolerance_mm <= 0:
        parser.error('--position-tolerance-mm must be finite and positive')
    if not np.isfinite(args.rotation_tolerance_deg) or args.rotation_tolerance_deg <= 0:
        parser.error('--rotation-tolerance-deg must be finite and positive')
    return args


def build_targets(robot, prediction, extrinsics):
    """World TCP -> URDF-root TCP -> cuRobo-base EE, for every original sample."""
    world_from_camera = np.linalg.inv(extrinsics)
    world, targets = {}, {}
    for side, geometry in robot.geometries.items():
        world[side] = world_from_camera @ prediction.camera_tcp[side]
        root_tcp = np.linalg.inv(robot.world_from_root[side]) @ world[side]
        base_ee = np.linalg.inv(geometry.footprint_from_base) @ root_tcp @ np.linalg.inv(geometry.link6_from_tcp)
        targets[side] = (root_tcp, base_ee)
    return world, targets


def main(name, argv=None):
    args = parse_args(name, argv)
    started = time.perf_counter()
    episode = args.episode.expanduser().resolve()
    if not episode.is_dir():
        raise SystemExit(f'Episode not found: {episode}')
    prediction_path = (args.prediction_json or episode / 'tcp_episode.json').expanduser().resolve()
    default_output = episode / 'TCP_prediction_ik' / slug(name)
    output = (args.output_dir or default_output).expanduser().resolve()
    if output == episode:
        raise SystemExit('--output-dir must differ from the source episode directory')
    if any((output / f).exists() for f in OUTPUT_FILENAMES) and not args.overwrite:
        raise SystemExit(f'Outputs already exist: {output}; pass --overwrite')
    try:
        prediction = load_prediction(prediction_path)
        count = len(prediction.timestamps)
        extrinsics_path = episode / 'extrinsics' / f'{prediction.camera}.npy'
        extrinsics = load_extrinsics(extrinsics_path, count)
        robot = load_embodiment(name, args.embodiments_root)
        configs = {s: robot.runtime for s in robot.geometries}
        step_limits = {side: args.max_joint_step_rad for side in robot.world_from_root}
        raw_offset = np.asarray(args.base_offset_mm, dtype=float) * 0.001
        offsets = {side: raw_offset for side in robot.world_from_root}
        if any(np.any(v) for v in offsets.values()):
            for side, offset in offsets.items():
                robot.world_from_root[side] = np.asarray(robot.world_from_root[side], dtype=float).copy()
                robot.world_from_root[side][:3, 3] += offset
            print('base offset applied (mm): ' + ', '.join(
                f'{side}={tuple(np.round(v * 1000, 1))}' for side, v in offsets.items()), flush=True)
        initial, provenance = load_initial_state(
            name, robot.model, {s: g.canonical_joint_names for s, g in robot.geometries.items()})
        _, targets = build_targets(robot, prediction, extrinsics)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    stages = {'input_preparation': time.perf_counter()-started}
    print(f'{name}: whole trajectory, {count} frames, joint step <= ' + ', '.join(
        f'{side} {value:g} rad' for side, value in step_limits.items()), flush=True)
    outputs, diagnostics, summaries, arms = {}, {}, {}, {}
    columns, offset = [], 0
    try:
        stamp = time.perf_counter()
        runtime = CuroboRuntime(args.device)
        stages['runtime_initialization'] = time.perf_counter()-stamp
        np.random.seed(args.random_seed)
        runtime.torch.manual_seed(args.random_seed)
        runtime.torch.cuda.manual_seed_all(args.random_seed)
        for side, geometry in robot.geometries.items():
            stamp = time.perf_counter()
            outputs[side], diagnostics[side], summaries[side] = solve_arm(
                runtime, configs[side], geometry, robot.model, *targets[side],
                initial[side], prediction.timestamps, args.ik_seeds, float(step_limits[side]),
                args.max_dropped_frames, args.position_tolerance_mm / 1000.0,
                float(np.deg2rad(args.rotation_tolerance_deg)), 0.0)
            stages[f'{side}_arm'] = time.perf_counter()-stamp
            dof = len(geometry.canonical_joint_names)
            arms[side] = {'active_joint_names': geometry.canonical_joint_names,
                          'joint_columns': list(range(offset, offset+dof)), 'gripper_column': offset+dof,
                          'base_link': geometry.base_link, 'ee_link': geometry.ee_link,
                          'world_from_root': robot.world_from_root[side],
                          'root_from_base': geometry.footprint_from_base, 'ee_from_tcp': geometry.link6_from_tcp,
                          'initial_joint_state': initial[side],
                          'static_target_hold': summaries[side].get('static_target_hold', False)}
            columns.extend([f'{side}/{joint}_rad' for joint in geometry.canonical_joint_names])
            columns.append(f'{side}/gripper_open_binary')
            offset += dof+1
            runtime.torch.cuda.empty_cache()
    except Exception as exc:
        # An environment/solver exception does not produce a fabricated candidate.
        # Keep previously published results coherent and report this attempt separately.
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / 'solver_error.json', {
            'schema_version': 3, 'status': 'error', 'embodiment': name,
            'prediction_json': str(prediction_path), 'error': f'{type(exc).__name__}: {exc}',
            'stage_wall_time_seconds': stages})
        raise SystemExit(f'{type(exc).__name__}: {exc}; see {output / "solver_error.json"}') from exc
    gripper_source = 'TCP4D prediction'
    state = np.column_stack([outputs['left'], prediction.gripper_open['left']]).astype(np.float32)
    if state.shape != (count, offset) or not np.isfinite(state).all():
        raise RuntimeError('Invalid generated trajectory')
    counts = {s: {k: summaries[s][k] for k in ('success', 'failure', 'dropped')} for s in arms}
    failures = [{'frame_index': i, 'side': s, 'reasons': diagnostics[s][i]['failure_reasons']}
                for i in range(count) for s in arms if not diagnostics[s][i]['success']]
    status = 'success' if all(s['trajectory_success'] for s in summaries.values()) else 'failed'
    elapsed = time.perf_counter()-started
    metadata = {
        'schema_version': 3, 'format': 'robotwin_closed_tcp_trajectory_v3',
        'embodiment': name, 'status': status, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'frame_count': count, 'frame_rate_hz': prediction.frame_rate_hz,
        'timestamps_seconds': prediction.timestamps, 'camera': prediction.camera, 'initialization': provenance,
        'inputs': {'episode': str(episode), 'prediction_json': str(prediction_path), 'prediction_sha256': sha256(prediction_path),
                   'extrinsics': str(extrinsics_path), 'extrinsics_sha256': sha256(extrinsics_path),
                   'extrinsics_semantics': 'world_to_camera', 'urdf': str(robot.urdf_path), 'urdf_sha256': sha256(robot.urdf_path),
                   'embodiments_root': str(robot.asset_dir.parent)},
        'coordinate_chain': 'camera closed TCP -> world -> target URDF root -> cuRobo base -> EE',
        'trajectory_retargeting': 'World closed TCP position, orientation and timestamps unchanged',
        'embodiment_base_offset_mm': [float(v) for v in args.base_offset_mm],
        'robot_config_snapshot': robot.config,
        'tcp_calibration': robot.calibration, 'runtime_robot_cfg': {s: c.robot_cfg for s, c in configs.items()},
        'runtime_config_corrections': {s: getattr(c, 'corrections', []) for s, c in configs.items()},
        'arms': arms, 'output': {'shape': list(state.shape), 'dtype': 'float32', 'columns': columns},
        'solver': {'type': 'whole_trajectory_candidate_graph_and_joint_optimization',
                   'position_tolerance_m': args.position_tolerance_mm / 1000.0,
                   'rotation_tolerance_deg': args.rotation_tolerance_deg,
                   'hold_static_arm_m': 0.0,
                   'ik_seeds': args.ik_seeds, 'maximum_ik_seeds': 256, 'maximum_initial_paths': 4,
                   'random_seed': args.random_seed, 'initial_seed_source': provenance['source'],
                   'initial_state_role': 'initialization_and_first_frame_soft_preference',
                   'max_joint_step_rad': {side: float(value) for side, value in step_limits.items()},
                   'max_dropped_frames': args.max_dropped_frames,
                   'velocity_and_acceleration': 'time_aware_smoothing_and_diagnostics_only',
                   'self_collision_check': True, 'world_obstacle_collision_check': False,
                   'inter_arm_collision_check': False, 'validation_samples': 'original_timestamps'},
        'gripper_source': gripper_source,
        'ground_truth_usage': 'Recorded target frame-0 values initialize search; no source episode states or future example states are used',
        'counts': counts,
        'dropped_frames': {s: summaries[s]['dropped_frames'] for s in arms},
        'dropped_frame_repair': ('A frame dropped under --max-dropped-frames keeps its timestamp and is replaced by '
                                 'joint-space interpolation between its valid neighbours; the payload stays full '
                                 'length and inside the joint limits. See diagnostics.json for its original reasons.'),
        'stage_wall_time_seconds': stages, 'total_wall_time_seconds': elapsed,
        'environment': {'python': sys.version.split()[0], 'torch': runtime.torch.__version__,
                        'torch_cuda': runtime.torch.version.cuda, 'curobo': package_version('curobo', runtime.curobo),
                        'gpu': runtime.torch.cuda.get_device_name(runtime.device)},
    }
    report = {'schema_version': 3, 'summary': {'status': status, 'frame_count': count, 'counts': counts,
              'failed_frames': failures, 'total_wall_time_seconds': elapsed}, 'arms': summaries,
              'frames': [{'frame_index': i, 'time_seconds': prediction.timestamps[i],
                          **{s: diagnostics[s][i] for s in arms}} for i in range(count)]}
    save_trajectory_output(output, state, metadata, report)
    (output / 'solver_error.json').unlink(missing_ok=True)
    print(f'Saved {output / metadata["output"]["state_file"]}: {state.shape}, {status}, {elapsed:.2f}s', flush=True)
    for side in arms:
        maximum = max((f['joint_step_max_abs_rad'] or 0 for f in diagnostics[side]), default=0.)
        print(f'  {side}: {counts[side]}, maximum joint step={maximum:.6f} rad', flush=True)
    if status != 'success':
        raise SystemExit(1)
