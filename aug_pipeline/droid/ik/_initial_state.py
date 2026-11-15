"""Named initial states for target-arm IK."""
import numpy as np


BUILTIN_SEED_SOURCE = 'builtin_target_example_frame_0'


def _arm(names, values):
    return {'left': dict(zip(names, values))}


# Exact float64 values read from joint_action/*_arm[0] of the supplied
# recordings (RoboTwin stores float32 drive targets as float64 in HDF5).
# These constants do not read config homestate or require the recordings.
BUILTIN_INITIAL_JOINTS = {
    'franka-panda': _arm(
        tuple(f'panda_joint{i}' for i in range(1, 8)),
        (0., 0.19634954631328583, 0., -2.6179938316345215, 0.,
         2.9415926933288574, 0.7853981852531433)),
    'ur5-wsg': _arm(
        ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
         'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'),
        (-1.544700026512146, -1.544700026512146, -1.544700026512146,
         -1.5793999433517456, 1.5793999433517456, 0.)),
    'sawyer': _arm(
        tuple(f'right_j{i}' for i in range(7)),
        (0., -1.2, 0., 1.4, 0., 0.3, 0.)),
    'kinova-gen3': _arm(
        tuple(f'joint_{i}' for i in range(1, 8)),
        (0., 0.4, 0., 1.0, 0., 0.7, 0.)),
    'jaco': _arm(tuple(f'jaco_joint_{i}' for i in range(1, 7)),
                       (0., 1.6, 2.6, 0., 0., 0.)),
}


def load_initial_state(embodiment, model, joint_names):
    if embodiment not in BUILTIN_INITIAL_JOINTS:
        raise ValueError(f'No built-in initial state for {embodiment}')
    arms = {}
    for side in joint_names:
        named_values = BUILTIN_INITIAL_JOINTS[embodiment][side]
        names = joint_names[side]
        if len(names) != len(named_values) or set(names) != set(named_values):
            raise ValueError(f'{embodiment}: {side} joint names disagree with built-in state')
        q = np.array([named_values[name] for name in names], dtype=np.float64)
        _validate_joint_values(q, model, names, f'built-in {embodiment}/{side}')
        arms[side] = q
    vector = np.concatenate([arms['left'], [1.]])
    return arms, {
        'source': BUILTIN_SEED_SOURCE, 'path': None, 'sha256': None,
        'original_example': f'{embodiment}.hdf5', 'frame_index': 0,
        'dataset': 'joint_action/vector',
        'joint_names': {s: list(joint_names[s]) for s in arms},
        'joint_order_basis': 'built-in named joints reordered to active_joint_names',
        'state_vector': vector.tolist(), 'gripper_open': {'left': 1.},
    }


def _validate_joint_values(q, model, names, source):
    if not np.isfinite(q).all():
        raise ValueError(f'{source}: nonfinite initial state')
    for name, value in zip(names, q):
        joint = model.joints_by_name[name]
        if ((joint.lower is not None and value < joint.lower - 1e-6) or
                (joint.upper is not None and value > joint.upper + 1e-6)):
            raise ValueError(f'{source}: initial {name}={value} outside URDF limits')

