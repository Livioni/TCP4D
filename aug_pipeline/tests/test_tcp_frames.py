from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from aug_pipeline.droid.ik import _embodiments as embodiments
from aug_pipeline.droid.ik._common import build_targets
from aug_pipeline.droid.urdf import rotation_matrix_from_rpy
from aug_pipeline.droid.solve import ROBOTIQ_TO_TARGET


@pytest.fixture
def gen3_assets(tmp_path, monkeypatch):
    """Minimal native +Z-approach/+X-jaw gripper with the released bad config."""
    asset = tmp_path / 'kinova-gen3'
    asset.mkdir()
    urdf = asset / 'robot.urdf'
    urdf.write_text('''<robot name="gen3_test">
      <link name="base"/><link name="wrist"/><link name="gen3_tcp"/><link name="finger"/>
      <joint name="arm_joint" type="revolute">
        <parent link="base"/><child link="wrist"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" effort="1" velocity="1"/>
      </joint>
      <joint name="tcp_joint" type="fixed">
        <parent link="wrist"/><child link="gen3_tcp"/><origin xyz="0 0 0.10937"/>
      </joint>
      <joint name="finger_joint" type="prismatic">
        <parent link="gen3_tcp"/><child link="finger"/><axis xyz="1 0 0"/>
        <limit lower="0" upper="0.04" effort="1" velocity="1"/>
      </joint>
    </robot>''')
    config = dict(urdf_path='robot.urdf', arm_joints_name=[['arm_joint']] * 2,
                  homestate=[[0.2]] * 2, robot_pose=[[0.1, -0.2, 0.3, 1, 0, 0, 0]],
                  move_group=['gen3_tcp'] * 2, ee_joints=['tcp_joint'] * 2,
                  gripper_name=[dict(base='finger_joint', mimic=[])] * 2,
                  gripper_scale=[0, 0.04], dual_arm=False,
                  delta_matrix=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                  global_trans_matrix=np.diag([1, -1, -1]).tolist())
    (asset / 'config.yml').write_text(yaml.safe_dump(config))
    (asset / 'spheres.yml').write_text('{}')
    kin = dict(base_link='base', ee_link='gen3_tcp', collision_spheres='spheres.yml',
               cspace=dict(joint_names=['arm_joint'], retract_config=[0],
                           null_space_weight=[1], cspace_distance_weight=[1]))
    (asset / 'curobo.yml').write_text(yaml.safe_dump(dict(robot_cfg=dict(kinematics=kin))))
    # Use the real project's axis calibration with this self-contained URDF.
    calibration = copy.deepcopy(embodiments.load_tcp_calibration('kinova-gen3'))
    calibration.update(urdf_sha256=embodiments.sha256(urdf), mesh_sha256={})
    monkeypatch.setattr(embodiments, 'load_tcp_calibration', lambda _: copy.deepcopy(calibration))
    return tmp_path, config, calibration


@pytest.mark.parametrize('jaw_flip', [False, True])
def test_gen3_physical_axes_follow_droid_through_target_chain(gen3_assets, jaw_flip):
    root, original, _ = gen3_assets
    robot = embodiments.load_embodiment('kinova-gen3', root)
    assert set(robot.geometries) == set(robot.world_from_root) == {'left'}
    source = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    source[:, :3, :3] = [rotation_matrix_from_rpy(rpy)
                         for rpy in ([0, 0, 0], [0.3, -0.7, 1.2], [-1.4, 0.5, -0.8])]
    source[:, :3, 3] = [[0.2, 0.1, 0.8], [-0.1, 0.4, 0.9], [0.5, -0.3, 1.1]]
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = rotation_matrix_from_rpy([0.2, 0.4, -0.5])
    extrinsics[:3, 3] = [0.1, -0.3, 0.5]
    correction = np.eye(4)
    correction[:3, :3] = ROBOTIQ_TO_TARGET
    if jaw_flip:
        correction[:3, :3] @= np.diag([1., -1., -1.])
    prediction = SimpleNamespace(camera_tcp={side: source @ correction for side in robot.geometries})
    _, targets = build_targets(robot, prediction, extrinsics)
    for side, geometry in robot.geometries.items():
        _, base_ee = targets[side]
        camera_ee = extrinsics @ robot.world_from_root[side] @ geometry.footprint_from_base @ base_ee
        # Physical native EE +Z approaches, and native EE +X spans the jaws.
        np.testing.assert_allclose(camera_ee[:, :3, 2], source[:, :3, 2], atol=1e-12)
        jaw_sign = -1 if jaw_flip else 1
        np.testing.assert_allclose(camera_ee[:, :3, 0], jaw_sign * source[:, :3, 1], atol=1e-12)
        camera_tcp = camera_ee @ geometry.link6_from_tcp
        np.testing.assert_allclose(camera_tcp[:, :3, 3], source[:, :3, 3], atol=1e-12)
        np.testing.assert_allclose(geometry.link6_from_tcp[:3, 3], [0, 0, 0.00244185])
        np.testing.assert_allclose(robot.config['delta_matrix'], geometry.link6_from_tcp[:3, :3])
    assert yaml.safe_load((root / 'kinova-gen3/config.yml').read_text()) == original
    assert robot.runtime.corrections


def test_assets_without_rotation_calibration_keep_their_tcp_convention(gen3_assets):
    root, original, calibration = gen3_assets
    calibration.pop('rotation_matrix')
    robot = embodiments.load_embodiment('kinova-gen3', root)
    for geometry in robot.geometries.values():
        np.testing.assert_allclose(geometry.link6_from_tcp[:3, :3], original['delta_matrix'])


@pytest.mark.parametrize('rotation', [np.eye(2), np.diag([1, 1, -1]), np.zeros((3, 3)),
                                      np.full((3, 3), float('nan'))])
def test_invalid_tcp_rotation_is_rejected(gen3_assets, rotation):
    root, _, calibration = gen3_assets
    calibration['rotation_matrix'] = rotation.tolist()
    with pytest.raises(ValueError, match='invalid EE-from-TCP rotation'):
        embodiments.load_embodiment('kinova-gen3', root)


@pytest.mark.parametrize('success', [True, False])
def test_single_arm_ik_publishes_and_reloads_one_trajectory(gen3_assets, monkeypatch, success):
    from aug_pipeline.config import write_json
    from aug_pipeline.data import adapt_prediction
    from aug_pipeline.droid.ik import _common as common, _initial_state
    from aug_pipeline.droid.retarget import ik_metrics
    from aug_pipeline.droid.robot import load_target
    from aug_pipeline.droid.render_target import render_target
    from aug_pipeline.droid.renderer import RobotRenderer

    root, _, _ = gen3_assets
    episode = root / 'episode'
    (episode / 'extrinsics').mkdir(parents=True)
    np.save(episode / 'extrinsics/third_views.npy', np.eye(4))
    np.save(episode / 'gripper_openness.npy', [.9, .3, .3])
    raw = dict(stage=1, camera='cam', episode_num_frames=3, frame_rate_hz=15,
        frames=[dict(frame_index=i, time_seconds=i/15, xyz_m=[.1, .2, 1.],
                     rpy_rad=[0, 0, 0], gripper_open=g, confidence=.9)
                for i, g in enumerate([.9, .3, .3])])
    write_json(episode / 'tcp_episode.json', adapt_prediction(raw, 3, 'cam', 15))
    monkeypatch.setitem(_initial_state.BUILTIN_INITIAL_JOINTS, 'kinova-gen3',
                        {'left': {'arm_joint': .2}})
    cuda = SimpleNamespace(manual_seed_all=lambda _: None, empty_cache=lambda: None,
                           get_device_name=lambda _: 'test GPU')
    runtime = SimpleNamespace(device='cpu', curobo=None, torch=SimpleNamespace(
        __version__='test', version=SimpleNamespace(cuda=None), cuda=cuda, manual_seed=lambda _: None))
    monkeypatch.setattr(common, 'CuroboRuntime', lambda _: runtime)
    calls = []
    def solve(*args):
        geometry = args[2]
        calls.append(geometry.side)
        diagnostics = [dict(success=success or i < 2,
                            failure_reasons=[] if success or i < 2 else ['position'],
                            joint_step_max_abs_rad=.1) for i in range(3)]
        summary = dict(success=3 if success else 2, failure=0 if success else 1,
                       dropped=0, dropped_frames=[], trajectory_success=success)
        return np.array([[.2], [.3], [.4]]), diagnostics, summary
    monkeypatch.setattr(common, 'solve_arm', solve)
    args = [str(episode), '--embodiments-root', str(root)]
    if success:
        common.main('kinova-gen3', args)
    else:
        with pytest.raises(SystemExit) as error:
            common.main('kinova-gen3', args)
        assert error.value.code == 1
    assert calls == ['left']
    output = episode / 'TCP_prediction_ik/kinova_gen3'
    metadata = json.loads((output / 'metadata.json').read_text())
    assert set(metadata['arms']) == set(metadata['counts']) == {'left'}
    assert metadata['output']['shape'] == [3, 2]
    assert len(metadata['initialization']['state_vector']) == 2
    report = json.loads((output / 'diagnostics.json').read_text())
    assert all('left' in frame and 'right' not in frame for frame in report['frames'])
    ep = SimpleNamespace(path=episode, count=3)
    if not success:
        assert not (output / 'robot_state.npy').exists()
        assert np.load(output / 'robot_state_candidate.npy').shape == (3, 2)
        with pytest.raises(ValueError, match='successful v3'):
            load_target(output, ep, root)
        return

    saved = np.load(output / 'robot_state.npy')
    np.testing.assert_allclose(saved, [[.2, 1], [.3, 0], [.4, 0]])
    target = load_target(output, ep, root)
    assert set(target.arms) == {'left'}
    assert ik_metrics(episode, 'kinova_gen3')['side'] == 'left'
    rendered = render_target(output, ep, root)
    assert set(rendered.arms) == {'left'}
    np.testing.assert_array_equal(np.load(output / 'robot_state.npy'), saved)
    assert not (output / 'render_side.json').exists()

    # Exercise renderer assembly without requiring a GPU or Vulkan driver.
    articulations = []
    def build():
        articulation = SimpleNamespace(get_active_joints=lambda: [
            SimpleNamespace(name=n) for n in ('arm_joint', 'finger_joint')])
        articulations.append(articulation)
        return articulation
    builder = SimpleNamespace(set_initial_pose=lambda _: None, build=build)
    loader = SimpleNamespace(load_file_as_articulation_builder=lambda _: builder)
    scene = SimpleNamespace(set_ambient_light=lambda _: None,
        add_directional_light=lambda *a, **k: None, add_camera=lambda *a: None,
        create_urdf_loader=lambda: loader)
    monkeypatch.setitem(sys.modules, 'sapien', SimpleNamespace(Scene=lambda: scene,
        Pose=lambda matrix: matrix, render=SimpleNamespace(
            set_camera_shader_dir=lambda _: None, set_msaa=lambda _: None)))
    with RobotRenderer(rendered, 48, 64) as renderer:
        assert len(renderer.instances) == len(articulations) == 1
        np.testing.assert_allclose(renderer.instances[0][1][:, 0], saved[:, 0])
        np.testing.assert_allclose(renderer.instances[0][1][:, 1], rendered.state[:, 1] * .04)


def test_dual_slot_ik_cannot_be_reused_as_single_arm(tmp_path):
    from aug_pipeline.config import write_json
    from aug_pipeline.droid.retarget import ik_metrics
    from aug_pipeline.droid.robot import validate_robot
    output = tmp_path / 'TCP_prediction_ik/kinova_gen3'
    arms = {side: dict(joint_columns=[i * 2], gripper_column=i * 2 + 1)
            for i, side in enumerate(('left', 'right'))}
    write_json(output / 'metadata.json', dict(status='success', frame_count=3, arms=arms))
    state = np.zeros((3, 4))
    np.save(output / 'robot_state.npy', state)
    assert ik_metrics(tmp_path, 'kinova_gen3') == {}
    with pytest.raises(ValueError, match='single-arm metadata'):
        validate_robot(SimpleNamespace(state=state, arms=arms), 3, True)


def test_base_search_samples_only_failed_single_arm_frames(tmp_path):
    from aug_pipeline.config import write_json
    from aug_pipeline.droid.search import subset_frames
    prediction = tmp_path / 'prediction.json'
    write_json(prediction, dict(frames=[{}] * 30))
    write_json(tmp_path / 'TCP_prediction_ik/kinova_gen3/diagnostics.json', dict(
        frames=[dict(frame_index=i, left=dict(success=i != 5)) for i in range(30)]))
    assert subset_frames(tmp_path, 'kinova_gen3', 10, prediction) == [0, 5, 10, 20]
