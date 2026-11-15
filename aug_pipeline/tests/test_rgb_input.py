from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import infer_stage1 as stage1
from aug_pipeline.config import Config, ROOT, write_json
from aug_pipeline.data import inspect_episode, prepare_rgb_input, read_input_frame
from aug_pipeline.droid.episode import load_episode
from aug_pipeline.worker import Worker
from tcp4d.stage1_geometry import decode_camera_geometry


@pytest.fixture
def rgb_input(tmp_path):
    source = tmp_path / 'rgb'
    source.mkdir()
    for number, value in [(1, 30), (2, 60), (10, 90), (11, 120)]:
        Image.new('RGB', (64, 48), (value, 0, 0)).save(source / f'frame_{number}.png')
    return source


def test_rgb_input_needs_no_metadata_labels_depth_or_calibration(rgb_input, tmp_path):
    # Unusable optional ground truth files must never be opened.
    for directory in ('observations', 'intrinsic', 'extrinsic', 'TCP/rgb'):
        target = rgb_input / directory
        target.mkdir(parents=True)
        (target / 'state.npy').write_bytes(b'not a numpy array')
    info = inspect_episode(rgb_input, fps=12.5)
    assert (info['count'], info['fps']) == (4, 12.5)
    assert 'gt_query' not in info
    assert [read_input_frame(info, i).getpixel((0, 0))[0] for i in range(4)] == [30, 60, 90, 120]
    out = tmp_path / 'prepared'
    prepare_rgb_input(info, out)
    episode = stage1.load_episode(out, 'third_views')
    assert episode.intrinsics is None
    assert episode.frame_rate == 12.5
    np.testing.assert_allclose(episode.timestamps, [0, .08, .16, .24])
    assert not (out / 'robot_state.npy').exists()
    assert not (out / 'intrinsics').exists()


def test_mp4_input_decodes_every_frame_and_uses_video_fps(tmp_path):
    import cv2
    path = tmp_path / 'demo.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.full((360, 640, 3), 40 * index, dtype=np.uint8))
    writer.release()
    info = inspect_episode(path)
    assert (info['input_type'], info['count'], info['fps'], info['width'], info['height']) == ('video', 5, 20, 320, 180)
    assert read_input_frame(info, 4).size == (320, 180)
    out = tmp_path / 'prepared'
    prepare_rgb_input(info, out)
    assert len(list((out / 'images/third_views').glob('*.png'))) == 5
    assert json.loads((out / 'metadata.json').read_text())['input_type'] == 'video'
    np.testing.assert_allclose(np.load(out / 'timestamps.npy'), np.arange(5) / 20)


def test_inconsistent_rgb_sizes_are_rejected(rgb_input, tmp_path):
    Image.new('RGB', (32, 24)).save(rgb_input / 'frame_11.png')
    with pytest.raises(ValueError, match='dimensions differ'):
        prepare_rgb_input(inspect_episode(rgb_input), tmp_path / 'prepared')


class PredictedScene:
    num_arms = 1

    def __init__(self):
        self.queries = []

    def __call__(self, views, **kwargs):
        self.queries.append(kwargs['tcp_query_points'].clone())
        count = len(views)
        height, width = views[0]['img'].shape[-2:]
        encoding = torch.tensor([1., 2., 3., 0., 0., 0., 1., .8, .9]).repeat(1, count, 1)
        return dict(pose_enc=encoding, depth=torch.stack([
            torch.full((height, width), float(v['idx'] + 1)) for v in views])[None],
            tcp_position=torch.tensor([.02, .01, 1.]).repeat(1, count, 1, 1),
            tcp_rotation=torch.eye(3).repeat(1, count, 1, 1, 1),
            tcp_gripper_logit=torch.full((1, count, 1), -1.),
            tcp_confidence=torch.ones((1, count, 1)))


def test_rgb_worker_consumes_only_tcp4d_predictions_across_windows(rgb_input, tmp_path, monkeypatch):
    import aug_pipeline.worker as module
    values = yaml.safe_load((ROOT / 'config.example.yaml').read_text())
    values.update(data_roots=[str(tmp_path)], work_dir=str(tmp_path / 'work'))
    config_path = tmp_path / 'config.yml'
    config_path.write_text(yaml.safe_dump(values))
    config = Config(config_path)
    directory = config.work / 'job'
    directory.mkdir(parents=True)
    info = inspect_episode(rgb_input)
    write_json(directory / 'job.json', dict(id='rgb-test', episode=info, request=dict(
        episode=str(rgb_input), camera='rgb', query=[20, 15], window_size=3, mask_method='sam3')))
    worker = Worker(config, directory)
    monkeypatch.setattr(module, 'video', lambda *args: None)
    model = PredictedScene()

    def predict(command):
        assert '--query-from-gt' not in command
        args = stage1.build_parser().parse_args([str(v) for v in command[2:]])
        episode = stage1.load_episode(args.input, args.camera)
        assert episode.intrinsics is None
        stage1.run_inference(args, episode, episode.image_paths, np.array([[20, 15]], dtype=np.float32),
                             'manual pixel', model, torch.device('cpu'), torch.float32)

    monkeypatch.setattr(module, 'run', predict)
    worker.track()
    episode = load_episode(worker.ep, True)
    assert episode.intrinsics.shape == (4, 3, 3)
    np.testing.assert_allclose(episode.extrinsics[:, :3, 3], np.tile([-1, -2, -3], (4, 1)))
    for index in range(4):
        _, depth = episode.read_frame(index)
        np.testing.assert_allclose(depth, index + 1)
    np.testing.assert_allclose(np.load(worker.ep / 'gripper_openness.npy'), torch.sigmoid(torch.tensor(-1.)).item())
    assert not (worker.ep / 'robot_state.npy').exists()
    assert len(model.queries) == 2
    point = episode.intrinsics[2] @ np.array([.02, .01, 1.])
    pad = stage1.load_episode(worker.ep).padding
    np.testing.assert_allclose(model.queries[1][0, 0].numpy(), point[:2] + [pad[0], pad[2]], atol=1e-5)
    prediction = json.loads(worker.prediction.read_text())
    assert len(prediction['frames']) == 4
    assert prediction['prediction']['initial_query_source'] == 'manual pixel'
    # Export model-estimated per-frame cameras alongside the target IK labels.
    from aug_pipeline.droid.export import export_arm_dataset
    ik = worker.ep / 'TCP_prediction_ik/kinova_gen3'
    ik.mkdir(parents=True)
    write_json(ik / 'metadata.json', dict(status='success', frame_count=4,
        embodiment='kinova-gen3', arms=dict(left=dict(joint_columns=[0], gripper_column=1,
                                                     active_joint_names=['joint_1']))))
    # A stale choice from a previous dual-slot run must not affect this export.
    write_json(ik / 'render_side.json', dict(render_side='right'))
    np.save(ik / 'robot_state.npy', np.zeros((4, 2), dtype=np.float32))
    exported = export_arm_dataset(frames_dir=worker.ep / 'images/third_views', ik_dir=ik,
        converted=worker.ep, arm='kinova_gen3', dataset='droid', task='', episode_name='rgb',
        prediction_json=worker.prediction, output_root=directory / 'datasets', mask_method='sam3')
    np.testing.assert_allclose(np.load(exported / 'intrinsics.npy'), episode.intrinsics)
    np.testing.assert_allclose(np.load(exported / 'extrinsics.npy'), episode.extrinsics)
    assert np.load(exported / 'states.npy').shape == (4, 2)

    # Rendering uses predicted gripper openness, without source joint states.
    from aug_pipeline.droid import render_target as target_module
    target = SimpleNamespace(name='kinova-gen3', config={}, state=np.zeros((4, 2)),
        arms=dict(left=dict(joint_columns=[0], gripper_column=1, world_from_root=np.eye(4))))
    monkeypatch.setattr(target_module, 'load_target', lambda *args: target)
    rendered = target_module.render_target(ik, episode, config['embodiments'])
    assert np.all(rendered.state[:, 1] > 0)
    assert set(rendered.arms) == {'left'}
    assert rendered.state.shape == (4, 2)
    np.testing.assert_array_equal(np.load(ik / 'robot_state.npy'), np.zeros((4, 2)))

    # SAM3 reads the prepared RGB directly; source-robot projection is absent.
    commands = []
    def mask(command):
        commands.append(command)
        out = Path(command[command.index('--output') + 1])
        for index in range(4):
            Image.new('L', (64, 48), 255).save(out / f'{index:06d}.png')
    monkeypatch.setattr(module, 'run', mask)
    worker.r.update(prompts=[], text='robot arm')
    worker.mask()
    assert len(commands) == 1 and commands[0][2] == 'aug_pipeline.droid.sam3'


def test_camera_decoder_removes_padding_and_inverts_c2w():
    episode = SimpleNamespace(source_width=320, source_height=180, padding=(1, 1, 1, 1))
    # +90 degrees around Z in camera-to-world coordinates.
    encoding = torch.tensor([[[1., 2., 3., 0., 0., 2 ** -.5, 2 ** -.5, .8, .9]]])
    k, e = decode_camera_geometry(dict(pose_enc=encoding), episode)
    np.testing.assert_allclose(k[0, :2, 2], [160, 90])
    expected = np.array([[0, 1, 0, -2], [-1, 0, 0, 1], [0, 0, 1, -3], [0, 0, 0, 1]])
    np.testing.assert_allclose(e[0], expected, atol=1e-6)


def test_tracking_requires_a_pixel_instead_of_falling_back_to_gt():
    worker = Worker.__new__(Worker)
    worker.r = dict(query=None)
    with pytest.raises(ValueError, match='Select the initial TCP'):
        worker.track()


def test_rgb_api_discovers_inputs_previews_frames_and_accepts_optional_fps(rgb_input, tmp_path):
    import io
    from fastapi.testclient import TestClient
    from aug_pipeline.web import create_app
    values = yaml.safe_load((ROOT / 'config.example.yaml').read_text())
    values.update(data_roots=[str(tmp_path)], work_dir=str(tmp_path / 'work'))
    config_path = tmp_path / 'api_config.yml'
    config_path.write_text(yaml.safe_dump(values))
    nested = tmp_path / 'pipeline' / 'episode'
    nested.mkdir(parents=True)
    video_path = nested / 'camera.mp4'
    video_path.write_bytes(b'listing only; decoding is tested separately')
    with TestClient(create_app(Config(config_path))) as client:
        listing = client.get('/api/episodes').json()
        assert any(item['path'] == str(rgb_input) for item in listing)
        assert any(item['path'] == str(video_path) for item in listing)
        info = client.post('/api/episodes/inspect', json=dict(episode=str(rgb_input), fps=24)).json()
        assert info['fps'] == 24 and 'gt_query' not in info
        frame = client.get('/api/episodes/frame', params=dict(episode=str(rgb_input), camera='rgb', index=2))
        assert frame.status_code == 200
        assert Image.open(io.BytesIO(frame.content)).getpixel((0, 0))[0] == 90
        job = client.post('/api/jobs', json=dict(episode=str(rgb_input), fps=24, query=[20, 15]))
        assert job.status_code == 201
        assert job.json()['episode']['fps'] == 24
        assert job.json()['request']['mask_method'] == 'sam3'
        assert client.post('/api/jobs', json=dict(episode=str(rgb_input), mask_method='urdf')).status_code == 422
