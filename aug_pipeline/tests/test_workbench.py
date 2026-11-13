from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aug_pipeline.config import Config, REPO, ROOT, read_json, write_json
from aug_pipeline.data import adapt_prediction, inspect_episode
from aug_pipeline.jobs import JobManager
from aug_pipeline.web import create_app
from aug_pipeline.worker import Worker

SAMPLE = ROOT.parent / 'assets/samples/droid/AUTOLab__Fri_Jul_14_16:44:09_2023'


@pytest.fixture
def config(tmp_path):
    import yaml
    values = yaml.safe_load((ROOT / 'config.example.yaml').read_text())
    values['data_roots'] = [str(SAMPLE.parent)]
    values['work_dir'] = str(tmp_path / 'work')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(values))
    return Config(path)


def test_config_relative_paths_use_project_root_outside_config_directory_and_cwd(tmp_path, monkeypatch):
    import yaml
    values = yaml.safe_load((ROOT / 'config.example.yaml').read_text())
    directory = tmp_path / 'settings'
    directory.mkdir()
    path = directory / 'config.yaml'
    path.write_text(yaml.safe_dump(values))
    cwd = tmp_path / 'unrelated'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = Config(path)
    for key in ('work_dir', 'checkpoint', 'embodiments', 'sam3_checkpoint', 'propainter_repo'):
        assert config[key] == str(REPO / values[key])
    assert config['data_roots'] == [str(REPO / 'assets/pipeline')]
    snapshot = tmp_path / 'runtime.yaml'
    config.snapshot(snapshot)
    assert Config(snapshot).values == config.values


def test_config_keeps_absolute_paths_and_expands_home_paths(tmp_path):
    import yaml
    values = yaml.safe_load((ROOT / 'config.example.yaml').read_text())
    values['work_dir'] = str(tmp_path / 'work')
    values['checkpoint'] = str(tmp_path / 'model')
    values['sam3_checkpoint'] = '~/checkpoints/sam3.pt'
    values['data_roots'] = ['assets/pipeline', str(tmp_path / 'episodes'), '~/datasets']
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(values))
    config = Config(path)
    assert config['work_dir'] == str(tmp_path / 'work')
    assert config['checkpoint'] == str(tmp_path / 'model')
    assert config['sam3_checkpoint'] == str(Path.home() / 'checkpoints/sam3.pt')
    assert config['data_roots'] == [str(REPO / 'assets/pipeline'), str(tmp_path / 'episodes'), str(Path.home() / 'datasets')]


def raw_prediction():
    return dict(stage=1, camera='cam', episode_num_frames=3, frame_rate_hz=15,
        frames=[dict(frame_index=i, time_seconds=i/15, xyz_m=[.1, .2, 1.0], rpy_rad=[0, .1, 0],
                     confidence=.9, gripper_open=.6) for i in range(3)])


def test_released_prediction_keeps_camera_coordinates_and_binary_labels():
    raw = raw_prediction()
    out = adapt_prediction(raw, 3, 'cam', 15)
    assert out['frames'][1]['left']['xyz_m'] == [.1, .2, 1.0]
    assert out['frames'][1]['left']['gripper_open'] is True
    assert out['frames'][1]['left']['gripper_probability'] == .6
    assert all('right' not in frame for frame in out['frames'])
    out['frames'][1]['left']['gripper_open'] = False
    assert raw['frames'][1]['gripper_open'] == .6


@pytest.mark.parametrize('kind', ['missing', 'shuffled', 'nan', 'camera', 'timestamps'])
def test_prediction_rejects_ambiguous_or_incomplete_trajectories(kind):
    raw = raw_prediction()
    if kind == 'missing': raw['frames'].pop(1)
    if kind == 'shuffled': raw['frames'].reverse()
    if kind == 'nan': raw['frames'][1]['xyz_m'][0] = float('nan')
    if kind == 'camera': raw['camera'] = 'other'
    if kind == 'timestamps': raw['frames'][1]['time_seconds'] = 0
    with pytest.raises(ValueError):
        adapt_prediction(raw, 3, 'cam', 15)


def test_sample_geometry_and_stream_alignment():
    info = inspect_episode(SAMPLE, '22008760')
    assert (info['count'], info['width'], info['height']) == (126, 320, 180)
    assert 'gt_query' not in info


def test_edits_invalidate_only_dependent_stages(config):
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'mask_method': 'sam3'})
        for stage in job['stages']:
            job['stages'][stage] = dict(status='complete', fingerprint=manager.fingerprint(job, stage))
        manager.save(job)
        changed = manager.update(job['id'], {'prompts': [{'frame': 10, 'points': [[30, 40]], 'labels': [0]}]})
        assert all(changed['stages'][s]['status'] == 'complete' for s in ('track', 'ik'))
        assert all(changed['stages'][s]['status'] == 'pending' for s in ('mask', 'background', 'inpaint', 'render', 'export'))
    finally:
        manager.close()


def test_tcp_calibration_changes_invalidate_ik_render_and_export(config, monkeypatch):
    from aug_pipeline.droid.ik import _embodiments as embodiments
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'arms': ['kinova_gen3'], 'mask_method': 'sam3'})
        for stage in job['stages']:
            job['stages'][stage] = dict(status='complete', fingerprint=manager.fingerprint(job, stage))
        calibration = copy.deepcopy(embodiments.load_tcp_calibration('kinova-gen3'))
        calibration['rotation_matrix'] = np.eye(3).tolist()
        monkeypatch.setattr(embodiments, 'load_tcp_calibration', lambda _: calibration)
        manager.invalidate(job)
        assert all(job['stages'][s]['status'] == 'pending' for s in ('ik', 'render', 'export'))
        assert all(job['stages'][s]['status'] == 'complete'
                   for s in ('track', 'mask', 'background', 'inpaint'))
    finally:
        manager.close()


def test_per_arm_ik_cache_recomputes_after_tcp_calibration_change(config, monkeypatch):
    from aug_pipeline.droid import retarget
    from aug_pipeline.droid.ik import _embodiments as embodiments
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'arms': ['kinova_gen3'], 'auto_base': False})
        worker = Worker(config, manager.directory(job['id']))
        worker.prediction.parent.mkdir(parents=True)
        worker.prediction.write_text('{}')
        (worker.ep / 'extrinsics').mkdir()
        np.save(worker.ep / 'extrinsics/third_views.npy', np.eye(4))
        calls = []
        monkeypatch.setattr(retarget, 'solve', lambda *args: calls.append(args) or True)
        monkeypatch.setattr(retarget, 'ik_metrics', lambda *args: {'drops': 0})
        monkeypatch.setattr(retarget, 'failure_reasons', lambda *args: {})
        worker.ik()
        worker.ik()
        assert len(calls) == 1
        calibration = copy.deepcopy(embodiments.load_tcp_calibration('kinova-gen3'))
        calibration['rotation_matrix'] = np.eye(3).tolist()
        monkeypatch.setattr(embodiments, 'load_tcp_calibration', lambda _: calibration)
        worker.ik()
        assert len(calls) == 2
    finally:
        manager.close()


def test_single_arm_schema_invalidates_ik_and_its_outputs(config, monkeypatch):
    import aug_pipeline.jobs as module
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'arms': ['kinova_gen3']})
        monkeypatch.setattr(module, 'IK_SCHEMA', 1)
        for stage in job['stages']:
            job['stages'][stage] = dict(status='complete', fingerprint=manager.fingerprint(job, stage))
        monkeypatch.setattr(module, 'IK_SCHEMA', 2)
        manager.invalidate(job)
        assert all(job['stages'][s]['status'] == 'pending' for s in ('ik', 'render', 'export'))
        assert all(job['stages'][s]['status'] == 'complete'
                   for s in ('track', 'mask', 'background', 'inpaint'))
    finally:
        manager.close()


@pytest.mark.parametrize('status', ['ready', 'complete', 'partial', 'failed', 'cancelled', 'interrupted'])
def test_target_robots_are_locked_for_saved_runs(config, status):
    with TestClient(create_app(config)) as client:
        job = client.post('/api/jobs', json={'episode': str(SAMPLE), 'arms': ['ur5_wsg']}).json()
        manager = client.app.state.manager
        job['status'] = status
        manager.save(job)
        root = f'/api/jobs/{job["id"]}'
        before = client.get(root).json()
        for arms in (['franka_panda'], ['ur5_wsg', 'franka_panda'], []):
            response = client.patch(root, json={'arms': arms, 'query': [20, 30]})
            assert response.status_code == 400
            assert 'Target robots are locked' in response.json()['detail']
            assert client.get(root).json() == before
        with pytest.raises(ValueError, match='Target robots are locked'):
            manager.update(job['id'], {'arms': ['franka_panda']})
        updated = client.patch(root, json={'arms': ['ur5_wsg'], 'query': [20, 30]})
        assert updated.status_code == 200
        assert updated.json()['request']['arms'] == ['ur5_wsg']
        assert updated.json()['request']['query'] == [20, 30]
        created = client.post('/api/jobs', json={'episode': str(SAMPLE), 'arms': ['franka_panda']})
        assert created.status_code == 201
        assert created.json()['id'] != job['id']
        assert created.json()['request']['arms'] == ['franka_panda']
        assert client.get(root).json()['request']['arms'] == ['ur5_wsg']


def test_single_workspace_lease_and_interrupted_recovery(config):
    manager = JobManager(config)
    job = manager.create({'episode': str(SAMPLE)})
    with pytest.raises(RuntimeError, match='already has'):
        JobManager(config)
    manager.close()
    job['status'] = 'running'
    job['stages']['mask']['status'] = 'running'
    write_json(config.work / 'jobs' / job['id'] / 'job.json', job)
    manager = JobManager(config)
    try:
        assert manager.get(job['id'])['status'] == 'interrupted'
        assert manager.get(job['id'])['stages']['mask']['status'] == 'interrupted'
    finally:
        manager.close()


def wait_until(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('Timed out waiting for runner')


def test_cancel_terminates_worker_and_retains_resume_state(config, monkeypatch):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [])
    manager = JobManager(config)
    try:
        manager.worker_command = lambda *_: [sys.executable, '-c', 'import time; time.sleep(30)']
        job = manager.create({'episode': str(SAMPLE), 'query': [20, 30]})
        manager.start(job['id'], 'track')
        assert Config(manager.directory(job['id']) / 'runtime.yaml').values == config.values
        wait_until(lambda: bool(manager.processes))
        manager.cancel(job['id'])
        wait_until(lambda: not manager.processes)
        restored = manager.get(job['id'])
        assert restored['status'] == 'cancelled'
        assert restored['stages']['track']['status'] == 'cancelled'
        assert (manager.directory(job['id']) / 'run.log').is_file()
    finally:
        manager.close()


def test_failed_worker_does_not_mark_downstream_complete(config, monkeypatch):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [])
    manager = JobManager(config)
    try:
        manager.worker_command = lambda *_: [sys.executable, '-c', 'raise SystemExit(7)']
        job = manager.create({'episode': str(SAMPLE), 'query': [20, 30]})
        manager.start(job['id'], 'track')
        wait_until(lambda: manager.get(job['id'])['status'] == 'failed')
        failed = manager.get(job['id'])
        assert failed['stages']['track']['status'] == 'failed'
        assert failed['stages']['ik']['status'] == 'pending'
    finally:
        manager.close()


def test_api_validates_prompts_paths_and_inputs(config):
    with TestClient(create_app(config)) as client:
        assert client.post('/api/episodes/inspect', json={'episode': '/etc'}).status_code == 400
        job = client.post('/api/jobs', json={'episode': str(SAMPLE), 'mask_method': 'sam3'}).json()
        root = f'/api/jobs/{job["id"]}'
        assert client.get(root + '/artifacts/%2E%2E%2F%2E%2E%2Fconfig.yaml').status_code == 404
        assert client.patch(root, json={'window_size': 'invalid'}).status_code == 422
        assert client.post('/api/jobs', json={'episode': str(SAMPLE), 'arms': ['invalid']}).status_code == 400
        bad = {'prompts': [{'frame': 126, 'points': [[1, 1]], 'labels': [1]}]}
        assert client.put(root + '/prompts', json=bad).status_code == 400
        bad['prompts'][0].update(frame=0, labels=[])
        assert client.put(root + '/prompts', json=bad).status_code == 400
        good = {'prompts': [{'frame': 0, 'points': [[20, 30]], 'labels': [0]}]}
        assert client.put(root + '/prompts', json=good).status_code == 200
        manager = client.app.state.manager
        (manager.directory(job['id']) / 'dataset.zip').write_bytes(b'old export')
        assert client.get(root + '/artifacts/dataset.zip').status_code == 409
        assert not any(a['name'] == 'dataset.zip' for a in client.get(root + '/artifacts').json())


def test_missing_sam3_checkpoint_is_rejected_before_queueing(config, tmp_path):
    config.values['sam3_checkpoint'] = str(tmp_path / 'missing.pt')
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'mask_method': 'sam3'})
        job['stages']['track'] = dict(status='complete', fingerprint=manager.fingerprint(job, 'track'))
        manager.save(job)
        with pytest.raises(ValueError, match='sam3_checkpoint'):
            manager.start(job['id'], 'mask')
        assert manager.get(job['id'])['status'] == 'ready'
        assert not manager.processes
    finally:
        manager.close()


def test_final_masks_control_temporal_plate_and_preserve_excluded_pixels(config, tmp_path):
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'mask_method': 'sam3'})
        directory = manager.directory(job['id'])
        worker = Worker(config, directory)
        source = worker.ep / 'images/third_views'
        source.mkdir(parents=True)
        (directory / 'masks').mkdir()
        for i in range(7):
            rgb = np.full((2, 2, 3), 80, np.uint8)
            rgb[0, 0] = 240 if i < 3 else 40  # robot pixels must not contaminate the plate
            rgb[0, 1] = 123  # reviewed negative area remains untouched
            mask = np.zeros((2, 2), np.uint8)
            mask[0, 0] = 255 if i < 3 else 0
            mask[1, 1] = 255  # never observed, requires inpainting
            Image.fromarray(rgb).save(source / f'{i:06d}.png')
            Image.fromarray(mask).save(directory / 'masks' / f'{i:06d}.png')
        import aug_pipeline.worker as module
        original = module.video
        module.video = lambda *args: None
        try:
            worker.background()
        finally:
            module.video = original
        image = np.asarray(Image.open(directory / 'prefill/000000.png'))
        assert image[0, 0].tolist() == [40, 40, 40]
        assert image[0, 1].tolist() == [123, 123, 123]
        assert image[1, 1].tolist() == [0, 0, 0]
        residual = np.asarray(Image.open(directory / 'residual/000000.png'))
        assert residual.tolist() == [[0, 0], [0, 255]]
    finally:
        manager.close()


def test_failed_or_removed_arms_do_not_survive_in_regenerated_dataset_zip(config, monkeypatch):
    import zipfile
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE)})
        directory = manager.directory(job['id'])
        worker = Worker(config, directory)
        old = directory / 'datasets/franka_panda/old.npy'
        old.parent.mkdir(parents=True)
        old.write_bytes(b'previously successful arm, now failed')
        write_json(directory / 'ik_summary.json', {'ur5_wsg': {'status': 'success'}, 'franka_panda': {'status': 'failed'}})
        from aug_pipeline.droid import export as export_dataset
        def export_one(**kwargs):
            destination = kwargs['output_root'] / kwargs['arm']
            destination.mkdir(parents=True)
            write_json(destination / 'metadata.json', {'tcp': {}})
            return destination
        monkeypatch.setattr(export_dataset, 'export_arm_dataset', export_one)
        worker.export()
        with zipfile.ZipFile(directory / 'dataset.zip') as archive:
            assert 'ur5_wsg/metadata.json' in archive.namelist()
            assert not any('franka_panda' in name for name in archive.namelist())
    finally:
        manager.close()


def test_sam3_no_visible_object_is_an_empty_mask_with_original_dimensions():
    from aug_pipeline.droid.sam3 import _pick_mask
    mask = _pick_mask({'out_binary_masks': np.zeros((0, 180, 320), dtype=bool), 'out_probs': np.array([])})
    assert mask.shape == (180, 320)
    assert not mask.any()


def test_default_mask_is_sam3_for_api_and_cli_jobs(config):
    with TestClient(create_app(config)) as client:
        assert client.get('/api/capabilities').json()['default_mask'] == 'sam3'
        response = client.post('/api/jobs', json={'episode': str(SAMPLE)})
        assert response.status_code == 201
        assert response.json()['request']['mask_method'] == 'sam3'
        job = client.app.state.manager.create({'episode': str(SAMPLE)})
        assert job['request']['mask_method'] == 'sam3'
        assert client.post('/api/jobs', json={'episode': str(SAMPLE), 'mask_method': 'unknown'}).status_code == 422


def test_saved_job_with_unsupported_mask_cannot_resume(config):
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE)})
        job['request']['mask_method'] = 'unknown'
        manager.save(job)
        with pytest.raises(ValueError, match='RGB input requires SAM3'):
            manager.start(job['id'], 'track')
        assert not manager.processes
        assert manager.get(job['id'])['status'] == 'ready'
    finally:
        manager.close()


def test_rgb_masking_requires_sam3_without_source_robot_resources(config, monkeypatch):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [dict(name='sam3_checkpoint', ok=False)])
    with pytest.raises(ValueError, match='sam3_checkpoint'):
        resources.check_stages(config, {'mask_method': 'sam3', 'prompts': [{'frame': 0}]}, ['mask'])


@pytest.mark.parametrize('fail', [False, True])
def test_sam3_text_pass_uses_native_model_and_releases_state(monkeypatch, tmp_path, fail):
    from contextlib import nullcontext
    import torch
    from aug_pipeline.droid.sam3 import run_text_pass
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())

    class Model:
        reset = False

        def init_state(self, resource_path, offload_video_to_cpu=False, use_cv2=False):
            assert resource_path == str(tmp_path)
            assert offload_video_to_cpu and use_cv2
            return {}

        def add_prompt(self, inference_state, frame_idx, text_str):
            inference_state['text'] = text_str
            assert frame_idx == 0

        def propagate_in_video(self, inference_state, start_frame_idx, reverse):
            assert inference_state['text'] == 'robot arm'
            if fail:
                raise RuntimeError('inference failed')
            yield 0, dict(out_binary_masks=np.ones((1, 2, 3), bool))

        def reset_state(self, state):
            self.reset = True

    model = Model()
    if fail:
        with pytest.raises(RuntimeError, match='inference failed'):
            run_text_pass(model, tmp_path, 'robot arm')
    else:
        masks = run_text_pass(model, tmp_path, 'robot arm')
        assert masks[0].shape == (2, 3) and masks[0].all()
    assert model.reset


@pytest.mark.parametrize('detected', [True, False])
def test_sam3_refinement_grounds_all_frames_before_normalized_corrections(
        config, monkeypatch, tmp_path, detected):
    from contextlib import nullcontext
    import torch
    from aug_pipeline.droid import sam3
    from aug_pipeline.segmentation import refine
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    episode = tmp_path / 'episode'
    frames = episode / 'images/third_views'
    frames.mkdir(parents=True)
    output = tmp_path / 'masks'
    output.mkdir()
    for i in range(3):
        Image.new('RGB', (20, 20)).save(frames / f'{i:06d}.png')

    class Model:
        reset = False
        corrected = False

        def init_state(self, resource_path, offload_video_to_cpu=False, use_cv2=False):
            assert len(list(Path(resource_path).glob('*.jpg'))) == 3
            return {'grounded': set()}

        def add_prompt(self, inference_state, frame_idx, text_str=None, points=None,
                       point_labels=None, obj_id=None, boxes_xywh=None, box_labels=None):
            if text_str is not None:
                assert frame_idx == 1
                np.testing.assert_allclose(boxes_xywh.numpy(), [[.1, .1, .8, .8]])
                ids = np.array([7] if detected else [])
                return frame_idx, dict(out_obj_ids=ids)
            assert inference_state['grounded'] == {0, 1, 2}
            assert obj_id == (7 if detected else 1)
            assert torch.all((points >= 0) & (points < 1))
            assert (point_labels == 0).any()
            self.corrected = True

        def propagate_in_video(self, inference_state, start_frame_idx, reverse):
            order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 3)
            for index in order:
                inference_state['grounded'].add(index)
                visible = detected or self.corrected
                yield index, dict(out_obj_ids=np.array([7 if detected else 1] if visible else []),
                    out_binary_masks=np.full((1 if visible else 0, 20, 20), self.corrected, bool))

        def reset_state(self, state):
            self.reset = True

    model = Model()
    monkeypatch.setattr(sam3, 'build_model', lambda _: model)
    refine(config, episode, output, dict(mask_method='sam3', text='robot arm',
        prompts=[dict(frame=1, points=[[5, 5], [15, 15]], labels=[1, 0], box=[2, 2, 18, 18])]))
    assert model.reset and model.corrected
    assert len(list(output.glob('*.png'))) == 3
    for path in output.glob('*.png'):
        assert np.asarray(Image.open(path)).all()


@pytest.mark.parametrize('fail', [False, True])
def test_current_frame_preview_is_isolated_from_saved_masks_and_stages(config, monkeypatch, fail):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [])
    calls = []

    def update(image, prompts, text):
        calls.append((image, prompts, text))
        if fail:
            raise ValueError('inference failed')
        return dict(image='data:image/png;base64,cHJldmlldw==')

    with TestClient(create_app(config)) as client:
        job = client.post('/api/jobs', json={'episode': str(SAMPLE)}).json()
        manager = client.app.state.manager
        monkeypatch.setattr(manager.preview_client, 'update', update)
        for stage in job['stages']:
            job['stages'][stage] = dict(status='complete', fingerprint=manager.fingerprint(job, stage))
        manager.save(job)
        directory = manager.directory(job['id'])
        (directory / 'masks').mkdir()
        (directory / 'masks/000005.png').write_bytes(b'existing mask')
        before = manager.get(job['id'])
        payload = dict(frame=5, prompts=[dict(frame=5, points=[[260, 100], [100, 130]], labels=[1, 0])])
        response = client.post(f'/api/jobs/{job["id"]}/mask-preview', json=payload)
        assert response.status_code == (400 if fail else 200)
        if not fail:
            assert response.json()['frame'] == 5
            assert response.json()['image'].startswith('data:image/png;base64,')
        assert manager.get(job['id']) == before
        assert (directory / 'masks/000005.png').read_bytes() == b'existing mask'
        assert not list(directory.glob('.mask-preview-*'))
        assert len(calls) == 1
        assert str(calls[0][0]).endswith('/000005.png')
        assert calls[0][1][0]['labels'] == [1, 0]


def test_preview_rejects_invalid_or_busy_frame_without_starting_inference(config, monkeypatch):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [])
    with TestClient(create_app(config)) as client:
        job = client.post('/api/jobs', json={'episode': str(SAMPLE)}).json()
        url = f'/api/jobs/{job["id"]}/mask-preview'
        prompt = dict(frame=0, points=[[10, 20]], labels=[1])
        assert client.post(url, json=dict(frame=126, prompts=[prompt])).status_code == 400
        assert client.post(url, json=dict(frame=1, prompts=[prompt])).status_code == 400
        assert client.post(url, json=dict(frame=0, prompts=[])).status_code == 400
        prompt['labels'] = []
        assert client.post(url, json=dict(frame=0, prompts=[prompt])).status_code == 400
        prompt['labels'] = [1]
        job['status'] = 'running'
        client.app.state.manager.save(job)
        assert client.post(url, json=dict(frame=0, prompts=[prompt])).status_code == 400
        assert not client.app.state.manager.processes


def test_preview_updates_positive_and_negative_points_without_propagation(config, monkeypatch, tmp_path):
    from contextlib import nullcontext
    import torch
    from aug_pipeline.droid import sam3
    from aug_pipeline.segmentation import preview_frame
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    image = tmp_path / 'image.png'
    Image.new('RGB', (20, 10), color='#888888').save(image)

    class Model:
        reset = False

        def init_state(self, resource_path, **kwargs):
            assert len(list(Path(resource_path).glob('*.jpg'))) == 1
            return {}

        def add_prompt(self, inference_state, frame_idx, text_str=None, points=None, point_labels=None, obj_id=None, clear_old_points=True):
            assert frame_idx == 0
            mask = np.ones((1, 10, 20), bool)
            if points is not None:
                np.testing.assert_allclose(points.numpy(), [[.25, .5], [.75, .5]])
                assert point_labels.tolist() == [1, 0]
                assert obj_id == 7
                mask[0, 5, 15] = False
            return 0, dict(out_obj_ids=np.array([7]), out_binary_masks=mask)

        def propagate_in_video(self, *args, **kwargs):
            raise AssertionError('A frame preview must not propagate')

        def reset_state(self, state):
            self.reset = True

    model = Model()
    monkeypatch.setattr(sam3, 'build_model', lambda _: model)
    output = tmp_path / 'preview'
    pixels = preview_frame(config, image, output, [dict(frame=5, points=[[5, 5], [15, 5]], labels=[1, 0])], 'robot arm')
    mask = np.asarray(Image.open(output / 'mask.png'))
    overlay = np.asarray(Image.open(output / 'overlay.png'))
    assert pixels == 199 and mask[5, 5] == 255 and mask[5, 15] == 0
    assert overlay[5, 15].tolist() == [136, 136, 136]
    assert overlay[5, 5].tolist() != [136, 136, 136]
    assert model.reset


def test_preview_session_reuses_weights_and_frame_features_and_replaces_points(config, monkeypatch, tmp_path):
    from contextlib import nullcontext
    import torch
    from aug_pipeline.droid import sam3
    from aug_pipeline.segmentation import FramePreviewSession
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    image = tmp_path / 'image.png'
    other = tmp_path / 'other.png'
    Image.new('RGB', (20, 10), '#888888').save(image)
    Image.new('RGB', (20, 10), '#999999').save(other)
    builds, clears = [], []
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: clears.append(True))

    class Model:
        states, seeds, resets = [], 0, 0

        def init_state(self, **kwargs):
            state = {}
            self.states.append(state)
            return state

        def add_prompt(self, inference_state, frame_idx, text_str=None, points=None,
                       point_labels=None, obj_id=None, clear_old_points=True):
            mask = np.ones((1, 10, 20), bool)
            if text_str:
                self.seeds += 1
            else:
                assert clear_old_points
                for (x, y), label in zip(points.tolist(), point_labels.tolist()):
                    mask[0, round(y*10), round(x*20)] = bool(label)
            return 0, dict(out_obj_ids=np.array([7]), out_binary_masks=mask)

        def reset_state(self, state):
            assert torch.is_inference_mode_enabled()
            self.resets += 1

    model = Model()
    def build(checkpoint):
        builds.append(checkpoint)
        return model
    monkeypatch.setattr(sam3, 'build_model', build)
    session = FramePreviewSession(config['sam3_checkpoint'])
    positive = [dict(frame=0, points=[[5, 5]], labels=[1])]
    negative = positive + [dict(frame=0, points=[[15, 5]], labels=[0])]
    try:
        session.update(image, positive, 'robot arm')
        state = session.state
        assert not session.frame_reused
        mask, _ = session.update(image, negative, 'robot arm')
        assert session.frame_reused and session.state is state and not mask[5, 15]
        mask, _ = session.update(image, positive, 'robot arm')
        assert session.frame_reused and mask[5, 15]
        assert len(builds) == len(model.states) == model.seeds == 1
        assert not clears
        session.update(other, positive, 'robot arm')
        assert not session.frame_reused
        assert len(builds) == 1 and len(model.states) == model.seeds == 2
    finally:
        session.close()
    assert model.resets == 2 and clears == [True]


def echo_preview_worker(connection, config):
    import os
    import time
    count = 0
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        count += 1
        if request['text'] == 'exit':
            connection.close()
            return
        if request['text'] == 'wait':
            time.sleep(30)
        if request['text'] == 'error':
            connection.send(dict(ok=False, error='test error'))
        else:
            connection.send(dict(ok=True, result=dict(pid=os.getpid(), count=count)))


def test_preview_worker_stays_alive_between_clicks_and_restarts_after_release(config):
    from aug_pipeline.preview import PreviewClient
    client = PreviewClient(config, worker=echo_preview_worker)
    try:
        first = client.update('image', [], 'robot arm')
        second = client.update('image', [], 'robot arm')
        assert first['pid'] == second['pid'] and second['count'] == 2
        with pytest.raises(ValueError, match='test error'):
            client.update('image', [], 'error')
        assert client.update('image', [], 'robot arm')['pid'] == first['pid']
        client.release()
        assert client.process is None
        restarted = client.update('image', [], 'robot arm')
        assert restarted['pid'] != first['pid'] and restarted['count'] == 1
    finally:
        client.shutdown()
    with pytest.raises(ValueError, match='shutting down'):
        client.update('image', [], 'robot arm')


@pytest.mark.parametrize('text', ['exit', 'wait'])
def test_preview_worker_failure_releases_process_and_can_be_retried(config, text):
    from aug_pipeline.preview import PreviewClient
    client = PreviewClient(config, worker=echo_preview_worker)
    try:
        client.update('image', [], 'robot arm')
        with pytest.raises(ValueError, match='try again'):
            client.update('image', [], text, timeout=.1)
        assert client.process is None
        assert client.update('image', [], 'robot arm')['count'] == 1
    finally:
        client.shutdown()


def test_tracking_needs_a_query_and_runs_without_a_prepare_stage(config, monkeypatch):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [])
    releases = []
    with TestClient(create_app(config)) as client:
        manager = client.app.state.manager
        monkeypatch.setattr(manager.preview_client, 'release', lambda: releases.append(True))
        manager.worker_command = lambda *_: [sys.executable, '-c', 'pass']
        job = manager.create({'episode': str(SAMPLE)})
        url = f'/api/jobs/{job["id"]}/run'
        assert list(job['stages'])[0] == 'track'
        assert 'prepare' not in client.get('/api/capabilities').json()['stages']
        assert client.post(url, json={}).status_code == 422
        assert client.post(url, json={'stage': 'all'}).status_code == 422
        assert client.post(url, json={'stage': 'prepare'}).status_code == 422
        response = client.post(url, json={'stage': 'track'})
        assert response.status_code == 400 and 'Click the initial TCP' in response.json()['detail']
        assert manager.get(job['id'])['status'] == 'ready'
        response = client.post(url, json={'stage': 'mask'})
        assert response.status_code == 400 and 'track' in response.json()['detail']
        assert not manager.processes and not releases
        assert all(s['status'] == 'pending' for s in manager.get(job['id'])['stages'].values())
        saved = client.patch(f'/api/jobs/{job["id"]}', json={'query': [20, 30]})
        assert saved.status_code == 200 and saved.json()['request']['query'] == [20, 30]
        assert client.post(url, json={'stage': 'track'}).status_code == 200
        wait_until(lambda: manager.get(job['id'])['stages']['track']['status'] == 'complete')
        assert releases == [True]
        assert all(s['status'] == 'pending' for name, s in manager.get(job['id'])['stages'].items() if name != 'track')
        assert client.post(url, json={'stage': 'mask'}).status_code == 200
        wait_until(lambda: manager.get(job['id'])['stages']['mask']['status'] == 'complete')
        assert releases == [True, True]
        assert manager.get(job['id'])['stages']['ik']['status'] == 'pending'


def test_legacy_jobs_drop_prepare_and_can_resume_tracking(config, monkeypatch):
    import aug_pipeline.resources as resources
    monkeypatch.setattr(resources, 'doctor', lambda *_: [])
    manager = JobManager(config)
    job = manager.create({'episode': str(SAMPLE)})
    job['stages']['prepare'] = {'status': 'complete'}
    job['stages']['track'] = {'status': 'failed'}
    job['status'] = 'failed'
    manager.save(job)
    manager.close()
    manager = JobManager(config)
    try:
        restored = manager.get(job['id'])
        assert 'prepare' not in restored['stages']
        manager.worker_command = lambda *_: [sys.executable, '-c', 'pass']
        manager.update(job['id'], {'query': [20, 30]})
        manager.start(job['id'], 'track')
        wait_until(lambda: manager.get(job['id'])['stages']['track']['status'] == 'complete')
        assert manager.get(job['id'])['status'] == 'ready'
    finally:
        manager.close()


def test_query_edits_invalidate_tracking_but_keep_the_source_frame_cache(config):
    manager = JobManager(config)
    try:
        job = manager.create({'episode': str(SAMPLE), 'query': [20, 30]})
        source = manager.source_fingerprint(job)
        for stage in job['stages']:
            job['stages'][stage] = dict(status='complete', fingerprint=manager.fingerprint(job, stage))
        manager.save(job)
        changed = manager.update(job['id'], {'query': [40, 50]})
        assert manager.source_fingerprint(changed) == source
        assert all(s['status'] == 'pending' for s in changed['stages'].values())
    finally:
        manager.close()


@pytest.mark.parametrize('changed', [False, True])
def test_prepare_migration_preserves_only_valid_completed_outputs(config, changed):
    manager = JobManager(config)
    job = manager.create({'episode': str(SAMPLE), 'query': [20, 30]})
    job['stages']['prepare'] = {'status': 'complete'}
    for stage in job['stages']:
        job['stages'][stage] = dict(status='complete', fingerprint=manager.fingerprint(job, stage, legacy=True))
    if changed:
        job['request']['query'] = [40, 50]
    manager.save(job)
    manager.close()
    manager = JobManager(config)
    try:
        migrated = manager.get(job['id'])
        assert 'prepare' not in migrated['stages']
        manager.invalidate(migrated)
        assert migrated['stages']['track']['status'] == ('pending' if changed else 'complete')
        assert migrated['stages']['ik']['status'] == ('pending' if changed else 'complete')
        if not changed:
            assert all(s['status'] == 'complete' for s in migrated['stages'].values())
    finally:
        manager.close()
