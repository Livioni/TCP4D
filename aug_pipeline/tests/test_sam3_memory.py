from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aug_pipeline import segmentation


@pytest.mark.parametrize('anchor', [0, 149])
def test_long_refinement_bounds_live_state_and_preserves_all_prompts(tmp_path, monkeypatch, anchor):
    import torch
    from aug_pipeline.droid import sam3

    frames = tmp_path / 'episode/images/third_views'
    frames.mkdir(parents=True)
    for index in range(240):
        Image.new('RGB', (20, 20)).save(frames / f'{index:06d}.png')
    request = dict(text='robot arm', prompts=[
        dict(frame=index, points=[[5, 5], [17, 17]], labels=[1, 0])
        for index in (anchor, anchor + 20, 220)])
    original = copy.deepcopy(request)
    seen = set()
    sessions, reset = [], []

    class Model:
        active = None

        def init_state(self, resource_path, **kwargs):
            assert self.active is None, 'Previous video state is still alive'
            files = sorted(Path(resource_path).glob('*.jpg'))
            assert len(files) <= segmentation.REFINEMENT_FRAMES
            assert kwargs['offload_video_to_cpu']
            indices = [int(p.resolve().stem) for p in files]
            self.active = dict(indices=indices, grounded=set())
            sessions.append(indices)
            return self.active

        def add_prompt(self, inference_state, frame_idx, **kwargs):
            assert 0 <= frame_idx < len(inference_state['indices'])
            if 'points' in kwargs:
                assert len(inference_state['grounded']) == len(inference_state['indices'])
                points = kwargs['points'].numpy() * 20
                labels = kwargs['point_labels'].numpy()
                assert np.all((points >= 0) & (points < 20))
                global_index = inference_state['indices'][frame_idx]
                if global_index in (anchor, anchor + 20, 220):
                    np.testing.assert_allclose(points, [[5, 5], [17, 17]])
                    np.testing.assert_array_equal(labels, [1, 0])
                    seen.add(global_index)
            return frame_idx, self.output()

        def output(self):
            mask = np.zeros((1, 20, 20), dtype=bool)
            mask[:, 3:15, 3:15] = True
            return dict(out_obj_ids=np.array([0]), out_binary_masks=mask)

        def propagate_in_video(self, inference_state, start_frame_idx, reverse):
            count = len(inference_state['indices'])
            order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, count)
            for index in order:
                inference_state['grounded'].add(index)
                yield index, self.output()

        def reset_state(self, state):
            assert state is self.active
            self.active = None
            reset.append(True)

    model = Model()
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(sam3, 'build_model', lambda _: model)
    output = tmp_path / 'masks'
    segmentation.refine(dict(sam3_checkpoint='unused'), tmp_path / 'episode', output, request)
    assert len(sessions) > 1 and len(reset) == len(sessions)
    assert model.active is None
    assert seen == {anchor, anchor + 20, 220}
    assert request == original
    assert [p.name for p in sorted(output.glob('*.png'))] == [f'{i:06d}.png' for i in range(240)]


def test_overlap_prompts_stay_inside_foreground_and_background():
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:45] = True
    prompt = segmentation._mask_prompt(mask, 7)
    assert prompt['frame'] == 7
    assert set(prompt['labels']) == {0, 1}
    for (x, y), label in zip(prompt['points'], prompt['labels']):
        assert bool(mask[int(y), int(x)]) == bool(label)
    assert segmentation._mask_prompt(np.zeros_like(mask), 0) is None


def test_failed_refinement_releases_the_clip_state(tmp_path, monkeypatch):
    import torch
    from aug_pipeline.droid import sam3

    frames = tmp_path / 'images/third_views'
    frames.mkdir(parents=True)
    for index in range(2):
        Image.new('RGB', (20, 20)).save(frames / f'{index:06d}.png')

    class Model:
        released = False

        def init_state(self, **kwargs):
            return {}

        def add_prompt(self, **kwargs):
            raise RuntimeError('inference failed')

        def reset_state(self, state):
            self.released = True

    model = Model()
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(sam3, 'build_model', lambda _: model)
    with pytest.raises(RuntimeError, match='inference failed'):
        segmentation.refine(dict(sam3_checkpoint='unused'), tmp_path, tmp_path / 'masks',
                            dict(text='robot arm', prompts=[dict(frame=0, points=[[5, 5]], labels=[1])]))
    assert model.released
