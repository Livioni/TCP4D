"""Refine SAM3 video masks from durable original-pixel prompts."""
from __future__ import annotations

import gc
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


class FramePreviewSession:
    """Keep SAM3 weights and the active frame's features resident between clicks."""

    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.model = None
        self.state = None
        self.temporary = None
        self.identity = None

    def update(self, image, prompts, text):
        import torch
        from .droid.sam3 import build_model, _pick_mask

        image = Path(image).resolve()
        stat = image.stat()
        points = [point for prompt in prompts for point in prompt.get('points', [])]
        labels = [label for prompt in prompts for label in prompt.get('labels', [])]
        box = next((p['box'] for p in reversed(prompts) if p.get('box')), None)
        identity = (str(image), stat.st_size, stat.st_mtime_ns, text, tuple(box) if box else None)
        self.frame_reused = identity == self.identity
        if self.model is None:
            self.model = build_model(self.checkpoint)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if not self.frame_reused:
                self.clear_frame()
                with Image.open(image) as source:
                    self.rgb = np.asarray(source.convert('RGB')).copy()
                height, width = self.rgb.shape[:2]
                self.temporary = tempfile.TemporaryDirectory(prefix='tcp4d_sam3_frame_')
                (Path(self.temporary.name) / '00000.jpg').symlink_to(image)
                self.state = self.model.init_state(resource_path=self.temporary.name,
                    offload_video_to_cpu=True, use_cv2=True)
                seed = dict(inference_state=self.state, frame_idx=0, text_str=text)
                if box:
                    x0, y0, x1, y1 = box
                    seed.update(boxes_xywh=torch.tensor([[x0/width, y0/height,
                        (x1-x0)/width, (y1-y0)/height]], dtype=torch.float32),
                        box_labels=torch.tensor([1], dtype=torch.int32))
                _, data = self.model.add_prompt(**seed)
                ids = np.asarray(data['out_obj_ids'])
                scores = np.asarray(data.get('out_probs', np.ones(len(ids))))
                self.obj_id = int(ids[int(scores.argmax())]) if len(ids) else 1
                self.seed_mask = _pick_mask(data).copy()
                self.identity = identity
            height, width = self.rgb.shape[:2]
            if points:
                _, data = self.model.add_prompt(inference_state=self.state, frame_idx=0, obj_id=self.obj_id,
                    points=torch.tensor([[x/width, y/height] for x, y in points], dtype=torch.float32),
                    point_labels=torch.tensor(labels, dtype=torch.int32), clear_old_points=True)
                mask = _pick_mask(data)
            else:
                mask = self.seed_mask.copy()
        return mask, self.rgb

    def clear_frame(self):
        if self.state is not None:
            import torch
            with torch.inference_mode():
                self.model.reset_state(self.state)
            self.state = None
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None
        self.identity = None
        self.rgb = None
        self.seed_mask = None

    def close(self):
        self.clear_frame()
        if self.model is not None:
            import torch
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()


def preview_frame(config, image, output, prompts, text):
    """Standalone single-frame preview; the Web UI keeps a session alive."""
    session = FramePreviewSession(config['sam3_checkpoint'])
    try:
        mask, source = session.update(image, prompts, text)
        rgb = source.copy()
    finally:
        session.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'mask.png')
    rgb[mask] = (rgb[mask] * .4 + np.array([255, 85, 100]) * .6).astype(np.uint8)
    Image.fromarray(rgb).save(output / 'overlay.png')
    return int(mask.sum())


# The multiplex tracker retains per-frame features and copies them when a user
# refines an object. Bound the entire session, not just the input RGB storage.
REFINEMENT_FRAMES = 32
REFINEMENT_OVERLAP = 8


def _refine_clip(model, files, prompts, text, width, height):
    import torch

    with tempfile.TemporaryDirectory(prefix='tcp4d_sam3_') as tmp:
        jpg = Path(tmp)
        for i, p in enumerate(files):
            (jpg / f'{i:05d}.jpg').symlink_to(p.resolve())
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = model.init_state(resource_path=str(jpg), offload_video_to_cpu=True, use_cv2=True)
            try:
                first = prompts[0]
                seed = dict(inference_state=state, frame_idx=first['frame'], text_str=text)
                if first.get('box'):
                    x0, y0, x1, y1 = first['box']
                    seed.update(boxes_xywh=torch.tensor([[x0/width, y0/height,
                        (x1-x0)/width, (y1-y0)/height]], dtype=torch.float32),
                        box_labels=torch.tensor([1], dtype=torch.int32))
                _, initial = model.add_prompt(**seed)
                ids = np.asarray(initial['out_obj_ids'])
                scores = np.asarray(initial.get('out_probs', np.ones(len(ids))))
                obj_id = int(ids[int(scores.argmax())]) if len(ids) else None

                # Ground every frame before instance refinement so propagation
                # can merge corrections with the existing per-frame predictions.
                for _, data in model.propagate_in_video(inference_state=state,
                        start_frame_idx=0, reverse=False):
                    ids = np.asarray(data['out_obj_ids'])
                    if obj_id is None and len(ids):
                        scores = np.asarray(data.get('out_probs', np.ones(len(ids))))
                        obj_id = int(ids[int(scores.argmax())])
                detected = obj_id is not None
                if obj_id is None:
                    obj_id = 1

                for p in prompts:
                    points, labels = list(p['points']), list(p['labels'])
                    # Repeated semantic boxes reset object IDs; use the box
                    # centre as a foreground point for later corrections.
                    if p.get('box') or (p is first and not detected):
                        if p.get('box'):
                            x0, y0, x1, y1 = p['box']
                            inferred = [[(x0+x1)/2, (y0+y1)/2]]
                        else:
                            inferred = []
                        negatives = [q for q, label in zip(points, labels) if label == 0]
                        inferred = [q for q in inferred if all(np.linalg.norm(np.asarray(q)-n) > 6 for n in negatives)]
                        points = inferred + points
                        labels = [1] * len(inferred) + labels
                    if not points:
                        continue
                    pts = [[float(x)/width, float(y)/height] for x, y in points]
                    model.add_prompt(inference_state=state, frame_idx=p['frame'], obj_id=obj_id,
                        points=torch.tensor(pts, dtype=torch.float32),
                        point_labels=torch.tensor(labels, dtype=torch.int32))
                masks = {}
                for reverse, start in [(False, prompts[0]['frame']), (True, prompts[-1]['frame'])]:
                    for index, data in model.propagate_in_video(inference_state=state,
                            start_frame_idx=start, reverse=reverse):
                        ids = list(data['out_obj_ids'])
                        masks[int(index)] = (np.asarray(data['out_binary_masks'])[ids.index(obj_id)].astype(bool)
                            if obj_id in ids else np.zeros((height, width), bool))
            finally:
                model.reset_state(state)
        if sorted(masks) != list(range(len(files))):
            raise ValueError('Interactive propagation did not cover every frame')
        return masks


def _mask_prompt(mask, frame):
    """Carry an object's identity across sessions using interior/exterior clicks."""
    import cv2

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    kernel = np.ones((5, 5), dtype=np.uint8)
    interior = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    exterior = ~cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    points, labels = [], []
    for region, label in ((interior if interior.any() else mask, 1), (exterior, 0)):
        ys, xs = np.nonzero(region)
        if len(xs):
            chosen = np.linspace(0, len(xs) - 1, min(9, len(xs))).astype(int)
            points.extend([[float(xs[i]), float(ys[i])] for i in chosen])
            labels.extend([label] * len(chosen))
    return dict(frame=frame, points=points, labels=labels)


def _refinement_windows(count, anchor):
    """Start at a user correction, then extend in both directions with overlap."""
    start = max(0, anchor - REFINEMENT_OVERLAP)
    end = min(count, start + REFINEMENT_FRAMES)
    yield start, end, None
    first_start = start
    while end < count:
        start = end - REFINEMENT_OVERLAP
        end = min(count, start + REFINEMENT_FRAMES)
        yield start, end, start
    start = first_start
    while start > 0:
        end = start + REFINEMENT_OVERLAP
        start = max(0, end - REFINEMENT_FRAMES)
        yield start, end, end - 1


def refine(config, episode, output, request):
    import torch
    from .droid.sam3 import build_model

    grouped = {}
    for p in request['prompts']:
        item = grouped.setdefault(p['frame'], dict(frame=p['frame'], points=[], labels=[]))
        item['points'].extend(p.get('points', []))
        item['labels'].extend(p.get('labels', []))
        if p.get('box'):
            item['box'] = p['box']
    prompts = sorted(grouped.values(), key=lambda p: p['frame'])
    if not prompts:
        raise ValueError('Interactive refinement requires at least one prompt')
    files = sorted((episode / 'images/third_views').glob('*.png'))
    if not files or any(not 0 <= p['frame'] < len(files) for p in prompts):
        raise ValueError('Refinement prompts must refer to existing video frames')
    with Image.open(files[0]) as image:
        width, height = image.size
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(Path(config['sam3_checkpoint']))
    covered = set()
    try:
        for start, end, carry_frame in _refinement_windows(len(files), prompts[0]['frame']):
            local = [dict(p, frame=p['frame'] - start) for p in prompts if start <= p['frame'] < end]
            if carry_frame is not None and carry_frame not in grouped:
                with Image.open(output / f'{carry_frame:06d}.png') as prior:
                    carry = _mask_prompt(np.asarray(prior) > 0, carry_frame - start)
                if carry is not None:
                    local.append(carry)
            # Reacquire from text if the carried object disappeared in the overlap.
            if not local:
                local = [dict(frame=0, points=[], labels=[])]
            local.sort(key=lambda p: p['frame'])
            masks = _refine_clip(model, files[start:end], local, request['text'], width, height)
            for index, mask in masks.items():
                global_index = start + index
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / f'{global_index:06d}.png')
                covered.add(global_index)
            del masks
            gc.collect()
            torch.cuda.empty_cache()
            print(f'SAM3 refinement frames {start}-{end - 1} ({len(covered)}/{len(files)})', flush=True)
        if covered != set(range(len(files))):
            raise ValueError('Interactive propagation did not cover every frame')
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    from .config import Config, read_json

    parser = argparse.ArgumentParser(description='Preview SAM3 corrections on one frame')
    parser.add_argument('--config', required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--text', default='robot arm')
    args = parser.parse_args()
    preview_frame(Config(args.config), args.image, args.output, read_json(args.prompts), args.text)
