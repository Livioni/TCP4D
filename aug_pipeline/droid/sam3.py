"""Automatic SAM3 text masks; interactive prompts live in segmentation.py."""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def build_model(checkpoint):
    from sam3.model_builder import build_sam3_predictor

    messages = io.StringIO()
    with contextlib.redirect_stdout(messages):
        predictor = build_sam3_predictor(checkpoint_path=str(checkpoint), version='sam3.1',
            use_fa3=False, use_rope_real=False, max_num_objects=1, multiplex_count=16)
    for line in messages.getvalue().splitlines():
        print(line[:500], flush=True)
    model = predictor.model
    # Single-frame batches fit a shared 24-GB GPU at the original resolution.
    model.batched_grounding_batch_size = 1
    model.postprocess_batch_size = 1
    return model


def _pick_mask(outputs):
    masks = np.asarray(outputs['out_binary_masks'])
    if not len(masks):
        return np.zeros(masks.shape[-2:], dtype=bool)
    scores = outputs.get('out_probs')
    index = (int(np.argmax(scores)) if scores is not None
             else int(np.argmax(masks.reshape(len(masks), -1).sum(axis=1))))
    return masks[index].astype(bool)


def run_text_pass(model, frame_dir, text):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = model.init_state(resource_path=str(frame_dir), offload_video_to_cpu=True, use_cv2=True)
        try:
            model.add_prompt(inference_state=state, frame_idx=0, text_str=text)
            return {int(index): _pick_mask(outputs)
                    for index, outputs in model.propagate_in_video(
                        inference_state=state, start_frame_idx=0, reverse=False)}
        finally:
            model.reset_state(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--text', default='robot arm')
    args = parser.parse_args()
    frames = sorted((args.episode / 'images/third_views').glob('*.png'))
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(args.checkpoint)
    masks = {}
    try:
        # Overlap avoids detector dropouts at session boundaries.
        for start in range(0, len(frames), 32):
            chunk = frames[start:start+40]
            with tempfile.TemporaryDirectory(prefix='tcp4d_sam3_') as temporary:
                directory = Path(temporary)
                for i, path in enumerate(chunk):
                    (directory / f'{i:05d}.jpg').symlink_to(path.resolve())
                for i, mask in run_text_pass(model, directory, args.text).items():
                    if mask.any() or start+i not in masks:
                        masks[start+i] = mask
            gc.collect()
            torch.cuda.empty_cache()
            print(f'SAM3 frames {start}-{start+len(chunk)-1}', flush=True)
            if start+len(chunk) == len(frames):
                break
    finally:
        del model
    if sorted(masks) != list(range(len(frames))):
        raise ValueError('SAM3 did not return every frame')
    for i, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8)*255).save(args.output / f'{i:06d}.png')


if __name__ == '__main__':
    main()
