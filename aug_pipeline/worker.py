"""One isolated process per pipeline stage; all model workers use tcp4d Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ASSETS, IK_SCHEMA, Config, REPO, read_json, write_json
from .data import adapt_prediction, prepare_rgb_input


def run(command, **kwargs):
    print('+', ' '.join(str(v) for v in command), flush=True)
    subprocess.run([str(v) for v in command], check=True, **kwargs)


def clean(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def paths(directory):
    return sorted(Path(directory).glob('*.png'), key=lambda p: int(p.stem))


def save_mask(path, mask):
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def read_mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def video(frames, output, fps):
    run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i',
         str(frames / '%06d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', str(output)])


class Worker:
    def __init__(self, config, directory):
        self.c = config
        self.d = Path(directory)
        self.job = read_json(self.d / 'job.json')
        self.r = self.job['request']
        self.info = self.job['episode']
        self.ep = self.d / 'converted' / f'{Path(self.r["episode"]).name}__{self.r["camera"]}'
        self.prediction = self.ep / 'tcp_episode_tcp4d.json'

    def track(self):
        if self.r['query'] is None:
            raise ValueError('Select the initial TCP in frame zero before tracking')
        prepare_rgb_input(self.info, clean(self.ep))
        video(self.ep / 'images/third_views', self.d / 'original.mp4', self.info['fps'])
        geometry = clean(self.ep / 'geometry')
        command = [sys.executable, REPO / 'infer_stage1.py', '--input', self.ep,
                   '--camera', 'third_views', '--checkpoint', self.c['checkpoint'],
                   '--window-size', str(self.r['window_size']), '--device', 'cuda',
                   '--output', self.d / 'tracking.json', '--geometry-output', geometry,
                   '--tcp-query-point', *[str(v) for v in self.r['query']]]
        run(command)
        raw = read_json(self.d / 'tracking.json')
        prediction = adapt_prediction(raw, self.info['count'], 'third_views', self.info['fps'])
        write_json(self.prediction, prediction)
        from .droid.episode import load_episode
        for source, target in [('intrinsics.npy', 'intrinsics/third_views.npy'),
                               ('extrinsics.npy', 'extrinsics/third_views.npy')]:
            destination = self.ep / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(geometry / source, destination)
        depth_dir = clean(self.ep / 'depths/third_views')
        for path in (geometry / 'depth').glob('*.npy'):
            shutil.copy2(path, depth_dir / path.name)
        episode = load_episode(self.ep, True)
        for index in range(episode.count):
            episode.read_frame(index)  # Validate every predicted depth before IK/render.
        np.save(self.ep / 'gripper_openness.npy',
                np.asarray([f['gripper_open'] for f in raw['frames']], dtype=np.float32))
        from PIL import ImageDraw
        out = clean(self.d / 'track_overlay')
        for index, (p, frame) in enumerate(zip(paths(self.ep / 'images/third_views'), raw['frames'])):
            image = Image.open(p).convert('RGB')
            xyz = np.asarray(frame['xyz_m'])
            if xyz[2] > 0:
                x, y = (episode.intrinsics[index] @ xyz)[:2] / xyz[2]
                draw = ImageDraw.Draw(image)
                draw.ellipse((x-3, y-3, x+3, y+3), fill='#41e6b3')
            image.save(out / p.name)
        video(out, self.d / 'tracking.mp4', self.info['fps'])

    def ik(self):
        from .droid import retarget
        from .droid.ik._embodiments import load_tcp_calibration
        results = {}
        previous = read_json(self.d / 'ik_summary.json') if (self.d / 'ik_summary.json').is_file() else {}
        for arm in self.r['arms']:
            spec = dict(retarget.SPECS[arm])
            assets = Path(self.c['embodiments']) / ASSETS[arm]
            signature = hashlib.sha256(json.dumps(dict(schema=IK_SCHEMA, spec=spec,
                tcp_calibration=load_tcp_calibration(ASSETS[arm]),
                prediction=hashlib.sha256(self.prediction.read_bytes()).hexdigest(),
                extrinsics=hashlib.sha256((self.ep / 'extrinsics/third_views.npy').read_bytes()).hexdigest(),
                assets=[(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(assets.rglob('*')) if p.is_file()]), sort_keys=True).encode()).hexdigest()
            cached = previous.get(arm, {})
            if ('ik' not in self.job.get('force_stages', []) and cached.get('signature') == signature
                    and cached.get('status') == 'success' and retarget.ik_metrics(self.ep, arm)):
                print(f'{arm}: reusing validated per-arm IK', flush=True)
                results[arm] = cached
                continue
            ok = retarget.solve(self.ep, arm, spec, self.prediction)
            metrics = retarget.ik_metrics(self.ep, arm)
            if (not ok or not metrics or metrics.get('drops', 1)) and self.r['auto_base']:
                best = retarget.search_base(self.ep, arm, self.prediction)
                if best and best.get('offset') is not None:
                    spec['offset'] = best['offset']
                    ok = retarget.solve(self.ep, arm, spec, self.prediction)
                    metrics = retarget.ik_metrics(self.ep, arm)
            accepted = bool(ok and metrics and metrics.get('drops') == 0)
            results[arm] = dict(status='success' if accepted else 'failed', metrics=metrics,
                                base=spec, reasons=retarget.failure_reasons(self.ep, arm), signature=signature)
            print(arm, results[arm], flush=True)
        write_json(self.d / 'ik_summary.json', results)
        if not any(r['status'] == 'success' for r in results.values()):
            raise RuntimeError('No target arm passed strict zero-drop IK; inspect ik_summary.json')

    def mask(self):
        method = self.r['mask_method']
        if method != 'sam3':
            raise ValueError('RGB input requires SAM3 masks')
        output = clean(self.d / 'masks')
        if self.r['prompts']:
            from .segmentation import refine
            refine(self.c, self.ep, output, self.r)
        else:
            run([sys.executable, '-m', 'aug_pipeline.droid.sam3', self.ep, '--output', output,
                 '--checkpoint', self.c['sam3_checkpoint'], '--text', self.r['text']])
        files = paths(output)
        if len(files) != self.info['count']:
            raise ValueError(f'Incomplete masks: {len(files)}/{self.info["count"]}')
        overlay = clean(self.d / 'mask_overlay')
        empty_frames = []
        for p, rgb in zip(files, paths(self.ep / 'images/third_views')):
            mask = read_mask(p)
            if not mask.any():
                empty_frames.append(int(p.stem))
            frame = np.array(Image.open(rgb).convert('RGB'))
            if mask.shape != frame.shape[:2]:
                raise ValueError('Mask resolution differs from the original image')
            frame[mask] = (frame[mask] * .4 + np.array([255, 85, 100]) * .6).astype(np.uint8)
            Image.fromarray(frame).save(overlay / p.name)
        write_json(self.d / 'mask_stats.json', dict(frames=len(files), empty_frames=empty_frames))
        if len(empty_frames) == len(files):
            raise ValueError('No source robot was segmented; review the text or add foreground prompts')
        if empty_frames:
            print(f'No detected source mask on frames {empty_frames}; review the overlay and refine if needed', flush=True)
        video(overlay, self.d / 'masks.mp4', self.info['fps'])

    def background(self):
        """Final reviewed masks are authoritative, including negative corrections."""
        images = paths(self.ep / 'images/third_views')
        masks = np.stack([read_mask(p) for p in paths(self.d / 'masks')])
        frames = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in images])
        # Estimate a fixed-camera plate using only pixels outside FINAL masks.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            plate = np.nanmedian(np.where(~masks[..., None], frames, np.nan), axis=0)
        valid = ((~masks).sum(axis=0) >= 3) & np.isfinite(plate).all(axis=2)
        plate = np.nan_to_num(plate).astype(np.uint8)
        frame_dir = clean(self.d / 'prefill')
        residual_dir = clean(self.d / 'residual')
        erased_dir = clean(self.d / 'erased')
        for index, (frame, mask) in enumerate(zip(frames, masks)):
            frame = frame.copy()
            frame[mask] = 0
            Image.fromarray(frame).save(erased_dir / f'{index:06d}.png')
            visible_plate = mask & valid
            frame[visible_plate] = plate[visible_plate]
            Image.fromarray(frame).save(frame_dir / f'{index:06d}.png')
            save_mask(residual_dir / f'{index:06d}.png', mask & ~valid)
        video(erased_dir, self.d / 'erased.mp4', self.info['fps'])
        video(frame_dir, self.d / 'prefill.mp4', self.info['fps'])

    def inpaint(self):
        output = clean(self.d / 'background')
        if not any(read_mask(p).any() for p in paths(self.d / 'residual')):
            for p in paths(self.d / 'prefill'):
                shutil.copy2(p, output / p.name)
        else:
            repo = Path(self.c['propainter_repo'])
            run([sys.executable, repo / 'inference_propainter.py',
                 '-i', self.d / 'prefill', '-m', self.d / 'residual', '-o', self.d / 'propainter',
                 '--save_fps', str(round(self.info['fps'])), '--fp16', '--save_frames', '--mask_dilation', '0',
                 '--subvideo_length', '40', '--neighbor_length', '6', '--ref_stride', '15'], cwd=repo)
            filled = paths(self.d / 'propainter/prefill/frames')
            if len(filled) != self.info['count']:
                raise ValueError('ProPainter returned an incomplete video')
            for index, p in enumerate(filled):
                rgb = np.array(Image.open(self.d / 'prefill' / f'{index:06d}.png').convert('RGB'))
                mask = read_mask(self.d / 'residual' / f'{index:06d}.png')
                fill = np.asarray(Image.open(p).convert('RGB'))
                if fill.shape != rgb.shape:
                    raise ValueError('ProPainter changed the output resolution')
                rgb[mask] = fill[mask]  # preserve every pixel the user did not erase
                Image.fromarray(rgb).save(output / f'{index:06d}.png')
        video(output, self.d / 'background.mp4', self.info['fps'])

    def render(self):
        from .droid.episode import load_episode
        from .droid.render_target import render_target
        from .droid.renderer import RobotRenderer
        episode = load_episode(self.ep, True)
        results = read_json(self.d / 'ik_summary.json')
        clean(self.d / 'targets')
        for arm in self.r['arms']:
            if results[arm]['status'] != 'success':
                continue
            target = render_target(self.ep / 'TCP_prediction_ik' / arm, episode, self.c['embodiments'])
            frames = clean(self.d / 'targets' / arm / 'images')
            with RobotRenderer(target, episode.height, episode.width) as renderer:
                for index in range(episode.count):
                    rgb, scene_depth = episode.read_frame(index)
                    new = renderer.render(index, episode.intrinsics[index], episode.extrinsics[index])
                    mask = read_mask(self.d / 'masks' / f'{index:06d}.png')
                    scene_ok = np.isfinite(scene_depth) & (scene_depth > 0)
                    saturated = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int) > 40
                    protect = scene_ok & saturated & (new.depth > scene_depth - .03) & ~mask
                    draw = (new.depth > 0) & ~protect & (mask | ~scene_ok | (new.depth < scene_depth - .02))
                    background = np.array(Image.open(self.d / 'background' / f'{index:06d}.png').convert('RGB'))
                    background[draw] = new.rgb[draw]
                    Image.fromarray(background).save(frames / f'{index:06d}.png')
            video(frames, frames.parent / 'video.mp4', self.info['fps'])

    def export(self):
        from .droid.export import export_arm_dataset
        results = read_json(self.d / 'ik_summary.json')
        clean(self.d / 'datasets')
        for arm in self.r['arms']:
            if results[arm]['status'] != 'success':
                continue
            output = export_arm_dataset(frames_dir=self.d / 'targets' / arm / 'images',
                ik_dir=self.ep / 'TCP_prediction_ik' / arm, converted=self.ep, arm=arm,
                dataset='droid', task='', episode_name=self.info['name'],
                prediction_json=self.prediction, output_root=self.d / 'datasets', mask_method=self.r['mask_method'])
            meta = read_json(output / 'metadata.json')
            meta['tcp']['source'] = 'TCP4D Stage1 prediction'
            meta['augmentation'] = dict(job=self.job['id'], reviewed_prompts=self.r['prompts'],
                                        source_camera=self.r['camera'], **results[arm])
            meta['geometry_source'] = 'TCP4D predicted depth and camera parameters'
            write_json(output / 'metadata.json', meta)
        shutil.make_archive(str(self.d / 'dataset'), 'zip', self.d / 'datasets')


def probe(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available to this process')
    print(torch.__version__, torch.cuda.get_device_name(0))
    import importlib.util
    import importlib
    if importlib.util.find_spec('sam3'):
        importlib.import_module('sam3')
        print('sam3 import passed')
    from curobo.curobolib import kinematics_fused_cu
    import sapien
    scene = sapien.Scene()
    scene.add_camera('probe', 32, 32, 1.0, .01, 10)
    scene.update_render()
    print('Available segmentation imports, cuRobo CUDA extension and SAPIEN render initialization passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    config = Config(args.config)
    if args.stage == 'probe':
        probe(config)
    else:
        from .metrics import StageMetrics
        with StageMetrics() as metrics:
            getattr(Worker(config, args.job), args.stage)()
        write_json(args.job / f'{args.stage}.metrics.json', metrics.result())


if __name__ == '__main__':
    main()
