"""CPU-only episode validation and the public Stage1-to-IK adapter."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .config import read_json


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}


def rgb_paths(directory):
    import re
    def key(path):
        return [int(v) if v.isdigit() else v.lower() for v in re.split(r'(\d+)', path.name)]
    return sorted((p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=key)


def inspect_episode(path, camera=None, fps=None):
    """Inspect RGB alone; calibration, depth, TCP labels and joints are never read."""
    path = Path(path).expanduser().resolve()
    metadata = {}
    if path.is_file():
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError('Choose an MP4/video file or an RGB frame directory')
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open video: {path}')
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            rate = float(capture.get(cv2.CAP_PROP_FPS))
            ok, first = capture.read()
            if not ok:
                raise ValueError('Video has no readable frames')
            native_height, native_width = first.shape[:2]
        finally:
            capture.release()
        cameras, camera, frame_dir = ['video'], 'video', None
        kind = 'video'
    elif path.is_dir():
        meta = path / 'metadata.json'
        metadata = read_json(meta) if meta.is_file() else {}
        image_root = path / 'images' if (path / 'images').is_dir() else path
        if rgb_paths(image_root):
            cameras, camera, frame_dir = ['rgb'], 'rgb', image_root
        else:
            cameras = sorted(p.name for p in image_root.iterdir() if p.is_dir() and rgb_paths(p))
            if not cameras:
                raise ValueError('No RGB frames found')
            camera = camera or cameras[0]
            if camera not in cameras:
                raise ValueError(f'Unknown camera {camera}; available: {cameras}')
            frame_dir = image_root / camera
        images = rgb_paths(frame_dir)
        count = len(images)
        with Image.open(images[0]) as first:
            native_width, native_height = first.size
        rate = metadata.get('fps', metadata.get('frequency_hz', 15))
        kind = 'rgb'
    else:
        raise FileNotFoundError(path)
    rate = fps if fps is not None else rate
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('Frame rate must be finite and positive')
    if count < 2:
        raise ValueError('At least two frames are required')
    # Bound model memory while preserving aspect ratio. UI queries and generated
    # outputs use this same resized image, so no hidden query rescaling is needed.
    scale = min(1.0, 320 / native_width, 240 / native_height)
    width, height = max(2, round(native_width * scale)), max(2, round(native_height * scale))
    instruction = metadata.get('language_instruction', [])
    if isinstance(instruction, str):
        instruction = [instruction]
    return dict(path=str(path), name=path.stem if path.is_file() else path.name,
                input_type=kind, frame_dir=str(frame_dir) if frame_dir else None,
                camera=camera, cameras=cameras, count=count, width=width, height=height,
                native_width=native_width, native_height=native_height,
                fps=float(rate), instruction=instruction)


def read_input_frame(info, index):
    if not 0 <= index < info['count']:
        raise ValueError('Frame outside episode')
    if info['input_type'] == 'video':
        import cv2
        capture = cv2.VideoCapture(info['path'])
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot decode video frame {index}')
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    else:
        with Image.open(rgb_paths(info['frame_dir'])[index]) as frame:
            image = frame.convert('RGB')
    if image.size != (info['native_width'], info['native_height']):
        raise ValueError(f'Frame {index} dimensions differ from frame zero')
    return image.resize((info['width'], info['height']), Image.Resampling.LANCZOS)


def prepare_rgb_input(info, destination):
    """Decode RGB into a private workspace without copying any source labels."""
    from .config import write_json
    destination = Path(destination)
    frames = destination / 'images/third_views'
    frames.mkdir(parents=True, exist_ok=True)
    timestamps = np.arange(info['count'], dtype=np.float64) / info['fps']
    if info['input_type'] == 'video':
        import cv2
        capture = cv2.VideoCapture(info['path'])
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if image.size != (info['native_width'], info['native_height']):
                    raise ValueError(f'Frame {index} dimensions differ from frame zero')
                image.resize((info['width'], info['height']), Image.Resampling.LANCZOS).save(frames / f'{index:06d}.png')
                index += 1
            if index != info['count']:
                raise ValueError(f'Video reports {info["count"]} frames but decoded {index}')
        finally:
            capture.release()
    else:
        # List once for long image sequences rather than rescanning per frame.
        files = rgb_paths(info['frame_dir'])
        if len(files) != info['count']:
            raise ValueError('RGB frame count changed since inspection')
        for index, path in enumerate(files):
            with Image.open(path) as image:
                if image.size != (info['native_width'], info['native_height']):
                    raise ValueError(f'Frame {index} dimensions differ from frame zero')
                image.convert('RGB').resize((info['width'], info['height']), Image.Resampling.LANCZOS).save(frames / f'{index:06d}.png')
    np.save(destination / 'timestamps.npy', timestamps)
    write_json(destination / 'metadata.json', dict(
        format_version='tcp4d_rgb_v1', frame_count=info['count'], num_frames=info['count'],
        fps=info['fps'], frequency_hz=info['fps'], width=info['width'], height=info['height'],
        image_width=info['width'], image_height=info['height'], source_file=info['path'],
        language_instruction=info['instruction'], input_type=info['input_type'],
        rgb=dict(format='PNG', channel_order='RGB'),
        extrinsics=dict(transform='world_to_camera', source='TCP4D predicted world frame'),
        depth=dict(format='NPY', unit='meter', invalid_value=0, source='TCP4D'),
    ))


def adapt_prediction(raw, count, camera, fps, threshold=0.5):
    """Strictly adapt released TCP4D's flat per-frame schema; never pad poses."""
    if raw.get('stage') != 1 or raw.get('camera') != camera:
        raise ValueError('Expected Stage1 output for the selected camera')
    frames = raw.get('frames', [])
    if (raw.get('episode_num_frames') != count or len(frames) != count or
            [f.get('frame_index') for f in frames] != list(range(count))):
        raise ValueError('Incomplete TCP prediction; padding and GT fallback are disabled')
    if not math.isclose(float(raw['frame_rate_hz']), fps, rel_tol=1e-6):
        raise ValueError('Prediction and episode frame rates disagree')
    output = []
    for f in frames:
        xyz, rpy = np.asarray(f['xyz_m']), np.asarray(f['rpy_rad'])
        grip, confidence, timestamp = float(f['gripper_open']), float(f['confidence']), float(f['time_seconds'])
        if (xyz.shape != (3,) or rpy.shape != (3,) or
                not np.isfinite(np.r_[xyz, rpy, grip, confidence, timestamp]).all() or not 0 <= grip <= 1):
            raise ValueError(f'Invalid TCP at frame {f["frame_index"]}')
        item = dict(xyz_m=xyz.tolist(), rpy_rad=rpy.tolist(), gripper_open=grip >= threshold,
                    gripper_probability=grip, prediction_confidence=confidence, padded=False)
        output.append(dict(frame_index=f['frame_index'], time_seconds=timestamp, left=item))
    times = [f['time_seconds'] for f in output]
    if np.any(np.diff(times) <= 0):
        raise ValueError('Prediction timestamps must increase')
    return dict(num_frames=count, frame_rate_hz=fps, view='third_views', camera=camera,
                position_unit='meter', rotation_unit='radian', source='tcp4d_stage1',
                coordinate_frame='OpenCV camera (+x right, +y down, +z forward)',
                prediction=dict(padded_frames=[], initial_query_source=raw.get('initial_query_source')),
                frames=output)
