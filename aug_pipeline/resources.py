from __future__ import annotations

import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .config import ARMS, ASSETS, ROOT, read_json, write_json


def check_stages(config, request, stages):
    """Check only work that needs execution, so completed stages can be reused."""
    needed = set()
    if 'track' in stages:
        needed.update(['ffmpeg', 'torch', 'torchvision', 'checkpoint'])
    if set(stages) & {'ik', 'render'}:
        needed.update(['sapien', 'embodiments', *['asset:' + a for a in request['arms']]])
    if 'ik' in stages:
        needed.add('nvidia-curobo')
    if 'mask' in stages:
        needed.update(['sam3', 'sam3_checkpoint'])
    if 'inpaint' in stages:
        needed.update(['propainter_repo', 'ProPainter.pth', 'recurrent_flow_completion.pth', 'raft-things.pth'])
    missing = [r['name'] for r in doctor(config) if r['name'] in needed and not r['ok']]
    if missing:
        raise ValueError('Missing resources: ' + ', '.join(missing) + '. Run tcp4d-aug doctor and prepare the corresponding resources.')


def fetch(url, destination):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.partial')
    print(f'Downloading {url}', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(destination)


def download(config, components, arms):
    from huggingface_hub import snapshot_download, hf_hub_download
    manifest = read_json(ROOT / 'resources.json')
    if 'all' in components:
        components = ['tcp4d', 'assets', 'sam3', 'propainter']
    for component in components:
        if component == 'tcp4d':
            spec = manifest['tcp4d']
            snapshot_download(spec['repo'], revision=spec['revision'], local_dir=config['checkpoint'])
        elif component == 'assets':
            spec = manifest['embodiments']
            snapshot_download(spec['repo'], revision=spec['revision'], local_dir=config['embodiments'],
                              allow_patterns=[ASSETS[a] + '/**' for a in set(arms)])
            missing = [a for a in arms if not (Path(config['embodiments']) / ASSETS[a] / 'config.yml').is_file()]
            if missing:
                raise FileNotFoundError(f'Public asset snapshot has no configuration for {missing}. '
                                        'Supply their URDF, meshes and IK configs under the configured embodiments directory.')
        elif component == 'sam3':
            spec = manifest['sam3']
            p = hf_hub_download(spec['repo'], spec['filename'], revision=spec['revision'])
            dest = Path(config['sam3_checkpoint'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        elif component == 'propainter':
            spec = manifest['propainter']
            repo = Path(config['propainter_repo'])
            if not repo.exists():
                subprocess.run(['git', 'clone', spec['repo'], str(repo)], check=True)
                subprocess.run(['git', '-C', str(repo), 'checkout', spec['revision']], check=True)
            for name in ('ProPainter.pth', 'recurrent_flow_completion.pth', 'raft-things.pth'):
                fetch(spec['weights_url'] + name, repo / 'weights' / name)
        else:
            raise ValueError(f'Unknown component: {component}')
    records = []
    roots = [Path(config[k]) for k in ('checkpoint', 'sam3_checkpoint')]
    roots += [Path(config['propainter_repo']) / 'weights']
    for root in roots:
        files = [root] if root.is_file() else [p for p in root.glob('*') if p.suffix in ('.pt', '.pth', '.safetensors')]
        for p in files:
            digest = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda: f.read(8 << 20), b''):
                    digest.update(chunk)
            records.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest.hexdigest()))
    write_json(config.work / 'download_receipt.json', dict(manifest=manifest, files=records))


def doctor(config, gpu=False):
    results = []
    def add(name, ok, detail):
        results.append(dict(name=name, ok=bool(ok), detail=str(detail)))
    for name in ('torch', 'torchvision', 'numpy', 'opencv-python', 'fastapi', 'scipy', 'sapien', 'sam3', 'nvidia-curobo'):
        try:
            add(name, True, importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            add(name, False, 'Install the corresponding aug extra')
    for key in ('checkpoint', 'embodiments', 'sam3_checkpoint', 'propainter_repo'):
        p = Path(config[key])
        ok = p.exists()
        if key == 'checkpoint':
            ok = (p / 'model.safetensors').is_file() or p.is_file()
        add(key, ok, p)
    for arm in ARMS:
        directory = Path(config['embodiments']) / ASSETS[arm]
        add(f'asset:{arm}', all((directory / f).is_file() for f in ('config.yml',)) and
            any((directory / f).is_file() for f in ('curobo_tmp.yml', 'curobo.yml')), directory)
    add('ffmpeg', shutil.which('ffmpeg'), shutil.which('ffmpeg') or 'Install system FFmpeg')
    for name in ('ProPainter.pth', 'recurrent_flow_completion.pth', 'raft-things.pth'):
        p = Path(config['propainter_repo']) / 'weights' / name
        add(name, p.is_file(), p)
    if gpu:
        command = [sys.executable, '-m', 'aug_pipeline.worker', '--config', str(config.path), '--stage', 'probe']
        try:
            result = subprocess.run(command, env=config.env(), capture_output=True, text=True, timeout=120)
            add('gpu_probe', result.returncode == 0, (result.stdout + result.stderr)[-5000:])
        except subprocess.TimeoutExpired:
            add('gpu_probe', False, 'Probe exceeded 120 seconds')
    return results
