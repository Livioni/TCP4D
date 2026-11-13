from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ARMS = ('ur5_wsg', 'franka_panda', 'jaco', 'sawyer', 'kinova_gen3')
ASSETS = {'ur5_wsg': 'ur5-wsg', 'franka_panda': 'franka-panda',
          'jaco': 'jaco', 'sawyer': 'sawyer', 'kinova_gen3': 'kinova-gen3'}
IK_SCHEMA = 2  # One physical arm, stored in the left slot.
STAGES = ('track', 'ik', 'mask', 'background', 'inpaint', 'render', 'export')
DEPS = {'track': (), 'ik': ('track',), 'mask': ('track',),
        'background': ('mask',), 'inpaint': ('background',),
        'render': ('ik', 'inpaint'), 'export': ('render',)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class Config:
    def __init__(self, path=None):
        import yaml
        path = Path(path or ROOT / 'config.example.yaml').expanduser().resolve()
        self.path = path
        self.values = yaml.safe_load(path.read_text()) or {}
        for key in ('work_dir', 'checkpoint', 'embodiments', 'sam3_checkpoint', 'propainter_repo'):
            value = Path(self.values[key]).expanduser()
            self.values[key] = str((REPO / value).resolve())
        self.values['data_roots'] = [str((REPO / Path(v).expanduser()).resolve())
                                     for v in self.values['data_roots']]

    def __getitem__(self, key):
        return self.values[key]

    @property
    def work(self):
        return Path(self['work_dir'])

    def episode(self, value):
        path = Path(value).expanduser().resolve()
        if not any(path.is_relative_to(Path(root)) for root in self['data_roots']):
            raise ValueError('Episode must be inside a configured data_root')
        return path

    def snapshot(self, path):
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.values, sort_keys=True))

    def env(self):
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(self['gpu']),
                   EMBODIMENTS_ROOT=self['embodiments'], PYTHONUNBUFFERED='1')
        env['PYTHONPATH'] = os.pathsep.join([str(REPO), env.get('PYTHONPATH', '')])
        return env
