"""Durable single-GPU job queue; subprocess groups make cancellation reliable."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ARMS, DEPS, IK_SCHEMA, STAGES, read_json, write_json
from .data import inspect_episode, read_input_frame, rgb_paths
from .preview import PreviewClient


def validate_request(config, value):
    request = dict(value)
    info = inspect_episode(config.episode(request['episode']), request.get('camera'), request.get('fps'))
    request.update(episode=info['path'], camera=info['camera'])
    request.setdefault('arms', ['ur5_wsg', 'franka_panda'])
    if not request['arms'] or any(a not in ARMS for a in request['arms']):
        raise ValueError('Select supported target arms')
    request['arms'] = list(dict.fromkeys(request['arms']))
    request.setdefault('mask_method', config['mask_method'])
    if request['mask_method'] != 'sam3':
        raise ValueError('RGB input requires SAM3 masks')
    request.setdefault('query', None)
    if request['query'] is not None:
        x, y = request['query']
        if not (0 <= x < info['width'] and 0 <= y < info['height']):
            raise ValueError('TCP query must be inside the original image')
    request.setdefault('text', 'robot arm')
    request.setdefault('window_size', 9)
    if not 2 <= request['window_size'] <= 18:
        raise ValueError('Window size must be 2..18')
    request.setdefault('auto_base', True)
    request.setdefault('prompts', [])
    return request, info


def validate_prompts(prompts, info):
    for p in prompts:
        if not 0 <= p['frame'] < info['count']:
            raise ValueError('Prompt frame outside episode')
        points, labels = p.get('points', []), p.get('labels', [])
        if len(points) != len(labels) or any(v not in (0, 1) for v in labels):
            raise ValueError('Each point needs a foreground/background label')
        for x, y in points:
            if not (0 <= x < info['width'] and 0 <= y < info['height']):
                raise ValueError('Prompt point outside original image')
        if p.get('box'):
            x0, y0, x1, y1 = p['box']
            if not (0 <= x0 < x1 < info['width'] and 0 <= y0 < y1 < info['height']):
                raise ValueError('Invalid prompt box')
        if not points and not p.get('box'):
            raise ValueError('Prompt needs points or a box')


class JobManager:
    def __init__(self, config):
        self.config = config
        self.root = config.work / 'jobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease = (config.work / 'runner.lock').open('a')
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close()
            raise RuntimeError('This workspace already has a running Web/CLI instance; use that server or another work_dir')
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cancelled = set()
        self.processes = {}
        self.preview_versions = {}
        self.preview_client = PreviewClient(config)
        for p in self.root.glob('*/job.json'):
            job = read_json(p)
            # Preserve valid completed outputs when folding preparation into tracking.
            if 'prepare' in job['stages'] and all(a in ARMS for a in job['request']['arms']):
                for stage in STAGES:
                    state = job['stages'][stage]
                    if state['status'] != 'complete':
                        continue
                    try:
                        if state.get('fingerprint') == self.fingerprint(job, stage, legacy=True):
                            state['fingerprint'] = self.fingerprint(job, stage)
                    except (OSError, ValueError, KeyError):
                        # Missing/changed inputs remain stale and are checked on resume.
                        continue
            changed = job['stages'].pop('prepare', None) is not None
            if job['status'] in ('queued', 'running'):
                job['status'] = 'interrupted'
                for state in job['stages'].values():
                    if state['status'] == 'running':
                        state['status'] = 'interrupted'
                changed = True
            if changed:
                write_json(p, job)

    def directory(self, job_id):
        if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
            raise ValueError('Invalid job ID')
        return self.root / job_id

    def get(self, job_id):
        return read_json(self.directory(job_id) / 'job.json')

    def save(self, job):
        job['updated_at'] = time.time()
        write_json(self.directory(job['id']) / 'job.json', job)

    def create(self, request):
        request, info = validate_request(self.config, request)
        validate_prompts(request['prompts'], info)
        job = dict(id=uuid.uuid4().hex, status='ready', created_at=time.time(), request=request,
                   episode=info, configuration=dict(self.config.values),
                   stages={s: {'status': 'pending'} for s in STAGES})
        self.save(job)
        return job

    def list(self):
        return sorted([read_json(p) for p in self.root.glob('*/job.json')],
                      key=lambda j: j['created_at'], reverse=True)

    def update(self, job_id, changes):
        with self.lock:
            job = self.get(job_id)
            if job['status'] in ('running', 'queued'):
                raise ValueError('Cancel or wait before editing a running job')
            allowed = {'query', 'arms', 'mask_method', 'text', 'window_size', 'auto_base', 'prompts'}
            if set(changes) - allowed:
                raise ValueError('Create a new job to change the input episode or camera')
            if 'arms' in changes and changes['arms'] != job['request']['arms']:
                raise ValueError('Target robots are locked for this run; create a new run to change them')
            request, info = validate_request(self.config, {**job['request'], **changes})
            validate_prompts(request['prompts'], info)
            job['request'] = request
            job['status'] = 'ready'
            self.invalidate(job)
            self.save(job)
            return job

    def source_fingerprint(self, job):
        r, info = job['request'], job['episode']
        base = Path(r['episode'])
        files = [base] if base.is_file() else rgb_paths(info.get('frame_dir') or base / 'images' / r['camera'])
        data = dict(episode=r['episode'], camera=r['camera'], fps=info['fps'],
                    width=info['width'], height=info['height'],
                    source_files=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files])
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def fingerprint(self, job, stage, *, legacy=False):
        r = job['request']
        keys = {'prepare': ['episode', 'camera'], 'track': ['query', 'window_size'],
                'ik': ['arms', 'auto_base'], 'mask': ['mask_method', 'text', 'prompts']}.get(stage, [])
        data = {k: r[k] for k in keys}
        data['pipeline_schema'] = 2 if legacy else 3
        data['config'] = self.config.values
        deps = {**DEPS, 'prepare': (), 'track': ('prepare',), 'mask': ('prepare',)} if legacy else DEPS
        data['dependencies'] = {s: self.fingerprint(job, s, legacy=legacy) for s in deps[stage]}
        if legacy and stage == 'prepare':
            base, info = Path(r['episode']), job['episode']
            files = [base] if base.is_file() else rgb_paths(info.get('frame_dir') or base / 'images' / r['camera'])
            data['fps'] = info['fps']
            data['source_files'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
        if stage == 'track':
            if not legacy:
                data['source'] = self.source_fingerprint(job)
            checkpoint = Path(self.config['checkpoint'])
            data['weights'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                               for p in ([checkpoint] if checkpoint.is_file() else sorted(checkpoint.glob('*'))) if p.is_file()]
        resources = []
        if stage == 'mask':
            data['refinement_schema'] = 2
            resources.append(Path(self.config['sam3_checkpoint']))
        if stage == 'ik':
            data['ik_schema'] = IK_SCHEMA
            from .config import ASSETS
            from .droid.ik._embodiments import load_tcp_calibration
            data['tcp_calibrations'] = {arm: load_tcp_calibration(ASSETS[arm]) for arm in r['arms']}
            for arm in r['arms']:
                resources.extend((Path(self.config['embodiments']) / ASSETS[arm]).rglob('*'))
        if resources:
            data['resources'] = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(resources) if p.is_file()]
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def invalidate(self, job):
        for stage in STAGES:
            state = job['stages'][stage]
            if state.get('fingerprint') != self.fingerprint(job, stage):
                job['stages'][stage] = {'status': 'pending'}

    def start(self, job_id, stage, force=False):
        if stage not in STAGES:
            raise ValueError('Unknown stage')
        with self.lock:
            job = self.get(job_id)
            if job['status'] in ('running', 'queued'):
                raise ValueError('Job is already queued or running')
            _, current = validate_request(self.config, job['request'])
            if stage == 'track' and job['request']['query'] is None:
                raise ValueError('Click the initial TCP in frame zero before running Track TCP')
            if any(current[k] != job['episode'][k] for k in ('count', 'width', 'height', 'fps')):
                raise ValueError('Source episode geometry changed; create a new run to keep its frame provenance consistent')
            job['episode'] = current
            self.cancelled.discard(job_id)
            self.invalidate(job)
            if force:
                names = {stage}
                for s in STAGES:
                    if s in names or any(d in names for d in DEPS[s]):
                        names.add(s)
                        job['stages'][s] = {'status': 'pending'}
            job['force_stages'] = sorted(names) if force else []
            missing = [s for s in DEPS[stage] if job['stages'][s]['status'] != 'complete']
            if missing:
                raise ValueError('Run prerequisite steps first: ' + ', '.join(missing))
            from .resources import check_stages
            check_stages(self.config, job['request'], [stage] if job['stages'][stage]['status'] != 'complete' else [])
            job['status'] = 'queued'
            job['configuration'] = dict(self.config.values)
            self.config.snapshot(self.directory(job_id) / 'runtime.yaml')
            self.save(job)
            self.pool.submit(self._run, job_id, stage)
            return job

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            self.cancelled.add(job_id)
            proc = self.processes.get(job_id)
            if proc and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                def kill_later():
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                threading.Thread(target=kill_later, daemon=True).start()
            if job['status'] in ('queued', 'running'):
                job['status'] = 'cancelled'
                self.save(job)
            return job

    def preview(self, job_id, frame, prompts):
        """Queue a single-frame preview without changing saved prompts or stages."""
        from .resources import doctor
        with self.lock:
            job = self.get(job_id)
            if job['status'] in ('running', 'queued'):
                raise ValueError('Wait for the current run before editing masks')
            _, info = validate_request(self.config, job['request'])
            if not 0 <= frame < info['count']:
                raise ValueError('Preview frame outside episode')
            validate_prompts(prompts, info)
            if not prompts or any(p['frame'] != frame for p in prompts):
                raise ValueError('Preview needs prompts for the current frame only')
            missing = [r['name'] for r in doctor(self.config)
                       if r['name'] in ('sam3', 'sam3_checkpoint', 'torch') and not r['ok']]
            if missing:
                raise ValueError('Missing resources: ' + ', '.join(missing))
            version = uuid.uuid4().hex
            self.preview_versions[job_id] = version
            return self.pool.submit(self._preview, job, frame, prompts, version)

    def _preview(self, job, frame, prompts, version):
        job_id = job['id']
        with self.lock:
            if self.preview_versions.get(job_id) != version:
                raise ValueError('Mask preview superseded by newer points')
        info = inspect_episode(job['request']['episode'], job['request']['camera'], job['request'].get('fps'))
        directory = self.directory(job_id) / '.source_frames' / self.source_fingerprint(job)
        directory.mkdir(parents=True, exist_ok=True)
        image = directory / f'{frame:06d}.png'
        if not image.is_file():
            read_input_frame(info, frame).save(image)
        result = self.preview_client.update(image, prompts, job['request']['text'])
        return dict(frame=frame, revision=version, **result)

    def _run(self, job_id, target):
        directory = self.directory(job_id)
        try:
            self.preview_client.release()
            for stage in STAGES:
                if stage != target:
                    continue
                with self.lock:
                    job = self.get(job_id)
                    if job_id in self.cancelled:
                        return
                    state = job['stages'][stage]
                    if state['status'] == 'complete':
                        continue
                    state.update(status='running', started_at=time.time())
                    job.update(status='running', error=None)
                    self.save(job)
                    command = self.worker_command(directory, stage)
                    log = (directory / 'run.log').open('ab', buffering=0)
                    log.write(f'\n[{stage}] starting\n'.encode())
                    proc = subprocess.Popen(command, env=self.config.env(), stdout=log, stderr=log,
                                            start_new_session=True)
                    self.processes[job_id] = proc
                code = proc.wait()
                log.close()
                with self.lock:
                    self.processes.pop(job_id, None)
                    job = self.get(job_id)
                    state = job['stages'][stage]
                    state['elapsed_s'] = round(time.time() - state['started_at'], 2)
                    if job_id in self.cancelled:
                        state['status'] = 'cancelled'
                        self.save(job)
                        return
                    if code:
                        state['status'] = 'failed'
                        job.update(status='failed', error=f'{stage} failed (exit {code}); see run.log')
                        self.save(job)
                        return
                    state.update(status='complete', fingerprint=self.fingerprint(job, stage))
                    metrics_path = directory / f'{stage}.metrics.json'
                    if metrics_path.is_file():
                        state['sampled_peak_gpu_mib'] = read_json(metrics_path)['sampled_peak_gpu_mib']
                    self.save(job)
            with self.lock:
                job = self.get(job_id)
                job['status'] = 'complete' if all(s['status'] == 'complete' for s in job['stages'].values()) else 'ready'
                summary = directory / 'ik_summary.json'
                if job['status'] == 'complete' and summary.is_file() and any(
                        result['status'] != 'success' for result in read_json(summary).values()):
                    job['status'] = 'partial'
                self.save(job)
        except Exception as exc:
            with self.lock:
                job = self.get(job_id)
                job.update(status='failed', error=str(exc))
                for state in job['stages'].values():
                    if state['status'] == 'running':
                        state['status'] = 'failed'
                self.save(job)

    def worker_command(self, directory, stage):
        return [sys.executable, '-m', 'aug_pipeline.worker', '--config', str(directory / 'runtime.yaml'),
                '--job', str(directory), '--stage', stage]

    def close(self):
        self.preview_client.shutdown()
        for job_id in list(self.processes):
            self.cancel(job_id)
        for job in self.list():
            if job['status'] in ('queued', 'running'):
                self.cancel(job['id'])
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.lease.close()
