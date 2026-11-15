from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import ARMS, DEPS, ROOT, STAGES
from .data import inspect_episode, read_input_frame, rgb_paths, VIDEO_SUFFIXES
from .jobs import JobManager
from .resources import doctor


class EpisodeInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    episode: str
    camera: str | None = None
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class JobInput(EpisodeInput):
    arms: list[str] = Field(default_factory=lambda: ['ur5_wsg', 'franka_panda'])
    mask_method: Literal['sam3'] = 'sam3'
    query: tuple[float, float] | None = None
    text: str = Field(default='robot arm', min_length=1, max_length=256)
    window_size: int = Field(default=9, ge=2, le=18)
    auto_base: bool = True


class RunInput(BaseModel):
    stage: Literal['track', 'ik', 'mask', 'background', 'inpaint', 'render', 'export']
    force: bool = False


class Prompt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frame: int = Field(ge=0)
    points: list[tuple[float, float]] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None


class PromptInput(BaseModel):
    prompts: list[Prompt]


class MaskPreviewInput(PromptInput):
    frame: int = Field(ge=0)


class JobChanges(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: tuple[float, float] | None = None
    arms: list[str] | None = None
    mask_method: Literal['sam3'] | None = None
    text: str | None = Field(default=None, min_length=1, max_length=256)
    window_size: int | None = Field(default=None, ge=2, le=18)
    auto_base: bool | None = None


def create_app(config):
    manager = JobManager(config)
    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(manager.close)
    app = FastAPI(title='TCP4D Augmentation', lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=400, content={'detail': str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def missing(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={'detail': str(exc)})

    @app.get('/api/capabilities')
    def capabilities():
        return dict(arms=ARMS, stages=STAGES, dependencies=DEPS, default_mask=config['mask_method'], checks=doctor(config),
                    data_roots=config['data_roots'])

    @app.get('/api/episodes')
    def episodes():
        results = []
        def visit(path):
            if path.is_relative_to(config.work):
                return
            if path.is_file():
                if path.suffix.lower() in VIDEO_SUFFIXES:
                    results.append(dict(path=str(path), name=path.name))
            elif path.is_dir():
                if rgb_paths(path) or (path / 'images').is_dir():
                    results.append(dict(path=str(path), name=path.name))
                else:
                    for child in sorted(path.iterdir()):
                        if not child.name.startswith('.'):
                            visit(child)
        for root in config['data_roots']:
            visit(Path(root))
        return results

    @app.post('/api/episodes/inspect')
    def inspect(value: EpisodeInput):
        return inspect_episode(config.episode(value.episode), value.camera, value.fps)

    @app.get('/api/episodes/frame')
    def frame(episode: str, camera: str, index: int = 0):
        info = inspect_episode(config.episode(episode), camera)
        if not 0 <= index < info['count']:
            raise ValueError('Frame outside episode')
        image = io.BytesIO()
        read_input_frame(info, index).save(image, format='PNG')
        return Response(image.getvalue(), media_type='image/png')

    @app.get('/api/jobs')
    def jobs():
        return manager.list()

    @app.post('/api/jobs', status_code=201)
    def create(value: JobInput):
        return manager.create(value.model_dump())

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        return manager.get(job_id)

    @app.patch('/api/jobs/{job_id}')
    def update(job_id: str, value: JobChanges):
        changes = value.model_dump(exclude_unset=True)
        if any(v is None for k, v in changes.items() if k != 'query'):
            raise ValueError('Only query may be null')
        return manager.update(job_id, changes)

    @app.post('/api/jobs/{job_id}/run')
    def run(job_id: str, value: RunInput):
        return manager.start(job_id, value.stage, value.force)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        return manager.cancel(job_id)

    @app.put('/api/jobs/{job_id}/prompts')
    def prompts(job_id: str, value: PromptInput):
        return manager.update(job_id, {'prompts': [p.model_dump() for p in value.prompts]})

    @app.post('/api/jobs/{job_id}/mask-preview')
    def mask_preview(job_id: str, value: MaskPreviewInput):
        return manager.preview(job_id, value.frame, [p.model_dump() for p in value.prompts]).result()

    @app.get('/api/jobs/{job_id}/events')
    async def events(job_id: str, request: Request, offset: int = 0):
        manager.get(job_id)
        async def stream():
            cursor = max(0, offset)
            while not await request.is_disconnected():
                log = manager.directory(job_id) / 'run.log'
                if log.exists():
                    with log.open('r', errors='replace') as f:
                        f.seek(cursor)
                        chunk = f.read(65536)
                        cursor = f.tell()
                    if chunk:
                        yield 'data: ' + json.dumps(dict(log=chunk, offset=cursor)) + '\n\n'
                yield 'event: status\ndata: ' + json.dumps(manager.get(job_id)) + '\n\n'
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/jobs/{job_id}/artifacts')
    def artifacts(job_id: str):
        directory = manager.directory(job_id)
        job = manager.get(job_id)
        available = []
        for p in sorted(directory.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix in ('.mp4', '.zip', '.json'):
                if p.name in ('job.json',) or p.is_relative_to(directory / 'converted') or p.is_relative_to(directory / 'datasets'):
                    continue
                phase = artifact_stage(str(p.relative_to(directory)))
                if phase and job['stages'][phase]['status'] != 'complete':
                    continue
                name = str(p.relative_to(directory))
                available.append(dict(name=name, url=f'/api/jobs/{job_id}/artifacts/{name}'))
        return available

    @app.get('/api/jobs/{job_id}/artifacts/{name:path}')
    def artifact(job_id: str, name: str):
        directory = manager.directory(job_id).resolve()
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(404, 'Artifact not found')
        phase = artifact_stage(str(path.relative_to(directory)))
        if phase and manager.get(job_id)['stages'][phase]['status'] != 'complete':
            raise HTTPException(409, 'Artifact is stale; rerun its stage first')
        # Converted RGB/depth symlinks intentionally point at approved data roots.
        return FileResponse(path)

    dist = ROOT / 'frontend/dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    else:
        @app.get('/')
        def build_notice():
            return {'message': 'Build the frontend: cd aug_pipeline/frontend && npm ci && npm run build',
                    'api_docs': '/docs'}
    return app


def artifact_stage(name):
    if name.endswith('.metrics.json') and name.split('.')[0] in STAGES:
        return name.split('.')[0]
    if name == 'original.mp4': return 'track'
    if name.startswith(('tracking.', 'track_overlay/')): return 'track'
    if name == 'ik_summary.json': return 'ik'
    if name == 'mask_stats.json' or name.startswith(('masks.', 'masks/', 'mask_overlay/')): return 'mask'
    if name.startswith(('erased.', 'erased/', 'prefill.', 'prefill/', 'residual/')): return 'background'
    if name.startswith(('background.', 'background/', 'propainter/')): return 'inpaint'
    if name.startswith('targets/'): return 'render'
    if name == 'dataset.zip' or name.startswith('datasets/'): return 'export'
    return None
