import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {api} from './api';
import {boxFrom, imagePoint, type Point} from './geometry';
import {previewKey, useMaskPreview, type Prompt} from './maskPreview';
import './style.css';

type Info = {path: string; name: string; camera: string; cameras: string[]; count: number; width: number; height: number; fps: number; instruction: string[]};
type Job = {id: string; status: string; created_at: number; request: {episode: string; camera: string; arms: string[]; mask_method: string; query: Point | null; text: string; prompts: Prompt[]}; episode: Info; stages: Record<string, {status: string; elapsed_s?: number}>; error?: string};
type Artifact = {name: string; url: string};
type Capabilities = {arms: string[]; stages: string[]; dependencies?: Record<string, string[]>; default_mask: string; data_roots: string[]; checks: {name: string; ok: boolean; detail: string}[]};
const names: Record<string, string> = {track: 'Track TCP', ik: 'Retarget', mask: 'Mask', background: 'Erase & prefill', inpaint: 'Inpaint', render: 'Render', export: 'Export'};
const dependencies: Record<string, string[]> = {track: [], ik: ['track'], mask: ['track'], background: ['mask'], inpaint: ['background'], render: ['ik', 'inpaint'], export: ['render']};

function App() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [episodes, setEpisodes] = useState<{path: string; name: string}[]>([]);
  const [episode, setEpisode] = useState('');
  const [fps, setFps] = useState<number | undefined>(undefined);
  const [info, setInfo] = useState<Info | null>(null);
  const [query, setQuery] = useState<Point | null>(null);
  const [arms, setArms] = useState(['ur5_wsg', 'franka_panda']);
  const [text, setText] = useState('robot arm');
  const [jobs, setJobs] = useState<Job[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [stage, setStage] = useState('track');
  const [tab, setTab] = useState('workspace');
  const [setupOpen, setSetupOpen] = useState(true);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [logs, setLogs] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [frame, setFrame] = useState(0);
  const [tool, setTool] = useState('positive');
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [showMask, setShowMask] = useState(true);
  const [selectedVideo, setSelectedVideo] = useState('');
  const canvas = useRef<HTMLCanvasElement>(null);
  const drag = useRef<Point | null>(null);
  const sourceVideo = useRef<HTMLVideoElement>(null);
  const targetVideo = useRef<HTMLVideoElement>(null);
  const running = job?.status === 'running' || job?.status === 'queued';
  const working = busy || running;
  const selectedArms = job?.request.arms ?? arms;
  const draft = !!job && JSON.stringify(prompts) !== JSON.stringify(job.request.prompts);
  const frameDraft = !!job && JSON.stringify(prompts.filter(p => p.frame === frame)) !== JSON.stringify(job.request.prompts.filter(p => p.frame === frame));
  const preview = useMaskPreview(job?.id, frame, prompts, frameDraft && stage === 'mask' && tab === 'workspace' && !working);
  const editedFrames = [...new Set(prompts.map(p => p.frame))];
  const reviewed = !!job && editedFrames.every(index => {
    const values = prompts.filter(p => p.frame === index);
    return JSON.stringify(values) === JSON.stringify(job.request.prompts.filter(p => p.frame === index))
      || !!preview.images[previewKey(job.id, index, values)];
  });
  const act = async (work: () => Promise<void>) => {
    setError(''); setBusy(true);
    try {await work();} catch (e) {setError(String(e));} finally {setBusy(false);}
  };

  useEffect(() => {void act(async () => {
    const [c, e, j] = await Promise.all([api<Capabilities>('/capabilities'), api<typeof episodes>('/episodes'), api<Job[]>('/jobs')]);
    setCaps(c); setEpisodes(e); setJobs(j);
    if (e.length) {
      setEpisode(e[0].path);
      setInfo(await api<Info>('/episodes/inspect', 'POST', {episode: e[0].path}));
    }
  });}, []);
  useEffect(() => {
    if (!job) return;
    setLogs('');
    const events = new EventSource(`/api/jobs/${job.id}/events`);
    events.onmessage = e => setLogs(old => (old + JSON.parse(e.data).log).slice(-100000));
    events.addEventListener('status', e => setJob(JSON.parse((e as MessageEvent).data)));
    return () => events.close();
  }, [job?.id]);
  const stageState = JSON.stringify(job?.stages);
  useEffect(() => {
    if (job) void api<Artifact[]>(`/jobs/${job.id}/artifacts`).then(setArtifacts).catch(e => setError(String(e)));
    else setArtifacts([]);
  }, [job?.id, stageState]);
  useEffect(() => {
    if (!info || !canvas.current) return;
    let active = true;
    const base = new Image();
    base.src = '/api/episodes/frame?' + new URLSearchParams({episode: info.path, camera: info.camera, index: String(frame)});
    base.onload = () => {
      if (!active || !canvas.current) return;
      const ctx = canvas.current.getContext('2d')!;
      ctx.drawImage(base, 0, 0);
      const paint = () => {
        if (!active) return;
        if (stage !== 'mask' && query) {
          ctx.strokeStyle = '#22c38e'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(query[0], query[1], 4, 0, Math.PI * 2); ctx.stroke();
        }
        if (stage === 'mask') for (const p of prompts.filter(p => p.frame === frame)) {
          if (p.box) {ctx.strokeStyle = '#efb34c'; ctx.lineWidth = 1; ctx.strokeRect(p.box[0], p.box[1], p.box[2]-p.box[0], p.box[3]-p.box[1]);}
          p.points.forEach(([x, y], i) => {
            ctx.fillStyle = p.labels[i] ? '#22c38e' : '#f05264'; ctx.strokeStyle = '#fff'; ctx.lineWidth = .6;
            ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          });
        }
      };
      const overlay = stage === 'mask' && showMask && ((frameDraft ? preview.image : undefined) ||
        (job?.stages.mask.status === 'complete' ? `/api/jobs/${job.id}/artifacts/mask_overlay/${String(frame).padStart(6, '0')}.png?v=${encodeURIComponent(stageState)}` : undefined));
      if (overlay) {
        const mask = new Image(); mask.src = overlay;
        mask.onload = () => {if (active) {ctx.drawImage(mask, 0, 0); paint();}};
        mask.onerror = paint;
      } else paint();
    };
    return () => {active = false;};
  }, [info, frame, query, prompts, stage, tab, showMask, job?.id, stageState, preview.image, frameDraft]);

  const resetDraft = () => {setPrompts([]); setQuery(null); setFrame(0); setStage('track'); setJob(null); setSetupOpen(true);};
  const loadEpisode = (camera?: string) => act(async () => {
    const value = await api<Info>('/episodes/inspect', 'POST', {episode, camera, fps});
    setInfo(value); setQuery(null); resetDraft();
  });
  const create = () => act(async () => {
    if (!info) throw new Error('Inspect an input first');
    const value = await api<Job>('/jobs', 'POST', {episode: info.path, camera: info.camera, arms, mask_method: 'sam3', query, text, fps: info.fps});
    setJob(value); setJobs(old => [value, ...old]); setQuery(value.request.query ?? null); setPrompts([]); setStage('track'); setFrame(0); setSetupOpen(false);
  });
  const execute = (value: string, force = false) => act(async () => {
    if (!job) throw new Error('Create a run first');
    if (draft && !['track', 'ik'].includes(value)) throw new Error('Propagate or discard mask edits first');
    setJob(await api<Job>(`/jobs/${job.id}/run`, 'POST', {stage: value, force}));
  });
  const selectJob = (value: Job) => {
    setJob(value); setInfo(value.episode); setEpisode(value.request.episode); setQuery(value.request.query ?? null);
    setArms(value.request.arms); setText(value.request.text);
    setPrompts(value.request.prompts); setFrame(0); setStage('track'); setTab('workspace'); setSetupOpen(false);
  };
  const point = (event: React.PointerEvent<HTMLCanvasElement>) => imagePoint(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), info!.width, info!.height);
  const pointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (working) return;
    const p = point(event);
    if (stage === 'track') {
      if (frame !== 0) return;
      const previous = query;
      setQuery(p);
      if (job) void act(async () => {
        try {
          const saved = await api<Job>(`/jobs/${job.id}`, 'PATCH', {query: p});
          setJob(saved); setJobs(old => old.map(value => value.id === saved.id ? saved : value));
        }
        catch (error) {setQuery(previous); throw error;}
      });
      return;
    }
    if (!job) return;
    if (tool === 'box') {drag.current = p; event.currentTarget.setPointerCapture(event.pointerId);}
    else setPrompts(old => [...old, {frame, points: [p], labels: [tool === 'positive' ? 1 : 0]}]);
  };
  const pointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag.current && !working) {
      const box = boxFrom(drag.current, point(event));
      if (box) setPrompts(old => [...old, {frame, points: [], labels: [], box}]);
    }
    drag.current = null;
  };
  const propagate = () => act(async () => {
    if (!job || !reviewed) return;
    const saved = await api<Job>(`/jobs/${job.id}/prompts`, 'PUT', {prompts});
    setPrompts(saved.request.prompts);
    setJob(saved);
    setJob(await api<Job>(`/jobs/${job.id}/run`, 'POST', {stage: 'mask'}));
  });
  const undo = () => setPrompts(old => {
    let index = old.length - 1;
    while (index >= 0 && old[index].frame !== frame) index--;
    return old.filter((_, i) => i !== index);
  });
  const videos = artifacts.filter(a => a.name.endsWith('.mp4') && a.name !== 'original.mp4');
  const videoChoice = videos.find(v => v.name === selectedVideo) || videos.find(v => v.name.startsWith('targets/')) || videos[0];
  const sync = () => {if (sourceVideo.current && targetVideo.current) targetVideo.current.currentTime = sourceVideo.current.currentTime;};
  const check = (name: string) => caps?.checks.find(c => c.name === name)?.ok;
  const framePrompts = prompts.filter(p => p.frame === frame);
  const missingSteps = ((caps?.dependencies || dependencies)[stage] || []).filter(s => job?.stages[s].status !== 'complete');
  const needsSavedMasks = !['track', 'ik'].includes(stage);
  const missingQuery = stage === 'track' && !query;

  return <div className="app">
    <header><a className="brand" href="/">TCP4D <span>Augmentation</span></a><nav aria-label="Main navigation">
      {['workspace', 'runs', 'resources'].map(t => <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>)}
    </nav></header>
    {error && <div className="error" role="alert">{error}<button onClick={() => setError('')}>Dismiss</button></div>}
    {tab === 'resources' ? <section className="panel resources">
      <div className="panel-title"><h1>Resources</h1><button onClick={() => void act(async () => setCaps(await api('/capabilities')))}>Refresh</button></div>
      <div className="checks">{caps?.checks.map(c => <div key={c.name}><span className={c.ok ? 'good' : 'missing'}>{c.ok ? 'Ready' : 'Missing'}</span><strong>{c.name}</strong><small>{c.detail}</small></div>)}</div>
    </section> : tab === 'runs' ? <section className="panel">
      <div className="panel-title"><h1>Saved runs</h1><button onClick={() => void act(async () => setJobs(await api('/jobs')))}>Refresh</button></div>
      {!jobs.length && <p>No saved runs.</p>}
      {jobs.map(j => <button className="run-row" key={j.id} onClick={() => selectJob(j)}><span><strong>{j.episode.name}</strong><small>{j.request.mask_method.toUpperCase()} · {j.request.arms.join(', ')}</small></span><span className="badge">{j.status}</span><small>{new Date(j.created_at * 1000).toLocaleString()}</small></button>)}
    </section> : <>
      <details className="panel source" open={setupOpen} onToggle={e => setSetupOpen(e.currentTarget.open)}>
        <summary><strong>Source & robots</strong><span>{info ? `${info.count} frames · ${info.fps} fps` : 'Choose an MP4 or RGB sequence'}</span></summary>
        <div className="setup-content">
          <label>Video or RGB frames<input list="episodes" value={episode} onChange={e => {setEpisode(e.target.value); setInfo(null); resetDraft();}} placeholder="/path/to/video.mp4 or /path/to/rgb/" disabled={working}/><datalist id="episodes">{episodes.map(e => <option key={e.path} value={e.path}>{e.name}</option>)}</datalist></label>
          <div className="form-row"><label>FPS (optional)<input type="number" min="0.001" step="any" value={fps ?? ''} disabled={working} placeholder="Video rate or 15" onChange={e => {setFps(e.target.value ? +e.target.value : undefined); setInfo(null);}}/></label><button disabled={working || !episode} onClick={() => void loadEpisode()}>Inspect input</button>{info && <label>Camera<select value={info.camera} disabled={working} onChange={e => void loadEpisode(e.target.value)}>{info.cameras.map(c => <option key={c}>{c}</option>)}</select></label>}</div>
          {info && <>
            <p className="instruction">{info.instruction[0] || info.name}</p>
            <label>Robot mask prompt<input value={text} disabled={working} onChange={e => setText(e.target.value)}/></label>
            <label>Target robots{job && ' · Locked'}</label><div className="arm-list">{caps?.arms.map(a => <label key={a}><input type="checkbox" checked={selectedArms.includes(a)} disabled={working || !!job || !check('asset:' + a)} onChange={e => setArms(e.target.checked ? [...arms, a] : arms.filter(x => x !== a))}/>{a.replaceAll('_', ' ')}{!check('asset:' + a) && <small>missing assets</small>}</label>)}</div>
            <div className="actions"><button className="primary" onClick={() => job ? resetDraft() : void create()} disabled={working || (!job && !arms.length)}>{job ? 'New run' : 'Create run'}</button>{job && <small>Target robots are locked for this run. Start a new run to choose different robots.</small>}</div>
          </>}
        </div>
      </details>
      <div className="run-bar"><span>{job ? <>Run <code>{job.id.slice(0, 8)}</code> <span className="badge">{job.status}</span></> : 'Create a run to get started.'}</span><div className="actions">
        {running && <button onClick={() => void act(async () => setJob(await api(`/jobs/${job!.id}/cancel`, 'POST', {})))}>Cancel</button>}
      </div></div>
      <div className="work-grid"><aside className="steps" aria-label="Pipeline steps">
        {(caps?.stages || Object.keys(names)).map((s, i) => <button key={s} aria-label={names[s]} className={stage === s ? 'selected' : ''} onClick={() => {setStage(s); if (s === 'track') setFrame(0);}}><span aria-hidden="true" className={'step-dot ' + (job?.stages[s].status || '')}>{job?.stages[s].status === 'complete' ? '✓' : i + 1}</span>{names[s]}</button>)}
      </aside><main>
        <section className="panel stage"><div className="panel-title"><h1>{names[stage]}</h1><div className="actions"><span className="badge">{job?.stages[stage].status || 'pending'}</span><button disabled={working || !job || missingQuery || (draft && needsSavedMasks) || !!missingSteps.length} onClick={() => void execute(stage)}>Run step</button><button disabled={working || !job || missingQuery || (draft && needsSavedMasks) || !!missingSteps.length} onClick={() => void execute(stage, true)}>Rerun</button></div></div>
          {job && missingSteps.length > 0 && <p className="hint">Complete {missingSteps.map(s => names[s]).join(' and ')} first, then run this step manually.</p>}
          {info && ['track', 'mask'].includes(stage) ? <>
            {stage === 'mask' && <div className="tools" aria-label="Mask tools">{[['positive', '+ Foreground'], ['negative', '− Background'], ['box', 'Box']].map(([value, label]) => <button key={value} aria-pressed={tool === value} className={tool === value ? 'active' : ''} disabled={working || !job} onClick={() => setTool(value)}>{label}</button>)}<span className="spacer"/><button disabled={working || !framePrompts.length} onClick={undo}>Undo</button><button disabled={working || !prompts.length} onClick={() => setPrompts([])}>Clear prompts</button></div>}
            <div className="canvas-wrap"><canvas ref={canvas} width={info.width} height={info.height} onPointerDown={pointerDown} onPointerUp={pointerUp} aria-label={stage === 'mask' ? 'Mask correction image' : 'Click to select the initial TCP'}/></div>
            {stage === 'mask' ? <>
              <div className="frame-row"><button aria-label="Previous frame" disabled={working || frame === 0} onClick={() => setFrame(frame - 1)}>‹</button><label>Frame {frame} / {info.count - 1}<input type="range" aria-label="Video frame" min="0" max={info.count - 1} value={frame} disabled={working} onChange={e => setFrame(+e.target.value)}/></label><button aria-label="Next frame" disabled={working || frame === info.count - 1} onClick={() => setFrame(frame + 1)}>›</button><label className="inline"><input type="checkbox" checked={showMask} onChange={e => setShowMask(e.target.checked)}/>Mask</label></div>
              <div className="mask-actions"><span role="status" className="preview-status">{preview.pending ? 'Updating current mask…' : preview.error ? 'Mask update failed.' : draft && !prompts.length ? 'Prompts cleared. Propagate to rebuild masks.' : preview.ready ? 'Current mask updated.' : 'Add foreground or background points to update this frame.'}</span>{preview.error && <button onClick={preview.retry}>Retry</button>}<button className="primary" disabled={working || !job || !reviewed || !draft || !!missingSteps.length} onClick={() => void propagate()}>Propagate</button></div>
              {preview.error && <p className="error" role="alert">{preview.error}</p>}
              {draft && <p className="hint">{reviewed ? 'Review the mask, then propagate to the full video.' : 'Wait for edited frames to finish updating before propagation.'}</p>}
            </> : <div className="query-row"><span>{query ? `TCP query: (${query[0].toFixed(1)}, ${query[1].toFixed(1)}) px` : 'Click the initial TCP in frame zero.'}</span></div>}
          </> : <p>{!info ? 'Inspect an input to begin.' : stage === 'ik' ? 'Solve motion for the selected target robots.' : stage === 'export' ? 'Export paired video and motion labels.' : 'Run this step to generate its output.'}</p>}
          {job?.error && <p className="error" role="alert">{job.error}</p>}
          {stage === 'ik' && artifacts.some(a => a.name === 'ik_summary.json') && <JsonPreview url={`/api/jobs/${job!.id}/artifacts/ik_summary.json?v=${stageState}`}/>}
          {stage === 'mask' && artifacts.some(a => a.name === 'mask_stats.json') && <JsonPreview title="Mask coverage" url={`/api/jobs/${job!.id}/artifacts/mask_stats.json?v=${stageState}`}/>}
        </section>
        {job && <details className="panel results" open={['render', 'export'].includes(stage)}><summary><strong>Results</strong><span>{videos.length} videos</span></summary><div className="results-content">
          {videoChoice ? <><select aria-label="Result video" value={videoChoice.name} onChange={e => setSelectedVideo(e.target.value)}>{videos.map(a => <option value={a.name} key={a.name}>{a.name.replace('targets/', '').replace('/video.mp4', '')}</option>)}</select><div className="video-grid"><div><small>Original</small><video ref={sourceVideo} controls src={`/api/jobs/${job.id}/artifacts/original.mp4`} onSeeked={sync} onPlay={() => {sync(); void targetVideo.current?.play().catch(() => {});}} onPause={() => targetVideo.current?.pause()}/></div><div><small>Result</small><video ref={targetVideo} controls src={videoChoice.url + '?v=' + encodeURIComponent(stageState)}/></div></div></> : <p>Results appear after running the pipeline.</p>}
          <div className="downloads">{artifacts.filter(a => a.name.endsWith('.zip') || a.name.endsWith('.mp4')).map(a => <a key={a.name} href={a.url} download>{a.name.endsWith('.zip') ? 'Download dataset' : a.name.replace('targets/', '')}</a>)}</div>
        </div></details>}
        <details className="panel logs"><summary>Run log</summary><pre aria-live="polite">{logs || 'No output yet.'}</pre></details>
      </main></div>
    </>}
  </div>;
}

function JsonPreview({url, title = 'IK details'}: {url: string; title?: string}) {
  const [value, setValue] = useState('');
  useEffect(() => {void fetch(url).then(r => r.json()).then(v => setValue(JSON.stringify(v, null, 2)));}, [url]);
  return <details><summary>{title}</summary><pre>{value}</pre></details>;
}
createRoot(document.getElementById('root')!).render(<App/>);
