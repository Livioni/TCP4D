import {useCallback, useEffect, useRef, useState} from 'react';
import {api} from './api';
import type {Point} from './geometry';

export type Prompt = {frame: number; points: Point[]; labels: number[]; box?: [number, number, number, number] | null};
export const previewKey = (job: string, frame: number, prompts: Prompt[]) => JSON.stringify([job, frame, prompts]);
type PreviewRequest = {job: string; frame: number; prompts: Prompt[]; key: string};

export function useMaskPreview(job: string | undefined, frame: number, prompts: Prompt[], enabled: boolean) {
  const [images, setImages] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [attempt, setAttempt] = useState(0);
  const desired = useRef<PreviewRequest | null>(null);
  const active = useRef(false);
  const cache = useRef<Record<string, string>>({});
  const lastFrameImage = useRef<Record<string, string>>({});
  const framePrompts = prompts.filter(p => p.frame === frame);
  const key = job ? previewKey(job, frame, framePrompts) : '';

  // Keep one request in flight. Rapid clicks replace the waiting request with
  // the latest complete set of points, so older results cannot replace it.
  const send = useCallback(async function send() {
    const next = desired.current;
    if (active.current || !next || cache.current[next.key]) return;
    active.current = true;
    try {
      const result = await api<{image: string}>(`/jobs/${next.job}/mask-preview`, 'POST',
        {frame: next.frame, prompts: next.prompts});
      cache.current[next.key] = result.image;
      lastFrameImage.current[`${next.job}:${next.frame}`] = result.image;
      setImages({...cache.current});
      setErrors(old => {const updated = {...old}; delete updated[next.key]; return updated;});
    } catch (error) {
      setErrors(old => ({...old, [next.key]: String(error)}));
    } finally {
      active.current = false;
      if (desired.current && desired.current.key !== next.key) void send();
    }
  }, []);

  useEffect(() => {
    desired.current = enabled && job && framePrompts.length ? {job, frame, prompts: framePrompts, key} : null;
    const timer = setTimeout(() => void send(), 80);
    return () => {clearTimeout(timer); desired.current = null;};
  }, [enabled, job, frame, key, attempt, send]);

  const retry = () => {
    setErrors(old => {const updated = {...old}; delete updated[key]; return updated;});
    setAttempt(value => value + 1);
  };
  return {
    image: framePrompts.length ? images[key] || lastFrameImage.current[`${job}:${frame}`] : undefined,
    ready: Boolean(images[key]),
    pending: enabled && framePrompts.length > 0 && !images[key] && !errors[key],
    error: errors[key],
    images,
    retry,
  };
}
