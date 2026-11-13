"""Persistent, isolated SAM3 preview worker sharing the pipeline's GPU queue."""
from __future__ import annotations

import multiprocessing
import threading


def serve(connection, config):
    import base64
    import contextlib
    import io
    import os
    import signal
    import time
    import traceback

    os.environ.update(config.env())
    config.work.mkdir(parents=True, exist_ok=True)

    def terminate(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    with (config.work / 'sam3-preview.log').open('a', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from PIL import Image
            import numpy as np
            from .segmentation import FramePreviewSession

            session = FramePreviewSession(config['sam3_checkpoint'])
            try:
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    started = time.perf_counter()
                    try:
                        mask, source = session.update(request['image'], request['prompts'], request['text'])
                        overlay = source.copy()
                        overlay[mask] = (overlay[mask] * .4 + np.array([255, 85, 100]) * .6).astype(np.uint8)
                        image = io.BytesIO()
                        Image.fromarray(overlay).save(image, format='PNG')
                        result = dict(image='data:image/png;base64,' + base64.b64encode(image.getvalue()).decode(),
                                      frame_reused=session.frame_reused,
                                      update_ms=round((time.perf_counter() - started) * 1000, 1))
                        print(f'Preview: {result["update_ms"]} ms; frame reused={result["frame_reused"]}', flush=True)
                        connection.send(dict(ok=True, result=result))
                    except Exception as exc:
                        traceback.print_exc()
                        session.clear_frame()
                        connection.send(dict(ok=False, error=str(exc)))
            finally:
                session.close()
                connection.close()


class PreviewClient:
    def __init__(self, config, worker=serve):
        self.config = config
        self.worker = worker
        self.process = None
        self.connection = None
        self.lock = threading.RLock()
        self.closed = False

    def update(self, image, prompts, text, timeout=120):
        with self.lock:
            if self.closed:
                raise ValueError('The preview worker is shutting down')
            if self.process is None or not self.process.is_alive():
                self.release()
                context = multiprocessing.get_context('spawn')
                self.connection, child = context.Pipe()
                self.process = context.Process(target=self.worker, args=(child, self.config), name='sam3-preview')
                try:
                    self.process.start()
                except Exception:
                    self.release()
                    raise
                finally:
                    child.close()
            connection = self.connection
        try:
            connection.send(dict(image=str(image), prompts=prompts, text=text))
            if not connection.poll(timeout):
                self.release()
                raise ValueError('SAM3 preview timed out; try again')
            response = connection.recv()
        except (EOFError, OSError) as exc:
            self.release()
            raise ValueError('SAM3 preview worker stopped; try again') from exc
        if not response['ok']:
            raise ValueError('SAM3 mask preview failed: ' + response['error'])
        return response['result']

    def release(self):
        """Give the GPU back before another pipeline stage needs it."""
        with self.lock:
            if self.process is not None:
                if self.process.pid is not None:
                    if self.process.is_alive():
                        self.process.terminate()
                    self.process.join(timeout=5)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join()
                self.process = None
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def shutdown(self):
        with self.lock:
            self.closed = True
            self.release()
