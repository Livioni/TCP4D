"""Sample GPU memory for this worker and its children, excluding other users' jobs."""
from __future__ import annotations

import os
import subprocess
import threading
import time


class StageMetrics:
    def __enter__(self):
        self.started = time.monotonic()
        self.peak = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        try:
            import psutil
            root = psutil.Process(os.getpid())
        except ImportError:
            return
        while not self.stop.is_set():
            try:
                pids = {root.pid, *[p.pid for p in root.children(recursive=True)]}
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                total = 0
                for row in result.stdout.splitlines():
                    columns = [v.strip() for v in row.split(',')]
                    if len(columns) == 2 and columns[0].isdigit() and columns[1].isdigit() and int(columns[0]) in pids:
                        total += int(columns[1])
                if total:
                    self.peak = max(self.peak or 0, total)
            except (OSError, subprocess.TimeoutExpired, psutil.Error):
                pass
            self.stop.wait(.5)

    def __exit__(self, *args):
        self.elapsed = time.monotonic() - self.started
        self.stop.set()
        self.thread.join(timeout=3)

    def result(self):
        return dict(elapsed_s=round(self.elapsed, 2), sampled_peak_gpu_mib=self.peak,
                    sampling_interval_s=.5, scope='worker and descendants; excludes unrelated processes')
