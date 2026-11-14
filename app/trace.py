"""Trace recording and replay (format from agents/context/protocol.md).

    {"frames": [{"ms": 0, "levels": [front, right, back, left]}, ...],
     "alerts": [{"startMs": 0, "endMs": 100}, ...]}

Replay never drives the physical motors: it only feeds frames to listeners
(dashboard, estimator).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .link import MicFrame


def load(path) -> dict:
    data = json.loads(Path(path).read_text())
    data.setdefault("alerts", [])
    for f in data["frames"]:
        if len(f["levels"]) != 4:
            raise ValueError(f"frame {f} needs 4 levels (front,right,back,left)")
    return data


def save(trace: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, separators=(",", ":")))
    return path


class Recorder:
    """Subscribe to a Link; collects MIC frames and alert windows."""

    def __init__(self):
        self.frames: list[dict] = []
        self.alerts: list[dict] = []
        self.active = False
        self._lock = threading.Lock()
        self._alert_start: Optional[int] = None

    def start(self) -> None:
        with self._lock:
            self.frames, self.alerts, self.active = [], [], True

    def __call__(self, ev) -> None:
        if self.active and isinstance(ev, MicFrame):
            with self._lock:
                self.frames.append({"ms": ev.ms, "levels": list(ev.levels)})

    def mark_alert(self, start_ms: int, end_ms: int) -> None:
        with self._lock:
            self.alerts.append({"startMs": int(start_ms), "endMs": int(end_ms)})

    def alert_down(self) -> None:
        """Start an alert window at the latest frame (e.g. Action held)."""
        if self.frames:
            self._alert_start = self.frames[-1]["ms"]

    def alert_up(self) -> None:
        if self._alert_start is not None and self.frames:
            self.mark_alert(self._alert_start, self.frames[-1]["ms"])
        self._alert_start = None

    def stop(self) -> dict:
        with self._lock:
            self.active = False
            return {"frames": list(self.frames), "alerts": list(self.alerts)}


class Replayer:
    """Feeds a trace's frames to a callback at their recorded pace (speed x)."""

    def __init__(self, trace: dict, on_frame: Callable[[MicFrame], None],
                 speed: float = 1.0, on_done: Optional[Callable[[], None]] = None):
        self.trace, self.on_frame, self.speed, self.on_done = trace, on_frame, speed, on_done
        self._stop = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> "Replayer":
        self.thread = threading.Thread(target=self.run, name="replay", daemon=True)
        self.thread.start()
        return self

    def cancel(self) -> None:
        self._stop.set()

    def run(self) -> None:
        frames = self.trace["frames"]
        if not frames:
            return
        t0, ms0 = time.monotonic(), frames[0]["ms"]
        for f in frames:
            if self._stop.is_set():
                return
            if self.speed > 0:
                due = t0 + (f["ms"] - ms0) / 1000 / self.speed
                d = due - time.monotonic()
                if d > 0:
                    time.sleep(d)
            self.on_frame(MicFrame(f["ms"], tuple(f["levels"]), time.monotonic()))
        if self.on_done and not self._stop.is_set():
            self.on_done()
