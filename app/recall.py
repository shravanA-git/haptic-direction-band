"""Alert log: every cue the band gives, with time, source, direction and
confidence. Saved as JSONL so tools/trial_metrics.py (H27 pitch table) and,
later, the Recall chatbot can read it.

One line per record:
  {"type":"alert","id":3,"t":1763100000.12,"time":"02:14:05","source":"name",
   "cue":"directional","motor":5,"direction":"back-left","confidence":0.91,
   "latency_ms":420,"dry":false, ...detail (reason, window, heard, target, repeat)}
  {"type":"label","ref":3,"truth":5,"t":...}      # operator: caller was back-left
  {"type":"label","ref":4,"truth":"false","t":...}  # nobody called: false alert
  {"type":"label","ref":null,"truth":2,"missed":true,"t":...}  # called, no cue

source: name | find | sound | manual | replay
cue:    directional | generic | centered | lost | sound
truth:  motor 0-7, null (no clear direction: generic is right), or "false"
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Optional

from .haptics import MOTOR_NAMES

SOURCES = ("name", "find", "sound", "manual", "replay", "avoid")
CUES = ("directional", "generic", "centered", "found", "lost", "sound", "ack", "unknown")


def _clean(v):
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


class AlertLog:
    def __init__(self, path=None, keep: int = 500):
        self.path = Path(path) if path else None
        self._recent: deque = deque(maxlen=keep)
        self._labels: dict = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict], None]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def add(self, source: str, cue: str, motor: Optional[int] = None,
            confidence: Optional[float] = None, latency_ms: Optional[float] = None,
            dry: bool = False, **detail) -> dict:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if cue not in CUES:
            raise ValueError(f"unknown cue {cue!r}")
        now = time.time()
        with self._lock:
            rec = {"type": "alert", "id": self._next_id, "t": round(now, 3),
                   "time": time.strftime("%H:%M:%S", time.localtime(now)),
                   "mono": round(time.monotonic(), 3), "source": source, "cue": cue,
                   "motor": None if motor is None else int(motor),
                   "direction": None if motor is None else MOTOR_NAMES[int(motor) % 8],
                   "confidence": None if confidence is None else round(float(confidence), 3),
                   "latency_ms": None if latency_ms is None else int(round(latency_ms)),
                   "dry": bool(dry)}
            rec.update({k: _clean(v) for k, v in detail.items() if v is not None})
            self._next_id += 1
            self._recent.append(rec)
            self._write(rec)
        self._notify(rec)
        return rec

    def label(self, ref: Optional[int], truth, missed: bool = False) -> dict:
        """Operator ground truth for alert `ref` (or a missed call, ref None)."""
        if truth not in (None, "false") and not (isinstance(truth, int) and 0 <= truth < 8):
            raise ValueError("truth must be a motor 0-7, null or 'false'")
        rec = {"type": "label", "ref": ref, "truth": truth, "t": round(time.time(), 3)}
        if missed:
            rec["missed"] = True
        with self._lock:
            if ref is not None:
                if not any(a["id"] == ref for a in self._recent):
                    raise ValueError(f"no alert {ref}")
                self._labels[ref] = truth
            self._write(rec)
        self._notify(rec)
        return rec

    def latest(self, sources: Iterable[str] = ("name", "sound", "find"),
               unlabelled: bool = True) -> Optional[dict]:
        with self._lock:
            for a in reversed(self._recent):
                if a["source"] in sources and not a["dry"] and (
                        not unlabelled or a["id"] not in self._labels):
                    return a
        return None

    def recent(self, n: int = 20, source: Optional[str] = None) -> list:
        """Newest last. Each alert carries "truth" once labelled."""
        with self._lock:
            items = [a for a in self._recent if source is None or a["source"] == source][-n:]
            return [dict(a, truth=self._labels[a["id"]]) if a["id"] in self._labels else dict(a)
                    for a in items]

    def _write(self, rec: dict) -> None:
        if not self.path:
            return
        try:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError as e:
            print(f"[recall] could not write {self.path}: {e}")

    def _notify(self, rec: dict) -> None:
        for fn in self._listeners:
            try:
                fn(rec)
            except Exception as e:
                print(f"[recall] listener error: {e}")


def load(path) -> tuple[list, list]:
    """Read a JSONL alert log -> (alerts, labels). Skips broken lines."""
    alerts, labels = [], []
    for line in Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        (alerts if r.get("type") == "alert" else labels if r.get("type") == "label" else []).append(r)
    return alerts, labels
