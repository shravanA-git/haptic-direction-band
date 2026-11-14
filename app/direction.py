"""Name-direction estimator, first pass (works on band-clock milliseconds).

Steps (agents/context/protocol.md): remove DC, match gains, per frame compute
(right-left, front-back) normalized by total level, combine across the alert
window, pick the nearest of 8 motors. Weak or conflicting evidence -> None
(caller plays the generic attention cue). Clock alignment from laptop speech
timestamps to ESP32 ms comes later (ClockSync below is the hook).
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional, Sequence

# Motor k sits at k*45 degrees clockwise from front.
MIN_LOUDNESS = 0.5      # mean excess over baseline, in baseline units
MIN_CONSISTENCY = 0.5   # |mean unit vector| over usable frames, 0..1
MIN_FRAMES = 2


@dataclass
class Estimate:
    motor: Optional[int]       # None -> generic attention
    angle_deg: Optional[float]
    strength: float            # mean loudness excess
    consistency: float         # 0..1 agreement of per-frame directions
    frames_used: int
    reason: str


def _baseline(frames: Sequence[dict], start: int, end: int):
    """Per-channel quiet level from frames outside the alert window."""
    quiet = [f["levels"] for f in frames if f["ms"] < start or f["ms"] > end]
    if len(quiet) < 3:
        quiet = [f["levels"] for f in frames]
    return [max(median(ch), 1e-6) for ch in zip(*quiet)]


CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"


def load_calibration(path=CALIBRATION_FILE) -> Optional[list]:
    """Per-channel sensitivity correction (front, right, back, left) written by
    `python -m tools.calibrate --save`. Absent file -> None (no correction)."""
    path = Path(path)
    if not path.exists():
        return None
    g = json.loads(path.read_text())["gains"]
    if len(g) != 4 or not all(0.2 <= float(x) <= 5 for x in g):
        raise ValueError(f"bad calibration gains {g}")
    return [float(x) for x in g]


def estimate(frames: Sequence[dict], start_ms: int, end_ms: int,
             gains: Optional[Sequence[float]] = None) -> Estimate:
    """gains: optional per-channel sensitivity correction from tools/calibrate.py,
    applied to each channel's excess over its baseline."""
    base = _baseline(frames, start_ms, end_ms)
    g = list(gains) if gains is not None else [1.0, 1.0, 1.0, 1.0]
    win = [f["levels"] for f in frames if start_ms <= f["ms"] <= end_ms]
    if not win:
        return Estimate(None, None, 0.0, 0.0, 0, "no frames in window (stale?)")
    sx = sy = ux = uy = strength = 0.0
    used = 0
    for lv in win:
        # DC removal + gain match: excess over each channel's own baseline,
        # in units of that baseline.
        e = [max(v / b - 1.0, 0.0) * k for v, b, k in zip(lv, base, g)]
        f, r, b, l = e
        total = f + r + b + l
        if total / 4 < MIN_LOUDNESS:
            continue
        x, y = (r - l) / total, (f - b) / total
        mag = math.hypot(x, y)
        if mag < 1e-6:
            continue
        used += 1
        strength += total / 4
        sx += x * total
        sy += y * total
        ux += x / mag
        uy += y / mag
    if used < MIN_FRAMES:
        return Estimate(None, None, 0.0, 0.0, used, "too few loud frames")
    consistency = math.hypot(ux, uy) / used
    strength /= used
    angle = math.degrees(math.atan2(sx, sy)) % 360
    if consistency < MIN_CONSISTENCY:
        return Estimate(None, angle, strength, consistency, used, "conflicting directions")
    motor = int(round(angle / 45)) % 8
    return Estimate(motor, angle, strength, consistency, used, "ok")


class ClockSync:
    """Maps laptop monotonic seconds to ESP32 ms using MIC frame receipts.
    offset = min over recent frames of (rx_ms - esp_ms): the least-delayed
    frame bounds the true offset best.
    """

    def __init__(self, window: int = 250):
        self.window = window
        self._d: list[float] = []

    def add(self, esp_ms: int, rx_monotonic: float) -> None:
        self._d.append(rx_monotonic * 1000 - esp_ms)
        if len(self._d) > self.window:
            self._d.pop(0)

    def to_esp_ms(self, laptop_monotonic: float) -> Optional[float]:
        if not self._d:
            return None
        return laptop_monotonic * 1000 - min(self._d)


class LiveEstimator:
    """Live name alerts: buffers recent MIC frames, maps laptop-monotonic
    speech timestamps onto the ESP32 clock and runs estimate() on the window.

    Feed it every MicFrame (it is a Link listener). Stale evidence (no frames
    yet, or the stream stopped more than stale_s ago) gives motor None, i.e.
    the generic attention cue.
    """

    def __init__(self, keep_ms: int = 8000, pad_ms: int = 100, stale_s: float = 1.0,
                 clock: Optional[ClockSync] = None, gains: Optional[Sequence[float]] = None):
        self.keep_ms, self.pad_ms, self.stale_s = keep_ms, pad_ms, stale_s
        self.gains = gains          # from load_calibration(); None = uncorrected
        self.clock = clock or ClockSync()
        self._frames: deque = deque()
        self._last_rx = 0.0
        self._lock = threading.Lock()

    def __call__(self, ev) -> None:
        ms, levels, rx = getattr(ev, "ms", None), getattr(ev, "levels", None), getattr(ev, "rx", 0.0)
        if ms is None or levels is None:
            return
        with self._lock:
            if self._frames and ms < self._frames[-1]["ms"]:   # ESP32 rebooted: clock restarted
                self._frames.clear()
                self.clock = ClockSync(self.clock.window)
            self.clock.add(ms, rx)
            self._frames.append({"ms": ms, "levels": list(levels)})
            while self._frames and ms - self._frames[0]["ms"] > self.keep_ms:
                self._frames.popleft()
            self._last_rx = rx

    def window(self, start_s: float, end_s: float):
        """Laptop monotonic seconds -> padded (start_ms, end_ms) on the ESP32 clock."""
        a, b = self.clock.to_esp_ms(start_s), self.clock.to_esp_ms(end_s)
        if a is None or b is None:
            return None
        return int(a) - self.pad_ms, int(b) + self.pad_ms

    def estimate(self, start_s: float, end_s: float, now: Optional[float] = None):
        """Returns (Estimate, esp_window or None)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            frames = list(self._frames)
            win = self.window(start_s, end_s)
            last_rx = self._last_rx
        if win is None or not frames:
            return Estimate(None, None, 0.0, 0.0, 0, "no MIC frames (band not streaming)"), None
        if now - last_rx > self.stale_s:
            return Estimate(None, None, 0.0, 0.0, 0, f"MIC stream stale ({now - last_rx:.1f} s)"), win
        return estimate(frames, *win, gains=self.gains), win
