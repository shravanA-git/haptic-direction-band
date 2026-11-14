"""Name-direction estimator, first pass (works on band-clock milliseconds).

Steps (agents/context/protocol.md): remove DC, match gains, per frame compute
(right-left, front-back) normalized by total level, combine across the alert
window, pick the nearest of 8 motors. Weak or conflicting evidence -> None
(caller plays the generic attention cue). Clock alignment from laptop speech
timestamps to ESP32 ms comes later (ClockSync below is the hook).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
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


def estimate(frames: Sequence[dict], start_ms: int, end_ms: int) -> Estimate:
    base = _baseline(frames, start_ms, end_ms)
    win = [f["levels"] for f in frames if start_ms <= f["ms"] <= end_ms]
    if not win:
        return Estimate(None, None, 0.0, 0.0, 0, "no frames in window (stale?)")
    sx = sy = ux = uy = strength = 0.0
    used = 0
    for lv in win:
        # DC removal + gain match: excess over each channel's own baseline,
        # in units of that baseline.
        e = [max(v / b - 1.0, 0.0) for v, b in zip(lv, base)]
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
