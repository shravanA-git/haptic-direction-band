"""Name-direction estimator (works on band-clock milliseconds).

estimate() (v2, robust), per name call:
  1. noise floor per channel: median + MAD of the 2 s of MIC frames BEFORE the name
     (frames while a motor was buzzing are skipped), in absolute units;
  2. a channel counts in a frame only if it is above floor + K_SIGMA x noise, so a
     small or jumpy mic (the front one on the real hat) cannot invent signal;
     excess = (level - floor) x gain, gains from tools/calibrate.py;
  3. work on the SPIKE: frames whose (3-frame smoothed) total excess is at least
     half of the name's peak;
  4. per frame (right-left, front-back) / total; drop frames whose direction is a
     circular outlier, Huber-weight the rest, average;
  5. pick the motor: nearest 45 degrees (analytic, the default), or nearest learned
     centroid with a margin when app/centroids.json exists (tools/learn_centroids.py
     from labelled trials; head and hat shadow asymmetrically, so data beats the
     ideal cosine response). Weak, conflicting or ambiguous -> None (generic cue).
estimate_v1() is the first-pass version, kept for before/after comparisons
(estimate(..., algo="v1") or HDB_ESTIMATOR=v1).
ClockSync maps laptop speech timestamps to ESP32 ms.
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
    strength: float            # v2: SNR (mean spike excess / noise); v1: mean excess over baseline
    consistency: float         # 0..1 agreement of per-frame directions
    frames_used: int
    reason: str
    margin: float = 0.0        # v2 centroid classifier: distance gap to the runner-up (0 = analytic)
    vector: Optional[tuple] = None   # v2: mean (east, north) of the spike, the centroid feature
    saturated: int = 0         # v2: window frames left out: the first clipped one (a mic at the ADC
                               # ceiling) and everything after it


def _baseline(frames: Sequence[dict], start: int, end: int):
    """Per-channel quiet level from frames outside the alert window."""
    quiet = [f["levels"] for f in frames if f["ms"] < start or f["ms"] > end]
    if len(quiet) < 3:
        quiet = [f["levels"] for f in frames]
    return [max(median(ch), 1e-6) for ch in zip(*quiet)]


CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration.json"            # shared, Protocol-approved
CALIBRATION_LOCAL_FILE = Path(__file__).resolve().parent / "calibration.local.json"   # this wearer's hat (gitignored)


def calibration_path() -> Optional[Path]:
    """The gains file in use: this wearer's local one if it exists, else the shared default, else None."""
    for p in (CALIBRATION_LOCAL_FILE, CALIBRATION_FILE):
        if p.exists():
            return p
    return None


def load_ceiling(path=None) -> Optional[float]:
    """This hat's learned saturation level from the same gains file (written by tools/calibrate), or None."""
    path = Path(path) if path is not None else calibration_path()
    if path is None or not path.exists():
        return None
    c = json.loads(path.read_text()).get("ceiling")
    return float(c) if c else None


def load_calibration(path=None) -> Optional[list]:
    """Per-channel sensitivity correction (front, right, back, left). With no path:
    app/calibration.local.json (written by `python -m tools.calibrate --save`, gitignored) in
    preference to the shared app/calibration.json. No file -> None (no correction)."""
    path = Path(path) if path is not None else calibration_path()
    if path is None or not path.exists():
        return None
    g = json.loads(path.read_text())["gains"]
    if len(g) != 4 or not all(0.2 <= float(x) <= 5 for x in g):
        raise ValueError(f"bad calibration gains {g}")
    return [float(x) for x in g]


def estimate_v1(frames: Sequence[dict], start_ms: int, end_ms: int,
                gains: Optional[Sequence[float]] = None) -> Estimate:
    """First pass (kept for comparison). gains: optional per-channel sensitivity correction from
    tools/calibrate.py, applied to each channel's excess over its baseline."""
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

    def __init__(self, keep_ms: int = 30000, pad_ms: int = 100, stale_s: float = 1.0,
                 clock: Optional[ClockSync] = None, gains: Optional[Sequence[float]] = None,
                 centroids: Optional[dict] = None):
        self.keep_ms, self.pad_ms, self.stale_s = keep_ms, pad_ms, stale_s
        self.gains = gains          # from load_calibration(); None = uncorrected
        self.centroids = centroids  # from load_centroids(); None = analytic 8-way
        self.ceiling: Optional[float] = None   # from load_ceiling(); None = the fixed flat-frame rule only
        self._motor_spans: deque = deque(maxlen=64)   # laptop-monotonic (start_s, end_s) of motor pulses
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

    def note_motor(self, start_s: float, duration_ms: float) -> None:
        """A motor was commanded at laptop time start_s for duration_ms: its buzz bleeds into the
        mics, so those frames are left out of the noise floor and the evidence."""
        with self._lock:
            self._motor_spans.append((start_s, start_s + duration_ms / 1000.0))

    def _skip_ms(self, lo_s: float, hi_s: float) -> list:
        """Motor spans near [lo_s, hi_s] as ESP32-ms ranges (with spin-down margin)."""
        out = []
        for a, b in list(self._motor_spans):
            if b < lo_s - 4.0 or a > hi_s + 0.5:
                continue
            ea, eb = self.clock.to_esp_ms(a), self.clock.to_esp_ms(b)
            if ea is not None and eb is not None:
                out.append((int(ea) - 30, int(eb) + MOTOR_MARGIN_MS))
        return out

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
        skip = self._skip_ms(start_s, end_s)
        return estimate(frames, *win, gains=self.gains, centroids=self.centroids, skip_ms=skip,
                        ceiling=self.ceiling), win



# --------------------------------------------------------------------------
# v2: robust estimator
# --------------------------------------------------------------------------

import os  # noqa: E402

ALGORITHM = os.environ.get("HDB_ESTIMATOR", "v2")
PRE_ROLL_MS = 2000
MIN_PRE_FRAMES = 6
K_SIGMA = 3.0             # a channel counts only if it is this many noise sigmas above its floor
V2_MIN_CONSISTENCY = 0.6  # whole-spike coherence for a direction to exist (swept on the sims: matches
                          # v1 on the diffuse control, 40 -> 153 correct on "noisy")
MIN_SNR = 3.0             # mean spike excess over the noise of all four channels
SPIKE_FRAC = 0.5
OUTLIER_DEG = 60.0
HUBER_DEG = 30.0
CENTROID_MARGIN = 0.15    # relative gap (d2 - d1) / d2 between the nearest and runner-up centroid; 0 = on the
                          # midline between two directions. Scale-free: centroids can sit close together
CENTROIDS_FILE = Path(__file__).resolve().parent / "centroids.json"
MOTOR_MARGIN_MS = 120     # frames this long after a motor pulse (spin-down, bleed) are skipped
SATURATED = 3900          # a frame with any channel at or above this is clipped (ADC max 4095): its
                          # ratios are meaningless, so it is excluded, not used
FLAT_LEVEL = 2200         # ...and the real hat flattens lower (preamp rails / 11 dB ADC range, ~2400-2650):
FLAT_CONTRAST = 0.15      # a frame above FLAT_LEVEL whose four channels are within 15% is saturated
CEILING_FRAC = 0.95       # with a learned ceiling (calibration.local.json "ceiling"), >= 95% of it is saturated
CONTRAST_FULL = 0.25      # frames with excess contrast below this are down-weighted, cubed (c/0.25)^3
SATURATED_REASON = "saturated: sound too loud for the mics"
SMOOTH_MIN_FRAMES = 8     # 3-frame smoothing of the spike only for windows long enough to need it
                          # (a 60-80 ms clap has 3-4 frames; smoothing left too few)


def _in_spans(ms: float, spans: Sequence) -> bool:
    return any(a <= ms <= b for a, b in spans)


def contrast(levels: Sequence[float]) -> float:
    """(max - min) / max over the four channels, 0 = perfectly flat."""
    m = max(levels)
    return (m - min(levels)) / m if m > 0 else 0.0


def is_saturated(levels: Sequence[float], ceiling: Optional[float] = None) -> bool:
    """A clipped frame: any channel at the ADC ceiling, or at the hat's learned ceiling, or well above
    FLAT_LEVEL with all four channels nearly equal (they have flattened against the rails)."""
    m = max(levels)
    if m >= SATURATED:
        return True
    if ceiling and m >= CEILING_FRAC * ceiling:
        return True
    return m >= FLAT_LEVEL and contrast(levels) < FLAT_CONTRAST


def noise_floor(frames: Sequence[dict], start_ms: int, end_ms: int, skip: Sequence = (),
                ceiling: Optional[float] = None):
    """(median, sigma) per channel, absolute units, from the quiet frames before the name."""
    ok = lambda f: not _in_spans(f["ms"], skip) and not is_saturated(f["levels"], ceiling)   # noqa: E731
    pre = [f["levels"] for f in frames if start_ms - PRE_ROLL_MS <= f["ms"] < start_ms and ok(f)]
    if len(pre) < MIN_PRE_FRAMES:     # short recording: everything outside the window, then everything
        pre = [f["levels"] for f in frames if (f["ms"] < start_ms or f["ms"] > end_ms) and ok(f)]
        if len(pre) < 3:
            pre = [f["levels"] for f in frames if not is_saturated(f["levels"], ceiling)] or \
                  [f["levels"] for f in frames]
    med, sig = [], []
    for ch in zip(*pre):
        m = median(ch)
        mad = median(abs(v - m) for v in ch)
        med.append(m)
        sig.append(max(1.4826 * mad, 0.02 * abs(m), 0.5))
    return med, sig


def _circ_dist(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def load_centroids(path=CENTROIDS_FILE) -> Optional[dict]:
    """{motor: (east, north)} learned by tools/learn_centroids.py, or None."""
    path = Path(path)
    if not path.exists():
        return None
    c = json.loads(path.read_text())["centroids"]
    out = {int(k): (float(v[0]), float(v[1])) for k, v in c.items()}
    if len(out) < 2:
        raise ValueError("centroids.json needs at least 2 classes")
    return out


def classify_centroid(vec, centroids: dict, margin: float = None) -> tuple:
    """(motor or None, relative margin) by nearest centroid in (east, north); None when the
    point is too close to the midline between two directions."""
    margin = CENTROID_MARGIN if margin is None else margin
    ranked = sorted((math.hypot(vec[0] - cx, vec[1] - cy), m) for m, (cx, cy) in centroids.items())
    if len(ranked) < 2:
        return ranked[0][1], 1.0
    d1, d2 = ranked[0][0], ranked[1][0]
    rel = (d2 - d1) / d2 if d2 > 1e-9 else 0.0
    return (ranked[0][1] if rel >= margin else None), rel


def estimate(frames: Sequence[dict], start_ms: int, end_ms: int,
             gains: Optional[Sequence[float]] = None, centroids: Optional[dict] = None,
             skip_ms: Sequence = (), algo: Optional[str] = None,
             min_frames: Optional[int] = None, ceiling: Optional[float] = None) -> Estimate:
    """Robust name direction (see the module docstring). gains: per-channel sensitivity
    correction (tools/calibrate.py), multiplies each channel's excess. centroids: {motor:
    (east, north)} from load_centroids(); skip_ms: [(a, b)] band-clock spans when a motor was on.
    min_frames: spike frames needed (default 2; calibration claps are one informative frame, so
    tools/calibrate and tools/learn_centroids pass 1). ceiling: this hat's learned saturation level
    (load_ceiling()); frames at 95% of it are clipped even when not perfectly flat."""
    if (algo or ALGORITHM) == "v1":
        return estimate_v1(frames, start_ms, end_ms, gains)
    skip = list(skip_ms)
    in_win = [f for f in frames if start_ms <= f["ms"] <= end_ms and not _in_spans(f["ms"], skip)]
    if not in_win:
        return Estimate(None, None, 0.0, 0.0, 0, "no frames in window (stale?)")
    # Clipped frames carry no direction, and neither does what follows them: after a clap or shout
    # pins the mics at 4095 the decay frames read near-equal on all four channels (real hat, Nov 14).
    # So only frames BEFORE the first clipped one are evidence.
    first_clip = next((f["ms"] for f in in_win if is_saturated(f["levels"], ceiling)), None)
    win = [f for f in in_win if first_clip is None or f["ms"] < first_clip]
    nsat = len(in_win) - len(win)
    min_frames = MIN_FRAMES if min_frames is None else max(1, int(min_frames))

    def generic(reason, angle=None, snr=0.0, cons=0.0, used=0, margin=0.0, vector=None):
        if nsat and reason in ("too few loud frames", "no frames in window (stale?)"):
            reason = SATURATED_REASON                 # too little unclipped signal left: say why
        return Estimate(None, angle, snr, cons, used, reason, margin, vector, nsat)
    if not win:
        return generic("no frames in window (stale?)")
    med, sig = noise_floor(frames, start_ms, end_ms, skip, ceiling)
    g = list(gains) if gains is not None else [1.0, 1.0, 1.0, 1.0]
    ex = []                                    # per frame: gated, gain-matched excess, front/right/back/left
    for f in win:
        ex.append([(v - m) * k if (v - m) > K_SIGMA * s else 0.0
                   for v, m, s, k in zip(f["levels"], med, sig, g)])
    tot = [sum(e) for e in ex]
    if max(tot) <= 0:
        return generic("too few loud frames")
    # A frame whose four channels are nearly equal says little about direction (loud decay, reverb, a
    # mic near its rails). Weight by the contrast of its excess over the floor, both when choosing the
    # spike and when averaging: otherwise loud flat frames outvote the one directional frame.
    fac = [min(1.0, contrast([max((v - m) * k, 0.0) for v, m, k in zip(f["levels"], med, g)])
               / CONTRAST_FULL) ** 3 for f in win]     # cubic: many flat frames must not add up to a majority
    tw = [t * c for t, c in zip(tot, fac)]
    if max(tw) <= 0:
        return generic("conflicting directions", snr=0.0)
    sm = ([sum(tw[max(0, i - 1):i + 2]) / 3 for i in range(len(tw))]        # 3-frame smoothing
          if len(tw) >= SMOOTH_MIN_FRAMES else list(tw))
    peak = max(sm)
    keep = [i for i in range(len(tw)) if sm[i] >= SPIKE_FRAC * peak and tw[i] > 0]
    noise = math.sqrt(sum(s * s for s in sig))
    snr = (sum(tot[i] for i in keep) / len(keep)) / noise if keep else 0.0
    if len(keep) < min_frames or snr < MIN_SNR:
        return generic("too few loud frames", snr=snr, used=len(keep))
    vec = []                                   # (x, y, weight) per spike frame
    for i in keep:
        f_, r_, b_, l_ = ex[i]
        x, y = (r_ - l_) / tot[i], (f_ - b_) / tot[i]
        if math.hypot(x, y) > 1e-6:
            vec.append((x, y, tw[i]))
    vec = [v for v in vec if v[2] > 0]
    if len(vec) < min_frames:
        return generic("too few loud frames", snr=snr, used=len(vec))
    w_all = sum(w for _, _, w in vec)
    # Is there a direction at all? Coherence of the WHOLE spike, before any rejection: real calls
    # are coherent, diffuse noise and rotating peaks are not (outlier rejection must not manufacture
    # a direction out of random frames).
    cx = sum(x / math.hypot(x, y) * w for x, y, w in vec) / w_all
    cy = sum(y / math.hypot(x, y) * w for x, y, w in vec) / w_all
    if math.hypot(cx, cy) < V2_MIN_CONSISTENCY:
        return generic("conflicting directions", math.degrees(math.atan2(cx, cy)) % 360, snr,
                       math.hypot(cx, cy), len(vec))
    for _ in range(2):                         # drop circular outliers around the weighted direction
        sx = sum(x * w for x, y, w in vec)
        sy = sum(y * w for x, y, w in vec)
        centre = math.degrees(math.atan2(sx, sy)) % 360
        kept = [v for v in vec if _circ_dist(math.degrees(math.atan2(v[0], v[1])) % 360, centre) <= OUTLIER_DEG]
        if len(kept) < min_frames:
            break
        vec = kept
    agree = sum(w for _, _, w in vec) / w_all   # share of the spike that agrees; a few outliers are fine,
    if agree < 0.5:                             # but with no majority (rotating peaks) there is no direction
        return generic("conflicting directions", centre, snr, 0.0, len(vec))
    ws = []                                    # Huber weights: far-from-centre frames count less
    for x, y, w in vec:
        d = _circ_dist(math.degrees(math.atan2(x, y)) % 360, centre)
        ws.append(w * (1.0 if d <= HUBER_DEG else HUBER_DEG / d))
    W = sum(ws)
    X = sum(x * w for (x, y, _), w in zip(vec, ws)) / W
    Y = sum(y * w for (x, y, _), w in zip(vec, ws)) / W
    ux = sum(x / math.hypot(x, y) * w for (x, y, _), w in zip(vec, ws)) / W
    uy = sum(y / math.hypot(x, y) * w for (x, y, _), w in zip(vec, ws)) / W
    consistency = math.hypot(ux, uy) * agree     # outliers that were dropped still lower confidence
    angle = math.degrees(math.atan2(X, Y)) % 360
    if len(vec) < min_frames:
        return generic("too few loud frames", angle, snr, consistency, len(vec), vector=(X, Y))
    if consistency < V2_MIN_CONSISTENCY:
        return generic("conflicting directions", angle, snr, consistency, len(vec), vector=(X, Y))
    if centroids:
        motor, margin = classify_centroid((X, Y), centroids)
        if motor is None:
            return generic("ambiguous (close to two directions)", angle, snr, consistency, len(vec),
                           margin, (X, Y))
        return Estimate(motor, angle, snr, consistency, len(vec), "ok", margin, (X, Y), nsat)
    return Estimate(int(round(angle / 45)) % 8, angle, snr, consistency, len(vec), "ok", 0.0, (X, Y), nsat)
