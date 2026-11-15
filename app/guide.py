"""Find-an-object cue policy: hot/cold guidance from the live vision track.

While a target is tracked (left / right / centered) the band pulses continuously:

  where   the motor points at the object (motor_for):
            centered                -> front 0
            off to one side         -> front-left 7 / front-right 1
            far to one side         -> left 6 / right 2
          The SIDE comes from the tracker's state (already dwell-filtered), the near/far
          choice from the smoothed offset with hysteresis around SIDE_FAR, so a target
          sitting near a threshold does not flicker between neighbouring motors.
  how hot the pulse rate rises as you get closer (interval_for): ~1.2 s when the
          object is far off-axis and not getting closer, ~0.5 s centered, down to
          ~0.18 s centered and close. "Close" is RELATIVE (closeness_from): the box size
          against the smallest stable size seen since tracking started, so a phone and a
          chair both read "closer" as you approach. Fast pulses are shorter (80 ms).
  centered  two front pulses once on entering, then the front pulses keep going
  FOUND     centered and close for FOUND_HOLD_S (1 s): one distinct long pattern, once,
            then quiet until the target moves away (the demo's ending)
  lost      lost cue once, then silence until the target is reacquired. A dropout shorter
            than LOST_GRACE_S keeps the last direction and keeps pulsing
  searching / idle   nothing

Vision feeds every frame through on_frame() (offset and box size change between state
changes; frames without a box are dropouts and change nothing); on_decision() handles
state changes; tick() fires the pulses. PROVISIONAL timings (approved by Shravan Nov 14).
"""
from __future__ import annotations

import math
import threading
import time
from statistics import median
from typing import Callable, Optional

from . import haptics

FRONT, FRONT_LEFT, LEFT, FRONT_RIGHT, RIGHT = 0, 7, 6, 1, 2
LEFT_MOTOR, RIGHT_MOTOR = LEFT, RIGHT      # far-side motors (kept for imports)
SIDE_FAR = 0.55        # |offset| beyond this -> side motor 6/2, else front-diagonal 7/1
SIDE_HYST = 0.08       # ...but only switch near<->far after crossing it by this much
SLOW_S, FAST_S = 1.2, 0.18                 # pulse interval: cold .. hot
LOG_EVERY_S = 1.0      # alert log: a repeat on the same motor at most this often
REPEAT_S = SLOW_S      # (old name)

TAU_OFFSET_S = 0.25    # smoothing time constants (EMA by elapsed time, so frame rate doesn't matter)
TAU_SIZE_S = 0.5
REF_FRAMES = 4         # frames to establish the starting size
GROW = 2.2             # a box 2.2x the starting size is fully "close"
FRAME_FULL = 0.6       # a box this share of the frame is always fully close
FOUND_CLOSE = 0.75     # closeness that counts as "close" for FOUND
FOUND_HOLD_S = 1.0     # centered and close for this long -> FOUND
FOUND_RELEASE = 0.5    # FOUND latch lifts when the target is off-centre or closeness is below this...
FOUND_AWAY_S = 0.5     # ...for this long
LOST_GRACE_S = 0.7     # no box for less than this never counts as lost


def motor_for(state: str, offset: Optional[float], far: Optional[bool] = None) -> int:
    """The motor that points at the object; offset -1 (left edge) .. +1 (right).
    far: the hysteresis-decided near/far flag; None decides from SIDE_FAR directly."""
    if state == "centered":
        return FRONT
    if offset is None:                       # no live offset: the side the state names
        return haptics.nearest_active(
            LEFT if state == "left" else RIGHT if state == "right" else FRONT)
    if far is None:
        far = abs(offset) > SIDE_FAR
    left = offset < 0
    if state in ("left", "right"):           # the tracker's (dwell-filtered) side beats a noisy sign
        left = state == "left"
    m = (LEFT if far else FRONT_LEFT) if left else (RIGHT if far else FRONT_RIGHT)
    return haptics.nearest_active(m, -1.0 if left else 1.0)    # 4-motor band: 7 -> 6, 1 -> 2


def closeness_from(size: float, ref: Optional[float]) -> float:
    """0 (as small as when tracking started) .. 1 (2.2x that, or filling the frame)."""
    if not ref or ref <= 0 or size <= 0:
        return 0.0
    big = max(min(ref * GROW, FRAME_FULL), ref * 1.15)
    return min(1.0, max(0.0, (size - ref) / (big - ref)))


def heat(state: str, offset: Optional[float], closeness: float) -> float:
    """0 (cold: far off-axis, not getting closer) .. 1 (hot: centered and close)."""
    centering = 1.0 if state == "centered" else 1.0 - min(1.0, abs(offset or 1.0))
    return 0.6 * centering + 0.4 * min(1.0, max(0.0, closeness))


def interval_for(state: str, offset: Optional[float], closeness: float) -> float:
    """Seconds between pulses: SLOW_S cold .. FAST_S hot."""
    return SLOW_S - (SLOW_S - FAST_S) * heat(state, offset, closeness)


def pulse_ms(interval: float) -> int:
    """Shorter pulses at high rates so they stay separate."""
    if interval < 0.35:
        return 80
    if interval < 0.6:
        return 120
    return haptics.DIRECTIONAL_MS


class ObjectGuide:
    def __init__(self, play: Callable[[list], object], repeat_s: float = REPEAT_S,
                 on_cue: Optional[Callable[[str, Optional[int], dict], None]] = None):
        self.play = play            # play(cue), e.g. lambda c: haptics.play(c, link.pulse)
        self.repeat_s = repeat_s    # unused since hot/cold; kept for callers
        # on_cue(kind, motor, info): cues played, for the alert log (repeats on the
        # same motor at most every LOG_EVERY_S). info: confidence, target,
        # latency_ms (decision frame -> cue), repeat, interval_s.
        self.on_cue = on_cue
        self._decision = None
        self.state = "idle"
        self.paused = False
        self._next: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._logged = (None, 0.0)          # (motor, time) of the last logged pulse
        # Optional app.avoid.AvoidLoop: before each pulse, steer around an
        # obstacle (veer/turn motor) or give the crash warning instead.
        self.avoid = None
        self._reset_track()

    # ---- the smoothed view of the target ----
    def _reset_track(self) -> None:
        self._raw_off: list = []                 # last 3 raw offsets / sizes: a median-of-3 rejects a
        self._raw_size: list = []                # single-frame spike before the EMA sees it
        self._off: Optional[float] = None        # smoothed offset
        self._size: Optional[float] = None       # smoothed box size
        self._ref: Optional[float] = None        # smallest stable size since tracking started
        self._ref_seed: list = []
        self._obs_t: Optional[float] = None
        self._last_box_t: Optional[float] = None
        self._far = False                         # near/far hysteresis state
        self._close_since: Optional[float] = None
        self._away_since: Optional[float] = None
        self._found = False                       # FOUND played; quiet until the target moves away
        self._lost_pending = None                 # (decision, deadline) while a short dropout is waited out

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        return 1.0 - math.exp(-max(dt, 0.0) / tau)

    def _observe(self, d, now: float) -> None:
        """Fold one vision frame into the smoothed offset/size. Frames without a box
        (detector dropouts) change nothing: the last direction is kept."""
        t = getattr(d, "t", None)
        t = now if t is None else t
        if d.state not in ("left", "right", "centered"):
            if d.state == "lost" and self._within_grace(d, now):
                return                             # a short dropout: keep the track, the grace decides
            self._reset_track()
            return
        if getattr(d, "bbox", True) is None:
            return                                 # dropout: hold
        self._last_box_t = t
        self._lost_pending = None                  # a box is back: whatever was pending is moot
        off, size = getattr(d, "offset", None), getattr(d, "size", 0.0) or 0.0
        if off is not None:
            self._raw_off = (self._raw_off + [off])[-3:]
            off = median(self._raw_off)            # one wild frame cannot move the offset
        if size > 0:
            self._raw_size = (self._raw_size + [size])[-3:]
            size = median(self._raw_size)
        dt = 0.0 if self._obs_t is None else t - self._obs_t
        self._obs_t = t
        if off is not None:
            self._off = off if self._off is None else self._off + self._alpha(dt, TAU_OFFSET_S) * (off - self._off)
            a = abs(self._off)
            if self._far and a < SIDE_FAR - SIDE_HYST:
                self._far = False
            elif not self._far and a > SIDE_FAR + SIDE_HYST:
                self._far = True
        if size > 0:
            self._size = size if self._size is None else self._size + self._alpha(dt, TAU_SIZE_S) * (size - self._size)
            if self._ref is None:
                self._ref_seed.append(self._size)
                if len(self._ref_seed) >= REF_FRAMES:
                    self._ref = max(median(self._ref_seed), 1e-3)
            else:
                self._ref = min(self._ref, self._size)          # moving away resets "far"; approaching heats up

    def _within_grace(self, d, now: float) -> bool:
        """A 'lost' verdict that arrives less than LOST_GRACE_S after the last box while we are
        tracking is a dropout, not a loss."""
        t = getattr(d, "t", None)
        return (self.state in ("left", "right", "centered") and self._last_box_t is not None
                and (now if t is None else t) - self._last_box_t < LOST_GRACE_S)

    def closeness(self) -> float:
        return closeness_from(self._size or 0.0, self._ref)

    # ---- inputs ----
    def on_frame(self, d, now: Optional[float] = None) -> None:
        """Every vision frame: keeps offset and box size current between state changes."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._observe(d, now)
            changed = d.state != self.state
            if not changed:
                self._decision = d
        if changed:
            self.on_decision(d, now, observe=False)

    def on_decision(self, d, now: Optional[float] = None, observe: bool = True, force: bool = False) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if observe:
                self._observe(d, now)
            if d.state == "lost" and not force and self._within_grace(d, now):
                self._lost_pending = (d, self._last_box_t + LOST_GRACE_S)   # a short dropout: keep going
                self._decision = self._decision or d
                return
            if d.state == self.state:
                self._decision = d
                return
            self.state, self._next = d.state, None
            self._decision = d
            self._found, self._close_since, self._away_since = False, None, None
            if self.paused:
                return
            if d.state in ("left", "right"):
                self._fire(now, repeat=False)
            elif d.state == "centered":
                self.play(haptics.centered())
                self._report("centered", None, now, False)
                self._next = now + self._interval()   # then keep pulsing (the hot signal)
            elif d.state == "lost":
                self.play(haptics.lost())
                self._report("lost", None, now, False)

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        pending = None
        with self._lock:
            if self._lost_pending and now >= self._lost_pending[1]:
                pending, self._lost_pending = self._lost_pending[0], None
            elif not self.paused and self.state in ("left", "right", "centered"):
                self._found_logic(now)
                if (not self._found and self._next is not None and now >= self._next):
                    self._fire(now, repeat=True)
        if pending is not None:                    # the dropout outlasted the grace period: really lost
            self._reset_track()
            self.on_decision(pending, now, observe=False, force=True)

    def pause(self) -> None:
        self.paused = True

    def resume(self, now: Optional[float] = None) -> None:
        with self._lock:
            self.paused = False
            self._close_since = self._away_since = None
            self._next = time.monotonic() if now is None else now  # pulses restart right away

    # ---- internals ----
    def _interval(self) -> float:
        return interval_for(self.state, self._off, self.closeness())

    def _found_logic(self, now: float) -> None:
        """FOUND: centered and close for FOUND_HOLD_S -> one long pattern, then quiet until the
        target moves away (off-centre or farther) for FOUND_AWAY_S."""
        close = self.state == "centered" and self.closeness() >= FOUND_CLOSE
        if not self._found:
            if not close:
                self._close_since = None
            elif self._close_since is None:
                self._close_since = now
            elif now - self._close_since >= FOUND_HOLD_S:
                self._found, self._close_since, self._away_since = True, None, None
                self.play(haptics.found())
                self._report("found", None, now, False)
            return
        away = self.state != "centered" or self.closeness() < FOUND_RELEASE
        if not away:
            self._away_since = None
        elif self._away_since is None:
            self._away_since = now
        elif now - self._away_since >= FOUND_AWAY_S:
            self._found, self._away_since = False, None            # the target moved: guide again
            self._next = now

    def _fire(self, now: float, repeat: bool) -> None:
        iv = self._interval()
        m = motor_for(self.state, self._off, self._far)
        ov = self.avoid.override(now) if self.avoid is not None else None
        self._next = now + iv
        if ov is not None and ov.action == "stop":
            return                      # AvoidLoop is playing the crash warning
        if ov is not None and ov.motor is not None:
            m = ov.motor                # steer around the obstacle toward the target
        self.play([(0, m, pulse_ms(iv))])
        last_m, last_t = self._logged
        if not repeat or m != last_m or now - last_t >= LOG_EVERY_S:
            self._logged = (m, now)
            self._report("directional", m, now, repeat, interval_s=round(iv, 2))

    def _report(self, kind: str, motor: Optional[int], now: float, repeat: bool, **extra) -> None:
        if not self.on_cue:
            return
        d = self._decision
        t = getattr(d, "t", None)
        info = {"confidence": getattr(d, "confidence", None), "target": getattr(d, "target", None),
                "state": self.state, "repeat": repeat,
                "latency_ms": (now - t) * 1000 if (t is not None and not repeat) else None, **extra}
        try:
            self.on_cue(kind, motor, info)
        except Exception as e:
            print(f"[guide] on_cue error: {e}")

    def start(self) -> "ObjectGuide":
        def run():
            while not self._stop.wait(0.02):    # fine enough for 0.18 s intervals
                self.tick()
        threading.Thread(target=run, name="guide", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()
