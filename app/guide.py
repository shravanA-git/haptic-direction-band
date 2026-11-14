"""Find-an-object cue policy: hot/cold guidance from the live vision track.

While a target is tracked (left / right / centered) the band pulses continuously:

  where   the motor points at the object (motor_for):
            centered                -> front 0
            off to one side         -> front-left 7 / front-right 1
            far to one side         -> left 6 / right 2
  how hot the pulse rate rises as you get closer (interval_for): ~1.2 s when the
          object is far off-axis and small, ~0.5 s centered but small, down to
          ~0.18 s centered and big (close). Fast pulses are shorter (80 ms) so
          they don't blur into one buzz.
  centered  two front pulses once on entering, then the front pulses keep going
  lost      lost cue once, then silence until the target is reacquired
  searching / idle   nothing

Vision feeds every frame through on_frame() (offset and box size change
between state changes); on_decision() handles state changes; tick() fires the
pulses. PROVISIONAL timings (approved by Shravan Nov 14; HDB-Protocol to check).
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from . import haptics

FRONT, FRONT_LEFT, LEFT, FRONT_RIGHT, RIGHT = 0, 7, 6, 1, 2
LEFT_MOTOR, RIGHT_MOTOR = LEFT, RIGHT      # far-side motors (kept for imports)
SIDE_FAR = 0.55        # |offset| beyond this -> side motor 6/2, else front-diagonal 7/1
SLOW_S, FAST_S = 1.2, 0.18                 # pulse interval: cold .. hot
SIZE_SMALL, SIZE_BIG = 0.05, 0.5           # box size (share of frame) -> 0..1 closeness
LOG_EVERY_S = 1.0      # alert log: a repeat on the same motor at most this often
REPEAT_S = SLOW_S      # (old name)


def motor_for(state: str, offset: Optional[float]) -> int:
    """The motor that points at the object; offset -1 (left edge) .. +1 (right)."""
    if state == "centered":
        return FRONT
    if offset is None:                       # no live offset: the side the state names
        return LEFT if state == "left" else RIGHT if state == "right" else FRONT
    far = abs(offset) > SIDE_FAR
    if offset < 0:
        return LEFT if far else FRONT_LEFT
    return RIGHT if far else FRONT_RIGHT


def heat(state: str, offset: Optional[float], size: float) -> float:
    """0 (cold: far off-axis, small) .. 1 (hot: centered and big)."""
    centering = 1.0 if state == "centered" else 1.0 - min(1.0, abs(offset or 1.0))
    closeness = min(1.0, max(0.0, (size - SIZE_SMALL) / (SIZE_BIG - SIZE_SMALL)))
    return 0.6 * centering + 0.4 * closeness


def interval_for(state: str, offset: Optional[float], size: float) -> float:
    """Seconds between pulses: SLOW_S cold .. FAST_S hot."""
    return SLOW_S - (SLOW_S - FAST_S) * heat(state, offset, size)


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

    # ---- inputs ----
    def on_frame(self, d, now: Optional[float] = None) -> None:
        """Every vision frame: keeps offset and box size current between state changes."""
        if d.state != self.state:
            self.on_decision(d, now)
            return
        with self._lock:
            self._decision = d

    def on_decision(self, d, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if d.state == self.state:
                self._decision = d
                return
            self.state, self._next = d.state, None
            self._decision = d
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
        with self._lock:
            if (not self.paused and self.state in ("left", "right", "centered")
                    and self._next is not None and now >= self._next):
                self._fire(now, repeat=True)

    def pause(self) -> None:
        self.paused = True

    def resume(self, now: Optional[float] = None) -> None:
        with self._lock:
            self.paused = False
            self._next = time.monotonic() if now is None else now  # pulses restart right away

    # ---- internals ----
    def _size(self) -> float:
        return float(getattr(self._decision, "size", 0.0) or 0.0)

    def _offset(self) -> Optional[float]:
        return getattr(self._decision, "offset", None)

    def _interval(self) -> float:
        return interval_for(self.state, self._offset(), self._size())

    def _fire(self, now: float, repeat: bool) -> None:
        iv = self._interval()
        m = motor_for(self.state, self._offset())
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
