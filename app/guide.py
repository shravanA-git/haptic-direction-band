"""Find-an-object cue policy: vision decisions -> haptic cues.

  left / right  directional pulse (motor 6 / motor 2) now, then every REPEAT_S
  centered      two front pulses once, on entering
  lost          lost cue once, then silence until the target is reacquired
  searching / idle   nothing

Vision calls on_decision() only on state changes; tick() supplies the repeats.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from . import haptics

LEFT_MOTOR, RIGHT_MOTOR = 6, 2
REPEAT_S = 0.7


class ObjectGuide:
    def __init__(self, play: Callable[[list], object], repeat_s: float = REPEAT_S,
                 on_cue: Optional[Callable[[str, Optional[int], dict], None]] = None):
        self.play = play            # play(cue), e.g. lambda c: haptics.play(c, link.pulse)
        self.repeat_s = repeat_s
        # on_cue(kind, motor, info): every cue played, for the alert log.
        # info: confidence, target, latency_ms (decision frame -> cue), repeat.
        self.on_cue = on_cue
        self._decision = None
        self.state = "idle"
        self.paused = False
        self._next: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Optional app.avoid.AvoidLoop: before each directional cue, steer around an
        # obstacle (veer/turn motor) or give the crash warning instead.
        self.avoid = None

    def on_decision(self, d, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if d.state == self.state:
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
            elif d.state == "lost":
                self.play(haptics.lost())
                self._report("lost", None, now, False)

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if (not self.paused and self.state in ("left", "right")
                    and self._next is not None and now >= self._next):
                self._fire(now, repeat=True)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self._next = time.monotonic()  # resume left/right repeats right away

    def _fire(self, now: float, repeat: bool) -> None:
        m = LEFT_MOTOR if self.state == "left" else RIGHT_MOTOR
        ov = self.avoid.override(now) if self.avoid is not None else None
        self._next = now + self.repeat_s
        if ov is not None and ov.action == "stop":
            return                      # AvoidLoop is playing the crash warning
        if ov is not None and ov.motor is not None:
            m = ov.motor                # steer around the obstacle toward the target
        self.play(haptics.directional(m))
        self._report("directional", m, now, repeat)

    def _report(self, kind: str, motor: Optional[int], now: float, repeat: bool) -> None:
        if not self.on_cue:
            return
        d = self._decision
        t = getattr(d, "t", None)
        info = {"confidence": getattr(d, "confidence", None), "target": getattr(d, "target", None),
                "state": self.state, "repeat": repeat,
                "latency_ms": (now - t) * 1000 if (t is not None and not repeat) else None}
        try:
            self.on_cue(kind, motor, info)
        except Exception as e:
            print(f"[guide] on_cue error: {e}")

    def start(self) -> "ObjectGuide":
        def run():
            while not self._stop.wait(0.05):
                self.tick()
        threading.Thread(target=run, name="guide", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()
