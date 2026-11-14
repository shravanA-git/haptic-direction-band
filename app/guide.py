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
    def __init__(self, play: Callable[[list], object], repeat_s: float = REPEAT_S):
        self.play = play            # play(cue), e.g. lambda c: haptics.play(c, link.pulse)
        self.repeat_s = repeat_s
        self.state = "idle"
        self.paused = False
        self._next: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def on_decision(self, d, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if d.state == self.state:
                return
            self.state, self._next = d.state, None
            if self.paused:
                return
            if d.state in ("left", "right"):
                self._fire(now)
            elif d.state == "centered":
                self.play(haptics.centered())
            elif d.state == "lost":
                self.play(haptics.lost())

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if (not self.paused and self.state in ("left", "right")
                    and self._next is not None and now >= self._next):
                self._fire(now)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self._next = time.monotonic()  # resume left/right repeats right away

    def _fire(self, now: float) -> None:
        m = LEFT_MOTOR if self.state == "left" else RIGHT_MOTOR
        self.play(haptics.directional(m))
        self._next = now + self.repeat_s

    def start(self) -> "ObjectGuide":
        def run():
            while not self._stop.wait(0.05):
                self.tick()
        threading.Thread(target=run, name="guide", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()
