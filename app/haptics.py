"""Haptic cue patterns, as timed PULSE steps.

Settled by the handoff: directional 200 ms on that motor, generic attention
300 ms on front (0), test pulse 150 ms, centered = two front pulses, lost =
one distinct cue. The two-pulse gap and the lost pattern are NOT settled yet
(HDB-Protocol proposal pending); the values marked PROVISIONAL are
placeholders kept in one place so they are a one-line change.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from .link import MAX_PULSE_MS

FRONT = 0
MOTOR_NAMES = ["front", "front-right", "right", "back-right",
               "back", "back-left", "left", "front-left"]

DIRECTIONAL_MS = 200
GENERIC_MS = 300
TEST_MS = 150
CENTERED_PULSE_MS = 100   # PROVISIONAL
CENTERED_GAP_MS = 100     # PROVISIONAL (HDB-Protocol suggestion, not agreed)
LOST_STEPS = [(0, FRONT, 600)]  # PROVISIONAL (HDB-Protocol suggestion, not agreed)
# Teach a Sound: a sweep around the cardinals, so it can't be read as a side.
SOUND_STEPS = [(0, 0, 80), (120, 2, 80), (240, 4, 80), (360, 6, 80)]  # PROVISIONAL
# One pattern per sound category (app.listen.categorize), so the wearer can
# tell taught sounds apart. Every pattern is symmetric (opposite motors
# together) or a full circle, so none reads as a direction; at most two motors
# at once; all end before notify.SOUND_DIR_OFFSET_MS (the pulse toward the
# sound). ALL PROVISIONAL (not yet reviewed by HDB-Protocol).
_FB, _LR = (0, 4), (2, 6)            # front+back, left+right


def _pair(off, motors, ms):
    return [(off, m, ms) for m in motors]


SOUND_PATTERNS = {
    # one long front+back buzz
    "tone": _pair(0, _FB, 300),
    # ding-dong: front+back, then left+right
    "two-tone": _pair(0, _FB, 150) + _pair(250, _LR, 150),
    # three short left+right beeps
    "repeating": _pair(0, _LR, 80) + _pair(160, _LR, 80) + _pair(320, _LR, 80),
    # a run once round all eight motors, clockwise from front
    "melody": [(i * 60, i, 60) for i in range(8)],
    # knock-knock on the two diagonals (an X)
    "percussive": _pair(0, (1, 5), 60) + _pair(180, (3, 7), 60),
    # fizz: front+back and left+right alternating fast
    "noise": [s for i in range(4) for s in _pair(i * 90, _FB if i % 2 == 0 else _LR, 60)],
    # two slow front+back pulses
    "voice": _pair(0, _FB, 150) + _pair(300, _FB, 150),
}
# Hold-to-talk feedback for a wearer who can't see the dashboard.
ACK_STEPS = [(0, 6, 120), (0, 2, 120)]                    # PROVISIONAL: "got it, searching"
UNKNOWN_STEPS = [(0, 4, 80), (160, 4, 80), (320, 4, 80)]  # PROVISIONAL: "can't look for that"

# A cue is a list of (offset_ms, motor, duration_ms).
Cue = list


def directional(motor: int) -> Cue:
    return [(0, int(motor) % 8, DIRECTIONAL_MS)]


def generic_attention() -> Cue:
    return [(0, FRONT, GENERIC_MS)]


def centered() -> Cue:
    return [(0, FRONT, CENTERED_PULSE_MS),
            (CENTERED_PULSE_MS + CENTERED_GAP_MS, FRONT, CENTERED_PULSE_MS)]


def lost() -> Cue:
    return list(LOST_STEPS)


def taught_sound(category: str = None) -> Cue:
    """The pattern for a taught sound's category; the shared sweep if the
    category is unknown or not given (e.g. a v1 sound with no signature)."""
    return list(SOUND_PATTERNS.get(category or "", SOUND_STEPS))


def target_ack() -> Cue:
    """Target set: both sides at once, not readable as a direction."""
    return list(ACK_STEPS)


def target_unknown() -> Cue:
    """Didn't understand / not an object we can look for."""
    return list(UNKNOWN_STEPS)


def test_pulse(motor: int) -> Cue:
    return [(0, int(motor) % 8, TEST_MS)]


def name_alert(motor) -> Cue:
    """Estimator result -> cue: a motor index, or None for generic attention."""
    return generic_attention() if motor is None else directional(motor)


# Intensity: scales a cue's pulse lengths *and* offsets, so multi-pulse
# patterns (centered, taught sound) keep their rhythm. Laptop-side only:
# the protocol has no duty-cycle field. Test pulses and bench sequences are
# never scaled, and STOP is a separate line that intensity never touches.
INTENSITY = {"low": 0.6, "medium": 1.0, "high": 1.5}
MIN_PULSE_MS = 60          # coin motors barely spin up below this


def scaled(cue: Cue, intensity: str = "medium") -> Cue:
    f = INTENSITY[intensity]
    if f == 1.0:
        return list(cue)
    return [(int(round(off * f)), m, max(MIN_PULSE_MS, min(MAX_PULSE_MS, int(round(ms * f)))))
            for off, m, ms in cue]


_epoch = 0
_epoch_lock = threading.Lock()


def cancel_all() -> None:
    """Drop every pending step of every cue/sequence in flight. Call this
    together with STOP so a half-played cue can't buzz again after STOP."""
    global _epoch
    with _epoch_lock:
        _epoch += 1


def epoch() -> int:
    """Changes whenever cancel_all() runs (lets timers notice a STOP)."""
    return _epoch


def play(cue: Cue, send: Callable[[int, int], object]) -> threading.Thread:
    """Send a cue's steps at their offsets on a background thread.
    `send(motor, ms)` is Link.pulse for the band, or a dry-run sink.
    cancel_all() stops any steps not yet sent.
    """
    born = _epoch

    def run():
        t0 = time.monotonic()
        for off, motor, ms in sorted(cue):
            d = t0 + off / 1000 - time.monotonic()
            if d > 0:
                time.sleep(d)
            if _epoch != born:
                return
            send(motor, ms)
    t = threading.Thread(target=run, name="cue", daemon=True)
    t.start()
    return t


# Bench sequences (dashboard buttons; same steps as tools/bench.py).
def walk(motors=range(8), ms: int = 250, every_ms: int = 700) -> Cue:
    """One pulse per motor, in order: check the physical mapping by feel."""
    return [(i * every_ms, int(m), ms) for i, m in enumerate(motors)]
