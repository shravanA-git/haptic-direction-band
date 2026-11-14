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

FRONT = 0
MOTOR_NAMES = ["front", "front-right", "right", "back-right",
               "back", "back-left", "left", "front-left"]

DIRECTIONAL_MS = 200
GENERIC_MS = 300
TEST_MS = 150
CENTERED_PULSE_MS = 100   # PROVISIONAL
CENTERED_GAP_MS = 120     # PROVISIONAL
LOST_STEPS = [(0, 4, 400)]  # PROVISIONAL: one long back pulse

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


def test_pulse(motor: int) -> Cue:
    return [(0, int(motor) % 8, TEST_MS)]


def name_alert(motor) -> Cue:
    """Estimator result -> cue: a motor index, or None for generic attention."""
    return generic_attention() if motor is None else directional(motor)


def play(cue: Cue, send: Callable[[int, int], object]) -> threading.Thread:
    """Send a cue's steps at their offsets on a background thread.
    `send(motor, ms)` is Link.pulse for the band, or a dry-run sink.
    """
    def run():
        t0 = time.monotonic()
        for off, motor, ms in sorted(cue):
            d = t0 + off / 1000 - time.monotonic()
            if d > 0:
                time.sleep(d)
            send(motor, ms)
    t = threading.Thread(target=run, name="cue", daemon=True)
    t.start()
    return t
