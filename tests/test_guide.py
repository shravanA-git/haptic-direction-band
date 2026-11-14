from types import SimpleNamespace as D

from app import haptics
from app.guide import ObjectGuide


def run(states_at, ticks):
    played = []
    g = ObjectGuide(played.append, repeat_s=0.7)
    events = sorted([(t, "d", s) for t, s in states_at] + [(t, "t", None) for t in ticks])
    for t, kind, s in events:
        g.on_decision(D(state=s), now=t) if kind == "d" else g.tick(now=t)
    return played


def test_left_repeats_until_centered_then_two_front_pulses():
    p = run([(0, "searching"), (1.0, "left"), (2.6, "centered")],
            [1.2, 1.7, 1.8, 2.5, 3.0, 4.0])
    assert p == [haptics.directional(6), haptics.directional(6),
                 haptics.directional(6), haptics.centered()]


def test_lost_fires_once_then_silent_until_reacquired():
    p = run([(0, "right"), (0.5, "lost"), (3.0, "right")], [0.6, 1.0, 2.0, 2.9, 3.8])
    assert p == [haptics.directional(2), haptics.lost(),
                 haptics.directional(2), haptics.directional(2)]


def test_repeated_same_state_is_ignored():
    assert run([(0, "centered"), (0.1, "centered")], [1, 2]) == [haptics.centered()]


def test_pause_suppresses_cues():
    played = []
    g = ObjectGuide(played.append)
    g.pause()
    g.on_decision(D(state="left"), now=0)
    g.tick(now=5)
    assert played == []
