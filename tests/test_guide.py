from types import SimpleNamespace as D

import pytest

from app import guide as G
from app import haptics
from app.guide import ObjectGuide


def dec(state, offset=None, size=0.0, **kw):
    return D(state=state, offset=offset, size=size, **kw)


def test_motor_points_where_the_object_is():
    assert G.motor_for("centered", 0.05) == 0
    assert G.motor_for("left", -0.3) == 7 and G.motor_for("right", 0.3) == 1
    assert G.motor_for("left", -0.8) == 6 and G.motor_for("right", 0.8) == 2
    assert G.motor_for("left", None) == 6 and G.motor_for("right", None) == 2   # no live offset


def test_interval_falls_as_you_get_hotter():
    cold = G.interval_for("right", 0.9, 0.05)            # far off-axis, small
    mid = G.interval_for("centered", 0.0, 0.05)          # centered but small
    hot = G.interval_for("centered", 0.0, 0.5)           # centered and big
    assert cold == pytest.approx(G.SLOW_S - 0.0, abs=0.35) and cold > mid > hot
    assert 0.4 <= mid <= 0.6 and hot == pytest.approx(G.FAST_S, abs=0.02)
    assert G.interval_for("right", 0.3, 0.3) < G.interval_for("right", 0.8, 0.3)    # closer to center
    assert G.interval_for("right", 0.5, 0.4) < G.interval_for("right", 0.5, 0.1)    # bigger box


def test_pulses_shorten_at_high_rates():
    assert G.pulse_ms(1.0) == haptics.DIRECTIONAL_MS and G.pulse_ms(0.5) == 120 and G.pulse_ms(0.18) == 80


def run(events):
    """events: (t, 'd'|'f'|'t', decision)"""
    played = []
    g = ObjectGuide(played.append)
    for t, kind, d in sorted(events, key=lambda e: e[0]):
        {"d": lambda: g.on_decision(d, now=t), "f": lambda: g.on_frame(d, now=t),
         "t": lambda: g.tick(now=t)}[kind]()
    return played


def motors(played):
    return [c[0][1] if len(c) == 1 else "two" for c in played]


def test_centered_keeps_pulsing_front_and_speeds_up_with_size():
    ev = [(0, "d", dec("centered", 0.0, 0.05))] + [(0.1 * k, "t", None) for k in range(1, 41)]
    ev += [(2.0, "f", dec("centered", 0.0, 0.5))]       # the box grows: hotter
    p = run(ev)
    assert p[0] == haptics.centered() and all(c == haptics.directional(0)[:1] or len(c) == 1 for c in p[1:])
    assert set(motors(p[1:])) == {0}
    slow = [c for c in p[1:] if c[0][2] >= 120]
    fast = [c for c in p[1:] if c[0][2] == 80]
    assert len(slow) >= 3 and len(fast) >= 8             # ~2 pulses/s small, ~5/s big
    assert len(p) - 1 >= 3 * 4                            # >= 3 pulses/s once held centered and big


def test_left_points_front_left_then_side_as_offset_changes():
    ev = [(0, "d", dec("left", -0.3, 0.2)), (0.5, "f", dec("left", -0.8, 0.2))]
    ev += [(0.05 * k, "t", None) for k in range(1, 80)]
    p = run(ev)
    assert motors(p)[0] == 7 and 6 in motors(p) and set(motors(p)) <= {6, 7}


def test_lost_fires_once_then_silent_until_reacquired():
    ev = [(0, "d", dec("right", 0.9)), (0.5, "d", dec("lost")), (3.0, "d", dec("right", 0.9))]
    ev += [(0.1 * k, "t", None) for k in range(1, 45)]
    p = run(ev)
    assert haptics.lost() in p
    after_lost = p[p.index(haptics.lost()) + 1:]
    assert after_lost[0] == haptics.directional(2)       # nothing between lost and reacquire
    assert sum(1 for c in p if c == haptics.lost()) == 1


def test_same_state_is_ignored_but_frames_update_the_track():
    p = run([(0, "d", dec("centered")), (0.1, "d", dec("centered"))])
    assert p == [haptics.centered()]


def test_pause_suppresses_cues():
    played = []
    g = ObjectGuide(played.append)
    g.pause()
    g.on_decision(dec("left", -0.5), now=0)
    g.tick(now=5)
    assert played == []
    g.resume(now=5.0)
    g.tick(now=5.1)
    assert played, "resume restarts the pulses"
