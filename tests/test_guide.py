"""ObjectGuide hot/cold: smoothing + hysteresis, relative closeness, FOUND, dropouts, safety."""
import random
from types import SimpleNamespace as D

import pytest

from app import guide as G
from app import haptics
from app.guide import ObjectGuide

BOX = (0, 0, 1, 1)


def dec(state, offset=None, size=0.0, **kw):
    return D(state=state, offset=offset, size=size, **kw)


class Run:
    """Feeds vision frames at `fps` and ticks the guide every 20 ms like the live thread."""

    def __init__(self, **kw):
        self.plays, self.t = [], 0.0
        self.g = ObjectGuide(lambda c: self.plays.append((round(self.t, 3), c)), **kw)

    def go(self, seconds, state, offset, size, fps=15, box=True, noise=0.0, size_noise=0.0, seed=0, grow=None):
        rnd = random.Random(seed)
        n = int(seconds * fps)
        step = 1.0 / fps
        for k in range(n):
            t0 = self.t
            sz = size if grow is None else size + (grow - size) * (k + 1) / n
            off = None if offset is None else offset + rnd.gauss(0, noise)
            d = D(state=state, offset=off, size=max(sz + rnd.gauss(0, size_noise), 0.001),
                  bbox=BOX if box else None, t=t0, confidence=0.9, target="cup")
            self.g.on_frame(d, now=t0)
            for j in range(int(step / 0.02)):
                self.t = t0 + (j + 1) * 0.02
                self.g.tick(now=self.t)
            self.t = t0 + step

    def pulses(self, since=0.0):
        return [(t, c[0][1], c[0][2]) for t, c in self.plays if len(c) == 1 and t >= since]

    def kinds(self):
        return [("found" if c == haptics.found() else "centered" if c == haptics.centered()
                 else "lost" if c == haptics.lost() else "pulse") for _, c in self.plays]


# ---------------------------------------------------------------- mapping and rate
def test_motor_points_where_the_object_is_and_the_state_decides_the_side():
    assert G.motor_for("centered", 0.05) == 0
    assert G.motor_for("left", -0.3) == 7 and G.motor_for("right", 0.3) == 1
    assert G.motor_for("left", -0.8) == 6 and G.motor_for("right", 0.8) == 2
    assert G.motor_for("left", None) == 6 and G.motor_for("right", None) == 2
    assert G.motor_for("right", -0.8) == 2                       # a noisy sign never beats the tracker's side
    assert G.motor_for("left", -0.3, far=True) == 6 and G.motor_for("left", -0.8, far=False) == 7


def test_interval_falls_as_you_get_hotter():
    cold = G.interval_for("right", 0.9, 0.0)
    mid = G.interval_for("centered", 0.0, 0.0)
    hot = G.interval_for("centered", 0.0, 1.0)
    assert cold > mid > hot and cold == pytest.approx(G.SLOW_S, abs=0.4)
    assert 0.4 <= mid <= 0.6 and hot == pytest.approx(G.FAST_S, abs=0.02)
    assert G.interval_for("right", 0.3, 0.5) < G.interval_for("right", 0.8, 0.5)
    assert G.interval_for("right", 0.5, 0.9) < G.interval_for("right", 0.5, 0.1)


def test_pulses_shorten_at_high_rates():
    assert G.pulse_ms(1.0) == haptics.DIRECTIONAL_MS and G.pulse_ms(0.5) == 120 and G.pulse_ms(0.18) == 80


# ------------------------------------------------- (2) closeness is relative to the object
def test_closeness_is_relative_to_the_starting_size_so_any_object_heats_up():
    assert G.closeness_from(0.06, 0.06) == 0.0
    assert G.closeness_from(0.132, 0.06) == pytest.approx(1.0)            # 2.2x a phone-sized start
    assert G.closeness_from(0.55, 0.25) == pytest.approx(1.0)             # 2.2x a chair-sized start
    assert G.closeness_from(0.6, 0.45) == 1.0                              # filling the frame is always close
    assert G.closeness_from(0.1, None) == 0.0 and G.closeness_from(0, 0.1) == 0.0


@pytest.mark.parametrize("start", [0.05, 0.12, 0.30])
def test_approaching_speeds_up_the_pulses_for_a_small_and_a_big_object(start):
    r = Run()
    r.go(1.2, "centered", 0.0, start)                                      # tracking starts: this is "far"
    far_iv = [b[0] - a[0] for a, b in zip(r.pulses(0.5), r.pulses(0.5)[1:])]
    n0 = len(r.plays)
    r.go(3.0, "centered", 0.0, start, grow=start * 2.4)                     # walk toward it
    late = [b[0] - a[0] for a, b in zip(r.pulses(r.t - 1.0), r.pulses(r.t - 1.0)[1:])]
    assert far_iv and late and sum(far_iv) / len(far_iv) > 0.4
    assert sum(late) / len(late) < 0.5 * (sum(far_iv) / len(far_iv))        # about twice as fast, any object
    assert len(r.plays) > n0


def test_interval_vs_distance_trace_is_the_same_curve_for_a_phone_and_a_chair():
    def curve(start):
        g = ObjectGuide(lambda c: None)
        out = []
        for k in range(40):                                                 # 40 frames closing from 1x to 2.4x
            size = start * (1 + 1.4 * k / 39)
            g.on_frame(D(state="centered", offset=0.0, size=size, bbox=BOX, t=k / 15.0, target="x"), now=k / 15.0)
            out.append(round(g._interval(), 2))
        return out
    phone, chair = curve(0.05), curve(0.25)
    assert phone[-1] == chair[-1] and phone[0] == chair[0] and phone[0] > 0.5 > phone[-1] - 0.0 + 0.0
    assert max(abs(a - b) for a, b in zip(phone, chair)) <= 0.05            # same curve, whatever the object
    assert phone[-1] < 0.25 and all(a >= b - 0.02 for a, b in zip(phone, phone[1:]))   # monotone: closer = faster


# ---------------------------------------- (1) steady motor: no flicker from noise
def test_no_motor_flips_from_noise_on_a_steady_offset_near_the_threshold():
    r = Run()
    r.go(12.0, "right", G.SIDE_FAR, 0.15, fps=15, noise=0.07, size_noise=0.01, seed=3)   # sits ON the boundary
    motors = [m for _, m, _ in r.pulses()]
    flips = sum(1 for a, b in zip(motors, motors[1:]) if a != b)
    assert len(motors) > 8 and flips == 0, (flips, set(motors))
    # the old stateless rule on the same noisy offsets flickers
    rnd = random.Random(3)
    raw = [G.motor_for("right", G.SIDE_FAR + rnd.gauss(0, 0.07)) for _ in range(180)]
    assert sum(1 for a, b in zip(raw, raw[1:]) if a != b) > 20


def test_hysteresis_switches_near_to_far_only_after_crossing_by_the_margin():
    r = Run()
    seq = []
    for off in (0.40, 0.50, 0.60, 0.65, 0.72, 0.60, 0.52, 0.48, 0.40):     # slow drift out and back
        r.go(1.5, "right", off, 0.15, fps=15)
        seq.append(r.pulses(r.t - 1.0)[-1][1])
    assert seq == [1, 1, 1, 2, 2, 2, 2, 2, 1]                             # 1->2 only past 0.63, back only below 0.47


def test_one_noisy_frame_does_not_flip_the_direction():
    r = Run()
    r.go(2.0, "right", 0.7, 0.15)
    r.g.on_frame(D(state="right", offset=-0.9, size=0.15, bbox=BOX, t=r.t, target="x"), now=r.t)   # one wild frame
    r.go(0.6, "right", 0.7, 0.15)
    assert {m for _, m, _ in r.pulses()} == {2}                          # right side motor throughout


# --------------------------------------------------------------- (3) FOUND
def test_found_plays_once_after_a_second_centered_and_close_then_goes_quiet():
    r = Run()
    r.go(1.5, "centered", 0.0, 0.10)                                       # starting size
    r.go(2.0, "centered", 0.0, 0.10, grow=0.26)                            # approach until close
    r.go(1.6, "centered", 0.0, 0.26)                                       # hold centered and close
    assert r.kinds().count("found") == 1
    t_found = next(t for t, c in r.plays if c == haptics.found())
    n = len(r.plays)
    r.go(4.0, "centered", 0.0, 0.26)                                       # stays there: nothing more
    assert len(r.plays) == n and r.t - t_found > 4
    # it moves away: pulses come back, and getting close again can find it again
    r.go(1.5, "right", 0.6, 0.12)
    assert len(r.plays) > n
    r.go(1.0, "centered", 0.0, 0.12)
    r.go(2.0, "centered", 0.0, 0.12, grow=0.30)
    r.go(1.6, "centered", 0.0, 0.30)
    assert r.kinds().count("found") == 2


def test_found_needs_the_full_second_and_closeness_and_centering():
    r = Run()
    r.go(1.2, "centered", 0.0, 0.10)
    r.go(0.6, "centered", 0.0, 0.10, grow=0.26)
    r.go(0.7, "centered", 0.0, 0.26)                                       # close for < 1 s only
    assert "found" not in r.kinds()
    r.go(0.8, "centered", 0.0, 0.10)                                       # backs off
    assert "found" not in r.kinds()
    r2 = Run()
    r2.go(1.2, "right", 0.5, 0.10)
    r2.go(3.0, "right", 0.5, 0.10, grow=0.30)                              # close but not centered
    assert "found" not in r2.kinds()


def test_found_pattern_respects_the_safety_limits():
    steps = haptics.found()
    assert all(0 < ms <= 1000 for _, _, ms in steps) and len(steps) >= 4
    for off, _, _ in steps:
        assert sum(1 for o, _, d in steps if o <= off < o + d) <= 2          # at most 2 motors at once
    assert steps != haptics.centered() and steps != haptics.lost()
    four = haptics.adapt(steps, (0, 2, 4, 6))
    assert {m for _, m, _ in four} == {0, 2, 6}


# ----------------------------------------------------------- (4) tracking loss
def test_short_dropout_keeps_the_last_direction_and_does_not_say_lost():
    r = Run()
    r.go(2.0, "left", -0.8, 0.12)
    before = r.pulses()
    r.go(0.6, "left", None, 0.12, box=False)                               # detector gap < 0.7 s
    r.go(1.0, "left", -0.8, 0.12)
    assert "lost" not in r.kinds()
    gap = [m for t, m, _ in r.pulses() if 2.0 <= t <= 2.6]
    assert gap and set(gap) == {6}                                         # still pulsing the last direction
    assert len(r.pulses()) > len(before)


def test_lost_waits_out_a_short_dropout_but_fires_after_a_real_loss():
    r = Run()
    r.go(1.5, "right", 0.8, 0.12)
    t_box = r.t - 1 / 15
    r.g.on_decision(D(state="lost", offset=None, size=0.0, bbox=None, t=t_box + 0.3, target="x"), now=r.t)
    r.go(0.2, "right", 0.8, 0.12)                                          # a box is back before the grace ends
    assert "lost" not in r.kinds() and r.g.state == "right"
    # a real loss: the tracker says lost and no box comes back
    r2 = Run()
    r2.go(1.5, "right", 0.8, 0.12)
    t_box = r2.t - 1 / 15
    r2.g.on_decision(D(state="lost", offset=None, size=0.0, bbox=None, t=t_box + 0.3, target="x"), now=r2.t)
    assert "lost" not in r2.kinds()
    r2.go(1.0, "lost", None, 0.0, box=False)                               # the tracker keeps saying lost
    assert r2.kinds().count("lost") == 1 and r2.g.state == "lost"
    n = len(r2.plays)
    r2.go(2.0, "lost", None, 0.0, box=False)
    assert len(r2.plays) == n                                              # silent after lost, until reacquired
    r2.go(0.5, "right", 0.8, 0.12)
    assert len(r2.plays) > n                                               # a box is back: guiding resumes


# ----------------------------------------------------------------- (5) existing behaviour
def run_events(events):
    played = []
    g = ObjectGuide(played.append)
    for t, kind, d in sorted(events, key=lambda e: e[0]):
        {"d": lambda: g.on_decision(d, now=t), "f": lambda: g.on_frame(d, now=t),
         "t": lambda: g.tick(now=t)}[kind]()
    return played


def test_lost_fires_once_then_silent_until_reacquired():
    ev = [(0, "d", dec("right", 0.9)), (0.5, "d", dec("lost")), (3.0, "d", dec("right", 0.9))]
    ev += [(0.1 * k, "t", None) for k in range(1, 45)]
    p = run_events(ev)
    assert haptics.lost() in p
    assert p[p.index(haptics.lost()) + 1] == haptics.directional(2)
    assert sum(1 for c in p if c == haptics.lost()) == 1


def test_same_state_is_ignored_but_frames_update_the_track():
    assert run_events([(0, "d", dec("centered")), (0.1, "d", dec("centered"))]) == [haptics.centered()]


def test_pause_suppresses_pulses_and_found_and_resume_restarts():
    r = Run()
    r.go(1.2, "centered", 0.0, 0.10)
    r.g.pause()
    n = len(r.plays)
    r.go(2.0, "centered", 0.0, 0.10, grow=0.3)
    r.go(2.0, "centered", 0.0, 0.3)
    assert len(r.plays) == n                                               # nothing while Action is held
    r.g.resume(now=r.t)
    r.g.tick(now=r.t + 0.01)
    assert len(r.plays) > n
    assert all(len(c) <= 5 and all(0 < ms <= 1000 for _, _, ms in c) for _, c in r.plays)


def test_avoid_stop_still_wins_over_a_pulse():
    r = Run()
    r.g.avoid = D(override=lambda now=None: D(action="stop", motor=None))
    r.go(2.0, "right", 0.6, 0.12)
    assert r.pulses() == []                                                # the crash warning owns the band
