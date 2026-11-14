"""Score the name-direction estimator on simulated and recorded MIC traces.

Backs the H15 gate: a called name gives the right side or a generic alert
4 out of 5 times. Wrong directions are the failure that matters.

  python -m tools.eval_estimator                      # 8 directions x all conditions, 10 trials each
  python -m tools.eval_estimator --trials 30 -v       # more trials, per-direction breakdown
  python -m tools.eval_estimator --conditions clean,noisy
  python -m tools.eval_estimator recordings/*.json    # also score recorded traces that carry "expected"
  python -m tools.eval_estimator --save /tmp/sim      # write the simulated traces as replayable files
  python -m tools.eval_estimator --gate               # exit 1 if any condition misses the gate

Each result is one of:
  correct  the expected motor
  near     one motor off (45 degrees). This still counts as right under the
           coarse fallback (front/back/left/right).
  wrong    any other motor. A wrong buzz is worse than a generic one.
  generic  no direction, so the wearer gets the generic attention cue
gate = (correct + generic) / n >= 0.80, and wrong <= 10%.
For the "diffuse" control (no real direction) the right answer is generic.

Recorded traces: a trace file can hold {"expected": {"motor": k}}, which applies
to all of its alerts, or a per-alert {"startMs":..,"endMs":..,"expected":k}.
Use null for "should be generic". Traces from the dashboard have no labels.
Add one by hand, or pass --expect k when scoring a single file.

Simulation model (tools.eval_estimator.simulate). Mics face front, right,
back and left. A talker at angle theta reaches mic i with head-shadow gain
((1 + cos(theta - phi_i)) / 2) ** SHADOW, blended with a diffuse (reverb)
share that reaches every mic equally. Each channel also gets an unknown gain
error, a noise floor, a syllable-like speech envelope and, if the condition
asks for it, motor bleed and a clock error on the alert window. These
numbers are guesses until real H27 recordings replace them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app import direction, trace

FRAME_MS = 50
TRACE_MS = 5000
NAME_START, NAME_LEN = 2000, 450         # name spoken 2.00-2.45 s
PAD_MS = 100                             # LiveEstimator pads the speech window by this
MIC_ANGLES = (0, 90, 180, 270)           # front, right, back, left
SHADOW = 1.5
DIRECTIONS = {k: k * 45 for k in range(8)}   # motor -> angle, clockwise from front
NAMES = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
GATE_OK, GATE_WRONG = 0.80, 0.10


@dataclass(frozen=True)
class Condition:
    name: str
    floor: float = 20.0          # quiet level per channel (before gain error)
    floor_jitter: float = 2.0    # +- uniform noise on every frame
    speech: tuple = (40, 80)     # speech excess at the facing mic, min/max
    diffuse: float = 0.15        # share of speech reaching all mics equally (reverb)
    gain_err: float = 0.15       # each channel's gain is off by up to +-15 %
    babble: float = 0.0          # background talkers: random per-channel bursts (x floor)
    motor_bleed: float = 0.0     # chance a frame gets a motor-noise spike on one channel
    clock_err_ms: int = 0        # alert window misaligned by up to +- this
    sources: int = 1             # 0 = fully diffuse control (expected generic)


CONDITIONS = {c.name: c for c in [
    Condition("clean"),
    Condition("noisy", floor=35, floor_jitter=6, speech=(25, 50), babble=0.6),
    Condition("reverb", diffuse=0.45),
    Condition("motor", motor_bleed=0.25),
    Condition("clock", clock_err_ms=150),
    Condition("hard", floor=35, floor_jitter=6, speech=(25, 50), babble=0.6, diffuse=0.35,
              gain_err=0.3, motor_bleed=0.15, clock_err_ms=100),
    Condition("diffuse", diffuse=1.0, sources=0),
]}


def mic_gains(theta_deg: float, diffuse: float) -> list:
    out = []
    for phi in MIC_ANGLES:
        shadow = ((1 + math.cos(math.radians(theta_deg - phi))) / 2) ** SHADOW
        out.append(diffuse + (1 - diffuse) * shadow)
    return out


def simulate(motor: Optional[int], cond: Condition, seed: int) -> dict:
    """One trace: a name called from motor's direction (None = diffuse)."""
    rnd = random.Random(seed)
    theta = DIRECTIONS[motor] + rnd.uniform(-10, 10) if motor is not None else 0.0
    gains = mic_gains(theta, 1.0 if cond.sources == 0 else cond.diffuse)
    err = [1 + rnd.uniform(-cond.gain_err, cond.gain_err) for _ in MIC_ANGLES]
    amp = rnd.uniform(*cond.speech)
    frames = []
    for i in range(TRACE_MS // FRAME_MS):
        ms = i * FRAME_MS
        lv = [cond.floor + rnd.uniform(-cond.floor_jitter, cond.floor_jitter) for _ in MIC_ANGLES]
        if cond.babble and rnd.random() < 0.3:
            ch = rnd.randrange(4)
            lv[ch] += rnd.uniform(0, cond.babble) * cond.floor
        t = ms - NAME_START
        if 0 <= t <= NAME_LEN:
            env = 0.35 + 0.65 * abs(math.sin(math.pi * t / 150))   # ~3 syllables
            env *= rnd.uniform(0.8, 1.2)
            if cond.sources == 0:            # diffuse: each mic fluctuates on its own
                lv = [v + amp * env * rnd.uniform(0.3, 1.0) for v in lv]
            else:
                lv = [v + amp * env * g for v, g in zip(lv, gains)]
        if cond.motor_bleed and rnd.random() < cond.motor_bleed:
            lv[rnd.randrange(4)] += rnd.uniform(0.5, 2.0) * cond.floor
        frames.append({"ms": ms, "levels": [round(max(v * e, 0), 1) for v, e in zip(lv, err)]})
    skew = rnd.randint(-cond.clock_err_ms, cond.clock_err_ms) if cond.clock_err_ms else 0
    start = NAME_START - PAD_MS + skew
    end = NAME_START + NAME_LEN + PAD_MS + skew
    return {"frames": frames,
            "alerts": [{"startMs": start, "endMs": end, "expected": motor}],
            "expected": {"motor": motor},
            "sim": {"condition": cond.name, "seed": seed, "angle": round(theta, 1)}}


def classify(got: Optional[int], expected: Optional[int]) -> str:
    if got is None:
        return "generic"
    if expected is None:
        return "wrong"               # buzzed a side when there was no direction
    d = min((got - expected) % 8, (expected - got) % 8)
    return "correct" if d == 0 else "near" if d == 1 else "wrong"


class Tally:
    KEYS = ("correct", "near", "wrong", "generic")

    def __init__(self):
        self.c = dict.fromkeys(self.KEYS, 0)

    def add(self, kind: str) -> None:
        self.c[kind] += 1

    @property
    def n(self) -> int:
        return sum(self.c.values())

    def score(self) -> float:
        return (self.c["correct"] + self.c["generic"]) / self.n if self.n else 0.0

    def wrong_rate(self) -> float:
        return self.c["wrong"] / self.n if self.n else 0.0

    def passes(self) -> bool:
        return self.score() >= GATE_OK and self.wrong_rate() <= GATE_WRONG

    def row(self, label: str) -> str:
        c = self.c
        verdict = "PASS" if self.passes() else "FAIL"
        return (f"{label:14s} {self.n:4d} {c['correct']:8d} {c['near']:5d} {c['wrong']:6d} "
                f"{c['generic']:8d}   {100 * self.score():5.1f}%  {verdict}")


HEADER = f"{'condition':14s} {'n':>4s} {'correct':>8s} {'near':>5s} {'wrong':>6s} {'generic':>8s}   {'gate':>6s}"


def run_trace(t: dict, expect_override=...) -> list:
    """[(alert, expected, Estimate, kind)] for every labelled alert in a trace."""
    out = []
    file_exp = (t.get("expected") or {}).get("motor", ...) if isinstance(t.get("expected"), dict) else ...
    for a in t["alerts"]:
        exp = expect_override if expect_override is not ... else a.get("expected", file_exp)
        if exp is ...:
            continue                 # unlabelled
        e = direction.estimate(t["frames"], a["startMs"], a["endMs"])
        out.append((a, exp, e, classify(e.motor, exp)))
    return out


def evaluate(conditions, trials: int, seed: int = 0, save: Optional[Path] = None):
    """{condition: (Tally, {motor: Tally})}"""
    results = {}
    for name in conditions:
        cond = CONDITIONS[name]
        ci = zlib.crc32(name.encode()) % 10_007   # same trials whichever conditions are picked
        motors = [None] if cond.sources == 0 else list(DIRECTIONS)
        total, per = Tally(), {}
        for m in motors:
            per[m] = Tally()
            for k in range(trials * (8 if m is None else 1)):
                s = seed * 1_000_003 + ci * 10_007 + (m if m is not None else 9) * 1009 + k
                t = simulate(m, cond, s)
                [(_, _, _, kind)] = run_trace(t)
                total.add(kind)
                per[m].add(kind)
                if save:
                    trace.save(t, save / f"{name}_{'none' if m is None else NAMES[m]}_{k:02d}.json")
        results[name] = (total, per)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", help="recorded trace files to score (need labels)")
    ap.add_argument("--trials", type=int, default=10, help="simulated trials per direction")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help=f"comma list from: {', '.join(CONDITIONS)}; 'none' to skip simulation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect", type=lambda s: None if s in ("none", "generic") else int(s),
                    default=..., help="label for every alert in the given traces (motor or 'generic')")
    ap.add_argument("--save", type=Path, help="write simulated traces here (replayable)")
    ap.add_argument("--gate", action="store_true", help="exit 1 if any row fails the gate")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-direction rows")
    a = ap.parse_args(argv)

    failed = False
    conds = [] if a.conditions == "none" else [c.strip() for c in a.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown condition(s) {bad}")
    if a.save:
        a.save.mkdir(parents=True, exist_ok=True)
    if conds:
        print(f"Simulated: {a.trials} trials per direction (diffuse: {8 * a.trials}), seed {a.seed}")
        print(HEADER)
        overall = Tally()
        for name, (total, per) in evaluate(conds, a.trials, a.seed, a.save).items():
            print(total.row(name))
            failed |= not total.passes()
            for k in Tally.KEYS:
                overall.c[k] += total.c[k]
            if a.verbose:
                for m, t in per.items():
                    print(t.row(f"  {'(none)' if m is None else NAMES[m]}"))
        print(overall.row("ALL"))
        if a.save:
            print(f"saved simulated traces to {a.save}")

    if a.traces:
        print(f"\nRecorded traces:")
        tally = Tally()
        for p in a.traces:
            rows = run_trace(trace.load(p), a.expect)
            if not rows:
                print(f"  {p}: no labelled alerts (add \"expected\" or pass --expect)")
            for al, exp, e, kind in rows:
                tally.add(kind)
                got = "generic" if e.motor is None else NAMES[e.motor]
                want = "generic" if exp is None else NAMES[exp]
                print(f"  {Path(p).name} {al['startMs']}-{al['endMs']} ms: expected {want}, got {got} "
                      f"[{kind}] ({e.reason}, agreement {e.consistency:.2f}, {e.frames_used} frames)")
        if tally.n:
            print(HEADER)
            print(tally.row("recorded"))
            failed |= not tally.passes()
    return 1 if (a.gate and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
