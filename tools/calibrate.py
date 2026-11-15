"""Mic calibration: claps at 8 (or 4) known positions around the band.

  python -m tools.calibrate                 # real band, 8 positions
  python -m tools.calibrate --cardinals     # front, right, back, left only
  python -m tools.calibrate --sim           # simulated band with mismatched mics
  python -m tools.calibrate --save          # write app/calibration.local.json without asking
  python -m tools.calibrate --reps 3        # 3 laps of 8 positions (24 claps), for learn_centroids

Steps:
  1. Quiet: stay silent for a few seconds; per-channel baseline and noise.
     Flags dead (constant) or clipping channels.
  2. For each position: stand ~1 m away in that direction, press Enter, clap
     once. The clap window is found automatically (loudness over baseline).
  3. Report: each clap's estimated direction before and after gain
     correction. Gains = how much less (or more) sensitive each mic is than
     the average, measured from claps spread evenly around the head.
  4. Save: recordings/calib-HHMMSS.json (labelled trace: replayable, and
     scored by tools/eval_estimator.py and tools/trial_metrics.py) and, if
     correction doesn't make things worse, app/calibration.local.json (gitignored: it
     describes this wearer's hat), which the app loads at start in preference to the
     shared app/calibration.json. A clap that is clipped, too loud for the hat (the mics flatten
     around 2400-2650) or has no direction (all four channels within 25%) is not counted: you are
     told to snap softer (aim for peaks 400-1500) and asked again, so every position gets the
     requested number of ACCEPTED claps. The hat's ceiling is learned and saved with the gains.
Close the Arduino Serial Monitor first: only one program can own the port.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from pathlib import Path
from statistics import median
from typing import Callable, Optional

from app import direction, trace
from app.direction import CALIBRATION_LOCAL_FILE
from app.link import Link, MicFrame, SimPort
from tools.bench import POS_NAMES, _yes, open_link
from tools.eval_estimator import classify, mic_gains

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"
CHANNELS = ("front", "right", "back", "left")
CLAP_MAX_PEAK = 2200     # a clap's loudest frame above this (hats flatten around 2400-2650) is too loud
CLAP_MIN_CONTRAST = 0.25  # (max - min) / max of the clap's excess over the quiet floor: below = no direction
TOO_LOUD = ("too loud / no direction: snap softer, aim for peaks 400-1500. "
            "This clap is not counted. Again:")
CLIP_ADC = 4090          # a channel at the 12-bit ADC ceiling (4095): the clap clipped, its direction is gone
CLAP_EXCESS = 1.0        # mean excess over baseline (baseline units) that counts as a clap
CLAP_MAX_MS = 400
GAIN_LIMITS = (0.4, 2.5)


# --------------------------------------------------------------- simulator
class ClapSimPort(SimPort):
    """Simulated band whose four mics differ: unequal noise floors (offset)
    and unequal sensitivities (gain), the case calibration has to fix.
    clap(position) adds a short, decaying clap from that direction."""

    FLOORS = (20.0, 26.0, 18.0, 23.0)
    SENS = (1.0, 0.55, 1.35, 0.9)

    def __init__(self, seed: int = 1, frame_ms: int = 20, diffuse: float = 0.15):
        super().__init__(frames=None, frame_ms=frame_ms)
        self.rnd = random.Random(seed)
        self.diffuse = diffuse
        self._clap: Optional[tuple] = None   # (start frame index, gains, amp)
        self.claps = 0

    def clap(self, position: int) -> None:
        theta = position * 45 + self.rnd.uniform(-8, 8)
        self._clap = (self._i + 1, mic_gains(theta, self.diffuse), self.rnd.uniform(90, 140))
        self.claps += 1

    def _next_frame(self):
        ms = self._i * self.frame_ms
        lv = [f + self.rnd.uniform(-1.5, 1.5) for f in self.FLOORS]
        if self._clap:
            i0, gains, amp = self._clap
            k = self._i - i0
            if 0 <= k * self.frame_ms < 160:
                env = math.exp(-k * self.frame_ms / 60)
                lv = [v + amp * env * g * s for v, g, s in zip(lv, gains, self.SENS)]
            elif k * self.frame_ms >= 160:
                self._clap = None
        return f"MIC {ms} " + " ".join(f"{max(v, 0):.0f}" for v in lv), self._t0 + ms / 1000


# ------------------------------------------------------------------ steps
class Collector:
    """Link listener keeping every MIC frame as trace dicts."""

    def __init__(self):
        self.frames: list = []
        self._lock = threading.Lock()

    def __call__(self, ev) -> None:
        if isinstance(ev, MicFrame):
            with self._lock:
                self.frames.append({"ms": ev.ms, "levels": list(ev.levels)})

    def since(self, n: int) -> list:
        with self._lock:
            return self.frames[n:]


def quiet_stats(frames: list) -> dict:
    chans = list(zip(*[f["levels"] for f in frames]))
    base = [max(median(c), 1e-6) for c in chans]
    noise = [median(abs(v - b) for v in c) for c, b in zip(chans, base)]
    flags = []
    for name, c in zip(CHANNELS, chans):
        if max(c) == min(c):
            flags.append(f"{name} mic is constant: unplugged or dead?")
        elif max(c) >= 4095:
            flags.append(f"{name} mic hits 4095: raw ADC or clipping?")
    return {"baseline": base, "noise": noise, "flags": flags}


def excess(levels, base) -> list:
    return [max(v / b - 1.0, 0.0) for v, b in zip(levels, base)]


def clap_peaks(frames: list, a: int, b: int) -> list:
    """Per-channel peak level (front, right, back, left) of the frames in [a - 40, b + 40] ms."""
    win = [f["levels"] for f in frames if a - 40 <= f["ms"] <= b + 40]
    return [max(c) for c in zip(*win)] if win else [0.0] * 4


def is_clipped(peaks: list) -> bool:
    return max(peaks) >= CLIP_ADC


def clap_frame(frames: list, a: int, b: int) -> Optional[list]:
    """The loudest frame (levels) of the clap window: a clap is one or two informative frames."""
    win = [f["levels"] for f in frames if a - 40 <= f["ms"] <= b + 40]
    return max(win, key=sum) if win else None


def learn_ceiling(frames: list) -> Optional[float]:
    """The level at which this hat's four mics flatten against the rails: the median top level of
    the frames that went flat (all channels within 15%, above 2200) but did not hit the ADC max.
    None until at least 2 such frames have been seen."""
    flat = [max(f["levels"]) for f in frames
            if direction.FLAT_LEVEL <= max(f["levels"]) < direction.SATURATED
            and direction.contrast(f["levels"]) < direction.FLAT_CONTRAST]
    return round(median(flat), 1) if len(flat) >= 2 else None


def reject_reason(frames: list, a: int, b: int, base: list, ceiling: Optional[float] = None) -> Optional[str]:
    """Why this clap should not count (message to show), or None. Clipped, too loud for this hat,
    or no direction (its four channels are nearly equal)."""
    peaks = clap_peaks(frames, a, b)
    if is_clipped(peaks):
        return (f"CLIPPED (peaks {fmt_peaks(peaks)}). Reduce the mic gain trimmer or snap softer / from "
                "farther away; this clap is not counted. Again:")
    lv = clap_frame(frames, a, b)
    if lv is None:
        return None
    limit = 0.9 * ceiling if ceiling else CLAP_MAX_PEAK
    if max(lv) > limit or direction.contrast([max(v - bb, 0.0) for v, bb in zip(lv, base)]) < CLAP_MIN_CONTRAST:
        return f"peaks {fmt_peaks(peaks)}: {TOO_LOUD}"
    return None


def fmt_peaks(peaks: list) -> str:
    return " ".join(f"{n[0].upper()}{p:.0f}" for n, p in zip(CHANNELS, peaks))


def find_clap(frames: list, base: list) -> Optional[tuple]:
    """First loud run in frames -> (start_ms, end_ms) or None."""
    start = last = None
    for f in frames:
        loud = sum(excess(f["levels"], base)) / 4 >= CLAP_EXCESS
        if loud:
            if start is None:
                start = f["ms"]
            last = f["ms"]
        elif start is not None and (f["ms"] - last > 60 or f["ms"] - start > CLAP_MAX_MS):
            return start, last
    return None


def wait_for_clap(col: Collector, n0: int, base: list, timeout: float) -> Optional[tuple]:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        hit = find_clap(col.since(n0), base)
        if hit:
            return hit
        time.sleep(0.03)
    return None


def compute_gains(frames: list, claps: list, base: list) -> Optional[list]:
    """claps: [(position, start, end)]. Mean excess per channel over claps
    spread evenly around the head -> gain to bring each channel to the mean.
    Needs all 8 positions or the 4 cardinals (otherwise directions bias it)."""
    positions = sorted({p for p, _, _ in claps})
    if positions not in (list(range(8)), [0, 2, 4, 6]):
        return None
    sens = [0.0] * 4
    for _, a, b in claps:
        win = [excess(f["levels"], base) for f in frames if a <= f["ms"] <= b]
        peak = max(win, key=sum)
        sens = [s + v for s, v in zip(sens, peak)]
    if min(sens) <= 0:
        return None
    mean = sum(sens) / 4
    lo, hi = GAIN_LIMITS
    return [round(min(hi, max(lo, mean / s)), 3) for s in sens]


def score(frames: list, claps: list, gains=None) -> list:
    out = []
    for p, a, b in claps:
        e = direction.estimate(frames, a - 40, b + 40, gains=gains, min_frames=1)   # a clap is 1 frame
        out.append((p, e, classify(e.motor, p)))
    return out


def run_calibration(link: Link, ask: Callable[[str], str], positions, out=print,
                    quiet_s: float = 3.0, clap_timeout: float = 4.0,
                    on_ready: Optional[Callable[[int], None]] = None) -> Optional[dict]:
    col = Collector()
    link.subscribe(col)
    out(f"[cal] step 1: stay quiet for {quiet_s:g} s...")
    time.sleep(quiet_s)
    if len(col.frames) < 5:
        out("[cal] FAIL: no MIC frames from the band. Is the firmware streaming? "
            "(python -m tools.bench --check)")
        return None
    q = quiet_stats(col.since(0))
    out("[cal]   baseline " + ", ".join(f"{n} {b:.1f}±{s:.1f}" for n, b, s in
                                        zip(CHANNELS, q["baseline"], q["noise"])))
    for fl in q["flags"]:
        out(f"[cal]   WARN {fl}")
    base = q["baseline"]
    claps = []
    attempts = {}
    out(f"[cal] step 2: clap once from each of {len(positions)} positions, ~1 m away "
        "(soft: aim for peaks 400-1500)")
    for n, p in enumerate(positions, 1):
        while True:
            r = ask(f"[cal]   ({n}/{len(positions)}) stand {POS_NAMES[p].upper()} of the band, press Enter, then clap "
                    "(s = skip) ").strip().lower()
            if r == "s":
                break
            n0 = len(col.frames)
            if on_ready:
                on_ready(p)
            hit = wait_for_clap(col, n0, base, clap_timeout)
            if hit:
                time.sleep(0.3)           # let the clap die down (and its tail reach the buffer)
                attempts[p] = attempts.get(p, 0) + 1
                new = col.since(n0)
                why = reject_reason(new, *hit, base, learn_ceiling(col.frames))
                if why:                   # clipped, too loud for this hat, or no direction in it
                    out(f"[cal]   {POS_NAMES[p]}: {why}")
                    continue
                claps.append((p, *hit))
                out(f"[cal]   {POS_NAMES[p]}: clap at {hit[0]}-{hit[1]} ms, "
                    f"peaks {fmt_peaks(clap_peaks(new, *hit))}")
                break
            out("[cal]   no clap heard: clap louder or closer, and try again")
    frames = col.since(0)
    ceiling = learn_ceiling(frames)
    out(f"[cal] accepted {len(claps)} of {len(positions)} claps (rejected {sum(attempts.values()) - len(claps)}"
        f"{'; hat ceiling about %.0f' % ceiling if ceiling else ''})")
    out(f"[cal]   {'position':12s} {'accepted':>8s} {'tries':>5s}  loudest frame of each accepted clap")
    for p in sorted({p for p, _, _ in claps} | set(attempts)):
        mine = [(a, b) for q, a, b in claps if q == p]
        want = sum(1 for q in positions if q == p)
        row = "  ".join(fmt_peaks(clap_frame(frames, a, b) or [0] * 4) for a, b in mine)
        out(f"[cal]   {POS_NAMES[p]:12s} {len(mine):>3d}/{want:<4d} {attempts.get(p, 0):>5d}  {row}")
    gains = compute_gains(frames, claps, base)
    before, after = score(frames, claps), score(frames, claps, gains) if gains else None
    out("[cal] step 3: results (estimated direction per clap)")
    out(f"[cal]   {'position':12s} {'raw':22s} calibrated")
    for i, (p, e, k) in enumerate(before):
        raw = f"{'generic' if e.motor is None else POS_NAMES[e.motor]} [{k}]"
        cal = ""
        if after:
            e2, k2 = after[i][1], after[i][2]
            cal = f"{'generic' if e2.motor is None else POS_NAMES[e2.motor]} [{k2}]"
        out(f"[cal]   {POS_NAMES[p]:12s} {raw:22s} {cal}")
    n_ok = sum(k == "correct" for _, _, k in before)
    n_ok2 = sum(k == "correct" for _, _, k in after) if after else None
    if gains:
        out("[cal]   gains " + ", ".join(f"{n} x{g}" for n, g in zip(CHANNELS, gains))
            + f"   correct: {n_ok}/{len(claps)} raw -> {n_ok2}/{len(claps)} calibrated")
    else:
        out(f"[cal]   correct: {n_ok}/{len(claps)}. No gains: need claps from all 8 positions "
            "or all 4 cardinals.")
    return {"baseline": base, "noise": q["noise"], "flags": q["flags"], "gains": gains,
            "ceiling": ceiling, "claps": claps, "correct_raw": n_ok, "correct_calibrated": n_ok2,
            "trace": {"frames": frames,
                      "alerts": [{"startMs": a - 40, "endMs": b + 40, "expected": p}
                                 for p, a, b in claps]}}


def save_calibration(res: dict, path=CALIBRATION_LOCAL_FILE) -> Path:
    path = Path(path)
    path.write_text(json.dumps({
        "gains": res["gains"], "ceiling": res.get("ceiling"), "channels": list(CHANNELS),
        "baseline": [round(b, 2) for b in res["baseline"]],
        "noise": [round(n, 2) for n in res["noise"]],
        "positions": [p for p, _, _ in res["claps"]],
        "correct_raw": res["correct_raw"], "correct_calibrated": res["correct_calibrated"],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "per-channel sensitivity correction; see tools/calibrate.py"}, indent=1) + "\n")
    return path


def main(argv=None, ask: Callable[[str], str] = input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--sim", action="store_true", help="simulated band with mismatched mics")
    ap.add_argument("--cardinals", action="store_true", help="4 positions instead of 8")
    ap.add_argument("--reps", type=int, default=1,
                    help="go around the positions this many times (3 gives tools/learn_centroids.py "
                         "enough to learn per-direction centroids)")
    ap.add_argument("--quiet-s", type=float, default=3.0)
    ap.add_argument("--save", action="store_true",
                    help="write app/calibration.local.json (this wearer's hat) without asking")
    ap.add_argument("--out", type=Path, default=CALIBRATION_LOCAL_FILE, help=argparse.SUPPRESS)
    ap.add_argument("--trace-dir", type=Path, default=RECORDINGS, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    on_ready = None
    if a.sim:
        port = ClapSimPort()
        link = Link(port)
        on_ready = lambda p: threading.Timer(0.1, port.clap, [p]).start()  # noqa: E731
    else:
        a.sim = False
        link = open_link(a)
        if link is None:
            return 1
    link.start()
    time.sleep(0.1 if a.sim else 2.0)       # opening the port resets the ESP32
    positions = ([0, 2, 4, 6] if a.cardinals else list(range(8))) * max(1, a.reps)
    try:
        res = run_calibration(link, ask, positions, quiet_s=a.quiet_s, on_ready=on_ready)
    finally:
        link.close()
    if res is None:
        return 1
    if res["claps"]:
        p = trace.save(res["trace"], a.trace_dir / time.strftime("calib-%H%M%S.json"))
        print(f"[cal] saved labelled trace {p}")
    g = res["gains"]
    if not g:
        return 2
    if res["correct_calibrated"] < res["correct_raw"]:
        print("[cal] calibrated gains score worse than raw: not saving them")
        return 2
    if a.save or _yes(ask(f"[cal] save gains to {a.out.name} for the app? [y/n] ")):
        print(f"[cal] saved {save_calibration(res, a.out)}; restart the app to use it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
