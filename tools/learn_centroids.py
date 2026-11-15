"""Learn where each direction lands in the estimator's (east, north) space.

The analytic estimator assumes an ideal cosine mic response; the real head and hat
shadow the mics unevenly. Record labelled claps, learn one centroid per direction, and
the live estimator classifies by nearest centroid (with a margin; ambiguous -> the
generic front cue). No centroids file -> the analytic path, unchanged.

  python -m tools.calibrate --reps 3            # 3 laps of the 8 positions (24 claps); --cardinals for 4
  python -m tools.learn_centroids               # all recordings/calib-*.json, leave-one-out report
  python -m tools.learn_centroids recordings/calib-101500.json --save
  python -m tools.learn_centroids --no-gains    # features without app/calibration.json

--save writes app/centroids.json (gitignored: it describes one wearer's head and hat),
but only if the leave-one-out score is at least as good as the analytic estimator's.
8-way needs claps from all 8 positions; the 4-motor band needs the 4 cardinals.
Each position needs 2+ repetitions (3 recommended) for a leave-one-out check.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Optional

from app import direction, trace
from tools.eval_estimator import NAMES, classify, confusion, print_matrix

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"


def gather(paths, gains) -> list:
    """[(true motor, estimate)] for every labelled (integer) alert in the traces."""
    out = []
    for p in paths:
        t = trace.load(p)
        for a in t.get("alerts", []):
            exp = a.get("expected")
            if not isinstance(exp, int):
                continue
            e = direction.estimate(t["frames"], a["startMs"], a["endMs"], gains=gains, algo="v2")
            out.append((exp, e))
    return out


def centroids_from(trials: list) -> dict:
    """{motor: (east, north)}: per-axis median over the trials of each direction."""
    by: dict = {}
    for m, e in trials:
        if e.vector is not None:
            by.setdefault(m, []).append(e.vector)
    return {m: (median(v[0] for v in vs), median(v[1] for v in vs)) for m, vs in by.items()}


def classify_with(vec, cents: dict, margin: float = None) -> Optional[int]:
    return direction.classify_centroid(vec, cents, margin)[0]


def leave_one_out(trials: list, margin: float = None) -> list:
    """[(true, analytic motor, centroid motor)] with each trial held out of its own centroids."""
    out = []
    for i, (m, e) in enumerate(trials):
        if e.vector is None or e.motor is None:       # a gate already said generic: no centroid step
            out.append((m, e.motor, None))
            continue
        rest = [t for j, t in enumerate(trials) if j != i]
        cents = centroids_from(rest)
        if m not in cents or len(cents) < 2:             # the only trial of its direction
            cents = centroids_from(trials)
        out.append((m, e.motor, classify_with(e.vector, cents, margin)))
    return out


def summarize(pairs: list) -> dict:
    c = {"correct": 0, "near": 0, "wrong": 0, "generic": 0}
    for true, got in pairs:
        c[classify(got, true)] += 1
    n = len(pairs) or 1
    c["score"] = (c["correct"] + c["generic"]) / n
    c["exact"] = c["correct"] / n
    return c


def report(trials: list, out=print) -> dict:
    loo = leave_one_out(trials)
    ana = [(t, a) for t, a, _ in loo]
    cen = [(t, c) for t, _, c in loo]
    sa, sc = summarize(ana), summarize(cen)
    ca, cc = confusion(ana), confusion(cen)
    out(f"{len(trials)} labelled claps, directions {sorted({m for m, _ in trials})}")
    out(f"{'':22s} {'exact':>6s} {'near':>5s} {'wrong':>6s} {'generic':>8s} {'front-bias':>11s}")
    for label, s, c in (("analytic (v2)", sa, ca), ("centroids (leave-1-out)", sc, cc)):
        out(f"{label:22s} {100 * s['exact']:5.1f}% {s['near']:5d} {s['wrong']:6d} {s['generic']:8d} "
            f"{100 * c['front_bias']:9.1f}%")
    print_matrix(ca, "analytic")
    print_matrix(cc, "centroids")
    return {"analytic": sa, "centroids": sc}


def save(trials: list, path: Path = direction.CENTROIDS_FILE) -> Path:
    cents = centroids_from(trials)
    counts: dict = {}
    for m, _ in trials:
        counts[m] = counts.get(m, 0) + 1
    path.write_text(json.dumps({
        "centroids": {str(m): [round(x, 4), round(y, 4)] for m, (x, y) in sorted(cents.items())},
        "trials": {str(m): n for m, n in sorted(counts.items())},
        "margin": direction.CENTROID_MARGIN,
        "note": "per-direction centroids in the estimator's (east, north) space; see tools/learn_centroids.py"},
        indent=1) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="*", type=Path, help="labelled traces (default recordings/calib-*.json)")
    ap.add_argument("--no-gains", action="store_true", help="ignore app/calibration.json")
    ap.add_argument("--save", action="store_true", help="write app/centroids.json if it beats analytic")
    ap.add_argument("--force", action="store_true", help="with --save: write it even if it does not")
    ap.add_argument("--out", type=Path, default=direction.CENTROIDS_FILE, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    paths = a.traces or sorted(RECORDINGS.glob("calib-*.json"))
    if not paths:
        print("no calib traces: python -m tools.calibrate --reps 3")
        return 1
    gains = None if a.no_gains else direction.load_calibration() if direction.CALIBRATION_FILE.exists() else None
    try:
        trials = gather(paths, gains)
    except (OSError, ValueError, KeyError) as e:
        print(f"cannot read the traces: {e}")
        return 1
    dirs = sorted({m for m, _ in trials})
    if len(dirs) < 2:
        print(f"need claps from at least 4 directions, got {dirs}")
        return 1
    thin = [NAMES[m] for m in dirs if sum(1 for t, _ in trials if t == m) < 2]
    if thin:
        print(f"WARNING: only one clap for {', '.join(thin)}: no leave-one-out check for those; record 3 laps")
    res = report(trials)
    if not a.save:
        return 0
    better = (res["centroids"]["exact"] >= res["analytic"]["exact"]
              and res["centroids"]["wrong"] <= res["analytic"]["wrong"])      # more exact, not just more generic
    if not better and not a.force:
        print("centroids do not beat the analytic estimator on these claps: not saving (--force to override)")
        return 2
    print(f"saved {save(trials, a.out)}; restart the app to use it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
