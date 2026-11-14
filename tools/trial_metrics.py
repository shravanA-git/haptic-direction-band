"""Trial metrics for the pitch (H27 freeze-and-measure): correct, wrong and
generic alerts, false alerts, misses and cue latency, from the alert log and
labelled traces.

  python -m tools.trial_metrics                            # newest recordings/alerts-*.jsonl
  python -m tools.trial_metrics recordings/alerts-*.jsonl  # several sessions together
  python -m tools.trial_metrics recordings/*.json          # + labelled MIC traces (offline estimator)
  python -m tools.trial_metrics --md                       # markdown table to paste into the pitch
  python -m tools.trial_metrics --json                     # machine-readable

Inputs, by file extension:
  .jsonl  alert log from the app (app/recall.py). Ground truth comes from the
          dashboard's label buttons: the caller's true side, "no clear side"
          (generic is right), "false" (nobody called) or "missed" (called, no cue).
  .json   recorded/simulated MIC traces with "expected" labels (same rules as
          tools/eval_estimator.py); scored by re-running the estimator.

Name alerts (per labelled call):
  correct  buzzed the caller's motor            near   one motor off (45 deg)
  wrong    any other motor, or a side when none  generic generic attention cue
  false    a cue when nobody called              missed called, no cue at all
  H15 gate: (correct + generic) / calls >= 80% and wrong <= 10%, calls
  including misses. "coarse" also counts near (4-position fallback).
Find cues: count of left/right, centered and lost cues, and decision->cue latency.
Taught sound: detections, false triggers, misses.
Latency is first cues only (repeats and dry-run replays are left out).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Optional

from app import recall, trace
from tools.eval_estimator import GATE_OK, GATE_WRONG, NAMES, classify, run_trace

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"
NAME_KEYS = ("correct", "near", "wrong", "generic", "false", "missed", "unlabelled")


def _pct(a: int, n: int) -> Optional[float]:
    return round(100 * a / n, 1) if n else None


def latency_stats(values) -> dict:
    v = sorted(float(x) for x in values if x is not None)
    if not v:
        return {"n": 0, "median_ms": None, "p90_ms": None, "max_ms": None}
    p90 = v[min(len(v) - 1, int(round(0.9 * (len(v) - 1))))]
    return {"n": len(v), "median_ms": round(median(v)), "p90_ms": round(p90), "max_ms": round(v[-1])}


def name_summary(c: dict) -> dict:
    """Gate numbers from correct/near/wrong/generic/missed counts."""
    calls = c["correct"] + c["near"] + c["wrong"] + c["generic"] + c["missed"]
    ok = c["correct"] + c["generic"]
    out = {"calls": calls,
           "score_pct": _pct(ok, calls),
           "coarse_pct": _pct(ok + c["near"], calls),
           "wrong_pct": _pct(c["wrong"], calls)}
    out["gate"] = bool(calls) and ok / calls >= GATE_OK and c["wrong"] / calls <= GATE_WRONG
    return out


def score_log(alerts: list, labels: list) -> dict:
    """Score one or more sessions of alert log records."""
    truth: dict = {}
    missed = 0
    for lb in labels:
        if lb.get("missed") and lb.get("ref") is None:
            missed += 1
        elif lb.get("ref") is not None:
            truth[(lb.get("_session"), lb["ref"])] = lb["truth"]   # last label wins

    live = [a for a in alerts if not a.get("dry") and a.get("source") not in ("replay", "manual")]
    name = dict.fromkeys(NAME_KEYS, 0)
    name["missed"] = missed
    sound = {"detections": 0, "false": 0, "unlabelled": 0}
    find = {"directional": 0, "centered": 0, "lost": 0, "repeats": 0}
    lat: dict = {"name": [], "find": [], "sound": []}
    rows = []
    for a in live:
        src = a["source"]
        key = (a.get("_session"), a["id"])
        t = truth.get(key, ...)
        kind = None
        if src == "name":
            if t is ...:
                kind = "unlabelled"
            elif t == "false":
                kind = "false"
            else:
                kind = classify(a.get("motor"), t)
            name[kind] += 1
        elif src == "sound":
            kind = "unlabelled" if t is ... else "false" if t == "false" else "detections"
            sound[kind] += 1
        elif src == "find":
            if a.get("repeat"):
                find["repeats"] += 1
                continue
            if a.get("cue") in find:
                find[a["cue"]] += 1
        if src in lat and not a.get("repeat"):
            lat[src].append(a.get("latency_ms"))
        if kind:
            rows.append({"id": a["id"], "time": a.get("time"), "source": src,
                         "got": a.get("direction") or a.get("cue"),
                         "truth": None if t is ... else t, "kind": kind,
                         "latency_ms": a.get("latency_ms")})
    return {"name": {**name, **name_summary(name)},
            "sound": sound, "find": find,
            "latency": {k: latency_stats(v) for k, v in lat.items()},
            "rows": rows}


def score_traces(paths) -> dict:
    c = dict.fromkeys(("correct", "near", "wrong", "generic", "missed"), 0)
    rows, skipped = [], []
    for p in paths:
        got = run_trace(trace.load(p))
        if not got:
            skipped.append(str(p))
        for al, exp, e, kind in got:
            c[kind] += 1
            rows.append({"file": Path(p).name, "window": [al["startMs"], al["endMs"]],
                         "expected": None if exp is None else NAMES[exp],
                         "got": None if e.motor is None else NAMES[e.motor], "kind": kind})
    del c["missed"]
    return {**c, **name_summary({**c, "missed": 0}), "rows": rows, "unlabelled_files": skipped}


def load_logs(paths) -> tuple[list, list]:
    alerts, labels = [], []
    for i, p in enumerate(paths):
        a, lb = recall.load(p)
        for r in a + lb:
            r["_session"] = i          # alert ids restart per session
        alerts += a
        labels += lb
    return alerts, labels


def report(logs, traces) -> dict:
    out: dict = {"logs": [str(p) for p in logs], "traces": [str(p) for p in traces]}
    if logs:
        out["live"] = score_log(*load_logs(logs))
    if traces:
        out["replay"] = score_traces(traces)
    return out


# ------------------------------------------------------------------ output
def _f(v, unit="") -> str:
    return "–" if v is None else f"{v}{unit}"


def _name_line(label: str, d: dict) -> str:
    return (f"{label:10s} {d['calls']:5d} {d['correct']:7d} {d['near']:5d} {d['wrong']:5d} "
            f"{d['generic']:7d} {d.get('missed', 0):6d} {d.get('false', 0):5d}   "
            f"{_f(d['score_pct'], '%'):>6s} {_f(d['coarse_pct'], '%'):>6s}  "
            f"{'PASS' if d['gate'] else 'FAIL'}")


def print_text(r: dict, verbose: bool = False) -> None:
    print("Hear your name: direction per call")
    print(f"{'':10s} {'calls':>5s} {'correct':>7s} {'near':>5s} {'wrong':>5s} {'generic':>7s} "
          f"{'missed':>6s} {'false':>5s}   {'gate':>6s} {'coarse':>6s}")
    live, rep = r.get("live"), r.get("replay")
    if live:
        print(_name_line("live", live["name"]))
    if rep:
        print(_name_line("traces", rep))
    if not (live or rep):
        print("  (no input: pass alert logs .jsonl and/or labelled traces .json)")
    if live:
        n = live["name"]
        if n["unlabelled"]:
            print(f"  {n['unlabelled']} name alert(s) not labelled yet: use the dashboard "
                  "label buttons after each call")
        s, f = live["sound"], live["find"]
        if any(s.values()):
            print(f"\nTaught sound: {s['detections']} detections, {s['false']} false triggers, "
                  f"{s['unlabelled']} unlabelled")
        if any(f.values()):
            print(f"\nFind an object: {f['directional']} left/right, {f['centered']} centered, "
                  f"{f['lost']} lost first cues ({f['repeats']} left/right repeats)")
        print("\nLatency (first cues)        n   median      p90      max")
        what = {"name": "name end -> cue", "find": "frame -> cue", "sound": "sound end -> cue"}
        for k, st in live["latency"].items():
            if st["n"]:
                print(f"  {k:5s} {what[k]:18s} {st['n']:4d} {_f(st['median_ms'], ' ms'):>8s} "
                      f"{_f(st['p90_ms'], ' ms'):>8s} {_f(st['max_ms'], ' ms'):>8s}")
    if rep and rep["unlabelled_files"]:
        print(f"\n{len(rep['unlabelled_files'])} trace(s) without labels skipped")
    if verbose:
        for row in (live or {}).get("rows", []) + (rep or {}).get("rows", []):
            print("  ", json.dumps(row))


def print_md(r: dict) -> None:
    live, rep = r.get("live"), r.get("replay")
    print("| Hear your name | calls | correct | near | wrong | generic | missed | false | "
          "right or generic |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, d in (("Live trials", live and live["name"]), ("Recorded traces", rep)):
        if d:
            print(f"| {label} | {d['calls']} | {d['correct']} | {d['near']} | {d['wrong']} | "
                  f"{d['generic']} | {d.get('missed', 0)} | {d.get('false', 0)} | "
                  f"{_f(d['score_pct'], '%')} |")
    if live:
        print("\n| Latency (first cue) | n | median | p90 |")
        print("|---|---|---|---|")
        for k, label in (("name", "Name end → buzz"), ("find", "Camera frame → buzz"),
                         ("sound", "Taught sound → buzz")):
            st = live["latency"][k]
            if st["n"]:
                print(f"| {label} | {st['n']} | {st['median_ms']} ms | {st['p90_ms']} ms |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help=".jsonl alert logs and/or .json traces")
    ap.add_argument("--md", action="store_true", help="markdown table for the pitch")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("-v", "--verbose", action="store_true", help="one line per scored alert")
    ap.add_argument("--gate", action="store_true", help="exit 1 if a name row fails the H15 gate")
    a = ap.parse_args(argv)
    files = list(a.files)
    if not files:
        newest = sorted(RECORDINGS.glob("alerts-*.jsonl"))[-1:]
        if not newest:
            print("No alert logs in recordings/. Run the app (python -m app.main) or pass files.")
            return 1
        files = newest
    logs = [p for p in files if p.suffix == ".jsonl"]
    traces = [p for p in files if p.suffix == ".json"]
    r = report(logs, traces)
    if a.json:
        print(json.dumps(r, indent=1))
    elif a.md:
        print_md(r)
    else:
        print_text(r, a.verbose)
    rows = [d for d in ((r.get("live") or {}).get("name"), r.get("replay")) if d and d["calls"]]
    return 1 if a.gate and (not rows or not all(d["gate"] for d in rows)) else 0


if __name__ == "__main__":
    sys.exit(main())
