"""Obstacle-avoidance scoreboard: does the band warn in time, and only then?

  python -m tools.eval_avoid                     # recordings/avoid/labels.json
  python -m tools.eval_avoid --labels path.json --json

Clips come from tools/record_avoid.py (same label format as the find eval:
{"clips": [{"object": scene, "file": "table.mp4", "fps": 15,
"segments": [{"label": "free", "f0": 0, "f1": 75}, ...]}]}). Labels:
  free        open path ahead                 -> no warning
  near        obstacle ~2 m ahead             -> no warning (steering is fine)
  blocked     obstacle under ~1 m ahead       -> crash warning expected
  side_left   obstacle on the left, path clear -> no warning; never veer left
  side_right  obstacle on the right            -> no warning; never veer right
Each clip runs offline through the COCO detector (all classes) and Avoider on
the clip's own timeline at ~10 fps. GRACE_S at each segment start is excused.

Metrics:
  warn_s       median delay from a blocked segment's start to the first warning
               (a miss counts as the segment length)
  missed       blocked segments with no warning at all
  false_warn   % of non-blocked frames (after grace) with a warning
  false_per_min warning onsets in non-blocked segments per minute
  steer_wrong  % of steer/turn decisions toward the obstacle in side_* segments
  det_fps      detector frames per second (all classes)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Callable, Optional

from tools import eval_find as ef

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "recordings" / "avoid" / "labels.json"
GRACE_S = 1.0
TARGETS = {
    "warn_s": ("<=", 1.0),
    "missed": ("<=", 0),
    "false_warn": ("<=", 2.0),
    "false_per_min": ("<=", 1.0),
    "steer_wrong": ("<=", 10.0),
    "det_fps": (">=", 9.0),
}
TOWARD = {"side_left": {7, 6}, "side_right": {1, 2}}


def run_clip(frames, fps: float, detect_all: Callable, mirror: bool = False) -> list:
    """frames: (index, width, frame) with frame.shape. -> per-frame records."""
    from app.avoid import Avoider
    av = Avoider()
    recs = []
    for i, w, frame in frames:
        t0 = time.perf_counter()
        dets = detect_all(frame)
        dt = time.perf_counter() - t0
        h = frame.shape[0]
        d = av.update(dets, w, h, i / fps, None, mirror)
        recs.append({"i": i, "action": d.action, "motor": d.motor, "dt": dt,
                     "nearest": None if d.nearest is None else
                     (d.nearest.cls, round(d.nearest.distance_m, 2))})
    return recs


def score_clip(recs: list, segments: list, fps: float, grace_s: float = GRACE_S) -> dict:
    g = int(round(grace_s * fps))
    by_i = {r["i"]: r for r in recs}
    warn, missed, fw_n, fw_bad, onsets, other_s, st_n, st_bad = [], 0, 0, 0, 0, 0.0, 0, 0
    for label, a, b in segments:
        rows = [by_i[i] for i in range(a, b + 1) if i in by_i]
        if label == "blocked":
            hit = next((r for r in rows if r["action"] == "stop"), None)
            warn.append((hit["i"] - a) / fps if hit else (b - a + 1) / fps)
            missed += hit is None
            continue
        scored = [r for r in rows if r["i"] - a >= g]
        other_s += max(0.0, (b - a + 1 - g) / fps)
        fw_n += len(scored)
        fw_bad += sum(r["action"] == "stop" for r in scored)
        prev = None
        for r in scored:
            onsets += r["action"] == "stop" and prev != "stop"
            prev = r["action"]
        if label in TOWARD:
            for r in scored:
                if r["action"] in ("steer", "turn"):
                    st_n += 1
                    st_bad += r["motor"] in TOWARD[label]
    return {"warn": warn, "missed": missed, "fw_n": fw_n, "fw_bad": fw_bad, "onsets": onsets,
            "other_s": other_s, "st_n": st_n, "st_bad": st_bad,
            "det_s": sum(r["dt"] for r in recs), "n": len(recs)}


def combine(per: list) -> dict:
    tot = {k: sum(p[k] for p in per) for k in ("missed", "fw_n", "fw_bad", "onsets", "other_s",
                                                 "st_n", "st_bad", "det_s", "n")}
    warn = [w for p in per for w in p["warn"]]
    return {"warn_s": round(median(warn), 2) if warn else None,
            "missed": tot["missed"] if warn else None,
            "false_warn": ef._pct(tot["fw_bad"], tot["fw_n"]),
            "false_per_min": round(tot["onsets"] / (tot["other_s"] / 60), 2) if tot["other_s"] else None,
            "steer_wrong": ef._pct(tot["st_bad"], tot["st_n"]) if tot["st_n"] else None,
            "det_fps": round(tot["n"] / tot["det_s"], 1) if tot["det_s"] else None}


def evaluate(labels: Path = LABELS, detect_all: Optional[Callable] = None, frames_for=None,
             grace_s: float = GRACE_S, mirror: Optional[bool] = None) -> dict:
    if not labels.exists():
        return {"skip": f"no clips yet ({labels} missing; record with tools/record_avoid.py)"}
    clips = [c for c in ef.parse_labels(json.loads(labels.read_text()), labels.parent)
             if c["segments"]]
    if not clips:
        return {"skip": "no labelled clips"}
    if detect_all is None:
        from app.avoid import MIN_SCORE
        from app.vision import Detector
        det = Detector()
        detect_all = lambda f: det.detect_all(f, min_score=MIN_SCORE)  # noqa: E731
    frames_for = frames_for or ef._video_frames
    per, rows = [], []
    for c in clips:
        vfps, frames = frames_for(c["video"])
        fps = float(c["fps"] or vfps)
        m = mirror if mirror is not None else bool(c.get("mirror"))
        recs = run_clip(frames, fps, detect_all, m)
        s = score_clip(recs, c["segments"], fps, grace_s)
        per.append(s)
        rows.append({"clip": c["object"], **combine([s])})
    metrics = combine(per)
    return {"metrics": metrics, "clips": rows, "verdict": ef.verdict(metrics, TARGETS)}


def report(res: dict) -> str:
    if "skip" in res:
        return f"avoid eval: SKIP - {res['skip']}"
    out = [f"{'metric':16} {'value':>8} {'target':>8}  verdict"]
    for k, (op, thr) in TARGETS.items():
        v = res["metrics"].get(k)
        out.append(f"{k:16} {'-' if v is None else v:>8} {op + str(thr):>8}  {res['verdict'][k]}")
    for r in res["clips"]:
        out.append("  clip {clip:12} warn {warn_s}s  missed {missed}  false {false_warn}%  "
                   "steer-wrong {steer_wrong}%".format(**r))
    ok = "FAIL" not in res["verdict"].values()
    out.append("RESULT: " + ("PASS" if ok else "FAIL"))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--grace", type=float, default=GRACE_S)
    ap.add_argument("--mirror", action="store_true", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    quiet = contextlib.redirect_stdout(sys.stderr) if a.json else contextlib.nullcontext()
    with quiet:
        res = evaluate(a.labels, grace_s=a.grace, mirror=a.mirror)
    print(json.dumps(res, indent=1) if a.json else report(res))
    return 0 if "skip" in res or "FAIL" not in res["verdict"].values() else 1


if __name__ == "__main__":
    sys.exit(main())
