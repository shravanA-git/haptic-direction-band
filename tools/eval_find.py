"""Voice-to-find scoreboard: is "find my X" demo-reliable?

  python -m tools.eval_find                 # Part A (text + spoken) and Part B (clips)
  python -m tools.eval_find --no-spoken     # text matching only (no whisper, ~1 s)
  python -m tools.eval_find --spoken 40     # spoken subset size (phrases; x 3 voices)
  python -m tools.eval_find --labels recordings/eval/labels.json --mirror
  python -m tools.eval_find --json          # machine-readable, same numbers
  python -m tools.eval_find --targets t.json  # override PASS/FAIL thresholds

Exit code 0 when every scored metric passes (SKIPped parts don't fail).

Part A, phrasing. TEMPLATES x OBJECTS through app.speech.match_target. The
expected answer is the COCO class (resolve_target) or, for open-vocabulary
objects, the object phrase itself (compared without spaces/plural s, so
"air pods" == "AirPods"). The spoken subset renders phrases with macOS `say`
in 3 voices and runs them through faster-whisper (app.speech.Transcriber),
then match_target.

Part B, clip replay. recordings/eval/labels.json (from Shravan's recorder),
videos <object>.mp4 beside it. Accepted shapes:
  {"fps": 15, "clips": [{"object": "banana", "video": "banana.mp4", "fps": 15,
     "segments": [{"label": "out", "start": 0, "end": 44}, ...]}]}
  a bare list of clips, or {"banana": {...clip...}}; segment keys may also be
  kind/type, start_frame/startFrame, end_frame/endFrame or "frames": [a, b];
  segments may be {"left": [[45, 90], ...], ...}. Frame ranges are inclusive.
Labels: out (object not in view), left / center / right (held still in that
part of the view), sweep (moving). Each clip runs offline through Vision's
detector + TargetTracker + ObjectGuide on the clip's own timeline (frame / fps),
so the numbers don't depend on how fast this laptop is.

Metrics (in-view = left/center/right/sweep):
  seen_coco / seen_open   % of in-view frames with a tracked box, COCO
                targets vs open-vocabulary (YOLO-World) targets
  side_acc      % of held-still frames whose side state matches the label
                (among frames where the tracker reports a side)
  cue_s         median delay from entering (the first frame the object is detected
                in a segment entered from out) to the first ObjectGuide cue pointing
                the right way (left motor for "left", centered for "center")
  cue_prompt_s  same from the recorder's prompt (info: includes the human's
                reaction, ~1.5 s in the Nov 14 clips)
  detect_s      same, to the first box (info)
  det_fps       detector frames per second on this machine
  false_out     % of out frames with a box
  flips_min     tracker state changes per minute while held still
  lost_s        median delay from the last box after leaving to "lost"
                (never = end of segment)
  wrong_cue     % of pulses on the opposite side while held left/right (motors 6,7
                are left; 1,2 are right)
  off_side      same, but front pulses (tracker still "centered") also count (info)
  centered_pps  front pulses per second while held centered (hot signal, >=3)
Part A also counts wrong_target: answers naming the wrong thing (a miss -> the
"can't look for that" cue is acceptable, a wrong target is not), including any
answer to a NEGATIVES phrase ("hello there").
  stale_cues    directional cues after the object left (info, no target)
GRACE_S at each segment start is excused from side_acc, flips, false_out and
wrong_cue: the recorder's segments are timed prompts and the object is still
moving into place for the first ~1-1.5 s. detect_s counts from the prompt, so
it includes the human's reaction time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "recordings" / "eval" / "labels.json"
GRACE_S = 1.5   # recorder segments are timed prompts; people take ~1-1.5 s to move
VOICES = ("Samantha", "Daniel", "Fred")

# Thresholds (PASS if value meets them). Override with --targets file.json.
# HDB-Supervisor's targets for Shravan, Nov 14.
TARGETS = {
    "text_acc": (">=", 95.0),
    "spoken_acc": (">=", 90.0),
    "wrong_target": ("<=", 0),        # a wrong object is worse than "can't look for that"
    "seen_coco": (">=", 80.0),
    "seen_open": (">=", 60.0),
    "side_acc": (">=", 95.0),
    "wrong_cue": ("<=", 2.0),
    "centered_pps": (">=", 1.5),      # hot/cold: pulses/s while held centered at a steady distance
                                      # (closeness is relative to the starting size: ~1.7/s steady, up to 5/s approaching)
    "cue_s": ("<=", 0.5),
    "lost_s": ("<=", 2.5),
    "flips_min": ("<=", 6.0),
    "false_out": ("<=", 0.0),
    "det_fps": (">=", 9.0),
}

TEMPLATES = [
    "where is my {x}", "where's my {x}", "where are my {x}", "find the {x}", "find my {x}",
    "find {x}", "can you find my {x}", "can you help me find {x}",
    "can you help me find my {x}", "help me find the {x}", "I lost my {x}",
    "I can't find my {x}", "look for the {x}", "look for my {x}", "please find my {x}",
    "locate my {x}", "where did I put my {x}", "where did I leave my {x}",
    "have you seen my {x}", "I'm looking for my {x}", "show me the {x}",
    "take me to my {x}", "get my {x}", "find the {x} for me", "okay find my {x} now",
    "Hey, where is my {x}?", "um where is my {x}", "uh find the {x}",
    "uh, um, where's my {x}", "hmm I need my {x}", "so um can you look for the {x}",
    "er where the heck is my {x}", "mm find my {x} please", "{x} please", "{x}",
]

# (as spoken, expected). Expected None = derive (COCO class or the phrase itself).
OBJECTS = [
    # COCO classes and synonyms
    "banana", "apple", "orange", "cup", "mug", "bottle", "water bottle", "cell phone", "phone",
    "laptop", "mouse", "keyboard", "remote", "book", "scissors", "backpack", "bag", "umbrella",
    "handbag", "purse", "chair", "teddy bear", "toothbrush", "clock", "vase", "spoon", "fork",
    "knife", "bowl", "sandwich", "sports ball", "potted plant", "TV",
    # open vocabulary
    "headphones", "keys", "wallet", "charger", "glasses", "AirPods", "sunglasses", "earbuds",
    "pen", "hat", "watch", "hearing aid", "pill bottle", "slippers", "ID card",
]

# Nothing asked for: must give None (the "can't look for that" cue), never a target.
NEGATIVES = ["hello there", "um", "uh never mind", "what time is it", "thank you",
             "okay", "stop", "can you hear me", "hmm", "I'm fine"]

# Other answers that still find the thing (the COCO detector sees a pill bottle).
ALSO_OK = {"pill bottle": {"bottle"}}

# Spoken subset: fixed, spread across templates and objects (deterministic).
SPOKEN_TEMPLATES = [
    "where is my {x}", "find the {x}", "I lost my {x}", "can you help me find {x}",
    "look for my {x}", "um where's my {x}", "uh find my {x} please", "I can't find my {x}",
    "have you seen my {x}", "where did I put my {x}",
]
SPOKEN_OBJECTS = [
    "banana", "cup", "water bottle", "cell phone", "phone", "laptop", "remote", "backpack",
    "book", "scissors", "headphones", "keys", "wallet", "charger", "glasses", "AirPods",
]


# --------------------------------------------------------------------------
# Part A: phrasing
# --------------------------------------------------------------------------

def expected_for(obj: str) -> str:
    from app.vision import resolve_target
    return resolve_target(obj) or obj.lower()


def _key(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    k = re.sub(r"[^a-z]", "", s.lower())
    return k[:-1] if k.endswith("s") else k


def is_match(got: Optional[str], want: str, obj: Optional[str] = None) -> bool:
    if got is None:
        return False
    ok = {want} | ALSO_OK.get(obj or "", set())
    return got in ok or _key(got) in {_key(w) for w in ok}


def score_text(templates=TEMPLATES, objects=OBJECTS,
               match: Optional[Callable[[str], Optional[str]]] = None,
               negatives: bool = True) -> dict:
    """Accuracy over templates x objects; "wrong" counts answers that name the
    wrong thing (including any answer to a NEGATIVES phrase), not misses."""
    if match is None:
        from app.speech import match_target as match
    n = ok = wrong = 0
    by_tpl, by_obj, fails = {}, {}, []
    for tpl in templates:
        for obj in objects:
            want = expected_for(obj)
            text = tpl.format(x=obj)
            got = match(text)
            good = is_match(got, want, obj)
            n += 1
            ok += good
            by_tpl.setdefault(tpl, [0, 0])
            by_obj.setdefault(obj, [0, 0])
            by_tpl[tpl][0] += good
            by_tpl[tpl][1] += 1
            by_obj[obj][0] += good
            by_obj[obj][1] += 1
            if not good:
                wrong += got is not None
                fails.append({"text": text, "want": want, "got": got})
    for text in (NEGATIVES if negatives else []):
        got = match(text)
        if got is not None:
            wrong += 1
            fails.append({"text": text, "want": None, "got": got})
    return {"n": n, "ok": ok, "acc": _pct(ok, n), "wrong": wrong,
            "worst_templates": _worst(by_tpl), "worst_objects": _worst(by_obj),
            "fails": fails}


def _worst(d: dict, k: int = 6) -> list:
    rows = [(name, _pct(a, b)) for name, (a, b) in d.items() if a < b]
    return sorted(rows, key=lambda r: r[1])[:k]


def spoken_phrases(limit: int) -> list:
    """Deterministic spread: walk templates and objects together."""
    out, i = [], 0
    nt, no = len(SPOKEN_TEMPLATES), len(SPOKEN_OBJECTS)
    seen = set()
    while len(out) < min(limit, nt * no):
        tpl, obj = SPOKEN_TEMPLATES[i % nt], SPOKEN_OBJECTS[(i + i // no) % no]
        if (tpl, obj) not in seen:
            seen.add((tpl, obj))
            out.append((tpl.format(x=obj), obj))
        i += 1
    return out


def available_voices(wanted=VOICES) -> list:
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True,
                                 timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    have = {line.split()[0] for line in listing.splitlines() if line.strip()}
    return [v for v in wanted if v in have]


def score_spoken(limit: int = 40, voices=None, transcriber=None) -> dict:
    """`say` -> wav -> faster-whisper -> match_target. SKIP if say/whisper missing."""
    voices = available_voices() if voices is None else voices
    if not voices:
        return {"skip": "macOS `say` voices not available"}
    try:
        from app.speech import PROMPT, Transcriber, load_audio, match_target
        tr = transcriber or Transcriber()
        tr.load()
    except Exception as e:
        return {"skip": f"faster-whisper unavailable ({type(e).__name__}: {e})"}
    n = ok = wrong = 0
    by_voice, fails, secs = {}, [], []
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "p.wav")
        for phrase, obj in spoken_phrases(limit):
            want = expected_for(obj)
            for v in voices:
                subprocess.run(["say", "-v", v, "-o", wav, "--data-format=LEI16@16000", phrase],
                               check=True)
                t = time.monotonic()
                text, _ = tr.transcribe(load_audio(wav), 0.0, PROMPT)
                secs.append(time.monotonic() - t)
                got = match_target(text)
                good = is_match(got, want, obj)
                n += 1
                ok += good
                by_voice.setdefault(v, [0, 0])
                by_voice[v][0] += good
                by_voice[v][1] += 1
                if not good:
                    wrong += got is not None
                    fails.append({"said": phrase, "voice": v, "heard": text, "want": want,
                                  "got": got})
    return {"n": n, "ok": ok, "acc": _pct(ok, n), "wrong": wrong, "voices": voices,
            "by_voice": {v: _pct(a, b) for v, (a, b) in by_voice.items()},
            "decode_median_s": round(median(secs), 3) if secs else None, "fails": fails}


# --------------------------------------------------------------------------
# Part B: clip replay
# --------------------------------------------------------------------------

IN_VIEW = ("left", "center", "right", "sweep")
HELD = ("left", "center", "right")
STATE_FOR = {"left": "left", "center": "centered", "right": "right"}
SEG_ALIASES = {"centre": "center", "centered": "center", "middle": "center", "none": "out",
               "away": "out", "gone": "out", "moving": "sweep"}


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_labels(data, base: Path) -> list:
    """-> [{"object", "video": Path, "fps": float|None, "segments": [(label, a, b)]}]"""
    gfps = data.get("fps") if isinstance(data, dict) else None
    gmirror = data.get("mirror") if isinstance(data, dict) else None
    if isinstance(data, dict) and "clips" in data:
        items = data["clips"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [dict(v, object=v.get("object", k)) for k, v in data.items()
                 if isinstance(v, dict)]
    else:
        items = []
    if isinstance(items, dict):
        items = [dict(v, object=v.get("object", k)) for k, v in items.items()]
    clips = []
    for c in items:
        obj = _first(c, "object", "target", "name")
        video = _first(c, "video", "file", "path") or f"{obj}.mp4"
        segs = []
        raw = c.get("segments") or c.get("labels") or []
        if isinstance(raw, dict):
            raw = [{"label": k, "frames": r} for k, rs in raw.items()
                   for r in (rs if rs and isinstance(rs[0], (list, tuple)) else [rs])]
        for s in raw:
            label = str(_first(s, "label", "kind", "type", "segment", "state")).lower()
            label = SEG_ALIASES.get(label, label)
            fr = s.get("frames")
            a = fr[0] if fr else _first(s, "f0", "start", "start_frame", "startFrame", "from")
            b = fr[1] if fr else _first(s, "f1", "end", "end_frame", "endFrame", "to")
            if a is None or b is None:
                continue
            segs.append((label, int(a), int(b)))
        segs.sort(key=lambda s: s[1])
        # The recorder writes half-open ranges (f1 == next f0): a shared boundary
        # frame belongs to the later segment.
        segs = [(lab, a, min(b, segs[k + 1][1] - 1) if k + 1 < len(segs) else b)
                for k, (lab, a, b) in enumerate(segs)]
        clips.append({"object": obj, "video": base / video, "fps": c.get("fps") or gfps,
                      "mirror": c.get("mirror", gmirror), "segments": segs})
    return clips


def label_at(segments, i: int):
    """(label, segment start frame) for frame i, or (None, None)."""
    for label, a, b in segments:
        if a <= i <= b:
            return label, a
    return None, None


def run_clip(frames: Iterable, fps: float, target: str, detect: Callable,
             config=None, mirror: bool = False) -> tuple:
    """frames: iterable of (index, width, frame). detect(frame, target) -> boxes.
    Returns (per-frame records, cues) on the clip's own timeline."""
    from app import haptics
    from app.guide import ObjectGuide
    from app.vision import TargetTracker
    tracker = TargetTracker(config)
    tracker.set_target(target, getattr(detect, "min_score", None),
                       getattr(detect, "confirm", 1))
    cues, now = [], [0.0]

    def play(cue):                  # every pulse the band would get, not the throttled log
        if cue == haptics.centered():
            cues.append((now[0], "centered", None))
        elif cue == haptics.lost():
            cues.append((now[0], "lost", None))
        else:
            cues.append((now[0], "directional", cue[0][1]))
    guide = ObjectGuide(play=play)
    recs = []
    prev_t = None
    for i, width, frame in frames:
        t = i / fps
        if prev_t is not None:      # the live guide thread ticks every 20 ms between frames
            k = prev_t + 0.02
            while k < t:
                now[0] = k
                guide.tick(now=k)
                k += 0.02
        prev_t = t
        now[0] = t
        t_det = time.perf_counter()
        boxes = detect(frame, target)
        dt = time.perf_counter() - t_det
        shape = getattr(frame, "shape", None)
        d = tracker.update(boxes, width, t, mirror, shape[0] if shape else None)
        guide.on_frame(d, now=t)    # live offset/size every frame; state changes inside
        guide.tick(now=t)
        recs.append({"i": i, "t": t, "state": d.state, "box": d.bbox is not None, "dt": dt})
    return recs, cues


LEFT_MOTORS, RIGHT_MOTORS = {6, 7}, {1, 2}      # front-left/left, front-right/right


def _cue_matches(label: str, kind: str, motor) -> bool:
    if label == "center":
        return kind == "centered"
    if label in ("left", "right"):
        return kind == "directional" and motor in (LEFT_MOTORS if label == "left" else RIGHT_MOTORS)
    return kind in ("directional", "centered")      # sweep: any pointing cue


def score_clip(recs: list, cues: list, segments: list, fps: float,
               grace_s: float = GRACE_S) -> dict:
    graceful = int(round(grace_s * fps))
    by_i = {r["i"]: r for r in recs}
    inview = [r for r in recs if label_at(segments, r["i"])[0] in IN_VIEW]
    seen = _pct(sum(r["box"] for r in inview), len(inview))

    side_n = side_ok = 0
    flips, held_s = 0, 0.0
    out_n = out_box = 0
    for label, a, b in segments:
        rows = [by_i[i] for i in range(a, b + 1) if i in by_i]
        scored = [r for r in rows if r["i"] - a >= graceful]
        if label in HELD:
            held_s += max(0.0, (b - a + 1 - graceful) / fps)
            prev = None
            for r in scored:
                if r["state"] in ("left", "right", "centered"):
                    side_n += 1
                    side_ok += r["state"] == STATE_FOR[label]
                if prev is not None and r["state"] != prev:
                    flips += 1
                prev = r["state"]
        elif label == "out":
            out_n += len(scored)
            out_box += sum(r["box"] for r in scored)

    # Entering: segment start (the recorder's prompt) -> first box. Leaving: last
    # box in the out segment (the object actually gone) -> "lost".
    detect, lost, cue_d, cue_p, stale = [], [], [], [], 0
    gone_at = {}
    for k, (label, a, b) in enumerate(segments):
        prev_label = segments[k - 1][0] if k else None
        rows = [by_i[i] for i in range(a, b + 1) if i in by_i]
        if label in IN_VIEW and prev_label in (None, "out"):
            hit = next((r for r in rows if r["box"]), None)
            detect.append((hit["i"] - a) / fps if hit else (b - a + 1) / fps)
            # first cue that points the right way for this segment (a "lost" buzz or
            # a repeat toward the wrong side doesn't count as finding it)
            first = next((t for t, kind, motor in cues if a / fps <= t <= (b + 1) / fps
                          and _cue_matches(label, kind, motor)), None)
            cue_p.append(first - a / fps if first is not None else (b - a + 1) / fps)
            if hit is not None:   # entering = the first frame the object is detected
                t_in = hit["i"] / fps
                first = next((t for t, kind, motor in cues if t_in <= t <= (b + 1) / fps
                              and _cue_matches(label, kind, motor)), None)
                cue_d.append(first - t_in if first is not None else (b + 1) / fps - t_in)
        if label == "out" and prev_label in IN_VIEW:
            lost_r = next((r for r in rows if r["state"] == "lost"), None)
            upto = lost_r["i"] if lost_r else b + 1
            boxes = [r["i"] for r in rows if r["box"] and r["i"] < upto]
            left_i = (boxes[-1] + 1) if boxes else a
            gone_at[a] = left_i / fps
            lost.append(((lost_r["i"] if lost_r else b + 1) - left_i) / fps)

    # hot signal: front pulses per second while the object is held centered
    cen_p, cen_s = 0, 0.0
    for label, a, b in segments:
        if label == "center":
            t0, t1 = a / fps + grace_s, (b + 1) / fps
            if t1 > t0:
                cen_s += t1 - t0
                cen_p += sum(1 for t, kind, m in cues if kind == "directional" and m == 0 and t0 <= t < t1)
    dir_n = dir_bad = dir_off = 0
    for t, kind, motor in cues:
        if kind != "directional":
            continue
        label, a = label_at(segments, int(round(t * fps)))
        if label in ("left", "right") and t - a / fps >= grace_s:
            dir_n += 1
            own, other = (LEFT_MOTORS, RIGHT_MOTORS) if label == "left" else (RIGHT_MOTORS, LEFT_MOTORS)
            dir_bad += motor in other          # pointing the opposite way
            dir_off += motor not in own        # opposite OR front (tracker still "centered")
        elif label == "out" and t > gone_at.get(a, a / fps):
            stale += 1   # pulsing toward an object that has already left the view
    return {
        "frames": len(recs), "seen": seen, "side_acc": _pct(side_ok, side_n),
        "detect_s": round(median(detect), 2) if detect else None,
        "cue_s": round(median(cue_d), 2) if cue_d else None,
        "cue_prompt_s": round(median(cue_p), 2) if cue_p else None,
        "false_out": _pct(out_box, out_n),
        "flips_min": round(flips / (held_s / 60), 2) if held_s else None,
        "lost_s": round(median(lost), 2) if lost else None,
        "wrong_cue": _pct(dir_bad, dir_n) if dir_n else None,
        "off_side": _pct(dir_off, dir_n) if dir_n else None,
        "stale_cues": stale,
        "centered_pps": round(cen_p / cen_s, 2) if cen_s else None,
        "det_fps": _fps(recs),
        "_counts": {"cue": cue_d, "cue_p": cue_p, "det_s": sum(r.get("dt", 0) for r in recs), "n": len(recs),"inview": len(inview), "seen": sum(r["box"] for r in inview),
                    "side_n": side_n, "side_ok": side_ok, "flips": flips, "held_s": held_s,
                    "out_n": out_n, "out_box": out_box, "detect": detect, "lost": lost,
                    "dir_n": dir_n, "dir_bad": dir_bad, "dir_off": dir_off, "stale": stale,
                    "cen_p": cen_p, "cen_s": cen_s},
    }


def _fps(recs) -> Optional[float]:
    total = sum(r.get("dt", 0) for r in recs)
    return round(len(recs) / total, 1) if total > 0 else None


def combine(per_clip: list, is_open: Optional[list] = None) -> dict:
    """Pool counts across clips (frame-weighted, not a mean of percentages).
    is_open[k]: clip k is an open-vocabulary (YOLO-World) target."""
    is_open = is_open or [False] * len(per_clip)
    split = {}
    for kind, flag in (("seen_coco", False), ("seen_open", True)):
        cs = [s["_counts"] for s, o in zip(per_clip, is_open) if o == flag]
        split[kind] = _pct(sum(c["seen"] for c in cs), sum(c["inview"] for c in cs))
    det_s = sum(s["_counts"]["det_s"] for s in per_clip)
    split["det_fps"] = round(sum(s["_counts"]["n"] for s in per_clip) / det_s, 1) if det_s else None
    cue = [x for s in per_clip for x in s["_counts"]["cue"]]
    split["cue_s"] = round(median(cue), 2) if cue else None
    split["off_side"] = _pct(sum(s["_counts"].get("dir_off", 0) for s in per_clip),
                             sum(s["_counts"]["dir_n"] for s in per_clip)) or 0.0
    cen_s = sum(s["_counts"].get("cen_s", 0.0) for s in per_clip)
    split["centered_pps"] = round(sum(s["_counts"].get("cen_p", 0) for s in per_clip) / cen_s, 2) if cen_s else None
    cue_p = [x for s in per_clip for x in s["_counts"].get("cue_p", [])]
    split["cue_prompt_s"] = round(median(cue_p), 2) if cue_p else None
    c = {k: 0 for k in ("inview", "seen", "side_n", "side_ok", "flips", "held_s", "out_n",
                        "out_box", "dir_n", "dir_bad")}
    detect, lost = [], []
    for s in per_clip:
        for k in c:
            c[k] += s["_counts"][k]
        detect += s["_counts"]["detect"]
        lost += s["_counts"]["lost"]
    return {"seen": _pct(c["seen"], c["inview"]), "side_acc": _pct(c["side_ok"], c["side_n"]),
            "detect_s": round(median(detect), 2) if detect else None,
            "false_out": _pct(c["out_box"], c["out_n"]),
            "flips_min": round(c["flips"] / (c["held_s"] / 60), 2) if c["held_s"] else None,
            "lost_s": round(median(lost), 2) if lost else None,
            "wrong_cue": _pct(c["dir_bad"], c["dir_n"]) if c["dir_n"] else None,
            "stale_cues": sum(s["_counts"]["stale"] for s in per_clip), **split}


LIVE_FPS = 10.0   # the app runs detection at ~9-10 fps on the demo laptop


def _video_frames(path: Path, stride: Optional[int] = None):
    """(fps, frames); every stride-th frame (default: thin to ~LIVE_FPS, like the app)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = stride or max(1, int(round(fps / LIVE_FPS)))

    def gen():
        i = 0
        try:
            while True:
                if i % step:
                    if not cap.grab():
                        break
                    i += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame.shape[1], frame
                i += 1
        finally:
            cap.release()
    return fps, gen()


class _VisionDetect:
    """Vision's own target resolution (COCO class or YOLO-World phrase) and detector."""

    def __init__(self, force_open: bool = False):
        from app.vision import Detector, Vision
        self.vision = Vision(camera=None)
        self.vision.detector = Detector()
        self.min_score = None
        self.force_open = force_open   # every clip through the open-vocabulary model

    def target(self, obj: str) -> Optional[str]:
        if self.force_open:
            from app import vision as V
            t = self.vision._open_phrase(obj)
            self.vision.tracker.set_target(t, V.OPEN_MIN_SCORE, V.OPEN_CONFIRM)
        else:
            t = self.vision.set_target(obj)
        self.min_score = self.vision.tracker.min_score
        self.confirm = getattr(self.vision.tracker, "confirm", 1)
        return t

    def __call__(self, frame, target):
        return self.vision.detector.detect(frame, target)


def score_clips(labels: Path = LABELS, mirror: Optional[bool] = None, detector=None,
                frames_for=None, grace_s: float = GRACE_S, force_open: bool = False) -> dict:
    if not labels.exists():
        return {"skip": f"no clips yet ({labels.relative_to(ROOT) if labels.is_relative_to(ROOT) else labels} missing)"}
    clips = parse_labels(json.loads(labels.read_text()), labels.parent)
    clips = [c for c in clips if c["segments"]]
    if not clips:
        return {"skip": f"no labelled clips in {labels.name}"}
    frames_for = frames_for or _video_frames
    try:
        detector = detector or _VisionDetect(force_open)
    except Exception as e:
        return {"skip": f"detector unavailable ({type(e).__name__}: {e})"}
    from app.vision import COCO_CLASSES, TrackerConfig
    cfg = TrackerConfig()
    per, rows, opens = [], [], []
    for c in clips:
        try:
            vfps, frames = frames_for(c["video"])
        except Exception as e:
            rows.append({"object": c["object"], "error": str(e)})
            continue
        fps = float(c["fps"] or vfps)
        target = detector.target(c["object"]) if hasattr(detector, "target") else c["object"]
        if not target:
            rows.append({"object": c["object"], "error": "vision can't look for it "
                         "(open vocabulary unavailable?)"})
            continue
        m = mirror if mirror is not None else bool(c.get("mirror"))
        recs, cues = run_clip(frames, fps, target, detector, cfg, m)
        s = score_clip(recs, cues, c["segments"], fps, grace_s)
        per.append(s)
        opens.append(force_open or target not in COCO_CLASSES)
        rows.append({"object": c["object"], "target": target,
                     **{k: v for k, v in s.items() if not k.startswith("_")}})
    out = {"clips": rows}
    if per:
        out["all"] = combine(per, opens)
        out["device"] = getattr(getattr(getattr(detector, "vision", None), "detector", None),
                                "device", None)
    return out


# --------------------------------------------------------------------------
# Verdict and report
# --------------------------------------------------------------------------

def _pct(a, n) -> Optional[float]:
    return round(100.0 * a / n, 1) if n else None


def verdict(metrics: dict, targets: dict = TARGETS) -> dict:
    """{metric: "PASS" | "FAIL" | "SKIP"}"""
    out = {}
    for k, (op, thr) in targets.items():
        v = metrics.get(k)
        if v is None:
            out[k] = "SKIP"
        else:
            out[k] = "PASS" if (v >= thr if op == ">=" else v <= thr) else "FAIL"
    return out


UNITS = {"text_acc": "%", "spoken_acc": "%", "wrong_target": "", "seen_coco": "%",
         "seen_open": "%", "side_acc": "%", "wrong_cue": "%", "centered_pps": "/s", "cue_s": " s", "lost_s": " s",
         "flips_min": "/min", "false_out": "%", "det_fps": " fps"}
LABEL = {"text_acc": "A  text match_target", "spoken_acc": "A  spoken (say+whisper)",
         "wrong_target": "A  wrong targets (any)", "seen_coco": "B  seen, COCO (in-view)",
         "seen_open": "B  seen, open-vocab (in-view)", "side_acc": "B  side correct (held)",
         "wrong_cue": "B  wrong-side pulses", "centered_pps": "B  pulses/s held centered", "cue_s": "B  first cue after entering",
         "lost_s": "B  lost cue after leaving", "flips_min": "B  flips/min (held still)",
         "false_out": "B  false detections (out)", "det_fps": "B  detector fps"}


def report(res: dict) -> str:
    a, s, b = res["text"], res["spoken"], res["clips"]
    lines = [f"Voice-to-find scoreboard  ({a['n']} phrasings: {len(TEMPLATES)} templates x "
             f"{len(OBJECTS)} objects)", ""]
    lines.append(f"{'metric':34} {'value':>10} {'target':>10}  verdict")
    for k, (op, thr) in res["targets"].items():
        v = res["metrics"].get(k)
        val = "-" if v is None else f"{v}{UNITS.get(k, '')}"
        lines.append(f"{LABEL.get(k, k):34} {val:>10} {op + str(thr):>10}  {res['verdict'][k]}")
    lines.append("")
    if a["worst_templates"]:
        lines.append("worst templates: " + "; ".join(f"'{t}' {p}%" for t, p in a["worst_templates"]))
    if a["worst_objects"]:
        lines.append("worst objects:   " + "; ".join(f"{o} {p}%" for o, p in a["worst_objects"]))
    if "skip" in s:
        lines.append(f"spoken: SKIP - {s['skip']}")
    else:
        lines.append(f"spoken: {s['ok']}/{s['n']} ({', '.join(f'{v} {p}%' for v, p in s['by_voice'].items())}; "
                     f"decode median {s['decode_median_s']} s)")
        for f in s["fails"][:8]:
            lines.append(f"  [{f['voice']}] said {f['said']!r} heard {f['heard']!r} -> "
                         f"{f['got']!r} (want {f['want']!r})")
    if "skip" in b:
        lines.append(f"clips: SKIP - {b['skip']}")
    else:
        if b.get("device"):
            lines.append(f"clips: detector on {b['device']} (fps target is for MPS; "
                         "HDB_VISION_DEVICE=mps to measure it)")
        for r in b["clips"]:
            if "error" in r:
                lines.append(f"  clip {r['object']}: ERROR {r['error']}")
            else:
                lines.append("  clip {object:12} seen {seen}%  side {side_acc}%  cue {cue_s}s "
                             "(from prompt {cue_prompt_s}s)  "
                             "false {false_out}%  flips {flips_min}/min  lost {lost_s}s  "
                             "wrong {wrong_cue}%  stale cues {stale_cues}".format(**r))
    lines.append("")
    lines.append("RESULT: " + ("PASS" if res["pass"] else "FAIL"))
    return "\n".join(lines)


def evaluate(spoken: int = 40, labels: Path = LABELS, mirror: Optional[bool] = None,
             targets: dict = TARGETS, **kw) -> dict:
    a = score_text()
    s = score_spoken(spoken) if spoken else {"skip": "disabled (--no-spoken)"}
    b = score_clips(labels, mirror, **kw)
    metrics = {"text_acc": a["acc"], "spoken_acc": s.get("acc"),
               "wrong_target": a["wrong"] + s.get("wrong", 0)}
    metrics.update(b.get("all", {}))
    v = verdict(metrics, targets)
    return {"metrics": metrics, "targets": targets, "verdict": v,
            "pass": "FAIL" not in v.values(), "text": a, "spoken": s, "clips": b}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spoken", type=int, default=40, help="spoken phrases (x voices)")
    ap.add_argument("--no-spoken", action="store_true")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--mirror", action="store_true", default=None,
                    help="clips from a webcam facing you (default: labels.json 'mirror')")
    ap.add_argument("--grace", type=float, default=GRACE_S,
                    help="seconds excused at each segment start")
    ap.add_argument("--open", action="store_true",
                    help="run every clip through the open-vocabulary model (scores seen_open)")
    ap.add_argument("--targets", type=Path, help="JSON {metric: [op, threshold]}")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    targets = dict(TARGETS)
    if a.targets:
        targets.update({k: tuple(v) for k, v in json.loads(a.targets.read_text()).items()})
    import contextlib
    # --json: keep stdout pure JSON ([vision] load logs go to stderr)
    quiet = contextlib.redirect_stdout(sys.stderr) if a.json else contextlib.nullcontext()
    with quiet:
        res = evaluate(0 if a.no_spoken else a.spoken, a.labels, a.mirror, targets,
                       grace_s=a.grace, force_open=a.open)
    if a.json:
        print(json.dumps(res, indent=1, default=str))
    else:
        print(report(res))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
