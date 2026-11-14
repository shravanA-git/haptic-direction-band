"""Scoreboard for app/sound_classes.py (common sounds via YAMNet) on recorded clips.

    python -m tools.eval_sounds                       # ESC-50 subset + spoken phrases
    python -m tools.eval_sounds --dir recordings/sounds/mine   # your own recordings
    python -m tools.eval_sounds --threshold 0.25

Data (all under recordings/, gitignored):
  * recordings/sounds/esc50/: ESC-50 clips (5 s, real recordings, CC BY-NC) with
    its esc50.csv. Categories in ESC_MAP must fire that label; every other
    category is a household distractor that must not fire anything.
  * recordings/sounds/speech/: phrases from macOS `say` (made on first run);
    must not fire.
  * --dir D: D/<label>/*.wav must fire <label>; D/none/*.wav must not fire.

One YAMNet pass per clip keeps every window's group scores; each threshold is
then replayed with the classifier's own rule (sound_classes.pick: one sound per
window, COOLDOWN_S per sound), so the sweep matches what the band would buzz.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

import numpy as np

from app import sound_classes as sc_mod

ROOT = Path(__file__).resolve().parent.parent
SOUNDS = ROOT / "recordings" / "sounds"
ESC_MAP = {"door_wood_knock": "knock", "clock_alarm": "alarm", "crying_baby": "baby crying",
           "dog": "dog", "siren": "siren", "car_horn": "car horn",
           "glass_breaking": "glass breaking"}
PHRASES = ["Hey, are you coming to dinner?", "The meeting starts at three o'clock.",
           "Can you pass me the remote?", "I think it's going to rain later today.",
           "Knock knock, who's there?", "Did you hear the doorbell?",
           "The dog needs to go for a walk.", "Turn the alarm off, please.",
           "Let's watch a movie tonight.", "Where did I leave my keys?"]
SWEEP = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def esc50_clips(d: Path) -> list:
    meta = d / "esc50.csv"
    if not meta.exists():
        return []
    out = []
    for r in csv.DictReader(meta.open()):
        f = d / r["filename"]
        if f.exists():
            out.append((f, ESC_MAP.get(r["category"]), r["category"]))
    return out


def speech_clips(d: Path) -> list:
    d.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(PHRASES):
        f = d / f"say-{i:02d}.wav"
        if not f.exists():
            subprocess.run(["say", "-o", str(f), "--file-format=WAVE",
                            "--data-format=LEI16@16000", text], check=True)
    return [(f, None, "speech (say)") for f in sorted(d.glob("say-*.wav"))]


def dir_clips(d: Path) -> list:
    return [(f, None if sub.name == "none" else sub.name, f"{d.name}/{sub.name}")
            for sub in sorted(p for p in d.iterdir() if p.is_dir())
            for f in sorted(sub.glob("*.wav"))]


def window_scores(clips: list) -> tuple:
    """Per clip: [(t, {label: score})] for every window; plus windows/s."""
    y = sc_mod.Yamnet()
    rows, n, t0 = [], 0, time.perf_counter()
    for f, want, group in clips:
        x, sr = sc_mod.load_wav(f)
        wins: list = []
        c = sc_mod.SoundClassifier(sample_rate=sr, model=y)
        c.on_window = lambda t, g, wins=wins: wins.append((t, g))
        c.sounds = {k: (v[0], v[1], 9.9) for k, v in c.sounds.items()}   # no hits here
        c.feed(np.concatenate([x, np.zeros(sr, np.float32)]), t_first=0.0)
        n += len(wins)
        rows.append((f, want, group, wins))
    return rows, n / max(1e-9, time.perf_counter() - t0)


def alerts(wins: list, thr) -> list:
    """Labels the classifier would fire, in order (same rule and cooldown)."""
    th = {k: (thr if isinstance(thr, (int, float)) else thr(k)) for k in sc_mod.COMMON}
    out, last = [], {}
    for t, g in wins:
        k = sc_mod.pick(g, th)
        if k and t - last.get(k, -1e9) >= sc_mod.COOLDOWN_S:
            last[k] = t
            out.append(k)
    return out


def fired(wins: list, thr) -> set:
    return set(alerts(wins, thr))


def score(rows: list, thr) -> dict:
    pos = [r for r in rows if r[1]]
    neg = [r for r in rows if not r[1]]
    first = [alerts(r[3], thr)[:1] for r in pos]
    hit = sum(r[1] in fired(r[3], thr) for r in pos)
    wrong = sum(bool(fired(r[3], thr) - {r[1]}) for r in pos)
    false = sum(bool(fired(r[3], thr)) for r in neg)
    return {"recall": hit / max(1, len(pos)), "wrong_label": wrong / max(1, len(pos)),
            "first_ok": sum(f == [r[1]] for f, r in zip(first, pos)) / max(1, len(pos)),
            "false_alert": false / max(1, len(neg)), "n_pos": len(pos), "n_neg": len(neg)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", action="append", default=[], help="label/*.wav folders")
    p.add_argument("--no-esc50", action="store_true")
    p.add_argument("--no-speech", action="store_true")
    p.add_argument("--threshold", type=float, default=None,
                   help="one threshold for every sound (default: each sound's own)")
    a = p.parse_args(argv)

    clips = [] if a.no_esc50 else esc50_clips(SOUNDS / "esc50")
    if not a.no_speech:
        clips += speech_clips(SOUNDS / "speech")
    for d in a.dir:
        clips += dir_clips(Path(d))
    if not clips:
        raise SystemExit("no clips: see the docstring for where to put them")
    rows, wps = window_scores(clips)
    print(f"{len(rows)} clips ({sum(1 for r in rows if r[1])} must fire, "
          f"{sum(1 for r in rows if not r[1])} must not), {wps:.0f} windows/s on CPU\n")

    print("threshold  recall  first-alert-right  also-a-wrong-label  false-alert")
    for t in SWEEP:
        s = score(rows, t)
        print(f"  {t:4.2f}     {s['recall']:5.0%}      {s['first_ok']:5.0%}             "
              f"{s['wrong_label']:5.0%}             {s['false_alert']:5.0%}")
    own = {k: v[2] for k, v in sc_mod.COMMON.items()}
    thr = a.threshold if a.threshold is not None else own.get
    s = score(rows, thr)
    print(f"  per-sound thresholds: recall {s['recall']:.0%}, first alert right "
          f"{s['first_ok']:.0%}, wrong label {s['wrong_label']:.0%}, "
          f"false alert {s['false_alert']:.0%}\n" if a.threshold is None else "")

    print(f"per group at {'each sound' if a.threshold is None else a.threshold}'s threshold:")
    groups: dict = {}
    for f, want, group, wins in rows:
        groups.setdefault((group, want), []).append(fired(wins, thr))
    for (group, want), fs in sorted(groups.items(), key=lambda kv: (kv[0][1] is None, kv[0][0])):
        if want:
            ok = sum(want in f for f in fs)
            others = sorted({k for f in fs for k in f - {want}})
            print(f"  {group:18s} -> {want:15s} {ok:2d}/{len(fs)}"
                  + (f"   also fired: {', '.join(others)}" if others else ""))
        else:
            bad = [k for f in fs for k in f]
            print(f"  {group:18s} (must not)      {sum(bool(f) for f in fs):2d}/{len(fs)} false"
                  + (f"   as: {', '.join(sorted(set(bad)))}" if bad else ""))


if __name__ == "__main__":
    main()
