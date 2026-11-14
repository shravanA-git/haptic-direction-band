"""Generate the two hand-made reference traces in tests/fixtures/.

  back_left.json  a call from back-left        -> expected motor 5
  unclear.json    loud but conflicting sides   -> expected generic attention

Deterministic (fixed seed). Run: python -m tools.make_fixtures
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FRAME_MS = 50
QUIET = [20, 18, 22, 19]          # front, right, back, left: unequal gains on purpose


def _trace(n: int, alert: tuple, loud_fn, seed: int) -> dict:
    rnd = random.Random(seed)
    frames = []
    for i in range(n):
        ms = i * FRAME_MS
        lv = [q + rnd.uniform(-2, 2) for q in QUIET]
        if alert[0] <= ms <= alert[1]:
            lv = [v + a for v, a in zip(lv, loud_fn(i, rnd))]
        frames.append({"ms": ms, "levels": [round(max(v, 0)) for v in lv]})
    return {"frames": frames, "alerts": [{"startMs": alert[0], "endMs": alert[1]}],
            "expected": None}


def back_left() -> dict:
    # Name spoken 1.00-1.45 s from behind-left: back and left loud, others a bit.
    t = _trace(50, (1000, 1450),
               lambda i, r: [6 + r.uniform(0, 3), 4 + r.uniform(0, 3),
                             60 + r.uniform(-8, 8), 55 + r.uniform(-8, 8)], seed=5)
    t["expected"] = {"motor": 5}
    return t


def unclear() -> dict:
    # Loud, but the dominant side flips every frame (reverb / two talkers).
    sides = [[60, 0, 0, 0], [0, 0, 60, 0], [0, 60, 0, 0], [0, 0, 0, 60]]
    t = _trace(50, (1000, 1450),
               lambda i, r: [s + r.uniform(0, 5) for s in sides[i % 4]], seed=9)
    t["expected"] = {"motor": None, "cue": "generic"}
    return t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("back_left", back_left), ("unclear", unclear)]:
        p = OUT / f"{name}.json"
        p.write_text(json.dumps(fn(), separators=(",", ":")) + "\n")
        print("wrote", p)
