"""Guided recorder: labelled obstacle clips for tools/eval_avoid.py.

Run from Terminal.app (camera permission), camera at chest height facing AWAY
from you, like the band:
    .venv/bin/python -m tools.record_avoid --camera 0
    .venv/bin/python -m tools.record_avoid --scenes table,person

For each scene it shows big on-screen instructions and records ~30 s:
    FREE (open path) -> NEAR (~2 m ahead) -> BLOCKED (under 1 m, dead ahead)
    -> SIDE LEFT (obstacle on the left, path clear) -> SIDE RIGHT -> FREE
Walk the camera toward the obstacle, or have a teammate step in front of it.
Press SPACE when ready; q skips a scene.

Output (gitignored): recordings/avoid/<scene>.mp4 plus labels.json, the same
format as recordings/eval (f0/f1 half-open frame ranges).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "recordings" / "avoid"
SCENES = ["table", "chair", "person"]
PLAN = [("free", 5, "Point at an OPEN path (nothing within 3 m)"),
        ("near", 5, "The {s} about 2 m AHEAD"),
        ("blocked", 5, "The {s} RIGHT IN FRONT (under 1 m)"),
        ("side_left", 5, "The {s} on the LEFT, path ahead clear"),
        ("side_right", 5, "The {s} on the RIGHT, path ahead clear"),
        ("free", 4, "OPEN path again")]


def banner(img, text, sub=""):
    import cv2
    for t, y, s in ((text, 50, 1.0), (sub, 90, 0.7)):
        if t:
            cv2.putText(img, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 5)
            cv2.putText(img, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2)
    return img


def main(argv=None) -> None:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--scenes", default=",".join(SCENES), help="comma-separated")
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = OUT_DIR / "labels.json"
    labels = {"camera": a.camera, "mirror": False, "clips": []}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
    cap = cv2.VideoCapture(a.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera} (camera permission for Terminal?)")
    win = "record obstacle clips (SPACE start, q skip)"

    def frame():
        ok, f = cap.read()
        if not ok:
            raise SystemExit("camera stopped delivering frames")
        return f

    for scene in [s.strip() for s in a.scenes.split(",") if s.strip()]:
        while True:
            cv2.imshow(win, banner(frame().copy(), f"Next: {scene}", "SPACE to start (q = skip)"))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord(" "), ord("q")):
                break
        if k == ord("q"):
            continue
        path = OUT_DIR / f"{scene.replace(' ', '_')}.mp4"
        frames, segs, n, t_start = [], [], 0, time.monotonic()
        for label, secs, text in PLAN:
            f0, t_end = n, time.monotonic() + secs
            while time.monotonic() < t_end:
                f = frame()
                frames.append(f)
                n += 1
                cv2.imshow(win, banner(f.copy(), text.format(s=scene),
                                       f"{t_end - time.monotonic():.1f}s   (REC)"))
                cv2.waitKey(1)
            segs.append({"label": label, "f0": f0, "f1": n})
        fps = n / (time.monotonic() - t_start)
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        labels["clips"] = [c for c in labels["clips"] if c["object"] != scene]
        labels["clips"].append({"object": scene, "file": path.name, "fps": round(fps, 2),
                                "segments": segs})
        labels_path.write_text(json.dumps(labels, indent=1))
        print(f"saved {path}  ({n} frames, {fps:.1f} fps)")
    cap.release()
    cv2.destroyAllWindows()
    print(f"labels: {labels_path}\nscore: python -m tools.eval_avoid")


if __name__ == "__main__":
    main()
