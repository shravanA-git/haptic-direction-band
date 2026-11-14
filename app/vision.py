"""Front-webcam target finder: "find the banana" -> left / right / centered / lost.

Owner: HDB-Vision (branch ``vision``). This module never touches the serial
port; it only reports decisions. HDB-Software's cue logic turns them into
PULSE/STOP.

Interface for app/main.py and app/haptics.py
-------------------------------------------
    from app.vision import Vision

    v = Vision(camera=0)                 # opens nothing until start()
    v.on_change(callback)                # callback(decision), called from the vision thread
    v.start()                            # False + v.error if camera/detector failed (no raise)
    v.set_target("banana")               # typed name; returns the COCO class or raises ValueError
    d = v.latest()                       # most recent VisionDecision (poll instead of callback)
    v.pause(); v.resume()                # e.g. while Action is held
    v.stop()

``VisionDecision`` fields:
    state       "searching" (target set, never seen yet), "left", "right",
                "centered", "lost" (was seen, gone for > lost_after_s), "idle"
                (no target / paused)
    offset      target center, -1.0 (left edge) .. +1.0 (right edge), smoothed; None if not visible
    confidence  detector score of the tracked box (0..1), 0 if not visible
    bbox        (x1, y1, x2, y2) in pixels or None
    target      COCO class being tracked, e.g. "banana"
    t           time.monotonic() of the frame the decision came from
    changed     True when ``state`` differs from the previous decision

Callbacks fire only when ``state`` changes. Left/right are from the wearer's
point of view for a forward-facing camera (no mirroring). For a laptop webcam
facing you, pass ``mirror=True``.

Centered uses a dead band with hysteresis: enter centered when
|offset| <= dead_band, leave it when |offset| > dead_band_exit.

Running it
----------
    pip install ultralytics opencv-python
    python -m app.vision --target banana --show          # webcam 0, live window
    python -m app.vision --target banana --source clip.mp4
    python -m app.vision --check                         # is this computer ready? camera, mic, weights
    python -m app.vision --selftest                      # tracker logic, no camera/weights
    python -m app.vision --target banana --show --log recordings/banana.csv   # tuning run
        (also --dead-band, --dead-band-exit, --lost-after, --smoothing, --min-score)
    python -m app.vision --analyze recordings/banana.csv  # gaps, jitter, suggested settings

Weights (yolo11s.pt, ~18 MB; HDB_YOLO_WEIGHTS to override) download at runtime into
models/ (gitignored). Tuned on Sophanit's live banana run (Nov 13): yolov8n@416 saw
the moving banana in ~25% of frames, median score 0.41, so: bigger model, 640 px,
lower score floors, longer lost timeout. Offset jitter was tiny (p95 0.012).
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

# --------------------------------------------------------------------------
# Target names
# --------------------------------------------------------------------------

COCO_CLASSES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush"
).split()
COCO_CLASSES = tuple(c.replace("_", " ") for c in COCO_CLASSES)

SYNONYMS = {
    "phone": "cell phone", "cellphone": "cell phone", "mobile": "cell phone",
    "smartphone": "cell phone", "iphone": "cell phone",
    "mug": "cup", "glass": "cup", "water bottle": "bottle",
    "sofa": "couch", "table": "dining table", "desk": "dining table",
    "television": "tv", "monitor": "tv", "screen": "tv",
    "computer": "laptop", "notebook": "laptop",
    "bag": "backpack", "purse": "handbag", "ball": "sports ball",
    "plant": "potted plant", "teddy": "teddy bear", "hairdryer": "hair drier",
    "people": "person", "someone": "person", "man": "person", "woman": "person",
    "doughnut": "donut", "fridge": "refrigerator", "bike": "bicycle",
}


def resolve_target(name: str) -> Optional[str]:
    """Map a typed or spoken target ("the bananas", "my phone") to a COCO class."""
    s = " ".join(name.lower().replace("_", " ").replace("-", " ").split())
    for prefix in ("find the ", "find my ", "find a ", "find ", "the ", "my ", "a ", "an "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    candidates = [s]
    if s.endswith("es"):
        candidates.append(s[:-2])
    if s.endswith("s"):
        candidates.append(s[:-1])
    for c in candidates:
        if c in COCO_CLASSES:
            return c
        if c in SYNONYMS:
            return SYNONYMS[c]
    return None


# --------------------------------------------------------------------------
# Tracker: pure logic, no camera, no model (unit-testable)
# --------------------------------------------------------------------------

Box = tuple  # (x1, y1, x2, y2, score)


@dataclass(frozen=True)
class VisionDecision:
    state: str
    offset: Optional[float]
    confidence: float
    bbox: Optional[tuple]
    target: Optional[str]
    t: float
    changed: bool = False


@dataclass
class TrackerConfig:
    dead_band: float = 0.12        # |offset| <= this enters "centered"
    dead_band_exit: float = 0.18   # |offset| > this leaves "centered"
    lost_after_s: float = 1.2      # no detection this long -> "lost" (bridges motion-blur misses)
    smoothing: float = 0.5         # EMA weight of the newest offset (1.0 = no smoothing)
    max_jump: float = 0.6          # prefer boxes within this offset distance of the last one
    min_score: float = 0.25
    min_dwell_s: float = 0.3       # a new side must hold this long before it's reported


class TargetTracker:
    """Turns per-frame detections of one class into left/right/centered/lost."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.cfg = config or TrackerConfig()
        self.target: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self._offset: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._ever_seen = False
        self._cand: Optional[str] = None     # pending side change (dwell)
        self._cand_since = 0.0
        self._state = "idle" if self.target is None else "searching"

    def set_target(self, target: Optional[str]) -> None:
        self.target = target
        self._reset()

    @property
    def state(self) -> str:
        return self._state

    def update(self, boxes: Sequence[Box], frame_width: int, t: float,
               mirror: bool = False) -> VisionDecision:
        prev = self._state
        if self.target is None:
            self._state = "idle"
            return VisionDecision("idle", None, 0.0, None, None, t, prev != "idle")

        cands = []
        for b in boxes:
            x1, y1, x2, y2, score = b[:5]
            if score < self.cfg.min_score:
                continue
            cx = (x1 + x2) / 2.0
            off = (cx / frame_width) * 2.0 - 1.0
            if mirror:
                off = -off
            cands.append((off, score, (x1, y1, x2, y2)))

        best = None
        if cands:
            if self._offset is not None:
                near = [c for c in cands if abs(c[0] - self._offset) <= self.cfg.max_jump]
                pool = near or cands
                # Stick with the same object: nearest first, score breaks ties.
                best = min(pool, key=lambda c: (abs(c[0] - self._offset), -c[1]))
            else:
                best = max(cands, key=lambda c: c[1])

        if best is not None:
            off, score, bbox = best
            a = self.cfg.smoothing
            self._offset = off if self._offset is None else a * off + (1 - a) * self._offset
            self._last_seen = t
            self._ever_seen = True
            o = self._offset
            if prev == "centered":
                centered = abs(o) <= self.cfg.dead_band_exit
            else:
                centered = abs(o) <= self.cfg.dead_band
            new = "centered" if centered else ("left" if o < 0 else "right")
            # Dwell: a side change must hold min_dwell_s before it is reported, so a
            # target swept through the middle doesn't fire left/centered/right cues
            # 0.2 s apart. First acquisition (from searching/lost) is immediate.
            if prev in ("left", "right", "centered"):
                if new == prev:
                    self._cand = None              # a blip back cancels the pending change
                else:
                    if self._cand != new:
                        self._cand, self._cand_since = new, t
                    if t - self._cand_since < self.cfg.min_dwell_s:
                        new = prev
                    else:
                        self._cand = None
            self._state = new
            return VisionDecision(self._state, round(o, 3), float(score), bbox,
                                  self.target, t, self._state != prev)

        # Nothing this frame: hold the last state briefly (detector flicker), then lost.
        if self._ever_seen and self._last_seen is not None:
            if t - self._last_seen > self.cfg.lost_after_s:
                self._state = "lost"
                self._offset = None
        else:
            self._state = "searching"
        return VisionDecision(self._state, None if self._state in ("lost", "searching") else
                              (round(self._offset, 3) if self._offset is not None else None),
                              0.0, None, self.target, t, self._state != prev)


# --------------------------------------------------------------------------
# Detector: ultralytics YOLO, weights downloaded at runtime
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_WEIGHTS = os.environ.get("HDB_YOLO_WEIGHTS", "yolo11s.pt")
DEFAULT_IMGSZ = 640
DETECT_CONF = 0.15  # detector floor; the tracker applies TrackerConfig.min_score


class Detector:
    def __init__(self, weights: str = None, imgsz: int = None,
                 device: Optional[str] = None):
        from ultralytics import YOLO  # imported lazily so the tracker works without it

        weights = weights or DEFAULT_WEIGHTS
        imgsz = imgsz or DEFAULT_IMGSZ

        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / weights
        if not path.exists():
            # ultralytics downloads known asset names into the cwd; do it inside models/.
            cwd = os.getcwd()
            try:
                os.chdir(MODELS_DIR)
                YOLO(weights)
            finally:
                os.chdir(cwd)
        self.model = YOLO(str(path))
        self._lock = threading.Lock()
        self.imgsz = imgsz
        if device is None:
            try:
                import torch
                device = "mps" if torch.backends.mps.is_available() else (
                    "cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                device = "cpu"
        self.device = device
        names = self.model.names
        self.class_ids = {(n.replace("_", " ")): i for i, n in names.items()}

    def detect_all(self, frame, min_score: float = 0.3) -> list:
        """Every COCO object in the frame: [(class, x1, y1, x2, y2, score)], best first."""
        with self._lock:  # shared by the vision thread and agent tools
            res = self.model.predict(frame, imgsz=self.imgsz, conf=min_score,
                                     device=self.device, verbose=False)[0]
        names = self.model.names
        out = [(names[int(c)].replace("_", " "), *xyxy, float(s)) for xyxy, c, s in
               zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist())]
        return sorted(out, key=lambda o: -o[5])

    def detect(self, frame, target: str) -> list:
        cid = self.class_ids.get(target)
        if cid is None:
            return []
        with self._lock:
            res = self.model.predict(frame, imgsz=self.imgsz, classes=[cid], conf=DETECT_CONF,
                                     device=self.device, verbose=False)[0]
        out = []
        for xyxy, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            out.append((*xyxy, float(conf)))
        return out


# --------------------------------------------------------------------------
# Vision: camera thread + detector + tracker
# --------------------------------------------------------------------------

class Vision:
    def __init__(self, camera=0, mirror: bool = False, weights: str = None,
                 imgsz: int = None, config: Optional[TrackerConfig] = None,
                 max_fps: float = 15.0, log_path: Optional[str] = None,
                 record_path: Optional[str] = None):
        self.camera = camera
        self.log_path = log_path  # per-frame CSV for tuning dead band / lost timing
        self.record_path = record_path  # raw video, to replay with --source later
        self.mirror = mirror
        self.weights = weights
        self.imgsz = imgsz
        self.max_fps = max_fps
        self.tracker = TargetTracker(config)
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[VisionDecision], None]] = []
        self._latest = VisionDecision("idle", None, 0.0, None, None, time.monotonic())
        self._frame = None
        self._raw = None
        self._paused = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.detector: Optional[Detector] = None
        self.fps = 0.0
        self.error: Optional[str] = None

    # ---- public API ----
    def on_change(self, cb: Callable[[VisionDecision], None]) -> None:
        self._callbacks.append(cb)

    def set_target(self, name: Optional[str]) -> Optional[str]:
        target = None
        if name:
            target = resolve_target(name)
            if target is None:
                raise ValueError(f"unknown target {name!r}; try one of the COCO classes")
        with self._lock:
            self.tracker.set_target(target)
        self._publish(VisionDecision(self.tracker.state, None, 0.0, None, target,
                                     time.monotonic(), True))
        return target

    def latest(self) -> VisionDecision:
        return self._latest

    def latest_frame(self):
        """Last annotated BGR frame (for the dashboard), or None."""
        return self._frame

    def latest_raw(self):
        """Last unannotated BGR frame (for scene queries such as the find agent), or None."""
        return self._raw

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        with self._lock:
            self.tracker.set_target(self.tracker.target)  # fresh start, no stale direction
        self._paused = False

    def start(self, strict: bool = False) -> bool:
        """Load the detector and open the camera. On failure, sets ``self.error``,
        prints it and returns False (raises RuntimeError if strict) so the app and
        dashboard keep running without vision."""
        if self._running:
            return True
        try:
            self.detector = Detector(self.weights, self.imgsz)
        except Exception as e:
            return self._fail(f"detector failed to load ({type(e).__name__}: {e}); "
                              "pip install ultralytics opencv-python, and be online on first run "
                              "(weights download into models/)", strict)
        # Open on the caller's (main) thread: macOS only shows the camera
        # permission prompt from the main run loop.
        cap = self._open()
        if not cap.isOpened():
            cap.release()
            return self._fail(f"cannot open camera {self.camera!r} (macOS: allow camera access "
                              "for your terminal app in System Settings > Privacy & Security > "
                              "Camera, then quit and reopen the terminal; or try --camera 1)",
                              strict)
        self.error = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, args=(cap,), name="vision",
                                        daemon=True)
        self._thread.start()
        return True

    def _fail(self, msg: str, strict: bool) -> bool:
        self.error = msg
        print(f"[vision] OFF: {msg}")
        if strict:
            raise RuntimeError(msg)
        return False

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ---- internals ----
    def _publish(self, d: VisionDecision) -> None:
        self._latest = d
        if d.changed:
            for cb in list(self._callbacks):
                try:
                    cb(d)
                except Exception as e:  # never let a consumer kill the camera loop
                    print(f"[vision] callback error: {e}")

    def _open(self):
        import cv2
        src = self.camera
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        if isinstance(src, int) and os.uname().sysname == "Darwin":
            cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def _loop(self, cap) -> None:
        period = 1.0 / self.max_fps if self.max_fps else 0
        last = time.monotonic()
        is_file = isinstance(self.camera, str) and not self.camera.isdigit()
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w") if self.log_path else None
        rec = None
        if log:
            log.write("t,target,n_boxes,raw_offset,raw_score,offset,confidence,state\n")
        try:
            while self._running:
                ok, frame = cap.read()
                t = time.monotonic()
                if not ok:
                    if is_file:
                        break
                    time.sleep(0.02)
                    continue
                self._raw = frame
                if self._paused:
                    self._frame = frame
                    if self._latest.state != "idle":
                        self._publish(replace(self._latest, state="idle", t=t, changed=True,
                                              offset=None, bbox=None, confidence=0.0))
                    time.sleep(0.03)
                    continue
                if self.record_path:
                    if rec is None:
                        import cv2
                        Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
                        rec = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                              self.max_fps or 15, (frame.shape[1], frame.shape[0]))
                    rec.write(frame)
                with self._lock:
                    target = self.tracker.target
                boxes = self.detector.detect(frame, target) if target else []
                with self._lock:
                    d = self.tracker.update(boxes, frame.shape[1], t, self.mirror)
                self._frame = annotate(frame, d, self.tracker.cfg, self.mirror)
                self._publish(d)
                if log:
                    raw = _raw_best(boxes, frame.shape[1], self.mirror)
                    log.write(f"{t:.3f},{target or ''},{len(boxes)},"
                              f"{'' if raw is None else f'{raw[0]:.3f}'},"
                              f"{'' if raw is None else f'{raw[1]:.2f}'},"
                              f"{'' if d.offset is None else d.offset},"
                              f"{d.confidence:.2f},{d.state}\n")
                dt = time.monotonic() - last
                self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt if dt > 0 else 0)
                last = time.monotonic()
                if period and dt < period:
                    time.sleep(period - dt)
        finally:
            cap.release()
            if log:
                log.close()
            if rec is not None:
                rec.release()
            self._running = False


def _raw_best(boxes, width, mirror=False):
    """(offset, score) of the highest-scoring raw box, unsmoothed; for tuning logs."""
    if not boxes:
        return None
    x1, _, x2, _, score = max(boxes, key=lambda b: b[4])[:5]
    off = (x1 + x2) / width - 1.0
    return (-off if mirror else off, score)


def annotate(frame, d: VisionDecision, cfg: TrackerConfig, mirror: bool = False):
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]
    # dead band
    for edge in (-cfg.dead_band, cfg.dead_band):
        x = int((edge * (-1 if mirror else 1) + 1) / 2 * w)
        cv2.line(img, (x, 0), (x, h), (200, 200, 200), 1)
    if d.bbox is not None:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    colors = {"centered": (0, 200, 0), "left": (0, 200, 255), "right": (0, 200, 255),
              "lost": (0, 0, 255), "searching": (200, 200, 0), "idle": (160, 160, 160)}
    label = f"{d.target or '-'}: {d.state.upper()}"
    if d.offset is not None:
        label += f"  off={d.offset:+.2f}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                colors.get(d.state, (255, 255, 255)), 2)
    return img


# --------------------------------------------------------------------------
# Self-test (tracker logic only) and CLI
# --------------------------------------------------------------------------

def _selftest() -> None:
    W = 640

    def box(cx, score=0.9):
        return (cx - 30, 100, cx + 30, 200, score)

    assert resolve_target("banana") == "banana"
    assert resolve_target("Find the bananas") == "banana"
    assert resolve_target("my phone") == "cell phone"
    assert resolve_target("unicorn") is None

    tr = TargetTracker(TrackerConfig(smoothing=1.0, lost_after_s=0.8, min_dwell_s=0.0))
    assert tr.update([box(100)], W, 0.0).state == "idle"
    tr.set_target("banana")
    d = tr.update([], W, 0.0)
    assert d.state == "searching", d
    d = tr.update([box(100)], W, 0.1)
    assert d.state == "left" and d.changed and d.offset < 0, d
    d = tr.update([box(540)], W, 0.2)
    assert d.state == "right", d
    d = tr.update([box(320)], W, 0.3)
    assert d.state == "centered" and d.changed, d
    # hysteresis: offset 0.15 is outside dead_band but inside dead_band_exit
    d = tr.update([box(320 + 0.15 * 320)], W, 0.4)
    assert d.state == "centered" and not d.changed, d
    d = tr.update([box(320 + 0.25 * 320)], W, 0.5)
    assert d.state == "right", d
    # flicker: one missing frame keeps state
    d = tr.update([], W, 0.9)
    assert d.state == "right" and not d.changed, d
    d = tr.update([], W, 1.4)
    assert d.state == "lost" and d.changed, d
    d = tr.update([], W, 2.0)
    assert d.state == "lost" and not d.changed, d
    d = tr.update([box(320)], W, 2.1)
    assert d.state == "centered" and d.changed, d
    # sticks to the nearby object rather than jumping to a higher-score far one
    d = tr.update([box(330, 0.5), box(600, 0.95)], W, 2.2)
    assert d.state == "centered", d
    # low scores ignored
    tr.set_target("banana")
    d = tr.update([box(100, 0.1)], W, 3.0)
    assert d.state == "searching", d
    # mirror flips sides
    d = tr.update([box(100)], W, 3.1, mirror=True)
    assert d.state == "right", d
    # dwell: a quick sweep through the middle does not flip states
    tr = TargetTracker(TrackerConfig(smoothing=1.0, min_dwell_s=0.3))
    tr.set_target("banana")
    assert tr.update([box(100)], W, 0.0).state == "left"          # first sight: immediate
    assert tr.update([box(320)], W, 0.1).state == "left"          # centered for 0.1 s: held
    assert tr.update([box(540)], W, 0.2).state == "left"          # right just started
    assert tr.update([box(540)], W, 0.4).state == "left"          # right for 0.2 s
    d = tr.update([box(540)], W, 0.55)
    assert d.state == "right" and d.changed, d                     # right held 0.35 s
    assert tr.update([box(320)], W, 0.6).state == "right"
    assert tr.update([box(540)], W, 0.7).state == "right"         # blip cleared the candidate
    assert tr.update([box(320)], W, 0.8).state == "right"
    assert tr.update([box(320)], W, 1.0).state == "right"         # 0.2 s since 0.8, not 0.6
    assert tr.update([box(320)], W, 1.15).state == "centered"
    print("vision selftest: OK")


def analyze(path: str) -> dict:
    """Summarize a --log CSV: detection gaps (-> lost_after_s), raw offset jitter
    (-> dead band hysteresis) and how often the state flips."""
    import csv
    import statistics as st
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 2:
        return {"frames": len(rows)}
    ts = [float(r["t"]) for r in rows]
    gaps, start = [], None
    for r, t in zip(rows, ts):
        seen = int(r["n_boxes"]) > 0
        if not seen and start is None:
            start = t
        elif seen and start is not None:
            gaps.append(t - start)
            start = None
    raws = [float(r["raw_offset"]) if r["raw_offset"] else None for r in rows]
    jitter = [abs(a - b) for a, b in zip(raws, raws[1:]) if a is not None and b is not None]
    flips = sum(1 for a, b in zip(rows, rows[1:]) if a["state"] != b["state"])
    dur = ts[-1] - ts[0]

    def pct(xs, q):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None

    short = [g for g in gaps if g < 3.0]  # longer gaps are the target really leaving
    out = {
        "frames": len(rows), "seconds": round(dur, 1),
        "fps": round((len(rows) - 1) / dur, 1) if dur else None,
        "seen_frac": round(sum(int(r["n_boxes"]) > 0 for r in rows) / len(rows), 2),
        "flicker_gaps_s": {"n": len(short), "p50": pct(short, .5), "p95": pct(short, .95),
                           "max": round(max(short), 3) if short else None},
        "raw_jitter": {"p50": pct(jitter, .5), "p95": pct(jitter, .95)},
        "state_flips_per_min": round(flips / dur * 60, 1) if dur else None,
        "states": {k: sum(r["state"] == k for r in rows) for k in
                   ("left", "right", "centered", "lost", "searching")},
    }
    if short:
        out["suggest_lost_after_s"] = round(max(0.5, 1.5 * pct(short, .95)), 2)
    if jitter:
        out["suggest_hysteresis"] = round(min(0.12, max(0.04, 2 * pct(jitter, .9))), 2)
    return out


def check(camera=0) -> bool:
    """Is this computer ready for vision + speech? Prints one line per item with a fix."""
    import importlib
    import socket
    import sys
    ok_all = True

    def line(ok, what, fix=""):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"  [{'ok' if ok else 'FAIL'}] {what}" + ("" if ok or not fix else f"\n         fix: {fix}"))

    print(f"python: {sys.executable} ({sys.version.split()[0]})")
    pip = f"{sys.executable} -m pip install -r requirements.txt"
    for mod in ("aiohttp", "serial", "numpy", "cv2", "ultralytics", "torch",
                "faster_whisper", "sounddevice"):
        try:
            m = importlib.import_module(mod)
            line(True, f"import {mod} {getattr(m, '__version__', '')}")
        except Exception as e:
            line(False, f"import {mod}: {e}", pip)
    try:
        det = Detector()
        line(True, f"detector {DEFAULT_WEIGHTS}@{det.imgsz} on {det.device}")
    except Exception as e:
        line(False, f"detector: {e}", "be online once so weights download into models/")
    try:
        import cv2
        v = Vision(camera=camera)
        cap = v._open()
        got = cap.isOpened() and cap.read()[0]
        shape = cap.read()[1].shape if got else None
        cap.release()
        line(got, f"camera {camera}" + (f" {shape[1]}x{shape[0]}" if shape else " not readable"),
             "System Settings > Privacy & Security > Camera: enable your terminal app, "
             "quit and reopen it; or try --camera 1")
    except Exception as e:
        line(False, f"camera: {e}")
    try:
        import numpy as np
        import sounddevice as sd
        rec = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(rec ** 2)))
        line(rms > 1e-5, f"microphone '{sd.query_devices(kind='input')['name']}' rms={rms:.5f}",
             "System Settings > Privacy & Security > Microphone: enable your terminal app "
             "(rms 0 means macOS is feeding silence)")
    except Exception as e:
        line(False, f"microphone: {e}")
    try:
        from .speech import Transcriber
        Transcriber().load()
        line(True, "whisper tiny.en loaded")
    except Exception as e:
        line(False, f"whisper: {e}", "be online once so weights download into models/whisper/")
    with socket.socket() as so:
        free = so.connect_ex(("127.0.0.1", 8765)) != 0
    line(free, "port 8765 free for the dashboard",
         "another app.main is already running: open http://localhost:8765 or stop it")
    print("READY" if ok_all else "NOT READY: fix the FAIL lines above")
    return ok_all


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="HDB vision: find a target in the front webcam")
    p.add_argument("--target", default="banana")
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--mirror", action="store_true", help="laptop webcam facing you")
    p.add_argument("--show", action="store_true", help="show an annotated window")
    p.add_argument("--imgsz", type=int, default=None, help=f"default {DEFAULT_IMGSZ}")
    p.add_argument("--weights", default=None, help=f"default {DEFAULT_WEIGHTS}")
    p.add_argument("--record", help="also save the raw camera video here (.mp4) for offline tuning")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = forever)")
    p.add_argument("--log", help="write a per-frame CSV here (for tuning)")
    d = TrackerConfig()
    p.add_argument("--dead-band", type=float, default=d.dead_band)
    p.add_argument("--dead-band-exit", type=float, default=d.dead_band_exit)
    p.add_argument("--lost-after", type=float, default=d.lost_after_s, help="seconds")
    p.add_argument("--smoothing", type=float, default=d.smoothing)
    p.add_argument("--min-score", type=float, default=d.min_score)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--analyze", metavar="CSV", help="summarize a --log file and exit")
    p.add_argument("--check", action="store_true",
                   help="is this computer ready? (imports, weights, camera, mic, port)")
    a = p.parse_args(argv)
    if a.check:
        raise SystemExit(0 if check(int(a.source) if a.source.isdigit() else a.source) else 1)
    if a.analyze:
        import json
        print(json.dumps(analyze(a.analyze), indent=2))
        return
    if a.selftest:
        _selftest()
        return

    cfg = TrackerConfig(dead_band=a.dead_band, dead_band_exit=a.dead_band_exit,
                        lost_after_s=a.lost_after, smoothing=a.smoothing,
                        min_score=a.min_score)
    v = Vision(camera=a.source, mirror=a.mirror, imgsz=a.imgsz, weights=a.weights,
               config=cfg, log_path=a.log, record_path=a.record)
    v.on_change(lambda d: print(f"{d.t:10.3f}  {d.state:9s} offset={d.offset} "
                                f"conf={d.confidence:.2f}"))
    try:
        v.start(strict=True)
    except RuntimeError:
        raise SystemExit(1)
    print(f"[vision] tracking {v.set_target(a.target)!r} on {a.source} "
          f"(device={v.detector.device}); Ctrl-C to quit")
    t0 = time.monotonic()
    try:
        while v._running or v._thread.is_alive():
            if a.show and v.latest_frame() is not None:
                import cv2
                cv2.imshow("HDB vision", v.latest_frame())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
            if a.seconds and time.monotonic() - t0 > a.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        v.stop()
        if v.error:
            print(f"[vision] error: {v.error}")
        print(f"[vision] ~{v.fps:.1f} fps")


if __name__ == "__main__":
    main()
