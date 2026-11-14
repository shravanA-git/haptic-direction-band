"""Reactive obstacle avoidance: steer around obstacles, warn before a collision.

Owner: HDB-Software. Reactive, no map: every update looks at one camera frame.

  1. Sense   every COCO detection of an obstacle class (person, chair, table, ...)
             -> distance from its box height and the class's typical real height
             (pinhole camera; a box cut off by the frame edge is treated as close).
  2. Sectors the camera's field of view split into N_SECTORS vertical bins; each
             bin takes the nearest obstacle overlapping it: free / near / blocked.
  3. Steer   pick the non-blocked bin closest to the goal bearing (the find
             target's bearing, or straight ahead). Veer = front-left / front-right
             motor (7 / 1); nothing open in view = turn cue left / right (6 / 2)
             toward the side with more room.
  4. Stop    centre bin closer than STOP_M on STOP_FRAMES updates in a row ->
             crash warning, overriding guidance. It repeats every STOP_REPEAT_S
             while the path stays blocked.

The crash warning uses existing cues only (generic attention: 300 ms front),
repeated quickly. A distinct "stop" pattern needs HDB-Protocol sign-off and
Shravan's approval first.

Use:
    av = Avoider()
    d = av.update(detections, frame_w, frame_h, t, goal_bearing=None)
    d.action   "clear" | "steer" | "turn" | "stop"
    d.motor    motor to cue for steer/turn (None for clear/stop)
    d.sectors  [(bearing_deg, state, distance_m | None)], left to right

    loop = AvoidLoop(vision, play=lambda cue: app.play_cue(cue, "avoid"), guide=guide)
    loop.start()   # runs the detector on the camera's latest frame, ~8 Hz

Limits: a chest/forehead webcam sees ~64 deg ahead, nothing beside or behind;
distance is a rough estimate (+-30 % for people, worse for odd furniture); only
COCO classes count as obstacles (walls, doors and stairs are invisible to it).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from . import haptics

HFOV_DEG = 64.0           # Logitech USB webcam, 640x480 (approx.)
N_SECTORS = 5             # bins across the field of view (~13 deg each)
NEAR_M = 2.5              # closer than this: "near" (still passable)
BLOCK_M = 1.3             # closer than this: "blocked" (steer around)
STOP_M = 0.9              # centre closer than this: crash warning
STOP_FRAMES = 2           # updates in a row before warning (one-frame flukes)
STOP_REPEAT_S = 0.8
STEER_REPEAT_S = 1.2      # walking with no find target: steer cue this often
PAD = 0.04                # widen each box by this share of the frame (body width)
SMOOTH_N = 3              # median of the last N distances per sector
MIN_SCORE = 0.35
VEER_LEFT, VEER_RIGHT, TURN_LEFT, TURN_RIGHT = 7, 1, 6, 2

# Typical real heights (m). Only these classes are obstacles; cups, phones and
# bananas on a table are not.
CLASS_HEIGHT_M = {
    "person": 1.7, "bicycle": 1.0, "motorcycle": 1.1, "car": 1.5, "bus": 3.0, "truck": 3.0,
    "chair": 0.9, "couch": 0.85, "bed": 0.6, "dining table": 0.75, "bench": 0.8,
    "potted plant": 0.7, "tv": 0.6, "refrigerator": 1.8, "oven": 0.9, "sink": 0.9,
    "toilet": 0.8, "suitcase": 0.65, "backpack": 0.5, "dog": 0.6, "cat": 0.3,
    "fire hydrant": 0.8, "parking meter": 1.3, "stop sign": 2.1, "umbrella": 1.0,
}


@dataclass(frozen=True)
class Obstacle:
    cls: str
    x1: float
    x2: float
    distance_m: float
    clipped: bool = False


@dataclass(frozen=True)
class AvoidDecision:
    action: str                      # clear | steer | turn | stop
    motor: Optional[int]
    bearing: Optional[float]         # degrees, + = right
    sectors: tuple = ()              # ((bearing, state, distance), ...)
    nearest: Optional[Obstacle] = None
    t: float = 0.0
    changed: bool = False

    def to_msg(self) -> dict:
        return {"type": "avoid", "action": self.action, "motor": self.motor,
                "bearing": None if self.bearing is None else round(self.bearing, 1),
                "sectors": [[round(b, 1), s, None if d is None else round(d, 2)]
                            for b, s, d in self.sectors],
                "nearest": None if self.nearest is None else
                {"cls": self.nearest.cls, "m": round(self.nearest.distance_m, 2)}}


def focal_px(frame_w: int, hfov_deg: float = HFOV_DEG) -> float:
    return (frame_w / 2) / math.tan(math.radians(hfov_deg) / 2)


def estimate_distance(cls: str, y1: float, y2: float, frame_w: int, frame_h: int,
                      hfov_deg: float = HFOV_DEG) -> tuple:
    """(metres, clipped). A box cut off by the top or bottom edge shows only part
    of the object, so its distance is an upper bound: tall cut-off boxes are close."""
    h_px = max(1.0, y2 - y1)
    d = focal_px(frame_w, hfov_deg) * CLASS_HEIGHT_M[cls] / h_px
    clipped = y1 <= 2 or y2 >= frame_h - 2
    if clipped and h_px >= 0.6 * frame_h:
        d = min(d, 0.8)
    return d, clipped


def obstacles(dets: Sequence, frame_w: int, frame_h: int, min_score: float = MIN_SCORE,
              hfov_deg: float = HFOV_DEG) -> list:
    """dets: [(cls, x1, y1, x2, y2, score)] as from Detector.detect_all."""
    out = []
    for cls, x1, y1, x2, y2, score in dets:
        if cls not in CLASS_HEIGHT_M or score < min_score:
            continue
        d, clipped = estimate_distance(cls, y1, y2, frame_w, frame_h, hfov_deg)
        out.append(Obstacle(cls, x1 / frame_w, x2 / frame_w, d, clipped))
    return out


def sector_bearings(n: int = N_SECTORS, hfov_deg: float = HFOV_DEG) -> list:
    return [((i + 0.5) / n * 2 - 1) * hfov_deg / 2 for i in range(n)]


def state_of(d: Optional[float]) -> str:
    if d is None or d >= NEAR_M:
        return "free"
    return "blocked" if d < BLOCK_M else "near"


class Avoider:
    def __init__(self, n: int = N_SECTORS, hfov_deg: float = HFOV_DEG):
        self.n, self.hfov = n, hfov_deg
        self.bearings = sector_bearings(n, hfov_deg)
        self._hist = [deque(maxlen=SMOOTH_N) for _ in range(n)]
        self._stop_run = 0
        self.last: Optional[AvoidDecision] = None

    def reset(self) -> None:
        for h in self._hist:
            h.clear()
        self._stop_run = 0

    def _centre(self) -> list:
        return [self.n // 2] if self.n % 2 else [self.n // 2 - 1, self.n // 2]

    def update(self, dets: Sequence, frame_w: int, frame_h: int, t: float,
               goal_bearing: Optional[float] = None, mirror: bool = False) -> AvoidDecision:
        obs = obstacles(dets, frame_w, frame_h, hfov_deg=self.hfov)
        if mirror:
            obs = [Obstacle(o.cls, 1 - o.x2, 1 - o.x1, o.distance_m, o.clipped) for o in obs]
        raw = [None] * self.n
        for o in obs:
            a = max(0, int((o.x1 - PAD) * self.n))
            b = min(self.n - 1, int((o.x2 + PAD) * self.n - 1e-9))
            for i in range(a, b + 1):
                if raw[i] is None or o.distance_m < raw[i]:
                    raw[i] = o.distance_m
        dist = []
        for i, d in enumerate(raw):   # median over the last frames: flicker-robust
            self._hist[i].append(d if d is not None else math.inf)
            m = sorted(self._hist[i])[len(self._hist[i]) // 2]
            dist.append(None if m == math.inf else m)
        states = [state_of(d) for d in dist]
        sectors = tuple(zip(self.bearings, states, dist))
        nearest = min(obs, key=lambda o: o.distance_m) if obs else None

        centre_d = min((raw[i] for i in self._centre() if raw[i] is not None), default=None)
        self._stop_run = self._stop_run + 1 if centre_d is not None and centre_d < STOP_M else 0

        goal = 0.0 if goal_bearing is None else max(-self.hfov / 2, min(self.hfov / 2, goal_bearing))
        gi = min(range(self.n), key=lambda i: abs(self.bearings[i] - goal))
        if self._stop_run >= STOP_FRAMES:
            action, motor, bearing = "stop", None, None
        elif self._stop_run > 0 and self.last is not None and self.last.action != "stop":
            # Something very close ahead for one frame: hold the last action rather
            # than steer now and warn on the next frame.
            action, motor, bearing = self.last.action, self.last.motor, self.last.bearing
        elif self._stop_run > 0:
            action, motor, bearing = "clear", None, goal
        elif states[gi] != "blocked":            # the way to the goal is open
            action, motor, bearing = "clear", None, goal
        else:
            open_ = [i for i in range(self.n) if states[i] != "blocked"]
            if open_:
                # prefer free over near, then closeness to the goal
                i = min(open_, key=lambda k: (states[k] != "free", abs(self.bearings[k] - goal)))
                bearing = self.bearings[i]
                action, motor = "steer", (VEER_RIGHT if bearing > 0 else VEER_LEFT)
                if abs(bearing) < 1e-6:
                    action, motor = "clear", None
            else:   # everything in view blocked: turn toward the roomier edge
                left = dist[0] if dist[0] is not None else math.inf
                right = dist[-1] if dist[-1] is not None else math.inf
                action = "turn"
                motor, bearing = (TURN_LEFT, -90.0) if left >= right else (TURN_RIGHT, 90.0)
        prev = self.last.action if self.last else None
        self.last = AvoidDecision(action, motor, bearing, sectors, nearest, t, action != prev)
        return self.last


def stop_cue() -> list:
    """Crash warning from existing cues: two generic-attention pulses (300 ms front)
    150 ms apart. PROVISIONAL until HDB-Protocol signs off a distinct pattern."""
    g = haptics.generic_attention()[0]
    return [g, (g[2] + 150, g[1], g[2])]


class AvoidLoop:
    """Runs the detector on the camera's latest frame and acts on the decision.

    With a find target being guided (guide.state left/right/centered), ObjectGuide
    asks override() before each directional cue: steer replaces the target's
    motor, stop replaces the cue with the crash warning. With no target, the loop
    itself plays the crash warning and steer cues when the path ahead is blocked."""

    def __init__(self, vision, play: Callable[[list], object], guide=None,
                 on_decision: Optional[Callable[[AvoidDecision], None]] = None,
                 hz: float = 8.0, avoider: Optional[Avoider] = None):
        self.vision, self.play, self.guide = vision, play, guide
        self.on_decision = on_decision
        self.period = 1.0 / hz
        self.avoider = avoider or Avoider()
        self.decision: Optional[AvoidDecision] = None
        self.enabled = True
        self._last_frame = None
        self._next_stop = 0.0
        self._next_steer = 0.0
        self._stop = threading.Event()
        self.fps = 0.0
        if guide is not None:
            guide.avoid = self

    # ---- for ObjectGuide ----
    def override(self, now: Optional[float] = None) -> Optional[AvoidDecision]:
        """The decision to apply instead of plain guidance, or None. Stale (>0.6 s)
        decisions are ignored so a stalled loop never steers."""
        d = self.decision
        now = time.monotonic() if now is None else now
        if not self.enabled or d is None or now - d.t > 0.6:
            return None
        return d if d.action in ("stop", "steer", "turn") else None

    def _goal_bearing(self) -> Optional[float]:
        d = self.vision.latest() if self.vision is not None else None
        if d is None or d.state not in ("left", "right", "centered") or d.offset is None:
            return None
        return d.offset * self.avoider.hfov / 2

    def guiding(self) -> bool:
        """ObjectGuide is repeating left/right cues (and asks override() each time).
        Centered gets no repeats, so then the loop gives steer cues itself."""
        return self.guide is not None and getattr(self.guide, "state", "idle") in (
            "left", "right") and not getattr(self.guide, "paused", False)

    def step(self, frame, now: Optional[float] = None) -> Optional[AvoidDecision]:
        now = time.monotonic() if now is None else now
        det = getattr(self.vision, "detector", None)
        if frame is None or det is None:
            return None
        dets = det.detect_all(frame, min_score=MIN_SCORE)
        h, w = frame.shape[:2]
        d = self.avoider.update(dets, w, h, now, self._goal_bearing(),
                                getattr(self.vision, "mirror", False))
        self.decision = d
        if d.changed and self.on_decision:
            try:
                self.on_decision(d)
            except Exception as e:
                print(f"[avoid] on_decision error: {e}")
        if self.enabled:
            self._act(d, now)
        return d

    def _act(self, d: AvoidDecision, now: float) -> None:
        if d.action == "stop":
            if now >= self._next_stop:          # warns even while guiding (override)
                self.play(stop_cue())
                self._next_stop = now + STOP_REPEAT_S
            return
        self._next_stop = 0.0
        if self.guiding():
            return                               # ObjectGuide applies steer via override()
        if d.action in ("steer", "turn") and now >= self._next_steer:
            self.play(haptics.directional(d.motor))
            self._next_steer = now + STEER_REPEAT_S
        elif d.action == "clear":
            self._next_steer = 0.0

    def start(self) -> "AvoidLoop":
        def run():
            last = time.monotonic()
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    if not getattr(self.vision, "_paused", False):
                        frame = self.vision.latest_raw()
                        if frame is not None and frame is not self._last_frame:
                            self._last_frame = frame
                            self.step(frame)
                except Exception as e:
                    print(f"[avoid] {type(e).__name__}: {e}")
                dt = time.monotonic() - last
                last = time.monotonic()
                self.fps = 0.8 * self.fps + 0.2 * (1 / dt if dt > 0 else 0)
                self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))
        threading.Thread(target=run, name="avoid", daemon=True).start()
        print(f"[avoid] on: {N_SECTORS} sectors over {HFOV_DEG:.0f} deg, warn < {STOP_M} m")
        return self

    def close(self) -> None:
        self._stop.set()
