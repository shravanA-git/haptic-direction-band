"""Visit a building before you go (CoStar track): step-by-step haptic guidance
through a Matterport digital twin while the wearer walks its 3D tour.

    nav = NavGuide(twin, play=lambda cue: app.play_cue(cue, "nav")).start()
    nav.set_goal("kitchen")                  # hold-to-talk "take me to the kitchen"
    nav.update_pose("sid", (x, y, z), yaw)   # the /nav page (Showcase SDK camera pose)
    nav.hold(1.2)                            # a name/sound alert is playing: stay quiet
    nav.pause(); nav.resume()                # Action held

Routes come from tools/costar_nav.py (walkable links between scan points that
cross no wall). Cues (ALL PROVISIONAL, not yet agreed with HDB-Protocol):
  next step   haptics.directional(motor): 200 ms toward the next scan point on the
              route, relative to where the tour camera faces; again when the
              wearer reaches a new scan point, when the turn needed changes
              motor, and every REPEAT_S while nothing changes
  arrived     haptics.centered() (two front pulses), then guidance stops
  can't       haptics.target_unknown() (unknown room, no route)
  got it      haptics.target_ack() when a route starts

Frames: the Model API is z-up (x, y on the floor); the Showcase SDK is y-up,
so tour (x, y, z) -> model (x, -z). Tour yaw -> model heading (clockwise from
+y) = YAW_SIGN * yaw. Both are ASSUMPTIONS until checked on a real model: the
/nav page shows the heading so the sign can be flipped with --nav-yaw-sign.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, Optional

from . import haptics
from tools.costar_nav import Twin, bearing, goal_panos, motor_for, shortest_path

REPEAT_S = 4.0          # repeat the current step while nothing changes (PROVISIONAL)
MIN_GAP_S = 0.8         # never two step cues closer than this
MATCH_M = 1.5           # tour position -> nearest scan point within this
YAW_SIGN = -1.0

_ROUTE = re.compile(
    r"\b(?:(?:take|bring|walk|lead|guide|get)\s+me|(?:let's|lets|i\s+want\s+to|i\s+wanna)?\s*"
    r"(?:go|head|navigate|get))\s+(?:back\s+)?(?:to|into|towards?)\s+(?:the\s+|my\s+|a\s+)?"
    r"(?P<place>[a-z][a-z \-']*?)\s*(?:please)?\s*[.!?]*$", re.I)


def route_request(text: str) -> Optional[str]:
    """'Take me to the kitchen.' -> 'kitchen'; 'go to the bathroom' -> 'bathroom'.
    None for anything else ('find my keys' stays a webcam find)."""
    m = _ROUTE.search((text or "").strip())
    if not m:
        return None
    place = m.group("place").strip(" -'").lower()
    return place or None


class NavGuide:
    def __init__(self, twin: Twin, play: Callable[[list], object],
                 on_status: Optional[Callable[[dict], None]] = None,
                 yaw_sign: float = YAW_SIGN, repeat_s: float = REPEAT_S):
        self.twin = twin
        self.play = play
        self.on_status = on_status
        self.yaw_sign = yaw_sign
        self.repeat_s = repeat_s
        self.goal: Optional[str] = None
        self.goals: set = set()
        self.pano: Optional[str] = None
        self.heading: Optional[float] = None
        self.paused = False
        self._hold_until = 0.0
        self._last_cue_t = -1e9
        self._last_motor: Optional[int] = None
        self._last_pano: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------- inputs
    def set_goal(self, place: str, now: Optional[float] = None) -> tuple:
        """(ok, message). Plays 'got it' or 'can't' and starts guiding."""
        now = time.monotonic() if now is None else now
        goals = goal_panos(self.twin, place)
        if not goals:
            self.play(haptics.target_unknown())
            rooms = sorted({r.label for r in self.twin.rooms.values() if r.label != "Void"})
            return False, f"no room like '{place}' (rooms: {', '.join(rooms) or 'none'})"
        with self._lock:
            self.goal, self.goals = place, goals
            self._last_motor = self._last_pano = None
            self._last_cue_t = -1e9
        self.play(haptics.target_ack())
        self._last_cue_t = now          # let 'got it' finish before the first step
        self._status()
        return True, f"route to {place}" + ("" if self.pano else " (waiting for the tour position)")

    def cancel(self, why: str = "") -> None:
        with self._lock:
            self.goal, self.goals = None, set()
        self._status(why)

    def update_pose(self, sweep: Optional[str], position=None, yaw: Optional[float] = None,
                    now: Optional[float] = None) -> None:
        pano = self._match(sweep, position)
        with self._lock:
            if pano:
                self.pano = pano
            if yaw is not None:
                self.heading = (self.yaw_sign * float(yaw)) % 360
        self.tick(now)

    def hold(self, seconds: float, now: Optional[float] = None) -> None:
        """A name/sound alert is playing: no step cue until it's over (then the
        current step is given again)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._hold_until = max(self._hold_until, now + seconds)
            self._last_motor = None          # re-cue when the hold ends

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self._last_motor = None

    # ------------------------------------------------------------- policy
    def next_step(self) -> Optional[dict]:
        """Where to go from here: {motor, to, metres, arrived} or None if unknown."""
        if not self.goals or self.pano is None:
            return None
        if self.pano in self.goals:
            return {"arrived": True}
        path = shortest_path(self.twin, self.pano, self.goals)
        if not path or len(path) < 2:
            return {"no_route": True}
        a, b = self.twin.panos[path[0]], self.twin.panos[path[1]]
        brg = bearing((a.x, a.y), (b.x, b.y))
        rel = brg - (self.heading if self.heading is not None else brg)
        metres = sum(_dist(self.twin, u, v) for u, v in zip(path, path[1:]))
        room = self.twin.rooms[b.room].label if b.room else "-"
        return {"arrived": False, "motor": motor_for(rel), "to": b.id, "room": room,
                "metres": round(metres, 1), "legs": len(path) - 1, "bearing": round(brg)}

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self.goals or self.paused or now < self._hold_until:
                return
            step = self.next_step()
            if step is None:
                return
            if step.get("arrived"):
                self.goal, self.goals = None, set()
                cue, info = haptics.centered(), "arrived"
            elif step.get("no_route"):
                self.goal, self.goals = None, set()
                cue, info = haptics.target_unknown(), "no walkable route from here"
            else:
                changed = step["motor"] != self._last_motor or self.pano != self._last_pano
                due = now - self._last_cue_t >= self.repeat_s
                if not (changed or due) or now - self._last_cue_t < MIN_GAP_S:
                    return
                self._last_motor, self._last_pano = step["motor"], self.pano
                cue, info = haptics.directional(step["motor"]), None
            self._last_cue_t = now
        self.play(cue)
        self._status(info, step)

    # ------------------------------------------------------------ helpers
    def _match(self, sweep, position) -> Optional[str]:
        if sweep and sweep in self.twin.panos:
            return sweep
        if position is None:
            return None
        if isinstance(position, dict):
            position = (position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0))
        x, _, z = position
        best, best_d = None, MATCH_M
        for p in self.twin.panos.values():
            d = ((p.x - x) ** 2 + (p.y + z) ** 2) ** 0.5   # tour (x, z) -> model (x, -z)
            if d < best_d:
                best, best_d = p.id, d
        return best

    def _status(self, info: Optional[str] = None, step: Optional[dict] = None) -> None:
        if not self.on_status:
            return
        p = self.twin.panos.get(self.pano) if self.pano else None
        here = self.twin.rooms[p.room].label if p and p.room else None
        st = {"goal": self.goal, "pano": self.pano, "room": here,
              "heading": None if self.heading is None else round(self.heading),
              "info": info}
        if step and not step.get("arrived") and not step.get("no_route"):
            st.update(motor=step["motor"], direction=haptics.MOTOR_NAMES[step["motor"]],
                      next_room=step["room"], metres=step["metres"], legs=step["legs"])
        try:
            self.on_status(st)
        except Exception:
            pass

    def start(self) -> "NavGuide":
        def run():
            while not self._stop.wait(0.25):
                try:
                    self.tick()
                except Exception as e:
                    print(f"[nav] {e}")
        threading.Thread(target=run, name="nav", daemon=True).start()
        return self

    def close(self) -> None:
        self._stop.set()


def _dist(twin: Twin, u: str, v: str) -> float:
    a, b = twin.panos[u], twin.panos[v]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# ====================================================================== route play
# "Plan once, play the route": no tour position needed (no Showcase SDK key).
STEP_S = 3.0            # Play: one step every STEP_S (PROVISIONAL)


class Speaker:
    """Speaks through macOS `say` on its own process, never while blocked
    (Action held: the laptop mic is listening to the wearer)."""

    def __init__(self, blocked: Callable[[], bool] = lambda: False):
        self.blocked = blocked
        self._proc = None
        import shutil
        self.available = shutil.which("say") is not None

    def say(self, text: str) -> bool:
        if not text or self.blocked() or not self.available:
            return False
        import subprocess
        self.cancel()
        self._proc = subprocess.Popen(["say", text], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        return True

    def cancel(self) -> None:
        p, self._proc = self._proc, None
        if p is not None and p.poll() is None:
            p.terminate()


def step_text(step, prev_room: Optional[str], last: bool) -> str:
    metres = max(1, int(round(step.metres)))
    words = {"ahead": "ahead", "ahead-right": "ahead right", "right": "right",
             "behind-right": "turn around, right", "behind": "turn around",
             "behind-left": "turn around, left", "left": "left", "ahead-left": "ahead left"}
    t = f"{words.get(step.turn, step.turn)} {metres} metre{'s' if metres != 1 else ''}"
    if step.room and step.room not in ("-", "Void") and step.room != prev_room:
        t += f", into the {step.room.lower()}"
    return t


class RoutePlayer:
    """Plans a route once (wearer facing the first leg) and plays it step by step:
    PULSE toward each leg, the step spoken, then the arrival cue.

        rp = RoutePlayer(twin, play=..., speak=Speaker(...).say, on_status=...)
        rp.plan("Living Room", "kitchen"); rp.play()   # or rp.next() per step
    Alerts win (hold), Action held pauses (and nothing is spoken)."""

    def __init__(self, twin: Twin, play: Callable[[list], object],
                 speak: Callable[[str], object] = lambda text: None,
                 on_status: Optional[Callable[[dict], None]] = None, step_s: float = STEP_S):
        self.twin = twin
        self.play_cue = play
        self.speak = speak
        self.on_status = on_status
        self.step_s = step_s
        self.steps: list = []
        self.texts: list = []
        self.intro = ""             # spoken once with the first step (arrival briefing)
        self._intro_due = False
        self.goal: Optional[str] = None
        self.start_pano: Optional[str] = None
        self.index = 0              # next step to give; len(steps) = arrival next
        self.done = False
        self.auto = False
        self.paused = False
        self._hold_until = 0.0
        self._want = 0              # steps requested by Next / auto, not yet given
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------- plan
    def starts(self) -> list:
        """[(label, pano id)]: each labelled room's first scan point, first pano first."""
        seen, out = set(), []
        for p in self.twin.panos.values():
            label = self.twin.rooms[p.room].label if p.room else None
            if label and label != "Void" and label not in seen:
                seen.add(label)
                out.append((label, p.id))
        return out

    def plan(self, start: Optional[str], goal: str) -> tuple:
        """(ok, message). start: room label, pano id/label, or None = first scan point."""
        from tools.costar_nav import NavError, find_pano, route_steps
        self.stop()
        try:
            a = find_pano(self.twin, start) if start else next(iter(self.twin.panos.values()))
        except (NavError, StopIteration) as e:
            return False, str(e) or "no scan points"
        goals = goal_panos(self.twin, goal)
        if not goals:
            rooms = sorted({r.label for r in self.twin.rooms.values() if r.label != "Void"})
            return False, f"no room like '{goal}' (rooms: {', '.join(rooms) or 'none'})"
        if a.id in goals:
            steps = []
        else:
            path = shortest_path(self.twin, a.id, goals)
            if not path:
                return False, f"no walkable route to {goal}"
            steps = route_steps(self.twin, path, None)
        here = self.twin.rooms[a.room].label if a.room else None
        texts, prev = [], here
        for k, s in enumerate(steps):
            texts.append(step_text(s, prev, k == len(steps) - 1))
            prev = s.room
        with self._lock:
            self.steps, self.texts, self.goal = steps, texts, goal
            self.start_pano, self.index, self.done, self._want = a.id, 0, False, 0
            self._intro_due = True          # the briefing goes with the first thing said
        self._status("planned")
        total = sum(s.metres for s in steps)
        return True, f"{len(steps)} steps, {total:.0f} m to the {goal}"

    def arrival_text(self) -> str:
        room = self.steps[-1].room if self.steps else self.goal
        return f"You've arrived at the {str(room or self.goal).lower()}."

    # --------------------------------------------------------------- play
    def play(self) -> None:
        with self._lock:
            if not self.goal or self.done:
                return
            self.auto = True
            self._want = max(self._want, 1)
        self._ensure_thread()
        self._wake.set()

    def next(self) -> None:
        with self._lock:
            if not self.goal or self.done:
                return
            self.auto = False
            self._want += 1
        self._ensure_thread()
        self._wake.set()

    def stop(self) -> None:
        with self._lock:
            playing = self.auto or self._want > 0
            self.auto, self._want = False, 0
        if playing and self.goal and not self.done:
            self._status("stopped")

    def hold(self, seconds: float, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._hold_until = max(self._hold_until, now + seconds)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="route", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        next_at = 0.0
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            now = time.monotonic()
            with self._lock:
                ready = (self._want > 0 and not self.done and not self.paused
                         and now >= self._hold_until and now >= next_at)
                if not ready:
                    continue
                self._want -= 1
                i = self.index
                self.index += 1
                if self.auto and i < len(self.steps):
                    self._want = max(self._want, 1)      # keep going
            self._give(i)
            next_at = time.monotonic() + (self.step_s if self.auto else 0.3)

    def _say(self, text: str) -> None:
        if self.paused:
            return
        if self._intro_due:
            self._intro_due = False
            if self.intro:
                text = f"{self.intro} {text}"
        self.speak(text)

    def _give(self, i: int) -> None:
        if i < len(self.steps):
            s = self.steps[i]
            self.play_cue(haptics.directional(s.motor))
            self._say(self.texts[i])
            self._status("step", i)
        else:
            with self._lock:
                self.done, self.auto, self._want = True, False, 0
            self.play_cue(haptics.centered())
            self._say(self.arrival_text())
            self._status("arrived", i)

    def _status(self, event: Optional[str], i: Optional[int] = None) -> None:
        if not self.on_status:
            return
        st = {"event": event, "goal": self.goal, "start": self.start_pano,
              "index": self.index, "done": self.done, "auto": self.auto,
              "steps": [{"n": k + 1, "turn": s.turn, "motor": s.motor, "metres": round(s.metres, 1),
                         "room": s.room, "pano": s.to_pano, "text": self.texts[k]}
                        for k, s in enumerate(self.steps)]}
        if event == "step" and i is not None:
            st["current"] = i
            st["motor"] = self.steps[i].motor
            st["text"] = self.texts[i]
        elif event == "arrived":
            st["current"] = len(self.steps)
            st["text"] = self.arrival_text()
        try:
            self.on_status(st)
        except Exception:
            pass
