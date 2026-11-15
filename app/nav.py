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
    r"(?P<place>[a-z0-9][a-z0-9 \-'&,.]*?)\s*(?:please)?\s*[.!?]*$", re.I)


def route_request(text: str) -> Optional[str]:
    """'Take me to the kitchen.' -> 'kitchen'; 'go to the bathroom' -> 'bathroom'.
    None for anything else ('find my keys' stays a webcam find)."""
    m = _ROUTE.search((text or "").strip())
    if not m:
        return None
    place = m.group("place").strip(" -',.").lower()
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
    def set_goal(self, place: str, now: Optional[float] = None, quiet: bool = False) -> tuple:
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
        if not quiet:
            self.play(haptics.target_ack())
        self._last_cue_t = now if not quiet else -1e9   # let 'got it' finish first
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
# Timing (ALL PROVISIONAL, see docs/costar_nav_cue_questions.md):
WALK_MPS = 0.8          # assumed walking speed; the /nav page picks slow / normal / demo
SPEEDS = {"slow": 0.6, "normal": 0.8, "demo": 1.5}
MIN_LEG_S = 2.5         # no leg shorter than this
PRE_TURN_S = 1.5        # 'in N metres, turn right' this long before a leg ends (spoken, no buzz)
KEEP_GOING_S = 4.0      # on long legs, a short front pulse this often = 'still straight'
KEEP_GOING = [(0, haptics.FRONT, 80)]   # PULSE 0 80 (not in the cue table yet)
PROGRESS_S = 0.25       # progress updates to the page


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


_WORDS = {"ahead": "ahead", "ahead-right": "ahead right", "right": "right",
          "behind-right": "turn around, right", "behind": "turn around",
          "behind-left": "turn around, left", "left": "left", "ahead-left": "ahead left"}
_TURN = {"ahead": "keep straight", "ahead-right": "bear right", "right": "turn right",
         "behind-right": "turn sharp right", "behind": "turn around",
         "behind-left": "turn sharp left", "left": "turn left", "ahead-left": "bear left"}


def _m(metres: float) -> str:
    n = max(1, int(round(metres)))
    return f"{n} metre{'s' if n != 1 else ''}"


def step_text(step, prev_room: Optional[str], last: bool = False) -> str:
    t = f"{_WORDS.get(step.turn, step.turn)} {_m(step.metres)}"
    if step.room and step.room not in ("-", "Void") and step.room != prev_room:
        t += f", into the {step.room.lower()}"
    return t


class RoutePlayer:
    """Plans a route once (wearer facing the first leg) and plays it like GPS:

      leg starts      PULSE toward it + the leg spoken ('right 3 metres, into the kitchen')
      during a leg    a short front pulse every KEEP_GOING_S on legs longer than that
      1.5 s before    'in 1 metre, turn right' (spoken only)
      leg ends        the next leg starts (its PULSE is the turn cue, at the turn)
      last leg ends   two front pulses + 'You've arrived at the kitchen'

    Each leg lasts max(MIN_LEG_S, metres / speed). Play runs the clock, Next skips
    to the next leg now, Stop stops. Name/sound alerts (hold) and Action held
    (pause) freeze the clock, so no cue fires and nothing is spoken until they end.

        rp = RoutePlayer(twin, play=..., speak=..., on_status=...)
        rp.plan("Living Room", "kitchen"); rp.play()
    threaded=False: no background thread; call tick(now) yourself (tests)."""

    def __init__(self, twin: Twin, play: Callable[[list], object],
                 speak: Callable[[str], object] = lambda text: None,
                 on_status: Optional[Callable[[dict], None]] = None,
                 speed: float = WALK_MPS, threaded: bool = True):
        self.twin = twin
        self.play_cue = play
        self.speak = speak
        self.on_status = on_status
        self.speed = speed
        self.threaded = threaded
        # on_move(pano id, heading deg): drive a 3D tour along the route (live mode):
        # through each scan point of a leg as the walker passes it, and at the leg
        # end to the turn point facing the next leg (same moment as the turn PULSE)
        self.on_move: Optional[Callable[[str, float], None]] = None
        self.short_hops = False     # live tour: route through more scan points (see shortest_path)
        self.legs: list = []        # per leg: start pano, first bearing, [(pano, metres in, bearing on)]
        self._sub_i = 0
        self.steps: list = []
        self.texts: list = []
        self.intro = ""             # spoken once with the first step (arrival briefing)
        self._intro_due = False
        self.goal: Optional[str] = None
        self.start_pano: Optional[str] = None
        self.leg = -1               # leg being walked; -1 = not started
        self.leg_t = 0.0            # seconds into it (frozen while held/paused)
        self.done = False
        self.auto = False
        self.paused = False
        self._hold_until = 0.0
        self._last_t: Optional[float] = None
        self._pre_turn_done = False
        self._next_keep = KEEP_GOING_S
        self._last_progress = -1e9
        self._skip = 0              # Next presses not yet handled (held/paused)
        self._lock = threading.RLock()
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
        steps, legs = [], []
        if a.id not in goals:
            path = shortest_path(self.twin, a.id, goals, 2.0 if self.short_hops else 1.0)
            if not path:
                return False, f"no walkable route to {goal}"
            steps = route_steps(self.twin, path, None)
            legs = self._legs(path, steps)
        here = self.twin.rooms[a.room].label if a.room else None
        texts, prev = [], here
        for s in steps:
            texts.append(step_text(s, prev))
            prev = s.room
        with self._lock:
            self.steps, self.texts, self.goal, self.legs = steps, texts, goal, legs
            self.start_pano, self.leg, self.leg_t, self.done = a.id, -1, 0.0, False
            self._intro_due, self._skip = True, 0
        self._status("planned")
        total = sum(s.metres for s in steps)
        return True, f"{len(steps)} steps, {total:.0f} m to the {goal}"

    def announce(self, text: str, cue: Optional[list] = None) -> None:
        """Say something instead of a route (e.g. the tenant is on another floor):
        no route is invented, the current one stops."""
        self.stop()
        with self._lock:
            self.steps, self.texts, self.goal, self.leg, self.done = [], [], None, -1, False
        if cue:
            self.play_cue(cue)
        self._say(text)
        self._status("info", text=text)

    def _legs(self, path: list, steps: list) -> list:
        """Scan points along each leg, for driving the tour."""
        from tools.costar_nav import bearing as brg
        pos = {p: (self.twin.panos[p].x, self.twin.panos[p].y) for p in path}
        legs, i = [], 0
        for s in steps:
            j = path.index(s.to_pano, i + 1)
            subs, cum = [], 0.0
            for k in range(i + 1, j):
                cum += _dist(self.twin, path[k - 1], path[k])
                subs.append((path[k], cum, brg(pos[path[k]], pos[path[k + 1]])))
            legs.append({"start": path[i], "bearing": brg(pos[path[i]], pos[path[i + 1]]),
                         "end": path[j], "last": brg(pos[path[j - 1]], pos[path[j]]), "subs": subs})
            i = j
        if legs:
            legs[-1]["into"] = self.into_room_bearing(legs[-1]["end"], legs[-1]["last"])
        return legs

    def into_room_bearing(self, pano_id: str, fallback: float) -> float:
        """Arrival view: from the arrival scan point toward the middle of its room
        (the other scan points in it), so the tour doesn't end facing a wall."""
        from tools.costar_nav import bearing as brg
        p = self.twin.panos[pano_id]
        mates = [q for q in self.twin.panos.values() if q.room and q.room == p.room]
        if len(mates) < 2:
            return fallback
        cx = sum(q.x for q in mates) / len(mates)
        cy = sum(q.y for q in mates) / len(mates)
        if ((cx - p.x) ** 2 + (cy - p.y) ** 2) ** 0.5 < 0.7:
            return fallback                    # already in the middle: keep walking direction
        return brg((p.x, p.y), (cx, cy))

    def demo_pair(self) -> Optional[dict]:
        """The longest showcase in this model: the start room and destination room
        whose route has the most legs (then the most metres), planned the way it
        will be played (short_hops). Named rooms only; a hallway is a poor goal."""
        from tools.costar_nav import route_steps
        vague = ("Void", "Unknown", "Other")
        labels = sorted({r.label for r in self.twin.rooms.values() if r.label not in vague})
        goals_ok = [g for g in labels if g.lower() not in ("hallway", "corridor")] or labels
        best = None
        for label, pano in self.starts():
            if label in vague:
                continue
            for goal in goals_ok:
                if goal == label:
                    continue
                goals = goal_panos(self.twin, goal)
                if not goals or pano in goals:
                    continue
                path = shortest_path(self.twin, pano, goals, 2.0 if self.short_hops else 1.0)
                if not path:
                    continue
                steps = route_steps(self.twin, path, None)
                key = (len(steps), sum(s.metres for s in steps))
                if best is None or key > best[0]:
                    best = (key, {"start": label, "start_pano": pano, "to": goal,
                                  "legs": len(steps), "metres": round(key[1], 1)})
        return best[1] if best else None

    def arrival_text(self) -> str:
        room = self.steps[-1].room if self.steps else self.goal
        return f"You've arrived at the {str(room or self.goal).lower()}."

    def leg_seconds(self, i: int) -> float:
        return max(MIN_LEG_S, self.steps[i].metres / max(0.1, self.speed))

    def set_speed(self, speed) -> float:
        self.speed = SPEEDS.get(speed, None) or float(speed)
        return self.speed

    # ------------------------------------------------------------- controls
    def play(self, now: Optional[float] = None) -> None:
        with self._lock:
            if not self.goal or self.done:
                return
            self.auto = True
            self._last_t = None
        self._ensure_thread()
        self.tick(now)

    def next(self, now: Optional[float] = None) -> None:
        """Skip to the next leg now (the first press starts the route)."""
        with self._lock:
            if not self.goal or self.done:
                return
            self._skip += 1
        self._ensure_thread()
        self.tick(now)

    def stop(self) -> None:
        with self._lock:
            playing = self.auto
            self.auto, self._skip = False, 0
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

    def close(self) -> None:
        self._stop.set()

    # --------------------------------------------------------------- clock
    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        events = []
        with self._lock:
            last, self._last_t = self._last_t, now
            frozen = self.paused or now < self._hold_until
            if not self.goal or self.done or frozen:
                return
            if self._skip:
                self._skip -= 1
                events += self._advance()
            elif self.auto:
                if self.leg < 0:
                    events += self._advance()
                else:
                    self.leg_t += max(0.0, now - last) if last is not None else 0.0
                    dur = self.leg_seconds(self.leg)
                    if self.leg < len(self.legs):            # tour: pass the scan points
                        walked = min(1.0, self.leg_t / dur) * self.steps[self.leg].metres
                        subs = self.legs[self.leg]["subs"]
                        while self._sub_i < len(subs) and subs[self._sub_i][1] <= walked:
                            pano, _, b = subs[self._sub_i]
                            self._sub_i += 1
                            events.append(("move", (pano, b)))
                    if dur > KEEP_GOING_S and self.leg_t >= self._next_keep \
                            and self.leg_t < dur - PRE_TURN_S:
                        self._next_keep += KEEP_GOING_S
                        events.append(("cue", list(KEEP_GOING)))
                    if not self._pre_turn_done and self.leg + 1 < len(self.steps) \
                            and self.leg_t >= dur - PRE_TURN_S:
                        self._pre_turn_done = True
                        nxt = self.steps[self.leg + 1]
                        left = max(0.0, (dur - self.leg_t) / dur * self.steps[self.leg].metres)
                        events.append(("say", f"in {_m(left)}, {_TURN.get(nxt.turn, nxt.turn)}"))
                    if self.leg_t >= dur:
                        events += self._advance()
            progress = self.auto and self.leg >= 0 and not self.done \
                and now - self._last_progress >= PROGRESS_S
            if progress:
                self._last_progress = now
        for kind, arg in events:
            if kind == "cue":
                self.play_cue(arg)
            elif kind == "say":
                self._say(arg)
            elif kind == "status":
                self._status(*arg)
            elif kind == "move" and self.on_move is not None:
                try:
                    self.on_move(*arg)
                except Exception as e:
                    print(f"[route] tour move: {e}")
        if progress and not any(k == "status" for k, _ in events):
            self._status("walking", self.leg)

    def _advance(self) -> list:
        """Start the next leg (its pulse is the turn cue) or arrive. Lock held."""
        self.leg += 1
        self.leg_t, self._pre_turn_done, self._next_keep = 0.0, False, KEEP_GOING_S
        self._sub_i = 0
        if self.leg < len(self.steps):
            s = self.steps[self.leg]
            out = [("cue", haptics.directional(s.motor))]
            if self.leg < len(self.legs):    # tour: at the turn point, facing the new leg
                out.append(("move", (self.legs[self.leg]["start"], self.legs[self.leg]["bearing"])))
            return out + [("say", self.texts[self.leg]), ("status", ("step", self.leg))]
        self.auto, self._skip = False, 0
        self.done = True
        out = [("cue", haptics.centered())]
        if self.legs:                        # tour: inside the goal room, facing into it
            last = self.legs[-1]
            out.append(("move", (last["end"], last.get("into", last["last"]))))
        return out + [("say", self.arrival_text()), ("status", ("arrived", self.leg))]

    def _say(self, text: str) -> None:
        if self.paused:
            return
        if self._intro_due:
            self._intro_due = False
            if self.intro:
                text = f"{self.intro} {text}"
        self.speak(text)

    def _ensure_thread(self) -> None:
        if not self.threaded or (self._thread is not None and self._thread.is_alive()):
            return

        def run():
            while not self._stop.wait(0.05):
                try:
                    self.tick()
                except Exception as e:
                    print(f"[route] {e}")
        self._thread = threading.Thread(target=run, name="route", daemon=True)
        self._thread.start()

    @property
    def index(self) -> int:
        """Next leg to start (compat): leg + 1."""
        return self.leg + 1

    def _status(self, event: Optional[str], i: Optional[int] = None, text: str = "") -> None:
        if not self.on_status:
            return
        st = {"event": event, "goal": self.goal, "start": self.start_pano,
              "index": self.leg + 1, "done": self.done, "auto": self.auto, "speed": self.speed,
              "steps": [{"n": k + 1, "turn": s.turn, "motor": s.motor, "metres": round(s.metres, 1),
                         "room": s.room, "pano": s.to_pano, "text": self.texts[k],
                         "seconds": round(self.leg_seconds(k), 1)}
                        for k, s in enumerate(self.steps)]}
        if event in ("step", "walking") and i is not None and 0 <= i < len(self.steps):
            dur = self.leg_seconds(i)
            st.update(current=i, motor=self.steps[i].motor, text=self.texts[i],
                      walked=round(min(1.0, self.leg_t / dur) * self.steps[i].metres, 1),
                      of=round(self.steps[i].metres, 1))
        elif event == "arrived":
            st.update(current=len(self.steps), text=self.arrival_text())
        elif event in ("info", "override"):
            st["text"] = text
        try:
            self.on_status(st)
        except Exception:
            pass


class TourDriver:
    """Tells our own tour moves' pose echoes from the user taking over (live mode).

    moved() when the route sends the tour somewhere; classify() each pose the
    page reports while a route is driving:
      echo      within SETTLE_S of a move, or at the scan point we moved to,
                looking roughly where we turned it: ignore (don't re-plan)
      override  anywhere else: the user dragged or clicked; pause Play and
                guide interactively from there
      free      nothing driven yet
    Yaw here is the tour's own (SDK) yaw in degrees."""

    SETTLE_S = 2.5          # fly transition + SDK pose echo
    YAW_TOL = 30.0

    def __init__(self):
        self.expect: Optional[tuple] = None

    def moved(self, pano: str, yaw: Optional[float], now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self.expect = (pano, yaw, now + self.SETTLE_S)

    def clear(self) -> None:
        self.expect = None

    def classify(self, pano: Optional[str], yaw: Optional[float],
                 now: Optional[float] = None) -> str:
        now = time.monotonic() if now is None else now
        if self.expect is None:
            return "free"
        want, want_yaw, until = self.expect
        if now < until:
            return "echo"
        same_view = yaw is None or want_yaw is None or \
            abs((float(yaw) - want_yaw + 180) % 360 - 180) <= self.YAW_TOL
        return "echo" if pano == want and same_view else "override"
