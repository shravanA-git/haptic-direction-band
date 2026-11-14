"""Dashboard server: serves dashboard/index.html and a WebSocket at /ws.

The Python app owns the serial port; the page only sends commands here.

WebSocket, page -> server (JSON):
  {"cmd":"pulse","motor":0,"ms":150}   {"cmd":"stop"}
  {"cmd":"cue","name":"generic|centered|lost|directional","motor":k}
  {"cmd":"record_start"}  {"cmd":"record_stop"}  {"cmd":"alert_down"} {"cmd":"alert_up"}
  {"cmd":"replay","name":"fixtures/back_left.json"}  {"cmd":"replay_stop"}
  {"cmd":"target","name":"banana"}   (typed target for vision)
  {"cmd":"action","down":bool}       (dashboard hold-to-talk = BTN ACTION DOWN/UP)
  {"cmd":"name","name":"Shravan"}    (name to listen for; comma-separated ok)
  {"cmd":"bench","name":"gate|walk|cardinals"}   (gate checks by feel; STOP cancels)
  {"cmd":"label","truth":k|null|"false","id":n?}  (ground truth for the latest/given alert)
  {"cmd":"missed","truth":k}         (a call the band gave no cue for)
server -> page:
  {"type":"mic","ms":..,"levels":[f,r,b,l],"replay":bool}
  {"type":"pulse","motor":k,"ms":..,"dry":bool}   {"type":"stop"}
  {"type":"btn","name":"ACTION","down":bool}      {"type":"raw","text":..}
  {"type":"status",...}  {"type":"estimate",...}  {"type":"traces","items":[..]}
  {"type":"state","object":..,"name":..}          {"type":"log","text":..}
  {"type":"alert",...AlertLog record}  {"type":"label",...}  {"type":"alerts","items":[..]}
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Optional

from aiohttp import WSMsgType, web

from . import direction, haptics, trace
from .link import ButtonEvent, Link, MicFrame, RawLine
from .notify import Notifier
from .recall import AlertLog

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
FIXTURES = ROOT / "tests" / "fixtures"
RECORDINGS = ROOT / "recordings"


class App:
    def __init__(self, link: Link, alert_log: Optional[AlertLog] = None):
        self.link = link
        self.alerts = alert_log or AlertLog()        # memory-only unless main gives a path
        self.notify = Notifier(link.pulse, self.alerts, dry_send=self._dry_pulse)
        self.alerts.subscribe(self.emit)
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.recorder = trace.Recorder()
        self.live = direction.LiveEstimator()   # MIC buffer + clock sync for name alerts
        self.names = None
        self.replayer: Optional[trace.Replayer] = None
        self.state = {"object": None, "name": None, "heard": None}
        self.vision = None
        self.guide = None
        self.speech = None
        link.subscribe(self._on_event)
        link.subscribe(self.live)
        link.subscribe(self.recorder)
        link.on_sent(self._on_sent)

    # ---------------------------------------------------------- broadcast
    def emit(self, msg: dict) -> None:
        """Thread-safe: may be called from the link or replay threads."""
        if self.loop is None or self.loop.is_closed():
            return
        data = json.dumps(msg)
        self.loop.call_soon_threadsafe(self._fanout, data)

    def _fanout(self, data: str) -> None:
        for ws in list(self.clients):
            if ws.closed:
                self.clients.discard(ws)
            else:
                asyncio.ensure_future(ws.send_str(data))

    # -------------------------------------------------------------- vision
    def attach_vision(self, vision, guide) -> None:
        """vision: app.vision.Vision (or anything with on_change/set_target/
        pause/resume); guide: app.guide.ObjectGuide driving the band."""
        self.vision, self.guide = vision, guide
        vision.on_change(self._on_vision)
        if getattr(guide, "on_cue", ...) is None:   # log find cues the guide plays
            guide.on_cue = self._on_find_cue

    def _on_find_cue(self, kind, motor, info) -> None:
        self.notify.cue("find", kind, motor, info.get("confidence"), info.get("latency_ms"),
                        play=False, target=info.get("target"), repeat=info.get("repeat") or None)

    def attach_speech(self, speech) -> None:
        """speech: app.speech.Speech (or anything with on_button and an
        on_result attribute). Action held = record; release = transcribe."""
        self.speech = speech
        speech.on_result = self._on_speech

    def _on_speech(self, r) -> None:
        if getattr(r, "error", None):
            self.emit({"type": "log", "text": f"speech: {r.error} (typed Find box still works)"})
            return
        self.state["heard"] = r.text or "–"
        self.emit({"type": "state", **self.state})
        if not r.target:
            self.emit({"type": "log", "text": f"heard \"{r.text}\": no known object; type it in Find"})
        elif self.vision:
            cls = self.vision.set_target(r.target)
            self.emit({"type": "log", "text": f"heard \"{r.text}\" -> target {cls or r.target}"})
        else:
            self.emit({"type": "log", "text": f"heard target {r.target}, but vision isn't running"})

    def attach_names(self, listener) -> None:
        """listener: app.speech.NameListener (or anything with on_hit,
        pause/resume, set_names). A hit -> direction estimate -> cue."""
        self.names = listener
        listener.on_hit = self.on_name_hit

    def on_name_hit(self, hit) -> None:
        """Name heard at hit.start..hit.end (laptop monotonic seconds):
        estimate direction from the band mics over that window and buzz."""
        e, win = self.live.estimate(hit.start, hit.end)
        latency_ms = (time.monotonic() - hit.end) * 1000    # end of the name -> cue sent
        self.notify.cue("name", "generic" if e.motor is None else "directional", e.motor,
                        e.consistency if e.motor is not None else None, latency_ms,
                        reason=e.reason, window=list(win) if win else None,
                        heard=getattr(hit, "heard", None), strength=e.strength)
        where = (f"{haptics.MOTOR_NAMES[e.motor]} (motor {e.motor})" if e.motor is not None
                 else f"generic: {e.reason}")
        self.state["name"] = f"{hit.name} → {where}"
        self.emit({"type": "state", **self.state})
        self.emit({"type": "estimate", "trace": f"live: {hit.name}", "motor": e.motor,
                   "angle": e.angle_deg, "consistency": round(e.consistency, 2),
                   "strength": round(e.strength, 2), "frames": e.frames_used,
                   "reason": e.reason, "window": win,
                   "latency": round(getattr(hit, "latency", 0.0), 2)})
        if win and self.recorder.active:   # recorded traces get the alert window for replay
            self.recorder.mark_alert(*win)

    def _action(self, down: bool) -> None:
        """Action button (board or dashboard): alert window while recording,
        no cues while held, hold-to-talk speech."""
        if self.recorder.active:
            (self.recorder.alert_down if down else self.recorder.alert_up)()
        if self.vision:
            for obj in (self.vision, self.guide):
                (obj.pause if down else obj.resume)()
        if self.names:   # hold-to-talk owns the mic while Action is held
            (self.names.pause if down else self.names.resume)()
        if self.speech:
            try:
                self.speech.on_button(down)
            except Exception as e:  # missing sounddevice, no mic permission...
                self.emit({"type": "log", "text": f"speech: {e}"})

    def _on_vision(self, d) -> None:
        self.guide.on_decision(d)
        target = getattr(d, "target", None)
        self.state["object"] = f"{target or '–'}: {d.state}"
        self.emit({"type": "state", **self.state})

    # ----------------------------------------------------------- link side
    def _on_event(self, ev) -> None:
        if isinstance(ev, MicFrame):
            self.emit({"type": "mic", "ms": ev.ms, "levels": list(ev.levels), "replay": False})
        elif isinstance(ev, ButtonEvent):
            self.emit({"type": "btn", "name": ev.name, "down": ev.down})
            if ev.name == "ACTION":
                self._action(ev.down)
        elif isinstance(ev, RawLine):
            self.emit({"type": "raw", "text": ev.text[:200]})

    def _on_sent(self, line: str) -> None:
        p = line.split()
        if p[0] == "PULSE":
            self.emit({"type": "pulse", "motor": int(p[1]), "ms": int(p[2]), "dry": False})
        elif p[0] == "STOP":
            self.emit({"type": "stop"})

    def _dry_pulse(self, motor: int, ms: int) -> None:
        self.emit({"type": "pulse", "motor": motor, "ms": ms, "dry": True})

    # ------------------------------------------------------------ commands
    def handle(self, m: dict) -> Optional[dict]:
        cmd = m.get("cmd")
        if cmd == "stop":
            haptics.cancel_all()
            self.link.stop()
        elif cmd == "pulse":
            self.link.pulse(int(m["motor"]), int(m.get("ms", haptics.TEST_MS)))
        elif cmd == "cue":
            name = m.get("name")
            if name in ("generic", "centered", "lost", "sound"):
                self.notify.cue("manual", name)
            else:
                self.notify.cue("manual", "directional", int(m.get("motor", 0)))
        elif cmd == "action":
            down = bool(m.get("down"))
            self.emit({"type": "btn", "name": "ACTION", "down": down})
            self._action(down)
        elif cmd == "name":
            if not self.names:
                return {"type": "log", "text": "name listener not running (start with --name)"}
            self.names.set_names(m.get("name") or "")
            return {"type": "log", "text": f"listening for: {', '.join(self.names.names) or 'nothing'}"}
        elif cmd == "bench":
            return self.bench(m.get("name"))
        elif cmd == "label":
            return self.label(m.get("truth"), m.get("id"))
        elif cmd == "missed":
            self.alerts.label(None, self._truth(m.get("truth")), missed=True)
            return {"type": "log", "text": "logged a missed call"}
        elif cmd == "alerts":
            return {"type": "alerts", "items": self.alerts.recent(30)}
        elif cmd == "record_start":
            self.recorder.start()
            return {"type": "log", "text": "recording…"}
        elif cmd == "alert_down":
            self.recorder.alert_down()
        elif cmd == "alert_up":
            self.recorder.alert_up()
        elif cmd == "record_stop":
            t = self.recorder.stop()
            path = trace.save(t, RECORDINGS / time.strftime("trace-%H%M%S.json"))
            self.emit(self.traces_msg())
            return {"type": "log", "text": f"saved {path.relative_to(ROOT)} "
                                           f"({len(t['frames'])} frames, {len(t['alerts'])} alerts)"}
        elif cmd == "replay":
            return self.replay(m["name"], float(m.get("speed", 1.0)))
        elif cmd == "replay_stop" and self.replayer:
            self.replayer.cancel()
        elif cmd == "traces":
            return self.traces_msg()
        elif cmd == "target":
            if not self.vision:
                return {"type": "log", "text": "vision not running (start with --target)"}
            cls = self.vision.set_target(m.get("name") or None)
            return {"type": "log", "text": f"target: {cls or 'none (not a known class)'}"}
        elif cmd == "state":  # vision/speech status from other code paths
            self.state.update({k: m[k] for k in ("object", "name", "heard") if k in m})
            self.emit({"type": "state", **self.state})
        return None

    def bench(self, name: str) -> dict:
        """H3 gate: short pulse, long pulse cut by STOP, finite pulse.
        walk/cardinals: one pulse per motor in order (H9 mapping by feel)."""
        if name == "gate":
            haptics.play([(0, 0, haptics.TEST_MS), (1000, 0, 1000)], self.link.pulse)
            born = haptics.epoch()

            def cut():  # skipped if someone pressed STOP first
                if haptics.epoch() == born:
                    self.handle({"cmd": "stop"})
                    haptics.play([(1200, 0, 300)], self.link.pulse)
            threading.Timer(1.3, cut).start()
            return {"type": "log", "text": "gate: motor 0 short buzz, long buzz cut by STOP "
                                           "at 0.3 s, then a 300 ms buzz that must end by itself"}
        motors = {"walk": range(8), "cardinals": (0, 2, 4, 6)}.get(name)
        if motors is None:
            return {"type": "log", "text": f"unknown bench sequence {name!r}"}
        haptics.play(haptics.walk(motors), self.link.pulse)
        return {"type": "log", "text": f"{name}: motors {list(motors)} in order, 0.7 s apart; "
                                       "each should buzz where the ring lights up"}

    @staticmethod
    def _truth(v):
        if v in (None, "none", "generic", ""):
            return None
        return "false" if v == "false" else int(v)

    def label(self, truth, ref=None) -> dict:
        """Ground truth for an alert (default: the newest unlabelled one)."""
        if ref is None:
            a = self.alerts.latest()
            if a is None:
                return {"type": "log", "text": "no unlabelled alert to label"}
            ref = a["id"]
        rec = self.alerts.label(int(ref), self._truth(truth))
        t = rec["truth"]
        what = ("false alert" if t == "false" else "no clear direction" if t is None
                else haptics.MOTOR_NAMES[t])
        return {"type": "log", "text": f"alert #{ref} labelled: {what}"}

    def traces_msg(self) -> dict:
        items = [f"fixtures/{p.name}" for p in sorted(FIXTURES.glob("*.json"))]
        items += [f"recordings/{p.name}" for p in sorted(RECORDINGS.glob("*.json"))]
        return {"type": "traces", "items": items}

    def _resolve(self, name: str) -> Path:
        base, _, fn = name.partition("/")
        root = {"fixtures": FIXTURES, "recordings": RECORDINGS}[base]
        p = (root / fn).resolve()
        if p.parent != root.resolve():
            raise ValueError("bad trace name")
        return p

    def replay(self, name: str, speed: float = 1.0) -> dict:
        """Replay a trace to the page and the estimator. Never drives motors."""
        if self.replayer:
            self.replayer.cancel()
        t = trace.load(self._resolve(name))

        def on_frame(f: MicFrame):
            self.emit({"type": "mic", "ms": f.ms, "levels": list(f.levels), "replay": True})

        def on_done():
            for a in t["alerts"]:
                e = direction.estimate(t["frames"], a["startMs"], a["endMs"])
                self.emit({"type": "estimate", "trace": name, "motor": e.motor,
                           "angle": e.angle_deg, "consistency": round(e.consistency, 2),
                           "strength": round(e.strength, 2), "reason": e.reason})
                self.notify.cue("replay", "generic" if e.motor is None else "directional",
                                e.motor, e.consistency if e.motor is not None else None,
                                dry=True, trace=name, window=[a["startMs"], a["endMs"]],
                                reason=e.reason)

        self.replayer = trace.Replayer(t, on_frame, speed, on_done).start()
        return {"type": "log", "text": f"replaying {name} (dry run, motors not driven)"}

    def status(self) -> dict:
        return {"type": "status", "device": self.link.device,
                "connected": self.link.connected(), "error": self.link.error,
                "frames": self.link.frames_rx, "recording": self.recorder.active,
                "health": self.link.health()}

    # ---------------------------------------------------------------- web
    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps(self.status()))
        await ws.send_str(json.dumps(self.traces_msg()))
        await ws.send_str(json.dumps({"type": "alerts", "items": self.alerts.recent(30)}))
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                reply = self.handle(json.loads(msg.data))
            except Exception as e:
                reply = {"type": "log", "text": f"error: {e}"}
            if reply:
                await ws.send_str(json.dumps(reply))
        self.clients.discard(ws)
        return ws

    async def index(self, request):
        return web.FileResponse(DASH / "index.html")

    async def _status_loop(self):
        while True:
            self.emit(self.status())
            await asyncio.sleep(0.5)

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)

        async def on_startup(_):
            self.loop = asyncio.get_running_loop()
            self._task = asyncio.ensure_future(self._status_loop())

        async def on_cleanup(_):
            self._task.cancel()
            for ws in list(self.clients):
                await ws.close()

        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)
        return app
