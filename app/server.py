"""Dashboard server: serves dashboard/index.html and a WebSocket at /ws.

The Python app owns the serial port; the page only sends commands here.

WebSocket, page -> server (JSON):
  {"cmd":"pulse","motor":0,"ms":150}   {"cmd":"stop"}
  {"cmd":"cue","name":"generic|centered|lost|directional","motor":k}
  {"cmd":"record_start"}  {"cmd":"record_stop"}  {"cmd":"alert_down"} {"cmd":"alert_up"}
  {"cmd":"replay","name":"fixtures/back_left.json"}  {"cmd":"replay_stop"}
server -> page:
  {"type":"mic","ms":..,"levels":[f,r,b,l],"replay":bool}
  {"type":"pulse","motor":k,"ms":..,"dry":bool}   {"type":"stop"}
  {"type":"btn","name":"ACTION","down":bool}      {"type":"raw","text":..}
  {"type":"status",...}  {"type":"estimate",...}  {"type":"traces","items":[..]}
  {"type":"state","object":..,"name":..}          {"type":"log","text":..}
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from aiohttp import WSMsgType, web

from . import direction, haptics, trace
from .link import ButtonEvent, Link, MicFrame, RawLine

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
FIXTURES = ROOT / "tests" / "fixtures"
RECORDINGS = ROOT / "recordings"


class App:
    def __init__(self, link: Link):
        self.link = link
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.recorder = trace.Recorder()
        self.clock = direction.ClockSync()
        self.replayer: Optional[trace.Replayer] = None
        self.state = {"object": None, "name": None}
        link.subscribe(self._on_event)
        link.subscribe(self.recorder)
        link.on_sent(self._on_sent)

    # ---------------------------------------------------------- broadcast
    def emit(self, msg: dict) -> None:
        """Thread-safe: may be called from the link or replay threads."""
        if self.loop is None:
            return
        data = json.dumps(msg)
        self.loop.call_soon_threadsafe(self._fanout, data)

    def _fanout(self, data: str) -> None:
        for ws in list(self.clients):
            if ws.closed:
                self.clients.discard(ws)
            else:
                asyncio.ensure_future(ws.send_str(data))

    # ----------------------------------------------------------- link side
    def _on_event(self, ev) -> None:
        if isinstance(ev, MicFrame):
            self.clock.add(ev.ms, ev.rx)
            self.emit({"type": "mic", "ms": ev.ms, "levels": list(ev.levels), "replay": False})
        elif isinstance(ev, ButtonEvent):
            self.emit({"type": "btn", "name": ev.name, "down": ev.down})
            if ev.name == "ACTION" and self.recorder.active:
                (self.recorder.alert_down if ev.down else self.recorder.alert_up)()
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
            self.link.stop()
        elif cmd == "pulse":
            self.link.pulse(int(m["motor"]), int(m.get("ms", haptics.TEST_MS)))
        elif cmd == "cue":
            name = m.get("name")
            cue = {"generic": haptics.generic_attention, "centered": haptics.centered,
                   "lost": haptics.lost}.get(name)
            cue = cue() if cue else haptics.directional(int(m.get("motor", 0)))
            haptics.play(cue, self.link.pulse)
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
        elif cmd == "state":  # vision/speech status from other code paths
            self.state.update({k: m[k] for k in ("object", "name") if k in m})
            self.emit({"type": "state", **self.state})
        return None

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
                haptics.play(haptics.name_alert(e.motor), self._dry_pulse)

        self.replayer = trace.Replayer(t, on_frame, speed, on_done).start()
        return {"type": "log", "text": f"replaying {name} (dry run, motors not driven)"}

    def status(self) -> dict:
        return {"type": "status", "device": self.link.device,
                "connected": self.link.connected(), "error": self.link.error,
                "frames": self.link.frames_rx, "recording": self.recorder.active}

    # ---------------------------------------------------------------- web
    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps(self.status()))
        await ws.send_str(json.dumps(self.traces_msg()))
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
