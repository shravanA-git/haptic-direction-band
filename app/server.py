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
  {"cmd":"ask","text":"who called me?"}   (Recall chatbot; reply {"type":"recall",..})
  {"cmd":"setting","setting":"intensity","value":"high"}   {"cmd":"settings"}
  {"cmd":"caregiver_test"}           (send a test text; caregiver_texts toggles via "setting")
  {"cmd":"teach","seconds":2.5,"label":"doorbell","add":bool}   (Teach a Sound: make the sound once;
                                     add=true adds an example to that label)
  {"cmd":"forget_sound","label":"doorbell"?}   (no label: forget all)   {"cmd":"sounds"}
  {"cmd":"avoid","on":bool}          (obstacle avoidance on/off, with --avoid)
HTTP GET /manifest.webmanifest, /icon.svg, /icon-{180,192,512}.png: "Add to
  Home Screen" (full-screen app on a phone). With a LAN token (main --lan), every
  other request from another device needs ?k=<token> once (then a cookie).
HTTP GET /video: MJPEG (multipart/x-mixed-replace) of the vision camera's
  annotated frames (tracked box, dead band, "target: STATE"); 503 without vision.
server -> page:
  {"type":"mic","ms":..,"levels":[f,r,b,l],"replay":bool}
  {"type":"pulse","motor":k,"ms":..,"dry":bool}   {"type":"stop"}
  {"type":"btn","name":"ACTION","down":bool}      {"type":"raw","text":..}
  {"type":"status",...,"faults":{"serial"|"camera"|"mic"|"speech": text}}  {"type":"estimate",...}  {"type":"traces","items":[..]}
  {"type":"state","object":..,"name":..,"heard":..,"sound":..}  {"type":"log","text":..}
  {"type":"sounds","items":[{"label","category","description","pitch_hz","examples"}]}
  {"type":"settings",...Settings.snapshot()}  {"type":"caregiver",...Caregiver.status()}  {"type":"recall","q":..,"a":..,"via":..,"actions":[..]}
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
from .recall_bot import RecallBot
from .settings import Settings

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"


def icon_png_bytes(size: int) -> bytes:
    """The app icon (belt ring seen from above, front motor lit) as a PNG, drawn
    pixel by pixel and compressed with zlib: no image libraries needed."""
    import math
    import struct
    import zlib
    bg, ring, pad, edge, lit = (18, 17, 16), (69, 66, 61), (46, 44, 41), (142, 136, 126), (240, 168, 48)
    f = size / 512.0
    cx, cy, rx, ry, half = 256, 276, 150, 125, 13
    pads = [(256 + rx * math.sin(k * math.pi / 4), cy - ry * math.cos(k * math.pi / 4)) for k in range(8)]
    rows = []
    for py in range(size):
        y = (py + 0.5) / f
        row = bytearray(b"\x00")
        for px in range(size):
            x = (px + 0.5) / f
            c = bg
            d = math.hypot((x - cx) / rx, (y - cy) / ry)
            if abs(d - 1) * min(rx, ry) <= half:
                c = ring
            if 88 <= y <= 124 and abs(x - 256) <= (y - 88) * 24 / 36:
                c = lit
            for k, (mx, my) in enumerate(pads):
                if abs(x - mx) <= 22 and abs(y - my) <= 22:
                    c = edge if (abs(x - mx) > 18 or abs(y - my) > 18) else (lit if k == 0 else pad)
            row += bytes(c)
        rows.append(bytes(row))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))
FIXTURES = ROOT / "tests" / "fixtures"
RECORDINGS = ROOT / "recordings"


class App:
    def __init__(self, link: Link, alert_log: Optional[AlertLog] = None,
                 recall_mode: str = "offline"):
        self.link = link
        self.alerts = alert_log or AlertLog()        # memory-only unless main gives a path
        self.settings = Settings()
        self.settings.on_change(self._on_setting)
        self.notify = Notifier(link.pulse, self.alerts, dry_send=self._dry_pulse,
                               settings=self.settings)
        self.recall_mode = recall_mode
        self._recall: Optional[RecallBot] = None
        self.caregiver = None
        # Watchdog (check_health, every 0.5 s): what is broken right now, for the
        # dashboard banner. The app keeps running whatever fails.
        self.faults: dict = {}
        self._action_down = False
        self._vision_stall_since: Optional[float] = None
        self._vision_restart_at = 0.0
        self._restarting = False
        self._mic_seen: dict = {}          # listener name -> (sample count, since)
        self.alerts.subscribe(self.emit)
        self.clients: set[web.WebSocketResponse] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.recorder = trace.Recorder()
        self.live = direction.LiveEstimator()   # MIC buffer + clock sync for name alerts
        self.names = None
        self.replayer: Optional[trace.Replayer] = None
        self.state = {"object": None, "name": None, "heard": None, "sound": None}
        self.sound = None
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
        if hasattr(vision, "on_frame"):          # live offset/size for hot/cold pulses
            vision.on_frame(guide.on_frame)
        if getattr(guide, "on_cue", ...) is None:   # log find cues the guide plays
            guide.on_cue = self._on_find_cue

    def attach_avoid(self, loop) -> None:
        """loop: app.avoid.AvoidLoop. Its cues play here (intensity, mute "avoid")
        and are logged; decisions go to the dashboard as {"type":"avoid",...}."""
        self.avoid = loop
        loop.play = self._play_avoid
        loop.halt = lambda: (haptics.cancel_all(), self.link.stop())   # STOP before a warning
        loop.on_decision = lambda d: self.emit(d.to_msg())

    def _play_avoid(self, cue) -> None:
        d = self.avoid.decision if self.avoid else None
        stop = d is not None and d.action == "stop"
        near = d.nearest if d is not None else None
        self.play_cue(cue, "avoid")
        # Log the first cue of each action (and every 5 s while it lasts), not every
        # repeat: a blocked path would otherwise flood the alert table at 1.25 Hz.
        key, now = (d.action if d else None, cue[0][1]), time.monotonic()
        last = getattr(self, "_avoid_logged", (None, 0.0))
        if last[0] == key and now - last[1] < 5.0:
            return
        self._avoid_logged = (key, now)
        # logged as existing cue kinds: crash warning = generic (front), steer = directional
        self.notify.cue("avoid", "generic" if stop else "directional",
                        None if stop else cue[0][1], play=False, action=d.action if d else None,
                        obstacle=near.cls if near else None,
                        metres=round(near.distance_m, 2) if near else None)

    def _on_find_cue(self, kind, motor, info) -> None:
        self.notify.cue("find", kind, motor, info.get("confidence"), info.get("latency_ms"),
                        play=False, target=info.get("target"), repeat=info.get("repeat") or None)

    def attach_speech(self, speech) -> None:
        """speech: app.speech.Speech (or anything with on_button and an
        on_result attribute). Action held = record; release = transcribe."""
        self.speech = speech
        speech.on_result = self._on_speech

    def _on_speech(self, r) -> None:
        """Hold-to-talk result. The wearer may not see the dashboard, so every
        outcome also buzzes: "got it, searching" or "can't look for that"."""
        if getattr(r, "error", None):
            if str(r.error).startswith("mic"):
                self.faults["speech"] = f"hold-to-talk {r.error}"
            self.emit({"type": "log", "text": f"speech: {r.error} (typed Find box still works)"})
            self.notify.cue("find", "unknown", reason=r.error)
            return
        self.faults.pop("speech", None)
        self.state["heard"] = r.text or "–"
        self.emit({"type": "state", **self.state})
        if not r.target:
            self.emit({"type": "log", "text": f"heard \"{r.text}\": no known object; type it in Find"})
            self.notify.cue("find", "unknown", heard=r.text, reason="no known object")
        elif self.vision:
            cls = self.vision.set_target(r.target)
            self.emit({"type": "log", "text": f"heard \"{r.text}\" -> target {cls or r.target}"})
            self._target_feedback(cls, heard=r.text)
        else:
            self.emit({"type": "log", "text": f"heard target {r.target}, but vision isn't running"})
            self.notify.cue("find", "unknown", heard=r.text, reason="vision not running")

    def _set_open_target(self, typed: str) -> None:
        cls = self.vision.set_target(typed)
        self._target_feedback(cls, typed=typed)
        self.emit({"type": "log", "text": f"target: {cls or 'none (can not look for that)'}"})

    def _target_feedback(self, cls, **detail) -> None:
        if cls:
            self.notify.cue("find", "ack", target=cls, **detail)
        else:
            self.notify.cue("find", "unknown", reason="not a known class", **detail)

    def attach_names(self, listener) -> None:
        """listener: app.speech.NameListener (or anything with on_hit,
        pause/resume, set_names). A hit -> direction estimate -> cue."""
        self.names = listener
        self.settings.names = ", ".join(getattr(listener, "names", []) or []) or None
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

    # ------------------------------------------------------------ caregiver
    def attach_caregiver(self, caregiver) -> None:
        """caregiver: app.notify.Caregiver. Gets every AlertLog record; texts
        only while settings.caregiver_texts is on."""
        self.caregiver = caregiver
        self.alerts.subscribe(caregiver)
        caregiver.on_status(lambda st: self.emit({"type": "caregiver", **st}))

    agent_status = None              # set by main when --agent is on: callable -> counters dict

    def caregiver_test(self) -> dict:
        if not self.caregiver:
            return {"type": "log", "text": "caregiver texting not set up"}

        def run():
            r = self.caregiver.send_test()
            self.emit({"type": "log", "text": "caregiver test text "
                       + ("sent" if r["ok"] else f"failed: {r['error']}")})
        threading.Thread(target=run, name="caregiver-test", daemon=True).start()
        return {"type": "log", "text": f"sending a test text via {self.caregiver.sender.name}..."}

    # ------------------------------------------------------------- settings
    def _on_setting(self, setting: str, value) -> None:
        if setting == "names" and self.names:
            self.names.set_names(value)
        elif setting == "sound_threshold" and self.sound:
            self.sound.threshold = float(value)
        elif setting == "caregiver_texts" and self.caregiver:
            self.emit({"type": "caregiver", **self.caregiver.status()})
            if value == "on" and not self.caregiver.configured:
                self.emit({"type": "log", "text": "caregiver texts on, but no phone is set "
                                                   "(CAREGIVER_PHONE in .env)"})
        self.emit({"type": "settings", **self.settings.snapshot()})

    def play_cue(self, cue, source: str):
        """Play a pattern someone else built (ObjectGuide) with the current
        intensity, unless that source is muted. Logging stays with the caller."""
        if self.settings.is_muted(source):
            return None
        return haptics.play(haptics.scaled(cue, self.settings.intensity), self.link.pulse)

    @property
    def recall(self) -> RecallBot:
        if self._recall is None:
            self._recall = RecallBot(self.alerts, self.settings, self.recall_mode)
        return self._recall

    def ask(self, text: str) -> None:
        """Recall chatbot; answers on a thread (Claude calls take seconds)."""
        def run():
            try:
                a = self.recall.ask(text)
                msg = {"type": "recall", "q": text, "a": a.text, "via": a.via,
                       "actions": [{"tool": n, "input": i} for n, i, _ in a.actions]}
            except Exception as e:
                msg = {"type": "recall", "q": text, "a": f"error: {e}", "via": "error",
                       "actions": []}
            self.emit(msg)
        threading.Thread(target=run, name="recall", daemon=True).start()

    def attach_sound(self, listener) -> None:
        """listener: app.listen.SoundListener. A detection -> its own pattern
        (haptics.taught_sound, no direction), logged with source "sound"."""
        self.sound = listener
        self.settings.sound_threshold = getattr(listener, "threshold", None)
        listener.on_detect = self.on_sound
        listener.on_taught = self._on_taught
        if hasattr(listener, "on_heard"):   # YAMNet's top classes for every sound (HDB_YAMNET=1)
            listener.on_heard = lambda text: self.emit({"type": "log", "text": f"sound: {text}"})
        if hasattr(listener, "on_classified"):   # every event's best taught match: near-misses
            listener.on_classified = self._on_classified
        self.state["sound"] = "taught, listening" if listener.template else "not taught yet"
        if getattr(listener, "yamnet", None) is not None:
            self.state["sound"] += " + YAMNet"

    def sounds_msg(self) -> dict:
        """Taught sounds for the dashboard, grouped by category, low pitch first."""
        sl, items = self.sound, []
        if sl is not None:
            for cat, rows in sl.categories().items():
                for label, pitch, desc in rows:
                    items.append({"label": label, "category": cat, "description": desc,
                                  "pitch_hz": round(pitch) if pitch else None,
                                  "examples": len(sl.sounds.get(label, []))})
        return {"type": "sounds", "items": items}

    def on_sound(self, hit) -> None:
        """Taught sound recognised (laptop mic) at hit.start..hit.end. If the band
        is streaming MIC frames, estimate where it came from over that window (as
        for names): the sound's own sweep, then a pulse toward it. Weak, stale or
        missing evidence: the sweep alone."""
        e, win = self.live.estimate(hit.start, hit.end)
        latency_ms = (time.monotonic() - hit.end) * 1000      # end of the sound -> cue sent
        self.notify.cue("sound", "sound", e.motor, hit.score, latency_ms,
                        duration_ms=round((hit.end - hit.start) * 1000),
                        label=getattr(hit, "label", None), category=getattr(hit, "category", None),
                        reason=e.reason, window=list(win) if win else None,
                        strength=e.strength)
        what = getattr(hit, "label", "") or "it"
        cat = getattr(hit, "category", "")
        where = (f" from {haptics.MOTOR_NAMES[e.motor]} (motor {e.motor})" if e.motor is not None
                 else "")
        self.state["sound"] = f"heard {what}" + (f" [{cat}]" if cat else "") + where + \
            f" (match {hit.score:.2f})"
        self.emit({"type": "state", **self.state})
        if e.motor is not None or win:
            self.emit({"type": "estimate", "trace": f"live: sound {what}", "motor": e.motor,
                       "angle": e.angle_deg, "consistency": round(e.consistency, 2),
                       "strength": round(e.strength, 2), "frames": e.frames_used,
                       "reason": e.reason, "window": win})
        if win and self.recorder.active:
            self.recorder.mark_alert(*win)

    def _on_classified(self, sig, label, score: float) -> None:
        """Every sound event vs the taught sounds, so a near-miss is visible."""
        thr = getattr(self.sound, "threshold", 0.8)
        what = getattr(sig, "description", "") or "sound"
        if label is None:
            text = f"sound: event {what}; nothing taught to compare"
        else:
            text = (f"sound: event {what} -> best '{label}' {score:.2f} "
                    f"({'match' if score >= thr else f'needs {thr:.2f}'})")
        print(f"[sound] {text[7:]}", flush=True)
        self.emit({"type": "log", "text": text})

    def _on_taught(self, tpl, msg: str) -> None:
        # a failed teach keeps the sounds taught before
        known = tpl or getattr(self.sound, "template", None)
        self.state["sound"] = (f"taught '{tpl.label}' [{tpl.category}], listening" if tpl
                               else "taught, listening" if known else "not taught yet")
        self.emit({"type": "state", **self.state})
        self.emit({"type": "log", "text": f"teach a sound: {msg}"})
        self.emit(self.sounds_msg())

    def _action(self, down: bool) -> None:
        """Action button (board or dashboard): alert window while recording,
        no cues while held, hold-to-talk speech."""
        self._action_down = down
        if self.recorder.active:
            (self.recorder.alert_down if down else self.recorder.alert_up)()
        if self.vision:
            for obj in (self.vision, self.guide):
                (obj.pause if down else obj.resume)()
        for lst in (self.names, self.sound):  # hold-to-talk owns the mic while held
            if lst:
                (lst.pause if down else lst.resume)()
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
            self.live.note_motor(time.monotonic(), int(p[2]))     # buzz bleeds into the mics
            self.emit({"type": "pulse", "motor": int(p[1]), "ms": int(p[2]), "dry": False})
        elif p[0] == "STOP":
            self.emit({"type": "stop"})

    def _dry_pulse(self, motor: int, ms: int) -> None:
        self.emit({"type": "pulse", "motor": motor, "ms": ms, "dry": True})

    # ------------------------------------------------------------ commands
    def handle(self, m: dict) -> Optional[dict]:
        cmd = m.get("cmd")
        if cmd == "avoid":
            if not getattr(self, "avoid", None):
                return {"type": "log", "text": "avoidance is off (start with --avoid)"}
            self.avoid.enabled = bool(m.get("on", True))
            return {"type": "log", "text": f"avoidance {'on' if self.avoid.enabled else 'off'}"}
        if cmd == "stop":
            haptics.cancel_all()
            self.link.stop()
        elif cmd == "pulse":
            motor, ms = int(m["motor"]), int(m.get("ms", haptics.TEST_MS))
            if haptics.ACTIVE is not None and motor not in haptics.ACTIVE:
                haptics.play([(0, motor, ms)], self.link.pulse)    # a diagonal: its two neighbours
            else:
                self.link.pulse(motor, ms)
        elif cmd == "cue":
            name = m.get("name")
            if name in ("generic", "centered", "lost", "sound", "ack", "unknown"):
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
        elif cmd == "ask":
            self.ask(str(m.get("text", "")))
        elif cmd == "setting":
            try:
                return {"type": "log", "text": self.settings.set(m.get("setting"), m.get("value"))}
            except ValueError as e:
                return {"type": "log", "text": f"setting: {e}"}
        elif cmd == "settings":
            return {"type": "settings", **self.settings.snapshot()}
        elif cmd == "caregiver_test":
            return self.caregiver_test()
        elif cmd == "teach":
            if not self.sound:
                return {"type": "log", "text": "sound listener not running (start with --sound)"}
            secs = min(max(float(m.get("seconds", 2.5)), 0.5), 10.0)
            label = str(m.get("label") or "").strip() or None
            add = bool(m.get("add"))
            self.sound.teach(secs, label=label, add=add)
            what = f"'{label}'" if label else "the sound"
            self.state["sound"] = (f"teaching {what}{' (another example)' if add else ''}: "
                                   f"make it now ({secs:g} s)")
            self.emit({"type": "state", **self.state})
        elif cmd == "forget_sound" and self.sound:
            label = str(m.get("label") or "").strip() or None
            if label and label not in self.sound.sounds:
                return {"type": "log", "text": f"no taught sound called '{label}'"}
            self.sound.forget(label)
            self.state["sound"] = "taught, listening" if self.sound.sounds else "not taught yet"
            self.emit({"type": "state", **self.state})
            self.emit(self.sounds_msg())
            return {"type": "log", "text": f"forgot {repr(label) if label else 'all taught sounds'}"}
        elif cmd == "sounds":
            return self.sounds_msg()
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
            typed = (m.get("name") or "").strip()
            from .vision import resolve_target
            if typed and resolve_target(typed) is None:
                # Not a COCO class: open-vocabulary lookup takes seconds (CLIP text
                # encoding); don't block the dashboard's event loop on it.
                threading.Thread(target=self._set_open_target, args=(typed,),
                                 name="open-target", daemon=True).start()
                return {"type": "log", "text": f"target: looking for '{typed}' (open vocabulary)..."}
            cls = self.vision.set_target(typed or None)
            if typed:
                self._target_feedback(cls, typed=typed)
            return {"type": "log", "text": f"target: {cls or 'none (not a known class)'}"}
        elif cmd == "state":  # vision/speech status from other code paths
            self.state.update({k: m[k] for k in ("object", "name", "heard", "sound") if k in m})
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
        every = haptics.ACTIVE if haptics.ACTIVE is not None else range(8)
        motors = {"walk": every, "cardinals": (0, 2, 4, 6)}.get(name)
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
                "health": self.link.health(), "faults": dict(self.faults),
                "motors": list(haptics.ACTIVE) if haptics.ACTIVE is not None else list(range(8)),
                "agent": self.agent_status() if self.agent_status else None}

    # ------------------------------------------------------------- watchdog
    VISION_STALL_S = 2.0      # no camera frame this long -> fault, target treated as lost
    VISION_RETRY_S = 8.0      # then restart the camera this often until frames return
    MIC_STALL_S = 2.5         # a live mic listener got no audio this long -> restart it

    def check_health(self, now: Optional[float] = None) -> dict:
        """One watchdog pass. Never raises; returns the current faults."""
        now = time.monotonic() if now is None else now
        f = {}
        try:
            if self.link.error:
                f["serial"] = f"band link: {self.link.error} (reconnecting; firmware stops motors)"
            if self.vision is not None:
                self._check_vision(now, f)
            for name, lst in (("names", self.names), ("sound", self.sound)):
                if lst is not None:
                    self._check_mic(name, lst, now, f)
            if self.faults.get("speech"):
                f["speech"] = self.faults["speech"]
        except Exception as e:      # the watchdog itself must never take the app down
            f["watchdog"] = f"{type(e).__name__}: {e}"
        changed = f != self.faults
        self.faults = f
        if changed:
            self.emit({"type": "log", "text": "faults: " + ("; ".join(f.values()) or "none")})
        return f

    def _check_vision(self, now: float, f: dict) -> None:
        v = self.vision
        last = getattr(v.latest(), "t", None) if hasattr(v, "latest") else None
        running = getattr(v, "_running", True)
        stalled = (not self._action_down and not getattr(v, "_paused", False)
                   and (not running or last is None or now - last > self.VISION_STALL_S))
        if not stalled:
            self._vision_stall_since = None
            if getattr(v, "error", None) and not running:
                f["camera"] = f"camera: {v.error}"
            return
        if self._vision_stall_since is None:
            self._vision_stall_since = now
            self._vision_restart_at = now + self.VISION_RETRY_S
            if self.guide is not None:   # stop directional repeats from a frozen decision
                from types import SimpleNamespace
                self.guide.on_decision(SimpleNamespace(
                    state="lost", target=getattr(v.latest(), "target", None), t=now,
                    confidence=0.0, offset=None, changed=True))
        age = now - (last if last is not None else self._vision_stall_since)
        f["camera"] = (f"camera: no frames for {age:.0f} s"
                       + (f" ({v.error})" if getattr(v, "error", None) else "")
                       + "; find-an-object paused, retrying")
        if now >= self._vision_restart_at and not self._restarting:
            self._vision_restart_at = now + self.VISION_RETRY_S
            self._restarting = True

            def restart():
                try:
                    v.stop()
                    ok = v.start()
                    self.emit({"type": "log", "text": "camera restarted" if ok else
                               f"camera restart failed: {getattr(v, 'error', '?')}"})
                except Exception as e:
                    self.emit({"type": "log", "text": f"camera restart failed: {e}"})
                finally:
                    self._restarting = False
            threading.Thread(target=restart, name="camera-restart", daemon=True).start()

    def _check_mic(self, name: str, lst, now: float, f: dict) -> None:
        if not getattr(lst, "live", True) or getattr(lst, "_stream", 1) is None \
                or getattr(lst, "paused", False) or self._action_down:
            self._mic_seen.pop(name, None)
            return
        n = getattr(lst, "_n", None)
        if n is None:
            return
        seen = self._mic_seen.get(name)
        if seen is None or seen[0] != n:
            self._mic_seen[name] = (n, now)
            return
        if now - seen[1] < self.MIC_STALL_S:
            return
        f["mic"] = f"mic: no audio for {now - seen[1]:.0f} s ({name} listener); restarting"
        self._mic_seen[name] = (n, now)          # retry at most every MIC_STALL_S
        try:
            lst.stop()
            lst.start()
        except Exception as e:
            f["mic"] = f"mic: {name} listener can't open the mic ({type(e).__name__}: {e})"

    def start_watchdog(self, every_s: float = 0.5) -> None:
        def run():
            while True:
                self.check_health()
                time.sleep(every_s)
        threading.Thread(target=run, name="watchdog", daemon=True).start()

    # ---------------------------------------------------------------- web
    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        self.clients.add(ws)
        await ws.send_str(json.dumps(self.status()))
        await ws.send_str(json.dumps(self.traces_msg()))
        await ws.send_str(json.dumps({"type": "alerts", "items": self.alerts.recent(30)}))
        await ws.send_str(json.dumps({"type": "settings", **self.settings.snapshot()}))
        if self.caregiver:
            await ws.send_str(json.dumps({"type": "caregiver", **self.caregiver.status()}))
        if self.sound:
            await ws.send_str(json.dumps(self.sounds_msg()))
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

    VIDEO_FPS = 12          # preview cap; the detector runs at ~10 fps anyway
    VIDEO_QUALITY = 70

    def _jpeg(self, frame) -> Optional[bytes]:
        """Encode once per camera frame, shared by every viewer."""
        key = id(frame)
        cached = getattr(self, "_jpeg_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.VIDEO_QUALITY])
        data = buf.tobytes() if ok else None
        self._jpeg_cache = (key, data, frame)   # keep frame alive so id() stays unique
        return data

    async def video(self, request):
        """Live camera preview for the dashboard <img src="/video">."""
        v = self.vision
        if v is None or not hasattr(v, "latest_frame"):
            return web.Response(status=503, text="vision is off")
        resp = web.StreamResponse(headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-cache, no-store", "Pragma": "no-cache"})
        await resp.prepare(request)
        loop = asyncio.get_running_loop()
        last = None
        try:
            while True:
                frame = v.latest_frame()
                if frame is not None and frame is not last:
                    last = frame
                    data = await loop.run_in_executor(None, self._jpeg, frame)
                    if data:
                        await resp.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(data)).encode()
                                         + b"\r\n\r\n" + data + b"\r\n")
                await asyncio.sleep(1.0 / self.VIDEO_FPS)
        except (ConnectionResetError, RuntimeError):
            pass   # viewer closed the tab
        return resp

    async def _status_loop(self):
        while True:
            self.emit(self.status())
            await asyncio.sleep(0.5)

    # ------------------------------------------------------- phone / LAN
    token: Optional[str] = None      # set by main --lan: phones need ?k=<token> once
    local_ok = True                  # this laptop itself never needs the token

    PUBLIC = ("/manifest.webmanifest", "/icon.svg", "/icon-180.png", "/icon-192.png", "/icon-512.png")

    @web.middleware
    async def _guard(self, request, handler):
        """No accounts: on --lan, a device other than this laptop must bring the
        token printed in the URL once; it's then kept in a cookie (the page's
        WebSocket and camera image send it automatically)."""
        local = self.local_ok and request.remote in ("127.0.0.1", "::1")
        if self.token and not local \
                and request.path not in self.PUBLIC:
            k = request.query.get("k")
            if k != self.token and request.cookies.get("hdb_k") != self.token:
                return web.Response(status=403, text="Open the address printed by the laptop "
                                                     "(it ends in ?k=...).")
            resp = await handler(request)
            if k == self.token and not getattr(resp, "prepared", False):
                resp.set_cookie("hdb_k", self.token, max_age=7 * 24 * 3600, samesite="Lax")
            return resp
        return await handler(request)

    async def manifest(self, request):
        k = request.query.get("k")
        start = f"/?k={k}" if self.token and k == self.token else "/"
        return web.json_response({
            "name": "Haptic Direction Band", "short_name": "HDB", "start_url": start,
            "display": "standalone", "orientation": "any",
            "background_color": "#121110", "theme_color": "#121110",
            "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"},
                      {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                      {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png",
                       "purpose": "any maskable"}]},
            content_type="application/manifest+json")

    @staticmethod
    def _icon_svg() -> str:
        import math
        pads = "".join(
            f'<rect x="{256 + 150 * math.sin(k * math.pi / 4) - 22:.0f}" '
            f'y="{276 - 125 * math.cos(k * math.pi / 4) - 22:.0f}" width="44" height="44" rx="6" '
            f'fill="{"#f0a830" if k == 0 else "#2e2c29"}" stroke="#8e887e" stroke-width="4"/>'
            for k in range(8))
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
                '<rect width="512" height="512" fill="#121110"/>'
                '<ellipse cx="256" cy="276" rx="150" ry="125" fill="none" stroke="#45423d" '
                'stroke-width="26"/><polygon points="256,88 232,124 280,124" fill="#f0a830"/>'
                + pads + '</svg>')

    async def icon_svg(self, request):
        return web.Response(text=self._icon_svg(), content_type="image/svg+xml")

    async def icon_png(self, request):
        """The same icon as PNG (iOS home screen ignores SVG): pure Python + zlib."""
        size = {"/icon-180.png": 180, "/icon-512.png": 512}.get(request.path, 192)
        cache = getattr(self, "_icons", {})
        if size not in cache:
            cache[size] = await asyncio.get_running_loop().run_in_executor(None, icon_png_bytes, size)
            self._icons = cache
        return web.Response(body=cache[size], content_type="image/png")

    def make_app(self) -> web.Application:
        app = web.Application(middlewares=[self._guard])
        app.router.add_get("/", self.index)
        app.router.add_get("/manifest.webmanifest", self.manifest)
        app.router.add_get("/icon.svg", self.icon_svg)
        app.router.add_get("/icon-180.png", self.icon_png)
        app.router.add_get("/icon-192.png", self.icon_png)
        app.router.add_get("/icon-512.png", self.icon_png)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_get("/video", self.video)

        async def on_startup(_):
            self.loop = asyncio.get_running_loop()
            self._task = asyncio.ensure_future(self._status_loop())
            if not getattr(self, "_watchdog_started", False):
                self._watchdog_started = True
                self.start_watchdog()

        async def on_cleanup(_):
            self._task.cancel()
            for ws in list(self.clients):
                await ws.close()

        app.on_startup.append(on_startup)
        app.on_cleanup.append(on_cleanup)
        return app
