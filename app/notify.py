"""One path for every cue: play the haptic pattern on the band and log it.

    n = Notifier(link.pulse, AlertLog(path))
    n.cue("name", "directional", motor=5, confidence=0.9, latency_ms=410)
    n.cue("name", "generic", reason="too few loud frames")
    n.cue("replay", "directional", motor=5, dry=True)    # dashboard only, no motors

Listeners on the AlertLog see each record: the dashboard panel, and
Caregiver (below), which texts a caregiver through Photon.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from . import haptics
from .recall import AlertLog
from .settings import Settings


def pattern(cue: str, motor: Optional[int] = None) -> list:
    if cue == "directional":
        if motor is None:
            raise ValueError("directional cue needs a motor")
        return haptics.directional(motor)
    return {"generic": haptics.generic_attention, "centered": haptics.centered,
            "lost": haptics.lost, "sound": haptics.taught_sound,
            "ack": haptics.target_ack, "unknown": haptics.target_unknown}[cue]()


class Notifier:
    def __init__(self, send: Callable[[int, int], object], log: Optional[AlertLog] = None,
                 dry_send: Optional[Callable[[int, int], object]] = None,
                 settings: Optional[Settings] = None):
        self.send = send                      # Link.pulse
        self.settings = settings or Settings()
        self.dry_send = dry_send or (lambda m, ms: None)
        self.log = log or AlertLog()

    def cue(self, source: str, cue: str, motor: Optional[int] = None,
            confidence: Optional[float] = None, latency_ms: Optional[float] = None,
            dry: bool = False, play: bool = True, **detail) -> dict:
        """play=False logs a cue someone else already played (e.g. ObjectGuide).
        A muted source is still logged (muted=True) but never buzzes."""
        if not dry and self.settings.is_muted(source):
            play, detail["muted"] = False, True
        if play:
            steps = haptics.scaled(pattern(cue, motor), self.settings.intensity)
            haptics.play(steps, self.dry_send if dry else self.send)
        return self.log.add(source, cue, motor, confidence, latency_ms, dry, **detail)


# ---------------------------------------------------------------- caregiver
ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "tools" / "photon" / "bridge.mjs"
E164 = re.compile(r"^\+[1-9]\d{6,14}$")


def load_env(path=ROOT / ".env") -> None:
    """KEY=value lines from the repo's .env (gitignored) into os.environ,
    without overriding anything already set. Secrets never live in the repo."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def mask_phone(p: Optional[str]) -> str:
    return f"{p[:2]}\u2022\u2022\u2022{p[-2:]}" if p and len(p) > 4 else "not set"


class DryRunSender:
    """Logs texts instead of sending (no credentials, tests, --sim)."""
    name = "dry run"

    def __init__(self):
        self.sent: list = []

    def send(self, to: str, text: str) -> tuple:
        self.sent.append((to, text))
        print(f"[caregiver] (dry run) to {mask_phone(to)}: {text}")
        return True, None

    def close(self) -> None:
        pass


class PhotonSender:
    """Sends through tools/photon/bridge.mjs (Photon Spectrum, iMessage).
    Credentials are read by the bridge from SPECTRUM_PROJECT_ID/SECRET in the
    environment. The bridge is started on first use and restarted if it dies."""
    name = "Photon"

    def __init__(self, node: str = "node", bridge: Path = BRIDGE, timeout_s: float = 30.0):
        self.node, self.bridge, self.timeout_s = node, Path(bridge), timeout_s
        self._p: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._replies: dict = {}
        self._cv = threading.Condition()
        self._next = 1
        self.error: Optional[str] = None

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("SPECTRUM_PROJECT_ID") and os.environ.get("SPECTRUM_PROJECT_SECRET")
                    and (BRIDGE.parent / "node_modules" / "spectrum-ts").exists())

    def _start(self) -> None:
        self._p = subprocess.Popen([self.node, str(self.bridge)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, bufsize=1, cwd=str(self.bridge.parent))
        ready = self._p.stdout.readline()
        try:
            r = json.loads(ready)
        except ValueError:
            r = {"ready": False, "error": f"bridge said {ready.strip()[:120]!r}"}
        if not r.get("ready"):
            self._p.kill()
            self._p = None
            raise RuntimeError(r.get("error") or "bridge failed to start")
        threading.Thread(target=self._read, args=(self._p,), name="photon", daemon=True).start()

    def _read(self, p) -> None:
        for line in p.stdout:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            with self._cv:
                self._replies[r.get("id")] = r
                self._cv.notify_all()

    def send(self, to: str, text: str) -> tuple:
        with self._lock:
            try:
                if self._p is None or self._p.poll() is not None:
                    self._start()
                mid, self._next = self._next, self._next + 1
                self._p.stdin.write(json.dumps({"id": mid, "to": to, "text": text}) + "\n")
                self._p.stdin.flush()
            except Exception as e:
                self.error = str(e)
                return False, self.error
        with self._cv:
            self._cv.wait_for(lambda: mid in self._replies, timeout=self.timeout_s)
            r = self._replies.pop(mid, None)
        if r is None:
            self.error = "no reply from Photon bridge (timeout)"
            return False, self.error
        self.error = None if r.get("ok") else r.get("error")
        return bool(r.get("ok")), self.error

    def close(self) -> None:
        if self._p and self._p.poll() is None:
            try:
                self._p.stdin.close()
                self._p.wait(timeout=5)
            except Exception:
                self._p.kill()


class Caregiver:
    """Texts a caregiver about the alerts that matter, from the AlertLog.

      name heard (directional or generic), object found (find/centered, once
      per target per FOUND_REPEAT_S), taught sound heard.
    Off unless settings.caregiver_texts is on. Rate limit: at most one text per
    min_interval_s (alerts in between are merged into one digest text) and
    max_per_hour texts; anything beyond is counted as suppressed. Replays,
    dry runs, manual test cues and repeats never text.
    """

    FOUND_REPEAT_S = 300

    def __init__(self, settings: Settings, sender=None, phone: Optional[str] = None,
                 min_interval_s: float = 60.0, max_per_hour: int = 10,
                 clock: Callable[[], float] = time.time):
        self.settings = settings
        self.sender = sender or DryRunSender()
        self.phone = phone
        self.min_interval_s, self.max_per_hour = min_interval_s, max_per_hour
        self.clock = clock
        self._pending: list = []
        self._sent_times: deque = deque()
        self._last_send = -1e18
        self._found: dict = {}            # target -> last found time
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self.sent = self.failed = self.suppressed = 0
        self.last: Optional[dict] = None  # {"time","text","ok","error"}
        self._listeners: list = []

    def on_status(self, fn) -> None:
        self._listeners.append(fn)

    @property
    def configured(self) -> bool:
        return bool(self.phone and E164.match(self.phone))

    def status(self) -> dict:
        return {"on": self.settings.caregiver_texts, "configured": self.configured,
                "sender": self.sender.name, "to": mask_phone(self.phone),
                "sent": self.sent, "failed": self.failed, "suppressed": self.suppressed,
                "pending": len(self._pending), "last": self.last,
                "limit": f"1 per {self.min_interval_s:g} s, {self.max_per_hour}/h"}

    # -- AlertLog listener
    def __call__(self, rec: dict) -> None:
        if rec.get("type") != "alert" or not self.settings.caregiver_texts:
            return
        line = self.describe(rec)
        if not line:
            return
        with self._lock:
            self._pending.append((rec.get("time", "")[:5], line))
        self._schedule()

    def describe(self, rec: dict) -> Optional[str]:
        if rec.get("dry") or rec.get("repeat") or rec.get("source") not in ("name", "find", "sound"):
            return None
        src, cue = rec["source"], rec.get("cue")
        if src == "name" and cue in ("directional", "generic"):
            return (f"name called from the {rec['direction']}" if rec.get("direction")
                    else "name called (direction unclear)")
        if src == "find" and cue == "centered":
            target = rec.get("target") or "object"
            now = self.clock()
            if now - self._found.get(target, -1e18) < self.FOUND_REPEAT_S:
                return None
            self._found[target] = now
            return f"found the {target}"
        if src == "sound" and cue == "sound":
            return "taught sound heard"
        return None

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                return                      # a flush is already scheduled
            wait = max(0.0, self._last_send + self.min_interval_s - self.clock())
            self._timer = threading.Timer(wait, self.flush)
            self._timer.daemon = True
            self._timer.start()

    def flush(self) -> Optional[dict]:
        """Send everything pending as one text (also called by the timer)."""
        with self._lock:
            self._timer = None
            items, self._pending = self._pending, []
            if not items:
                return None
            now = self.clock()
            while self._sent_times and now - self._sent_times[0] > 3600:
                self._sent_times.popleft()
            if len(self._sent_times) >= self.max_per_hour:
                self.suppressed += len(items)
                self._emit()
                return None
            self._sent_times.append(now)
            self._last_send = now
        return self._send(self.format(items))

    @staticmethod
    def format(items: list) -> str:
        if len(items) == 1:
            t, line = items[0]
            return f"[Band] {t} {line[0].upper()}{line[1:]}."
        span = f"{items[0][0]}\u2013{items[-1][0]}" if items[0][0] != items[-1][0] else items[0][0]
        return f"[Band] {len(items)} alerts {span}: " + "; ".join(l for _, l in items) + "."

    def send_test(self) -> dict:
        return self._send("[Band] Test: caregiver texts are working.")

    def _send(self, text: str) -> dict:
        if not self.configured:
            ok, err = False, "caregiver phone not set (CAREGIVER_PHONE in .env, E.164)"
        else:
            try:
                ok, err = self.sender.send(self.phone, text)
            except Exception as e:
                ok, err = False, str(e)
        with self._lock:
            self.sent += ok
            self.failed += not ok
            self.last = {"time": time.strftime("%H:%M:%S"), "text": text, "ok": ok, "error": err}
        self._emit()
        return self.last

    def _emit(self) -> None:
        st = self.status()
        for fn in self._listeners:
            try:
                fn(st)
            except Exception as e:
                print(f"[caregiver] listener error: {e}")

    def close(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.sender.close()


def caregiver_from_env(settings: Settings, dry: bool = False) -> Caregiver:
    """Phone and Photon credentials from the environment / .env (never the repo)."""
    load_env()
    phone = os.environ.get("CAREGIVER_PHONE")
    sender = PhotonSender() if (not dry and PhotonSender.available()) else DryRunSender()
    return Caregiver(settings, sender, phone)
