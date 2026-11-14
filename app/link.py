"""USB serial link to the ESP32 band (115200 baud, one text line per message).

Contract (agents/context/protocol.md, Nov 13 handoff):
  ESP32 -> laptop   MIC <ms> <front> <right> <back> <left>
                    BTN ACTION DOWN | BTN ACTION UP
  laptop -> ESP32   PULSE <motor> <duration_ms>
                    STOP

Exactly one Link owns the port. Everything else (dashboard, vision, speech,
estimator) goes through it: call pulse()/stop() and subscribe() for events.
"""
from __future__ import annotations

import glob
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

BAUD = 115200
NUM_MOTORS = 8
MAX_PULSE_MS = 1000  # laptop-side cap (matches proposed firmware cap)


# ---------------------------------------------------------------- messages
@dataclass(frozen=True)
class MicFrame:
    ms: int                      # ESP32 monotonic milliseconds
    levels: tuple                # (front, right, back, left), nonnegative
    rx: float = 0.0              # laptop time.monotonic() at receipt


@dataclass(frozen=True)
class ButtonEvent:
    name: str                    # "ACTION" (others not yet in the protocol)
    down: bool
    rx: float = 0.0


@dataclass(frozen=True)
class RawLine:
    """A line we could not parse (boot noise, firmware debug prints)."""
    text: str
    rx: float = 0.0


Event = Union[MicFrame, ButtonEvent, RawLine]


def parse_line(line: str, rx: Optional[float] = None) -> Event:
    rx = time.monotonic() if rx is None else rx
    parts = line.strip().split()
    try:
        if len(parts) == 6 and parts[0] == "MIC":
            ms = int(parts[1])
            levels = tuple(float(p) for p in parts[2:])
            if ms >= 0 and all(v >= 0 for v in levels):
                return MicFrame(ms, levels, rx)
        elif len(parts) == 3 and parts[0] == "BTN" and parts[2] in ("DOWN", "UP"):
            return ButtonEvent(parts[1], parts[2] == "DOWN", rx)
    except ValueError:
        pass
    return RawLine(line.strip(), rx)


def format_pulse(motor: int, duration_ms: int) -> str:
    motor, duration_ms = int(motor), int(duration_ms)
    if not 0 <= motor < NUM_MOTORS:
        raise ValueError(f"motor {motor} out of range 0..{NUM_MOTORS - 1}")
    if not 1 <= duration_ms <= MAX_PULSE_MS:
        raise ValueError(f"duration {duration_ms} ms out of range 1..{MAX_PULSE_MS}")
    return f"PULSE {motor} {duration_ms}"


def format_stop() -> str:
    return "STOP"


# ------------------------------------------------------------------- ports
class SerialPort:
    """Real USB serial port (pyserial). exclusive=True enforces one owner."""

    def __init__(self, device: str, baud: int = BAUD):
        import serial  # imported lazily so tests and sim need no pyserial
        self.device = device
        self._s = serial.Serial(device, baud, timeout=0.1, exclusive=True)

    def readline(self) -> Optional[str]:
        raw = self._s.readline()
        return raw.decode("ascii", "replace") if raw else None

    def write_line(self, line: str) -> None:
        self._s.write((line + "\n").encode("ascii"))
        self._s.flush()

    def close(self) -> None:
        self._s.close()


def find_esp32_port() -> Optional[str]:
    """Best guess at the ESP32's USB serial device on macOS/Linux."""
    pats = ["/dev/cu.usbserial*", "/dev/cu.SLAB_USBtoUART*", "/dev/cu.wchusbserial*",
            "/dev/cu.usbmodem*", "/dev/ttyUSB*", "/dev/ttyACM*"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None


class SimPort:
    """Simulated ESP32. Emits MIC lines (from a trace, or quiet noise) on a
    real-time schedule, accepts PULSE/STOP and keeps a log of them so tests
    can assert on exactly what the laptop sent.
    """

    def __init__(self, frames: Optional[list] = None, loop: bool = False,
                 frame_ms: int = 50, realtime: bool = True):
        self.device = "sim"
        self.frames = frames          # list of {"ms":..,"levels":[f,r,b,l]}
        self.loop = loop
        self.frame_ms = frame_ms
        self.realtime = realtime
        self.sent: list[str] = []     # every line the laptop wrote
        self.active: dict[int, float] = {}  # motor -> sim end time
        self._inject: "queue.Queue[str]" = queue.Queue()
        self._t0 = time.monotonic()
        self._i = 0
        self._closed = False
        self._lock = threading.Lock()

    # -- laptop side
    def readline(self) -> Optional[str]:
        if self._closed:
            return None
        try:
            return self._inject.get_nowait()
        except queue.Empty:
            pass
        line, due = self._next_frame()
        if line is None:
            time.sleep(0.02)
            return None
        if self.realtime:
            wait = due - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 0.1))
                if due - time.monotonic() > 0:
                    return None  # not due yet; caller polls again
        self._i += 1
        return line + "\n"

    def write_line(self, line: str) -> None:
        with self._lock:
            self.sent.append(line)
            now = time.monotonic()
            parts = line.split()
            if parts and parts[0] == "PULSE":
                self.active[int(parts[1])] = now + int(parts[2]) / 1000
            elif parts == ["STOP"]:
                self.active.clear()

    def close(self) -> None:
        self._closed = True

    # -- test helpers
    def inject(self, line: str) -> None:
        self._inject.put(line.rstrip("\n") + "\n")

    def motors_on(self) -> set:
        now = time.monotonic()
        with self._lock:
            return {m for m, end in self.active.items() if end > now}

    def _next_frame(self):
        if self.frames is None:  # synthetic quiet room
            ms = self._i * self.frame_ms
            k = self._i % 7
            line = f"MIC {ms} {20 + k} {19 + (k * 3) % 5} {21 - k % 3} {20 + (k * 2) % 4}"
            return line, self._t0 + ms / 1000
        if not self.frames:
            return None, 0
        n = len(self.frames)
        if self._i >= n and not self.loop:
            return None, 0
        lap, j = divmod(self._i, n)
        f = self.frames[j]
        span = self.frames[-1]["ms"] - self.frames[0]["ms"] + self.frame_ms
        ms = f["ms"] - self.frames[0]["ms"] + lap * span
        lv = " ".join(str(round(v)) for v in f["levels"])
        return f"MIC {ms} {lv}", self._t0 + ms / 1000


# -------------------------------------------------------------------- link
Listener = Callable[[Event], None]


class Link:
    """Single owner of the serial port. A reader thread parses incoming lines
    and fans events out to subscribers; writes are serialized by a lock.
    """

    def __init__(self, port, on_sent: Optional[Callable[[str], None]] = None):
        self.port = port
        self.device = getattr(port, "device", "?")
        self._listeners: list[Listener] = []
        self._sent_listeners: list[Callable[[str], None]] = [on_sent] if on_sent else []
        self._wlock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_rx: float = 0.0
        self.frames_rx = 0
        self.error: Optional[str] = None

    # subscriptions
    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def on_sent(self, fn: Callable[[str], None]) -> None:
        self._sent_listeners.append(fn)

    # lifecycle
    def start(self) -> "Link":
        self._thread = threading.Thread(target=self._run, name="link-rx", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        try:
            self.stop()  # leave the band quiet
        except Exception:
            pass
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.port.close()

    # commands
    def pulse(self, motor: int, duration_ms: int) -> str:
        return self._send(format_pulse(motor, duration_ms))

    def stop(self) -> str:
        return self._send(format_stop())

    def _send(self, line: str) -> str:
        with self._wlock:
            self.port.write_line(line)
        for fn in self._sent_listeners:
            _safe(fn, line)
        return line

    def connected(self, stale_s: float = 1.0) -> bool:
        return self.error is None and (time.monotonic() - self.last_rx) < stale_s

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = self.port.readline()
            except Exception as e:  # unplugged, etc.
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.5)
                continue
            if not line or not line.strip():
                continue
            ev = parse_line(line)
            self.last_rx = ev.rx
            if isinstance(ev, MicFrame):
                self.frames_rx += 1
            for fn in self._listeners:
                _safe(fn, ev)


def _safe(fn, arg) -> None:
    try:
        fn(arg)
    except Exception as e:  # one bad listener must not kill the reader
        print(f"[link] listener error: {e!r}")
