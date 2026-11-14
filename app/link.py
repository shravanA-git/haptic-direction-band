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
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Optional, Sequence, Union

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


class NullPort:
    """Placeholder while no band is plugged in: every read fails, so Link keeps
    trying its reopen factory until the ESP32 shows up."""
    device = "no band"

    def readline(self):
        raise OSError(2, "no ESP32 serial port found")

    def write_line(self, line: str) -> None:
        raise OSError(2, "no ESP32 serial port found")

    def close(self) -> None:
        pass


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
        self.broken = False           # test helper: simulate an unplugged cable
        self._lock = threading.Lock()

    # -- laptop side
    def readline(self) -> Optional[str]:
        if self.broken:
            raise OSError(6, "Device not configured (sim unplugged)")
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
        if self.broken:
            raise OSError(6, "Device not configured (sim unplugged)")
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

    def reopen(self) -> "SimPort":
        """Reconnect factory for Link: fails while 'unplugged', else reopens."""
        if self.broken:
            raise OSError(2, "No such file or directory (sim unplugged)")
        self._closed = False
        return self

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


# ------------------------------------------------------------ motor map
MOTOR_MAP_FILE = Path(__file__).resolve().parent / "motor_map.json"


def load_motor_map(path=MOTOR_MAP_FILE) -> Optional[list]:
    """Logical motor (position: 0 front .. 7 front-left) -> physical channel
    number sent in PULSE. Written by `python -m tools.bench --map` when the
    wiring doesn't match the pin map; absent file = identity (no remap).
    """
    path = Path(path)
    if not path.exists():
        return None
    m = json.loads(path.read_text())["map"]
    validate_motor_map(m)
    return [int(c) for c in m]


def validate_motor_map(m: Sequence[int]) -> None:
    if len(m) != NUM_MOTORS or sorted(int(c) for c in m) != list(range(NUM_MOTORS)):
        raise ValueError(f"motor map must be a permutation of 0..{NUM_MOTORS - 1}, got {list(m)}")


def save_motor_map(m: Sequence[int], path=MOTOR_MAP_FILE) -> Path:
    validate_motor_map(m)
    path = Path(path)
    path.write_text(json.dumps({"map": [int(c) for c in m],
                                "note": "map[position] = physical channel; see tools/bench.py --map"}) + "\n")
    return path


# -------------------------------------------------------------------- link
Listener = Callable[[Event], None]


class Link:
    """Single owner of the serial port. A reader thread parses incoming lines
    and fans events out to subscribers; writes are serialized by a lock.
    """

    def __init__(self, port, on_sent: Optional[Callable[[str], None]] = None,
                 motor_map: Optional[Sequence[int]] = None,
                 reopen: Optional[Callable[[], object]] = None, retry_s: float = 1.0):
        """reopen: called (every retry_s) after the port fails, to get a fresh
        port, e.g. lambda: SerialPort(find_esp32_port()). Without it a failed
        port stays failed."""
        self.port = port
        self.reopen, self.retry_s = reopen, retry_s
        self.reconnects = 0
        self.dropped_writes = 0
        self.device = getattr(port, "device", "?")
        if motor_map is not None:
            validate_motor_map(motor_map)
        self.motor_map = list(motor_map) if motor_map is not None else None
        self.opened = time.monotonic()
        self._listeners: list[Listener] = []
        self._sent_listeners: list[Callable[[str], None]] = [on_sent] if on_sent else []
        self._wlock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_rx: float = 0.0
        self.last_mic_rx: float = 0.0
        self.frames_rx = 0
        self.lines_rx = 0
        self.raw_rx = 0
        self.btn_rx = 0
        self.last_raw: Optional[str] = None
        self._mic_gaps: deque = deque(maxlen=100)   # ESP32 ms between frames
        self._last_mic_ms: Optional[int] = None
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
        """Pulse a logical motor (0 front, clockwise). Goes through the motor
        map if one is loaded; listeners always see the logical motor."""
        line = format_pulse(motor, duration_ms)
        if self.motor_map is None:
            return self._send(line)
        return self._send(format_pulse(self.motor_map[int(motor)], duration_ms), line)

    def pulse_channel(self, channel: int, duration_ms: int) -> str:
        """Pulse a physical channel, bypassing the motor map (mapping checks)."""
        line = format_pulse(channel, duration_ms)
        logical = line
        if self.motor_map is not None:
            logical = format_pulse(self.motor_map.index(int(channel)), duration_ms)
        return self._send(line, logical)

    def stop(self) -> str:
        return self._send(format_stop())

    def _send(self, wire: str, logical: Optional[str] = None) -> str:
        """Never raises for a dead port: the line is dropped and the error shown
        (firmware stops all motors by itself when the USB link is lost)."""
        with self._wlock:
            try:
                self.port.write_line(wire)
            except Exception as e:
                self.dropped_writes += 1
                self.error = self.error or f"{type(e).__name__}: {e}"
                print(f"[link] not sent ({self.error}): {wire}")
                return wire
        for fn in self._sent_listeners:
            _safe(fn, logical or wire)
        return wire

    def health(self) -> dict:
        """Link diagnostics for the bench and the dashboard header.
        state: "error" (port failed), "silent" (open, nothing received yet or
        for >1 s), "lines" (lines arrive but no MIC frames: early firmware),
        "streaming" (MIC frames arriving)."""
        now = time.monotonic()
        gaps = list(self._mic_gaps)
        if self.error:
            state = "error"
        elif self.last_mic_rx and now - self.last_mic_rx < 1.0:
            state = "streaming"
        elif self.last_rx and now - self.last_rx < 3.0:
            state = "lines"
        else:
            state = "silent"
        return {
            "state": state,
            "lines": self.lines_rx, "mic": self.frames_rx, "btn": self.btn_rx, "raw": self.raw_rx,
            "last_raw": self.last_raw,
            "last_rx_age": round(now - self.last_rx, 2) if self.last_rx else None,
            "mic_gap_ms": round(median(gaps), 1) if gaps else None,
            "mic_gap_max_ms": max(gaps) if gaps else None,
            "mapped": self.motor_map is not None,
            "reconnects": self.reconnects, "dropped_writes": self.dropped_writes,
            "error": self.error,
        }

    def connected(self, stale_s: float = 1.0) -> bool:
        return self.error is None and (time.monotonic() - self.last_rx) < stale_s

    def _reconnect(self) -> None:
        """Close the dead port and retry the reopen factory until it works."""
        with self._wlock:
            try:
                self.port.close()
            except Exception:
                pass
        while not self._stop.is_set():
            if self.reopen is None:
                self._stop.wait(0.5)
                return
            try:
                port = self.reopen()
            except Exception as e:
                self.error = f"waiting for band: {type(e).__name__}: {e}"
                self._stop.wait(self.retry_s)
                continue
            with self._wlock:
                self.port = port
                self.device = getattr(port, "device", self.device)
            self.error = None
            self.reconnects += 1
            self._last_mic_ms = None
            print(f"[link] reconnected to {self.device}")
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                line = self.port.readline()
            except Exception as e:  # unplugged, etc.
                self.error = f"{type(e).__name__}: {e}"
                self._reconnect()
                continue
            if not line or not line.strip():
                continue
            ev = parse_line(line)
            self.last_rx = ev.rx
            self.lines_rx += 1
            if isinstance(ev, MicFrame):
                self.frames_rx += 1
                self.last_mic_rx = ev.rx
                if self._last_mic_ms is not None and ev.ms > self._last_mic_ms:
                    self._mic_gaps.append(ev.ms - self._last_mic_ms)
                self._last_mic_ms = ev.ms
            elif isinstance(ev, ButtonEvent):
                self.btn_rx += 1
            else:
                self.raw_rx += 1
                self.last_raw = ev.text[:120]
            for fn in self._listeners:
                _safe(fn, ev)


def _safe(fn, arg) -> None:
    try:
        fn(arg)
    except Exception as e:  # one bad listener must not kill the reader
        print(f"[link] listener error: {e!r}")
