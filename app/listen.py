"""Teach a Sound (prototype): record one sound on the laptop mic, then buzz
its own pattern (haptics.taught_sound, no direction) whenever it happens.

    sl = SoundListener(on_detect=cb)      # cb(SoundHit), from the audio thread
    sl.start()                            # laptop mic via sounddevice
    sl.teach(2.5)                         # next 2.5 s: make the sound once
    sl.feed(samples)                      # or push float32 16 kHz audio yourself (tests)

How it works (numpy only, no model):
  * 32 ms frames. A sound "event" is a run of frames louder than
    RATIO x the noise floor (20th percentile of the last ~3 s), closed by
    END_QUIET quiet frames or MAX_EVENT_S.
  * Each event becomes a fingerprint: the mean log energy in NBANDS
    log-spaced bands (100 Hz-7 kHz), mean-removed and unit length, so it
    describes the *shape* of the spectrum, not how loud it was.
  * Teaching keeps the loudest event in the teach window. Detection scores a
    new event by cosine similarity with the taught fingerprint, times a
    duration factor; >= threshold (default 0.80) fires, with a cooldown.
Direction is not estimated yet (timeline: add it once recognition is
reliable in noise). The taught fingerprint is saved to recordings/taught_sound.json.
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512                    # 32 ms
NBANDS = 20
FMIN, FMAX = 100.0, 7000.0
RATIO = 4.0                    # event = frame RMS above RATIO x noise floor
MIN_RMS = 0.004                # ... and above this absolute floor
END_QUIET = 3                  # quiet frames that close an event (~96 ms)
MIN_EVENT_FRAMES = 2
MAX_EVENT_S = 1.5
ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_FILE = ROOT / "recordings" / "taught_sound.json"

_EDGES = np.geomspace(FMIN, FMAX, NBANDS + 1)
_FREQS = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
_BAND = np.digitize(_FREQS, _EDGES) - 1        # FFT bin -> band (out of range: -1/NBANDS)
_WINDOW = np.hanning(FRAME).astype(np.float32)


def band_energies(frame: np.ndarray) -> np.ndarray:
    """log10 energy per band for one FRAME-sample frame."""
    spec = np.abs(np.fft.rfft(frame * _WINDOW)) ** 2
    out = np.zeros(NBANDS)
    ok = (_BAND >= 0) & (_BAND < NBANDS)
    np.add.at(out, _BAND[ok], spec[ok])
    return np.log10(out + 1e-10)


def fingerprint(bands: list) -> np.ndarray:
    """Mean band log-energies over an event -> mean-removed unit vector."""
    v = np.mean(np.asarray(bands), axis=0)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def similarity(fp: np.ndarray, dur_s: float, tpl: "Template") -> float:
    """Cosine of the spectral shapes x a duration factor (1.0 within 2x of
    the taught length, falling off beyond)."""
    cos = float(np.dot(fp, tpl.fp))
    ratio = max(dur_s, 1e-3) / max(tpl.duration_s, 1e-3)
    dur = min(1.0, 2.0 / max(ratio, 1 / ratio))
    return max(0.0, cos) * dur


@dataclass
class Template:
    fp: np.ndarray
    duration_s: float
    level: float
    taught_at: float = 0.0

    def to_json(self) -> dict:
        return {"fp": [round(float(x), 5) for x in self.fp], "duration_s": round(self.duration_s, 3),
                "level": round(self.level, 5), "taught_at": self.taught_at,
                "bands": NBANDS, "frame": FRAME, "rate": SAMPLE_RATE}

    @classmethod
    def from_json(cls, d: dict) -> "Template":
        if d.get("bands") != NBANDS or d.get("frame") != FRAME or d.get("rate") != SAMPLE_RATE:
            raise ValueError("taught sound was saved with different settings; teach it again")
        return cls(np.asarray(d["fp"], dtype=float), float(d["duration_s"]),
                   float(d["level"]), float(d.get("taught_at", 0.0)))


@dataclass(frozen=True)
class SoundHit:
    score: float
    start: float      # laptop time.monotonic() seconds
    end: float
    level: float


@dataclass
class _Event:
    start_n: int
    bands: list
    peak: float
    quiet: int = 0


class SoundListener:
    def __init__(self, on_detect: Optional[Callable[[SoundHit], None]] = None,
                 threshold: float = 0.80, cooldown_s: float = 1.5, device=None,
                 template_file: Optional[Path] = TEMPLATE_FILE):
        self.on_detect = on_detect
        self.on_taught: Optional[Callable[[Optional[Template], str], None]] = None
        self.on_event: Optional[Callable[[float, float], None]] = None   # debug: (score, dur)
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.device = device
        self.template_file = Path(template_file) if template_file else None
        self.template: Optional[Template] = None
        self.paused = False
        self.last: Optional[SoundHit] = None
        self._levels: deque = deque(maxlen=int(3 * SAMPLE_RATE / FRAME))
        self._buf = np.zeros(0, dtype=np.float32)
        self._n = 0                       # samples consumed
        self._offset: Optional[float] = None
        self._ev: Optional[_Event] = None
        self._teach_until: Optional[int] = None   # sample index where teaching ends
        self._teach_events: list = []
        self._last_hit = -1e9
        self._lock = threading.Lock()
        self._stream = None
        if self.template_file and self.template_file.exists():
            try:
                self.template = Template.from_json(json.loads(self.template_file.read_text()))
            except (ValueError, KeyError) as e:
                print(f"[listen] ignoring {self.template_file.name}: {e}")

    # ------------------------------------------------------------- public
    @property
    def noise_floor(self) -> float:
        return float(np.percentile(self._levels, 20)) if self._levels else 0.0

    def start(self) -> "SoundListener":
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                      blocksize=FRAME, device=self.device, callback=self._cb)
        self._stream.start()
        return self

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def pause(self) -> None:
        self.paused = True
        self._ev = None

    def resume(self) -> None:
        self._ev = None
        self.paused = False

    def teach(self, seconds: float = 2.5) -> None:
        """Make the sound once within the next `seconds`; the loudest event wins."""
        with self._lock:
            self._teach_events = []
            self._teach_until = self._n + len(self._buf) + int(seconds * SAMPLE_RATE)

    @property
    def teaching(self) -> bool:
        return self._teach_until is not None

    def forget(self) -> None:
        self.template = None
        if self.template_file and self.template_file.exists():
            self.template_file.unlink()

    def feed(self, samples, t_first: Optional[float] = None) -> None:
        """Push float32 mono 16 kHz audio; t_first = monotonic time of samples[0]."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if t_first is None:
            t_first = time.monotonic() - len(samples) / SAMPLE_RATE
        with self._lock:
            off = t_first - (self._n + len(self._buf)) / SAMPLE_RATE
            self._offset = off if self._offset is None else min(self._offset + 1e-6, off)
            self._buf = np.concatenate([self._buf, samples])
            while len(self._buf) >= FRAME:
                f, self._buf = self._buf[:FRAME], self._buf[FRAME:]
                self._frame(f)

    # ---------------------------------------------------------- internals
    def _cb(self, indata, frames, time_info, status) -> None:
        lat = self._stream.latency if self._stream is not None else 0.0
        self.feed(indata[:, 0].copy(), time.monotonic() - frames / SAMPLE_RATE - lat)

    def _t(self, n: int) -> float:
        return (self._offset or 0.0) + n / SAMPLE_RATE

    def _frame(self, f: np.ndarray) -> None:
        n0 = self._n
        self._n += FRAME
        rms = float(math.sqrt(float(np.mean(f * f))) + 1e-9)
        floor = self.noise_floor
        loud = rms > max(MIN_RMS, RATIO * floor) and len(self._levels) >= 5
        if self._ev is None or not loud:
            self._levels.append(rms)            # floor tracks the room, not the event
        if self.paused and not self.teaching:
            return
        ev = self._ev
        if ev is None:
            if loud:
                self._ev = _Event(n0, [band_energies(f)], rms)
        elif loud:
            ev.bands.append(band_energies(f))
            ev.peak = max(ev.peak, rms)
            ev.quiet = 0
            if len(ev.bands) * FRAME >= MAX_EVENT_S * SAMPLE_RATE:
                self._close(self._n)
        else:
            ev.quiet += 1
            if ev.quiet >= END_QUIET:
                self._close(self._n - END_QUIET * FRAME)
        if self._teach_until is not None and self._n >= self._teach_until and self._ev is None:
            self._finish_teach()

    def _close(self, end_n: int) -> None:
        ev, self._ev = self._ev, None
        if ev is None or len(ev.bands) < MIN_EVENT_FRAMES:
            return
        dur = (end_n - ev.start_n) / SAMPLE_RATE
        fp = fingerprint(ev.bands)
        if self._teach_until is not None:
            self._teach_events.append(Template(fp, dur, ev.peak, time.time()))
            return
        if self.template is None:
            return
        score = similarity(fp, dur, self.template)
        if self.on_event:
            self.on_event(score, dur)
        end_t = self._t(end_n)
        if score >= self.threshold and end_t - self._last_hit >= self.cooldown_s:
            self._last_hit = end_t
            hit = SoundHit(round(score, 3), self._t(ev.start_n), end_t, ev.peak)
            self.last = hit
            if self.on_detect:
                try:
                    self.on_detect(hit)
                except Exception as e:
                    print(f"[listen] on_detect error: {e}")

    def _finish_teach(self) -> None:
        events, self._teach_events, self._teach_until = self._teach_events, [], None
        if not events:
            msg = "no sound heard while teaching; make it louder or closer to the laptop"
            tpl = None
        else:
            tpl = max(events, key=lambda e: e.level)
            self.template = tpl
            msg = f"taught a {tpl.duration_s * 1000:.0f} ms sound"
            if self.template_file:
                try:
                    self.template_file.parent.mkdir(parents=True, exist_ok=True)
                    self.template_file.write_text(json.dumps(tpl.to_json()) + "\n")
                except OSError as e:
                    msg += f" (not saved: {e})"
        if self.on_taught:
            try:
                self.on_taught(tpl, msg)
            except Exception as e:
                print(f"[listen] on_taught error: {e}")


def main(argv=None) -> int:
    """python -m app.listen [--teach] [--seconds 20]: try it in the terminal."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--teach", action="store_true", help="teach a new sound first")
    ap.add_argument("--seconds", type=float, default=30.0, help="listen this long")
    ap.add_argument("--threshold", type=float, default=0.80)
    a = ap.parse_args(argv)
    done = threading.Event()
    sl = SoundListener(lambda h: print(f"  DETECTED score {h.score:.2f}"), threshold=a.threshold)
    sl.on_taught = lambda tpl, msg: (print(f"[listen] {msg}"), done.set())
    sl.on_event = lambda s, d: print(f"  event {d * 1000:.0f} ms, score {s:.2f}")
    sl.start()
    time.sleep(1.0)                       # learn the room's noise floor
    if a.teach or sl.template is None:
        print("[listen] make the sound now (2.5 s)...")
        sl.teach(2.5)
        done.wait(5)
    print(f"[listen] listening {a.seconds:.0f} s (Ctrl-C to stop)")
    try:
        time.sleep(a.seconds)
    except KeyboardInterrupt:
        pass
    sl.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
