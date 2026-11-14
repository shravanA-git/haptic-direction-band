"""Teach a Sound: record sounds on the laptop mic (a doorbell, a microwave beep),
then buzz a pattern (haptics.taught_sound, no direction) whenever one happens.

    sl = SoundListener(on_detect=cb)      # cb(SoundHit), from the audio thread
    sl.start()                            # laptop mic via sounddevice
    sl.teach(2.5)                         # next 2.5 s: make the sound once (slot "sound")
    sl.teach(2.5, label="doorbell")       # named sound; teaching a name again replaces it
    sl.teach(2.5, label="doorbell", add=True)   # ...or adds another example of it
    sl.forget("doorbell"); sl.forget()    # one sound / all of them
    sl.summary()                          # taught sounds grouped by category and pitch
    sl.feed(samples)                      # or push float32 16 kHz audio yourself (tests)

SoundHit: score, start/end (laptop monotonic s), level, label, category, pitch_hz.

How it works (numpy only, no model):
  * 32 ms frames. An event starts when a frame is louder than RATIO x the noise
    floor (20th percentile of the last ~3 s) and continues while frames stay
    above HOLD_RATIO x the floor (hysteresis), so a ringing chime stays one event
    while it decays. Gaps shorter than END_QUIET frames (~220 ms) are kept inside
    the event, so "beep-beep-beep" is one pattern. MAX_EVENT_S caps it.
  * Every event gets a Signature:
      - pitch: per frame, the strongest spectral peak (2048-point zero-padded
        FFT, parabolic interpolation). A frame is tonal when its 3 strongest
        peaks hold >= TONAL_PEAK_FRAC of the energy; claps, knocks and hiss don't.
      - notes: runs of tonal frames at a steady pitch (within NOTE_CENTS), e.g.
        a ding-dong is two notes, 659 Hz then 523 Hz.
      - onsets: frames where the level jumps (a new strike or beep).
      - category, from frequency and pattern: tone, two-tone, melody, repeating
        (beeps), percussive (clap/knock), noise. See categorize().
    plus the old fingerprint (mean log energy in NBANDS bands, mean-removed, unit
    length: the spectrum's shape, not its loudness).
  * Matching an event against each taught example:
      tonal template:  0.5 pitch + 0.2 pattern + 0.3 spectrum, and an untuned
                       event (a clap) is capped far below threshold;
      untuned template: 0.75 spectrum + 0.25 pattern, capped if the event is tonal;
    all times a duration factor. The best sound >= threshold (default 0.80) fires,
    with a cooldown.
YAMNet (HDB_YAMNET=1; pip install ai-edge-litert): sounds that match no taught
sound go to Google's pretrained AudioSet classifier on a worker thread; its top
classes are reported (on_heard) and a watched class (YAMNET_WATCH: doorbell,
knock, door, smoke alarm, ...) >= HDB_YAMNET_MIN buzzes like a taught sound,
label = the AudioSet class, category "yamnet". Weights download at runtime into
models/yamnet/ (gitignored).
With band MIC frames, app/server.py adds the direction. Taught sounds are saved to
recordings/taught_sound.json (v2: a list of labelled sounds; a v1 single
fingerprint still loads, as label "sound", spectrum-only).
"""
from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512                    # 32 ms
NBANDS = 20
FMIN, FMAX = 100.0, 7000.0
RATIO = 4.0                    # event starts: frame RMS above RATIO x noise floor
HOLD_RATIO = 1.5               # ... and continues while above HOLD_RATIO x floor,
HOLD_ROOM = 1.25               # HOLD_ROOM x the room's 90th-percentile level
HOLD_PEAK = 0.15               # and HOLD_PEAK x the event's peak (-16 dB): talk nearby
                               # doesn't keep a loud chime's event open
MIN_RMS = 0.004                # start also needs this absolute level
END_QUIET = 7                  # quiet frames that close an event (~224 ms)
MIN_EVENT_FRAMES = 1           # a clap can be a single 32 ms frame
MAX_EVENT_S = 3.0
DEFAULT_LABEL = "sound"
MAX_EXAMPLES = 5               # per label, with teach(add=True)

# pitch analysis
NFFT = 2048                    # zero-padded: 7.8 Hz bins
PITCH_MIN, PITCH_MAX = 150.0, 6000.0
PEAK_HALF_BINS = 8             # +-62 Hz: a Hann main lobe at 512 samples
TONAL_PEAK_FRAC = 0.5          # 3 strongest peaks hold this share of the energy
TONAL_EVENT_FRAC = 0.35        # share of tonal frames for a "tonal" event
NOTE_CENTS = 60                # same note if within this many cents
ONSET_JUMP = 1.8               # level jump over the recent minimum that marks an onset
NOISE_SUB = 2.0                # subtract this x the room spectrum before pitch picking
NOISE_ALPHA = 0.05             # room-spectrum update rate (quiet frames)
PITCH_RMS_RATIO = 2.5          # only frames this far above the floor get a pitch
NOTE_MIN_SHARE = 0.3           # notes shorter than this x the longest note are dropped
ONSET_MIN_PEAK = 0.25          # onsets need this x the event's peak level

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_FILE = ROOT / "recordings" / "taught_sound.json"

_EDGES = np.geomspace(FMIN, FMAX, NBANDS + 1)
_FREQS = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
_BAND = np.digitize(_FREQS, _EDGES) - 1        # FFT bin -> band (out of range: -1/NBANDS)
_WINDOW = np.hanning(FRAME).astype(np.float32)
_PFREQS = np.fft.rfftfreq(NFFT, 1 / SAMPLE_RATE)
_PLO, _PHI = int(np.searchsorted(_PFREQS, PITCH_MIN)), int(np.searchsorted(_PFREQS, PITCH_MAX))
_NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()


def band_energies(frame: np.ndarray) -> np.ndarray:
    """log10 energy per band for one FRAME-sample frame."""
    spec = np.abs(np.fft.rfft(frame * _WINDOW)) ** 2
    out = np.zeros(NBANDS)
    ok = (_BAND >= 0) & (_BAND < NBANDS)
    np.add.at(out, _BAND[ok], spec[ok])
    return np.log10(out + 1e-10)


def pitch_spectrum(frame: np.ndarray) -> np.ndarray:
    """Power spectrum over PITCH_MIN..PITCH_MAX (zero-padded NFFT)."""
    return np.abs(np.fft.rfft(frame * _WINDOW, n=NFFT))[_PLO:_PHI] ** 2


def frame_pitch(frame: np.ndarray, noise_psd: Optional[np.ndarray] = None,
                spec: Optional[np.ndarray] = None) -> tuple:
    """(pitch_hz or None, tonality 0..1) for one frame. Tonality = share of the
    150 Hz-6 kHz energy held by the 3 strongest spectral peaks. With noise_psd
    (the room's average spectrum), the room is subtracted first, so steady hum,
    fans and HVAC don't count as the sound's pitch."""
    raw = pitch_spectrum(frame) if spec is None else spec
    p = raw if noise_psd is None else np.maximum(raw - NOISE_SUB * noise_psd, 0.0)
    total = float(p.sum())
    if total <= 1e-12 or total < 0.2 * float(raw.sum()):
        return None, 0.0        # nothing clearly above the room
    work = p.copy()
    held, top = 0.0, None
    for i in range(3):
        k = int(np.argmax(work))
        lo, hi = max(0, k - PEAK_HALF_BINS), k + PEAK_HALF_BINS + 1
        held += float(work[lo:hi].sum())
        if i == 0:
            top = k
        work[lo:hi] = 0.0
    tonality = held / total
    if tonality < TONAL_PEAK_FRAC:
        return None, tonality
    if top <= 2 or top >= len(p) - 3:
        return None, tonality   # rumble/hiss leaking in at the range edge, not a pitch
    k = top
    if 0 < k < len(p) - 1:   # parabolic interpolation on log power
        a, b, c = np.log(p[k - 1] + 1e-20), np.log(p[k] + 1e-20), np.log(p[k + 1] + 1e-20)
        den = a - 2 * b + c
        k = k + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    return float((_PLO + k) * SAMPLE_RATE / NFFT), tonality


def cents(a: float, b: float) -> float:
    return 1200.0 * math.log2(max(a, 1e-6) / max(b, 1e-6))


def note_name(hz: float) -> str:
    m = int(round(69 + 12 * math.log2(max(hz, 1e-6) / 440.0)))
    return f"{_NOTE_NAMES[m % 12]}{m // 12 - 1}"


def fingerprint(bands: list) -> np.ndarray:
    """Mean band log-energies over an event -> mean-removed unit vector."""
    v = np.mean(np.asarray(bands), axis=0)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _dur_factor(a: float, b: float) -> float:
    """1.0 within 2x of each other, falling off beyond."""
    ratio = max(a, 1e-3) / max(b, 1e-3)
    return min(1.0, 2.0 / max(ratio, 1 / ratio))


# --------------------------------------------------------------------------
# Signature: pitch + pattern + category
# --------------------------------------------------------------------------

@dataclass
class Signature:
    duration_s: float
    tonal: float                                # share of loud frames with a clear pitch
    notes: list = field(default_factory=list)   # [(hz, start_s, dur_s)]
    onsets: list = field(default_factory=list)  # [start_s] of each strike / beep
    category: str = ""
    description: str = ""

    @property
    def is_tonal(self) -> bool:
        return self.tonal >= TONAL_EVENT_FRAC and bool(self.notes)

    @property
    def pitch_hz(self) -> Optional[float]:
        """Main pitch: the longest note."""
        return max(self.notes, key=lambda n: n[2])[0] if self.notes else None

    def to_json(self) -> dict:
        return {"duration_s": round(self.duration_s, 3), "tonal": round(self.tonal, 3),
                "notes": [[round(h, 1), round(s, 3), round(d, 3)] for h, s, d in self.notes],
                "onsets": [round(t, 3) for t in self.onsets],
                "category": self.category, "description": self.description}

    @classmethod
    def from_json(cls, d: dict) -> "Signature":
        return cls(float(d["duration_s"]), float(d["tonal"]),
                   [tuple(n) for n in d.get("notes", [])], list(d.get("onsets", [])),
                   d.get("category", ""), d.get("description", ""))


def _distinct_pitches(notes: list) -> list:
    """Note pitches merged within NOTE_CENTS, in order of first appearance."""
    out: list = []
    for hz, _, _ in notes:
        if not any(abs(cents(hz, o)) <= NOTE_CENTS for o in out):
            out.append(hz)
    return out


def categorize(sig: Signature) -> tuple:
    """(category, description) from pitch and pattern.

    tone        one steady pitch, one strike            "single mid tone 1000 Hz (B5)"
    repeating   one pitch, several beeps                "repeating beep x3 at 2000 Hz, every 250 ms"
    two-tone    two pitches (doorbell ding-dong)        "two-tone falling 659->523 Hz (E5->C5)"
    melody      three or more pitches                   "4-note melody 523-659-784-1047 Hz"
    voice       low gliding pitch (speech, singing)     "voice-like, gliding 150-210 Hz"
    percussive  untuned or low short strikes            "2 untuned hits (knocking)", "low thump x2"
    noise       untuned and sustained (hiss, voice)     "untuned noise, 900 ms"
    """
    ms = round(sig.duration_s * 1000)
    hits = max(1, len(sig.onsets))
    if not sig.is_tonal:
        if sig.duration_s <= 0.35 * hits:
            return "percussive", ("short untuned hit (clap/knock)" if hits == 1
                                  else f"{hits} untuned hits (knocking)")
        return "noise", f"untuned noise, {ms} ms"
    pitches = _distinct_pitches(sig.notes)
    if max(pitches) < 400 and max(d for _, _, d in sig.notes) < 0.13 \
            and sig.duration_s <= 0.25 * hits:
        return "percussive", (f"low thump x{hits} (~{pitches[0]:.0f} Hz, knocking)" if hits > 1
                              else f"low thump (~{pitches[0]:.0f} Hz)")
    if max(pitches) < 350 and len(sig.notes) >= 3:
        return "voice", f"voice-like, gliding {min(pitches):.0f}-{max(pitches):.0f} Hz, {ms} ms"
    if len(pitches) == 1:
        hz = pitches[0]
        strikes = max(hits, len(sig.notes))
        if strikes >= 2:
            gaps = np.diff(sig.onsets) if len(sig.onsets) >= 2 else \
                np.diff([n[1] for n in sig.notes])
            every = f", every {round(float(np.median(gaps)) * 1000)} ms" if len(gaps) else ""
            return "repeating", f"repeating beep x{strikes} at {hz:.0f} Hz ({note_name(hz)}){every}"
        band = "low" if hz < 400 else ("high" if hz > 2000 else "mid")
        return "tone", f"single {band} tone {hz:.0f} Hz ({note_name(hz)}), {ms} ms"
    if len(pitches) == 2:
        a, b = pitches
        way = "falling" if b < a else "rising"
        return "two-tone", (f"two-tone {way} {a:.0f}->{b:.0f} Hz "
                            f"({note_name(a)}->{note_name(b)}), {ms} ms")
    return "melody", (f"{len(pitches)}-note melody "
                      + "-".join(f"{p:.0f}" for p in pitches[:6]) + " Hz")


def describe_frames(rms: list, pitches: list, on: list) -> Signature:
    """Signature from per-frame RMS, (pitch or None) and loud flags of one event."""
    dt = FRAME / SAMPLE_RATE
    loud = [i for i, o in enumerate(on) if o]
    last = loud[-1] if loud else 0
    dur = (last + 1) * dt
    tonal = sum(1 for i in loud if pitches[i] is not None) / max(1, len(loud))
    peak = max(rms[:last + 1]) if rms else 0.0
    # onsets: first frame, then level jumps over the recent minimum after a dip
    onsets = [0.0]
    for i in range(2, last + 1):
        recent = min(rms[max(0, i - 4):i - 1])
        if on[i] and rms[i] > ONSET_JUMP * recent and rms[i] > rms[i - 1] * 1.3 \
                and rms[i] >= ONSET_MIN_PEAK * peak and i * dt - onsets[-1] >= 0.08:
            onsets.append(i * dt)
    onset_frames = {int(round(t / dt)) for t in onsets[1:]}
    # notes: runs of tonal frames at a steady pitch (1-frame dropouts tolerated)
    notes, cur, gap = [], None, 0

    def close(c):
        if c and len(c["p"]) >= 2:
            notes.append((float(np.median(c["p"])), c["i0"] * dt, (c["i1"] - c["i0"] + 1) * dt))

    for i in range(last + 1):
        p = pitches[i]
        if p is None:
            gap += 1
            if gap > 1 and cur:
                close(cur)
                cur = None
            continue
        gap = 0
        if cur and abs(cents(p, float(np.median(cur["p"][-4:])))) <= NOTE_CENTS \
                and i not in onset_frames:
            cur["p"].append(p)
            cur["i1"] = i
        else:
            close(cur)
            cur = {"p": [p], "i0": i, "i1": i}
    close(cur)
    if notes:   # drop fragments (background talk, room) next to the sound's main notes
        longest = max(d for _, _, d in notes)
        notes = [n for n in notes if n[2] >= NOTE_MIN_SHARE * longest]
    sig = Signature(dur, tonal, notes, onsets)
    sig.category, sig.description = categorize(sig)
    return sig


def pitch_score(t: Signature, e: Signature) -> float:
    """How well the event's notes match the template's (duration-weighted),
    with penalties for a different note order or count."""
    if not t.notes or not e.notes:
        return 0.0
    total = sum(d for _, _, d in t.notes)
    s = 0.0
    for hz, _, d in t.notes:
        c = min(abs(cents(eh, hz)) for eh, _, _ in e.notes)
        s += d / total * math.exp(-(c / 100.0) ** 2)     # 100 cents (a semitone) -> 0.37
    td, ed = _distinct_pitches(t.notes), _distinct_pitches(e.notes)
    # order of the distinct pitches (ding-dong vs dong-ding); repeats of one pitch don't count
    order = [min(range(len(ed)), key=lambda j: abs(cents(ed[j], hz))) for hz in td]
    if order != sorted(order):
        s *= 0.4                  # same notes, other order: a different chime
    return s * math.sqrt(min(len(td), len(ed)) / max(len(td), len(ed)))


def pattern_score(t: Signature, e: Signature) -> float:
    """Same number of strikes, at similar spacing."""
    nt, ne = len(t.onsets), len(e.onsets)
    s = min(nt, ne) / max(nt, ne)
    if nt >= 2 and ne >= 2:
        gt, ge = float(np.median(np.diff(t.onsets))), float(np.median(np.diff(e.onsets)))
        s *= _dur_factor(gt, ge) ** 0.5
    return s


# --------------------------------------------------------------------------
# YAMNet: Google's pretrained AudioSet classifier (521 classes), for sounds
# nobody taught ("Doorbell", "Knock", "Smoke detector, smoke alarm", ...)
# --------------------------------------------------------------------------

YAMNET_URL = ("https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
              "task_library/audio_classification/android/"
              "lite-model_yamnet_classification_tflite_1.tflite")
YAMNET_FILE = ROOT / "models" / "yamnet" / "yamnet.tflite"      # gitignored, ~4 MB
# AudioSet classes that buzz (HDB_YAMNET_CLASSES="Doorbell;Knock" overrides; ';'-separated).
# Doorbell only for now (Sophanit, Nov 14): in a live run real rings scored
# 0.89-0.96 while "Door" fired 7 false alerts at 0.33-0.41.
YAMNET_WATCH = ("Doorbell", "Ding-dong")
# Candidates for later: "Knock", "Smoke detector, smoke alarm", "Fire alarm", "Alarm",
# "Alarm clock", "Siren", "Buzzer", "Telephone bell ringing", "Ringtone",
# "Baby cry, infant cry", "Bark", "Microwave oven", "Beep, bleep", "Shatter"
YAMNET_MIN = 0.50              # class score that buzzes (HDB_YAMNET_MIN)
# Haptic pattern category (haptics.SOUND_PATTERNS) per class; others use "yamnet"
# (no pattern of its own: the shared sweep).
YAMNET_CATEGORY = {"Doorbell": "two-tone", "Ding-dong": "two-tone", "Knock": "percussive",
                   "Smoke detector, smoke alarm": "repeating", "Fire alarm": "repeating",
                   "Alarm": "repeating", "Siren": "melody", "Baby cry, infant cry": "voice",
                   "Bark": "noise", "Shatter": "percussive"}


class YamNet:
    """YAMNet TFLite on the LiteRT runtime (pip install ai-edge-litert). Weights
    download at runtime into models/yamnet/. scores() takes 16 kHz float audio
    and returns the max over 0.975 s windows (hop 0.49 s) per class."""

    def __init__(self, path: Path = YAMNET_FILE):
        self.path = Path(path)
        self._it = None
        self.labels: list = []
        self._lock = threading.Lock()

    def load(self) -> "YamNet":
        with self._lock:
            if self._it is not None:
                return self
            if not self.path.exists():
                import urllib.request
                try:
                    import certifi, ssl     # python.org macOS builds ship no CA bundle
                    ctx = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    ctx = None
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".part")
                with urllib.request.urlopen(YAMNET_URL, context=ctx, timeout=60) as r, \
                        open(tmp, "wb") as f:
                    f.write(r.read())
                tmp.rename(self.path)
            import zipfile
            from ai_edge_litert.interpreter import Interpreter
            self.labels = zipfile.ZipFile(self.path).read(
                "yamnet_label_list.txt").decode().splitlines()
            it = Interpreter(model_path=str(self.path))
            it.allocate_tensors()
            self._in, self._out = it.get_input_details()[0], it.get_output_details()[0]
            self._it = it
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        self.load()
        n = int(self._in["shape"][-1])          # 15600 samples = 0.975 s
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        out = None
        with self._lock:
            for i in range(0, len(x) - n + 1, n // 2):
                self._it.set_tensor(self._in["index"], x[i:i + n].reshape(self._in["shape"]))
                self._it.invoke()
                s = self._it.get_tensor(self._out["index"]).reshape(-1)
                out = s if out is None else np.maximum(out, s)
        return out

    def top(self, x: np.ndarray, k: int = 5) -> list:
        s = self.scores(x)
        return [(self.labels[i], float(s[i])) for i in np.argsort(s)[::-1][:k]]


def yamnet_available() -> Optional[str]:
    """None if YAMNet can run here, else the reason."""
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        return "pip install ai-edge-litert"
    return None


# --------------------------------------------------------------------------
# Templates (one example of a taught sound)
# --------------------------------------------------------------------------

@dataclass
class Template:
    fp: np.ndarray
    duration_s: float
    level: float
    taught_at: float = 0.0
    label: str = DEFAULT_LABEL
    sig: Optional[Signature] = None      # None: a v1 template (spectrum-only matching)

    def to_json(self) -> dict:
        d = {"fp": [round(float(x), 5) for x in self.fp], "duration_s": round(self.duration_s, 3),
             "level": round(self.level, 5), "taught_at": self.taught_at,
             "bands": NBANDS, "frame": FRAME, "rate": SAMPLE_RATE}
        if self.sig is not None:
            d["sig"] = self.sig.to_json()
        return d

    @classmethod
    def from_json(cls, d: dict, label: str = DEFAULT_LABEL) -> "Template":
        if d.get("bands") != NBANDS or d.get("frame") != FRAME or d.get("rate") != SAMPLE_RATE:
            raise ValueError("taught sound was saved with different settings; teach it again")
        sig = Signature.from_json(d["sig"]) if d.get("sig") else None
        return cls(np.asarray(d["fp"], dtype=float), float(d["duration_s"]),
                   float(d["level"]), float(d.get("taught_at", 0.0)), label, sig)

    @property
    def category(self) -> str:
        return self.sig.category if self.sig else "unknown"

    @property
    def description(self) -> str:
        return self.sig.description if self.sig else f"{self.duration_s * 1000:.0f} ms sound"


def similarity(fp: np.ndarray, dur_s: float, tpl: Template,
               sig: Optional[Signature] = None) -> float:
    """Match score 0..1 of an event (fingerprint, duration, signature) against one
    template. Without signatures: spectral cosine x duration factor (v1)."""
    spectral = max(0.0, float(np.dot(fp, tpl.fp)))
    dur = _dur_factor(dur_s, tpl.duration_s)
    t = tpl.sig
    if t is None or sig is None:
        return spectral * dur
    pat = pattern_score(t, sig)
    if t.is_tonal:
        if sig.tonal < 0.6 * TONAL_EVENT_FRAC or not sig.notes:
            return min(spectral, 0.5) * 0.6 * dur      # a clap can't be a chime
        s = 0.5 * pitch_score(t, sig) + 0.2 * pat + 0.3 * spectral
    else:
        s = 0.75 * spectral + 0.25 * pat
        if sig.is_tonal and sig.tonal > 0.7:
            s = min(s, 0.6)                           # a chime isn't a knock
    return s * dur


@dataclass(frozen=True)
class SoundHit:
    score: float
    start: float      # laptop time.monotonic() seconds
    end: float
    level: float
    label: str = DEFAULT_LABEL
    category: str = ""
    pitch_hz: Optional[float] = None


@dataclass
class _Event:
    start_n: int
    bands: list                      # band energies of loud frames (fingerprint)
    peak: float
    rms: list = field(default_factory=list)
    pitches: list = field(default_factory=list)
    on: list = field(default_factory=list)
    quiet: int = 0
    audio: list = field(default_factory=list)   # raw frames incl. pre-roll (for YAMNet)


PRE_ROLL_FRAMES = 6            # ~190 ms of audio kept before an event starts


class SoundListener:
    def __init__(self, on_detect: Optional[Callable[[SoundHit], None]] = None,
                 threshold: float = 0.80, cooldown_s: float = 1.5, device=None,
                 template_file: Optional[Path] = TEMPLATE_FILE, yamnet=None):
        self.on_detect = on_detect
        # YAMNet for untaught sounds: yamnet=True/False, a YamNet-like object, or None
        # -> on when HDB_YAMNET=1. Runs on its own thread, never in the audio callback.
        import os
        if yamnet is None:
            yamnet = os.environ.get("HDB_YAMNET", "0").lower() in ("1", "on", "true", "yes")
        if yamnet is True:
            why = yamnet_available()
            yamnet = YamNet() if why is None else None
            if why:
                print(f"[listen] YAMNet off: {why}")
        self.yamnet = yamnet or None
        env = os.environ.get("HDB_YAMNET_CLASSES")
        self.yamnet_watch = tuple(c.strip() for c in env.split(";") if c.strip()) if env \
            else YAMNET_WATCH
        self.yamnet_min = float(os.environ.get("HDB_YAMNET_MIN", YAMNET_MIN))
        self.on_heard: Optional[Callable[[str], None]] = None   # "YAMNet: Knock 0.41, ..."
        self.last_yamnet: list = []
        self._yq = None
        if self.yamnet is not None:
            import queue
            self._yq = queue.Queue(maxsize=8)
            threading.Thread(target=self._yamnet_worker, name="yamnet", daemon=True).start()
            threading.Thread(target=self._yamnet_load, name="yamnet-load", daemon=True).start()
        self.on_taught: Optional[Callable[[Optional[Template], str], None]] = None
        self.on_event: Optional[Callable[[float, float], None]] = None   # debug: (score, dur)
        # debug: (Signature, best label or None, best score) for every event
        self.on_classified: Optional[Callable[[Signature, Optional[str], float], None]] = None
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.device = device
        self.template_file = Path(template_file) if template_file else None
        self.sounds: dict = {}            # label -> [Template] (examples)
        self.paused = False
        self.last: Optional[SoundHit] = None
        self.last_event_start = 0.0       # monotonic start of the last closed event
        self._latest: Optional[Template] = None
        self._levels: deque = deque(maxlen=int(3 * SAMPLE_RATE / FRAME))
        self._noise_psd: Optional[np.ndarray] = None   # room spectrum (quiet frames)
        self._pre: deque = deque(maxlen=PRE_ROLL_FRAMES)  # recent frames (event pre-roll)
        self._buf = np.zeros(0, dtype=np.float32)
        self._n = 0                       # samples consumed
        self._offset: Optional[float] = None
        self._ev: Optional[_Event] = None
        self._teach_until: Optional[int] = None   # sample index where teaching ends
        self._teach_label = DEFAULT_LABEL
        self._teach_add = False
        self._teach_events: list = []
        self._last_hit = -1e9
        self._lock = threading.Lock()
        self._stream = None
        if self.template_file and self.template_file.exists():
            try:
                self._load(json.loads(self.template_file.read_text()))
            except (ValueError, KeyError, TypeError) as e:
                print(f"[listen] ignoring {self.template_file.name}: {e}")

    # ------------------------------------------------------------- public
    @property
    def template(self) -> Optional[Template]:
        """The most recently taught example (None if nothing is taught)."""
        if self._latest is not None and self._latest.label in self.sounds:
            return self._latest
        allt = [t for ex in self.sounds.values() for t in ex]
        return max(allt, key=lambda t: t.taught_at) if allt else None

    @template.setter
    def template(self, tpl: Optional[Template]) -> None:
        if tpl is None:
            self.sounds.clear()
        else:
            self.sounds[tpl.label] = [tpl]
        self._latest = tpl

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

    def teach(self, seconds: float = 2.5, label: Optional[str] = None, add: bool = False) -> None:
        """Make the sound once within the next `seconds`; the loudest event wins.
        Same label again: replaces it, or adds an example with add=True."""
        with self._lock:
            self._teach_events = []
            self._teach_label = (label or DEFAULT_LABEL).strip() or DEFAULT_LABEL
            self._teach_add = add
            self._teach_until = self._n + len(self._buf) + int(seconds * SAMPLE_RATE)

    @property
    def teaching(self) -> bool:
        return self._teach_until is not None

    def forget(self, label: Optional[str] = None) -> None:
        """Forget one taught sound, or all of them."""
        if label is None:
            self.sounds.clear()
        else:
            self.sounds.pop(label, None)
        self._latest = None
        self._save()

    def categories(self) -> dict:
        """category -> [(label, main pitch Hz or None, description)], sorted by pitch."""
        out: dict = {}
        for label, ex in self.sounds.items():
            t = max(ex, key=lambda x: x.taught_at)
            pitch = t.sig.pitch_hz if t.sig else None
            out.setdefault(t.category, []).append((label, pitch, t.description))
        for v in out.values():
            v.sort(key=lambda r: (r[1] is None, r[1] or 0))
        return dict(sorted(out.items()))

    def summary(self) -> str:
        if not self.sounds:
            return "no sounds taught"
        lines = []
        for cat, rows in self.categories().items():
            lines.append(f"{cat}:")
            for label, pitch, desc in rows:
                n = len(self.sounds[label])
                lines.append(f"  {label}: {desc}" + (f"  [{n} examples]" if n > 1 else ""))
        return "\n".join(lines)

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
        ready = len(self._levels) >= 5
        start = ready and rms > max(MIN_RMS, RATIO * floor)
        hold = ready and rms > self._hold_level(floor)
        ev = self._ev
        spec = pitch_spectrum(f)
        if (ev is None and not start) or (ev is not None and not hold):
            self._levels.append(rms)            # floor tracks the room, not the event
            self._noise_psd = spec if self._noise_psd is None else \
                (1 - NOISE_ALPHA) * self._noise_psd + NOISE_ALPHA * spec
        if self.paused and not self.teaching:
            return
        if ev is None:
            if start:
                ev = self._ev = _Event(n0, [], rms)
                ev.audio = list(self._pre)
                self._add(ev, f, rms, True, spec)
            else:
                self._pre.append(f)
        elif hold:
            self._add(ev, f, rms, True, spec)
            ev.peak = max(ev.peak, rms)
            ev.quiet = 0
            if len(ev.on) * FRAME >= MAX_EVENT_S * SAMPLE_RATE:
                self._close(self._n)
        else:
            self._add(ev, f, rms, False)
            ev.quiet += 1
            if ev.quiet >= END_QUIET:
                self._close(self._n - END_QUIET * FRAME)
        if self._teach_until is not None and self._n >= self._teach_until and self._ev is None:
            self._finish_teach()

    def _hold_level(self, floor: float) -> float:
        """An open event continues while frames stay above this: above the room's
        usual ups and downs (HOLD_ROOM x its 90th percentile, and HOLD_RATIO x the
        floor) and above HOLD_PEAK x the event's own peak (a decaying chime ends
        when it fades into the room, not on its first dip)."""
        room = float(np.percentile(self._levels, 90)) if self._levels else 0.0
        peak = self._ev.peak if self._ev is not None else 0.0
        return max(0.5 * MIN_RMS, HOLD_RATIO * floor, HOLD_ROOM * room, HOLD_PEAK * peak)

    def _add(self, ev: _Event, f: np.ndarray, rms: float, on: bool,
             spec: Optional[np.ndarray] = None) -> None:
        ev.rms.append(rms)
        ev.on.append(on)
        ev.audio.append(f)
        if on:
            ev.bands.append(band_energies(f))
            clear = rms >= PITCH_RMS_RATIO * self.noise_floor   # well above the room
            ev.pitches.append(frame_pitch(f, self._noise_psd, spec)[0] if clear else None)
        else:
            ev.pitches.append(None)

    def _close(self, end_n: int) -> None:
        ev, self._ev = self._ev, None
        if ev is None or len(ev.bands) < MIN_EVENT_FRAMES:
            return
        sig = describe_frames(ev.rms, ev.pitches, ev.on)
        self.last_event_start = self._t(ev.start_n)
        dur = sig.duration_s
        fp = fingerprint(ev.bands)
        if self._teach_until is not None:
            self._teach_events.append(Template(fp, dur, ev.peak, time.time(),
                                               self._teach_label, sig))
            return
        end_n = ev.start_n + int(round(dur * SAMPLE_RATE))
        if not self.sounds:
            if self.on_classified:
                self.on_classified(sig, None, 0.0)
            self._to_yamnet(ev, sig, end_n)
            return
        best, best_label = 0.0, None
        for label, examples in self.sounds.items():
            for tpl in examples:
                s = similarity(fp, dur, tpl, sig)
                if s > best:
                    best, best_label = s, label
        if self.on_event:
            self.on_event(best, dur)
        if self.on_classified:
            self.on_classified(sig, best_label, best)
        end_t = self._t(end_n)
        if best < self.threshold:
            self._to_yamnet(ev, sig, end_n)
        if best >= self.threshold and end_t - self._last_hit >= self.cooldown_s:
            self._last_hit = end_t
            hit = SoundHit(round(best, 3), self._t(ev.start_n), end_t, ev.peak,
                           best_label, sig.category, sig.pitch_hz)
            self.last = hit
            if self.on_detect:
                try:
                    self.on_detect(hit)
                except Exception as e:
                    print(f"[listen] on_detect error: {e}")

    # ------------------------------------------------------------- YAMNet
    def _to_yamnet(self, ev: _Event, sig: Signature, end_n: int) -> None:
        if self._yq is None or self.paused:
            return
        try:
            self._yq.put_nowait((np.concatenate(ev.audio), self._t(ev.start_n),
                                 self._t(end_n), ev.peak, sig))
        except Exception:
            pass                       # busy: drop rather than lag behind the room

    def _yamnet_load(self) -> None:
        try:
            self.yamnet.load()
        except Exception as e:
            print(f"[listen] YAMNet load failed: {e}")

    def _yamnet_worker(self) -> None:
        while True:
            audio, start_t, end_t, peak, sig = self._yq.get()
            try:
                top = self.yamnet.top(audio, 5)
            except Exception as e:
                print(f"[listen] YAMNet error: {e}")
                continue
            self.last_yamnet = top
            if self.on_heard:
                try:
                    self.on_heard("YAMNet: " + ", ".join(f"{n} {s:.2f}" for n, s in top[:3]))
                except Exception:
                    pass
            watch = [(n, s) for n, s in top if n in self.yamnet_watch and s >= self.yamnet_min]
            if not watch:
                continue
            name, score = max(watch, key=lambda x: x[1])
            with self._lock:
                if self.paused or end_t - self._last_hit < self.cooldown_s:
                    continue
                self._last_hit = end_t
            hit = SoundHit(round(score, 3), start_t, end_t, peak, name,
                           YAMNET_CATEGORY.get(name, "yamnet"), sig.pitch_hz)
            self.last = hit
            if self.on_detect:
                try:
                    self.on_detect(hit)
                except Exception as e:
                    print(f"[listen] on_detect error: {e}")

    def _finish_teach(self) -> None:
        events, self._teach_events, self._teach_until = self._teach_events, [], None
        label = self._teach_label
        if not events:
            msg = "no sound heard while teaching; make it louder or closer to the laptop"
            tpl = None
        else:
            tpl = max(events, key=lambda e: e.level)
            ex = self.sounds.get(label, []) if self._teach_add else []
            self.sounds[label] = (ex + [tpl])[-MAX_EXAMPLES:]
            self._latest = tpl
            n = len(self.sounds[label])
            msg = (f"taught a {tpl.duration_s * 1000:.0f} ms sound '{label}': {tpl.description}"
                   f" [{tpl.category}]" + (f" ({n} examples)" if n > 1 else ""))
            if tpl.duration_s < 0.15:
                msg += "; very short, so try louder or closer for the whole sound"
            err = self._save()
            if err:
                msg += f" (not saved: {err})"
        if self.on_taught:
            try:
                self.on_taught(tpl, msg)
            except Exception as e:
                print(f"[listen] on_taught error: {e}")

    def _save(self) -> Optional[str]:
        if not self.template_file:
            return None
        try:
            if not self.sounds:
                if self.template_file.exists():
                    self.template_file.unlink()
                return None
            self.template_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"version": 2, "sounds": [
                {"label": label, "examples": [t.to_json() for t in ex]}
                for label, ex in self.sounds.items()]}
            self.template_file.write_text(json.dumps(data) + "\n")
        except OSError as e:
            return str(e)
        return None

    def _load(self, d: dict) -> None:
        if "fp" in d:                                  # v1: one fingerprint
            t = Template.from_json(d)
            self.sounds = {t.label: [t]}
        else:
            self.sounds = {s["label"]: [Template.from_json(x, s["label"]) for x in s["examples"]]
                           for s in d.get("sounds", [])}


def analyze(samples, sr: int = SAMPLE_RATE) -> list:
    """Every event in a clip -> [(start_s, Signature)] (no teaching, no matching).
    For tuning and for checking what a recording looks like."""
    if sr != SAMPLE_RATE:
        raise ValueError(f"need {SAMPLE_RATE} Hz audio")
    out = []
    sl = SoundListener(template_file=None)
    sl.on_classified = lambda sig, label, score: out.append((sl.last_event_start, sig))
    sl.feed(samples, t_first=0.0)
    sl.feed(np.zeros(SAMPLE_RATE, np.float32))
    return out


def main(argv=None) -> int:
    """python -m app.listen [--teach [--label doorbell] [--add]] [--seconds 20]
    [--list] [--forget LABEL|all] [--analyze clip.wav]"""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--teach", action="store_true", help="teach a sound first")
    ap.add_argument("--label", default=None, help="name for the taught sound, e.g. doorbell")
    ap.add_argument("--add", action="store_true", help="with --teach: add another example")
    ap.add_argument("--seconds", type=float, default=30.0, help="listen this long")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--list", action="store_true", help="print taught sounds by category")
    ap.add_argument("--forget", metavar="LABEL", help="forget one sound (or 'all')")
    ap.add_argument("--analyze", metavar="FILE", help="describe every sound in a wav/npy clip")
    a = ap.parse_args(argv)
    if a.analyze:
        if a.analyze.endswith(".npy"):
            x = np.load(a.analyze).astype(np.float32)
        else:
            from faster_whisper import decode_audio
            x = decode_audio(a.analyze, sampling_rate=SAMPLE_RATE)
        for t0, sig in analyze(x):
            print(f"  {t0:6.2f}s  {sig.duration_s * 1000:5.0f} ms  [{sig.category}] "
                  f"{sig.description}  (tonal {sig.tonal:.2f}, {len(sig.onsets)} onsets)")
        return 0
    done = threading.Event()
    sl = SoundListener(lambda h: print(f"  DETECTED '{h.label}' score {h.score:.2f}"),
                       threshold=a.threshold)
    if a.forget:
        sl.forget(None if a.forget == "all" else a.forget)
        print(f"[listen] forgot {a.forget}")
    if a.list:
        print(sl.summary())
        return 0
    sl.on_taught = lambda tpl, msg: (print(f"[listen] {msg}"), done.set())
    sl.on_classified = lambda sig, label, s: print(
        f"  event {sig.duration_s * 1000:.0f} ms [{sig.category}] {sig.description}"
        + (f" -> best '{label}' score {s:.2f}" if label else ""))
    sl.start()
    time.sleep(1.0)                       # learn the room's noise floor
    if a.teach or sl.template is None:
        print("[listen] make the sound now (2.5 s)...")
        sl.teach(2.5, a.label, a.add)
        done.wait(6)
    print(sl.summary())
    print(f"[listen] listening {a.seconds:.0f} s (Ctrl-C to stop)")
    try:
        time.sleep(a.seconds)
    except KeyboardInterrupt:
        pass
    sl.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
