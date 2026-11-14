"""Common household sounds with no teaching: YAMNet (MediaPipe Audio Classifier).

    sc = SoundClassifier(on_detect=cb)   # cb(listen.SoundHit), from the audio thread
    sc.start()                           # laptop mic via sounddevice (16 kHz mono)
    sc.feed(samples, t_first)            # or push float32 audio yourself (tests, files)
    sc.pause(); sc.resume()              # Action held: no alerts

PROTOTYPE beside app/listen.py (Teach a Sound), not wired into the server yet.
Hits have listen.SoundHit's shape (score, start/end on the laptop monotonic
clock, level, label, category) so App.on_sound can take them unchanged: the
category picks the haptic pattern (haptics.SOUND_PATTERNS) and start/end the
band-mic window for the direction estimate.

How it works:
  * YAMNet scores 521 AudioSet classes on 0.975 s windows; we classify a window
    every HOP_S (half a window, so a short knock is never split badly).
  * Each common sound (COMMON) is a group of YAMNet classes; its score in a
    window is the best class score in the group. A group fires when its score
    >= its threshold, then stays quiet for COOLDOWN_S.
  * YAMNet only knows "somewhere in this window": start/end are tightened to the
    32 ms frames louder than TIGHTEN x the window's loudest frame, because the
    direction estimator wants the sound itself, not the second around it.

The model (4 MB) downloads into models/ (gitignored) on first use; never commit it.
Thresholds: see tools/eval_sounds.py (ESC-50 clips + speech) for how they were set.
"""
from __future__ import annotations

import argparse
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .listen import SoundHit

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "yamnet.tflite"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/audio_classifier/"
             "yamnet/float32/1/yamnet.tflite")

SAMPLE_RATE = 16000
WINDOW_S = 0.975              # YAMNet's input length
HOP_S = 0.4875
COOLDOWN_S = 3.0
TIGHTEN = 0.3
FRAME_S = 0.032
DEFAULT_THRESHOLD = 0.3

# label -> (YAMNet class names, haptics category, threshold). Thresholds PROVISIONAL.
COMMON = {
    "alarm": (["Smoke detector, smoke alarm", "Fire alarm", "Alarm", "Alarm clock",
               "Buzzer", "Beep, bleep"], "repeating", 0.3),
    "doorbell": (["Doorbell", "Ding-dong"], "two-tone", 0.3),
    "knock": (["Knock"], "percussive", 0.3),
    "baby crying": (["Baby cry, infant cry"], "voice", 0.3),
    "dog": (["Dog", "Bark"], "noise", 0.3),
    "siren": (["Siren", "Police car (siren)", "Ambulance (siren)",
               "Fire engine, fire truck (siren)", "Civil defense siren"], "melody", 0.3),
    "car horn": (["Vehicle horn, car horn, honking", "Air horn, truck horn", "Car alarm"],
                 "tone", 0.3),
    "phone": (["Telephone bell ringing", "Ringtone"], "repeating", 0.3),
    "glass breaking": (["Shatter"], "percussive", 0.3),
}


def fetch(path: Path = MODEL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"[sounds] downloading YAMNet -> {path.relative_to(ROOT)}")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


def pick(g: dict, thresholds: dict):
    """The one sound a window fires (best score relative to its threshold), or None."""
    label, score = max(g.items(), key=lambda kv: kv[1] / thresholds[kv[0]])
    return label if score >= thresholds[label] else None


class Yamnet:
    """521 class scores for one window (any sample rate; MediaPipe resamples)."""

    def __init__(self, model: Path = MODEL):
        from mediapipe.tasks.python import audio
        from mediapipe.tasks.python.components.containers import audio_data
        from mediapipe.tasks.python.core import base_options
        self._ad = audio_data
        opts = audio.AudioClassifierOptions(
            base_options=base_options.BaseOptions(model_asset_path=str(fetch(model))),
            max_results=-1, running_mode=audio.RunningMode.AUDIO_CLIPS)
        self._clf = audio.AudioClassifier.create_from_options(opts)
        self._lock = threading.Lock()
        cats = self._classify(np.zeros(int(WINDOW_S * SAMPLE_RATE), np.float32), SAMPLE_RATE)
        self.names = [c.category_name for c in cats]
        self.index = {n: i for i, n in enumerate(self.names)}

    def _classify(self, x: np.ndarray, sr: int) -> list:
        with self._lock:
            res = self._clf.classify(self._ad.AudioData.create_from_array(
                np.asarray(x, np.float32), sr))
        return sorted(res[0].classifications[0].categories, key=lambda c: c.index)

    def scores(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        return np.array([c.score for c in self._classify(x, sr)], np.float32)

    def top(self, x: np.ndarray, sr: int = SAMPLE_RATE, k: int = 3) -> list:
        s = self.scores(x, sr)
        return [(self.names[i], round(float(s[i]), 2)) for i in np.argsort(-s)[:k]]


class SoundClassifier:
    def __init__(self, on_detect: Optional[Callable[[SoundHit], None]] = None,
                 sounds: Optional[dict] = None, sample_rate: int = SAMPLE_RATE,
                 model: Optional[Yamnet] = None, device=None):
        self.on_detect = on_detect
        self.on_window: Optional[Callable[[float, dict], None]] = None   # debug/eval
        self.sounds = dict(sounds or COMMON)
        self.sr = sample_rate
        self.device = device
        self.yamnet = model or Yamnet()
        missing = [n for cls, _, _ in self.sounds.values() for n in cls
                   if n not in self.yamnet.index]
        if missing:
            raise ValueError(f"not YAMNet classes: {missing}")
        self._groups = {k: np.array([self.yamnet.index[n] for n in v[0]])
                        for k, v in self.sounds.items()}
        self.paused = False
        self.last: Optional[SoundHit] = None
        self._win = int(round(WINDOW_S * self.sr))
        self._hop = int(round(HOP_S * self.sr))
        self._buf = np.zeros(0, np.float32)
        self._n0 = 0                    # sample index of _buf[0]
        self._offset: Optional[float] = None
        self._last_hit: dict = {}
        self._stream = None

    # ------------------------------------------------------------- public
    def group_scores(self, window: np.ndarray) -> dict:
        s = self.yamnet.scores(window, self.sr)
        return {k: float(s[idx].max()) for k, idx in self._groups.items()}

    def start(self) -> "SoundClassifier":
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      blocksize=self._hop, device=self.device,
                                      callback=self._cb)
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

    def resume(self) -> None:
        self._buf, self._n0 = np.zeros(0, np.float32), self._n0 + len(self._buf)
        self.paused = False

    def feed(self, samples, t_first: Optional[float] = None) -> None:
        """Push float32 mono audio at self.sr; t_first = monotonic time of samples[0]."""
        x = np.asarray(samples, np.float32).reshape(-1)
        if t_first is not None:
            off = t_first - (self._n0 + len(self._buf)) / self.sr
            self._offset = off if self._offset is None else min(self._offset + 1e-6, off)
        if self.paused:
            self._n0 += len(self._buf) + len(x)
            self._buf = np.zeros(0, np.float32)
            return
        self._buf = np.concatenate([self._buf, x])
        while len(self._buf) >= self._win:
            self._window(self._buf[:self._win], self._n0)
            self._buf, self._n0 = self._buf[self._hop:], self._n0 + self._hop

    # ---------------------------------------------------------- internals
    def _cb(self, indata, frames, time_info, status) -> None:
        lat = self._stream.latency if self._stream is not None else 0.0
        self.feed(indata[:, 0].copy(), time.monotonic() - frames / self.sr - lat)

    def _t(self, n: float) -> float:
        return (self._offset or 0.0) + n / self.sr

    def _window(self, w: np.ndarray, n0: int) -> None:
        g = self.group_scores(w)
        t0 = self._t(n0)
        if self.on_window:
            self.on_window(t0, g)
        label = pick(g, {k: v[2] for k, v in self.sounds.items()})
        if label is None:
            return
        score = g[label]
        if t0 - self._last_hit.get(label, -1e9) < COOLDOWN_S:
            return
        self._last_hit[label] = t0
        f = max(1, int(FRAME_S * self.sr))
        rms = np.sqrt(np.mean(w[:len(w) // f * f].reshape(-1, f) ** 2, axis=1))
        loud = np.nonzero(rms >= TIGHTEN * rms.max())[0] if rms.max() > 0 else [0]
        start, end = n0 + loud[0] * f, n0 + (loud[-1] + 1) * f
        hit = SoundHit(round(score, 3), self._t(start), self._t(end), float(rms.max()),
                       label, self.sounds[label][1], None)
        self.last = hit
        if self.on_detect:
            try:
                self.on_detect(hit)
            except Exception as e:
                print(f"[sounds] on_detect error: {e}")


# ---------------------------------------------------------------- files / CLI
def load_wav(path) -> tuple:
    """(float32 mono samples, sample rate) from a 16-bit PCM WAV."""
    import wave
    with wave.open(str(path), "rb") as f:
        sr, ch, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        raw = f.readframes(f.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), sr


def run_file(path, sc: Optional[SoundClassifier] = None) -> list:
    """Hits for one file (padded with 1 s of silence so its end gets a window)."""
    x, sr = load_wav(path)
    hits = []
    sc = sc or SoundClassifier(sample_rate=sr)
    sc.on_detect = hits.append
    sc.feed(np.concatenate([x, np.zeros(sr, np.float32)]), t_first=0.0)
    return hits


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Common sounds with YAMNet (no teaching)")
    p.add_argument("--file", nargs="*", help="WAV file(s): print hits and top classes")
    p.add_argument("--listen", action="store_true", help="laptop mic, print hits")
    p.add_argument("--seconds", type=float, default=0, help="with --listen: stop after N s")
    p.add_argument("--top", action="store_true",
                   help="with --listen: also print YAMNet's top 3 classes for each non-silent window")
    p.add_argument("--fetch", action="store_true", help="download the model and exit")
    a = p.parse_args(argv)
    if a.fetch:
        print(fetch())
        return
    if a.file:
        y = Yamnet()
        for f in a.file:
            x, sr = load_wav(f)
            hits = run_file(f, SoundClassifier(sample_rate=sr, model=y))
            print(f"{f}: top {y.top(x[:int(sr * WINDOW_S * 5)], sr)}")
            for h in hits:
                print(f"   {h.label} [{h.category}] {h.start:.2f}-{h.end:.2f}s score {h.score}")
        return
    if a.listen:
        sc = SoundClassifier(on_detect=lambda h: print(
            f"[sounds] {h.label} [{h.category}] score {h.score} "
            f"({(h.end - h.start) * 1000:.0f} ms)", flush=True))
        if a.top:
            def show(t, g, sc=sc):
                w = sc._buf[:sc._win]
                top = sc.yamnet.top(w)
                if top[0][0] != "Silence":
                    best = max(g, key=g.get)
                    print(f"  {top}   best sound: {best} {g[best]:.2f}", flush=True)
            sc.on_window = show
        sc.start()
        print("[sounds] listening; Ctrl-C to stop", flush=True)
        t0 = time.monotonic()
        try:
            while not a.seconds or time.monotonic() - t0 < a.seconds:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        sc.stop()
        return
    p.print_help()


if __name__ == "__main__":
    main()
