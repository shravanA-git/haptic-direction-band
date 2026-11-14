"""Hold-to-talk target selection: hold Action, say "find the banana", release.

Owner: HDB-Vision (branch ``vision``). Never touches the serial port.

Interface for app/main.py / app/server.py
-----------------------------------------
    from app.speech import Speech

    sp = Speech(on_result=callback)    # callback(SpeechResult), from a worker thread
    sp.load()                          # optional: download/load the model up front
    sp.begin()                         # BTN ACTION DOWN: start recording the laptop mic
    sp.end()                           # BTN ACTION UP: stop, transcribe in the background
    sp.on_button(down)                 # same as begin()/end()

``SpeechResult`` fields:
    text      full transcript, e.g. "Find the banana."
    target    COCO class matched in the text ("banana"), or None -> keep the typed
              fallback (dashboard Find box / --target)
    words     [Word(text, start, end)], start/end on the laptop's time.monotonic()
              clock (seconds), for name timestamps later
    t0        time.monotonic() of the first audio sample
    seconds   audio length

Typical wiring (vision + guide are already paused while Action is held):
    sp = Speech(on_result=lambda r: r.target and vision.set_target(r.target))

Running it
----------
    pip install faster-whisper sounddevice
    python -m app.speech --file clip.wav           # transcribe a file, print target
    python -m app.speech --hold 3                  # record 3 s from the mic
    python -m app.speech --selftest                # target matching (no model)
    python -m app.speech --selftest --tts          # + macOS `say` round trip (needs model)

Whisper weights (``tiny.en`` by default, ~75 MB; ~0.2 s per phrase on CPU) download at runtime into
models/whisper/ (gitignored). Set HDB_WHISPER_MODEL to override.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .vision import COCO_CLASSES, SYNONYMS

SAMPLE_RATE = 16000
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models" / "whisper"
DEFAULT_MODEL = os.environ.get("HDB_WHISPER_MODEL", "tiny.en")
MIN_SECONDS = 0.3     # shorter holds are treated as accidental presses
MAX_SECONDS = 15.0    # cap a stuck button


# --------------------------------------------------------------------------
# Target matching: transcript -> COCO class (pure, testable)
# --------------------------------------------------------------------------

def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z ]+", " ", text.lower().replace("-", " ")).split()


def _singular(w: str) -> list[str]:
    out = [w]
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s"):
        out.append(w[:-1])
    return out


_VOCAB = {c: c for c in COCO_CLASSES}
_VOCAB.update(SYNONYMS)
# Common mishearings of short demo words.
_VOCAB.update({"bananas": "banana", "banan": "banana", "nana": "banana",
               "cellphone": "cell phone", "cop": "cup"})


def match_target(text: str) -> Optional[str]:
    """Find the object asked for in free speech. Prefers the words after
    "find"/"where"/"look for"; longest phrase wins ("cell phone" over "phone")."""
    words = _norm(text)
    if not words:
        return None
    for i, w in enumerate(words):  # focus on what follows the command word
        if w in ("find", "where", "locate", "for", "get"):
            tail = words[i + 1:]
            hit = _scan(tail)
            if hit:
                return hit
    return _scan(words)


def _scan(words: list[str]) -> Optional[str]:
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = words[i:i + n]
            for last in _singular(phrase[-1]):
                key = " ".join(phrase[:-1] + [last])
                if key in _VOCAB:
                    return _VOCAB[key]
    return None


# --------------------------------------------------------------------------
# Transcriber: faster-whisper, weights downloaded at runtime
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    text: str
    start: float   # time.monotonic() seconds
    end: float


@dataclass
class SpeechResult:
    text: str
    target: Optional[str]
    words: list = field(default_factory=list)
    t0: float = 0.0
    seconds: float = 0.0
    error: Optional[str] = None


class Transcriber:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self._model = None
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                self._model = WhisperModel(self.model_name, device="cpu",
                                           compute_type="int8",
                                           download_root=str(MODELS_DIR))
        return self._model

    def transcribe(self, audio, t0: float = 0.0, prompt: Optional[str] = None):
        """audio: float32 mono at 16 kHz. Returns (text, [Word]) with word times
        shifted onto the monotonic clock by t0."""
        model = self.load()
        segs, _ = model.transcribe(audio, language="en", beam_size=1,
                                   word_timestamps=True, vad_filter=True,
                                   condition_on_previous_text=False,
                                   initial_prompt=prompt)
        text, words = [], []
        for s in segs:
            text.append(s.text.strip())
            for w in (s.words or []):
                words.append(Word(w.word.strip(), t0 + float(w.start), t0 + float(w.end)))
        return " ".join(text).strip(), words


def load_audio(path: str):
    """Any file ffmpeg/PyAV can read -> float32 mono 16 kHz."""
    from faster_whisper import decode_audio
    return decode_audio(path, sampling_rate=SAMPLE_RATE)


# --------------------------------------------------------------------------
# Hold-to-talk recorder
# --------------------------------------------------------------------------

# An initial prompt made whisper hallucinate trailing words ("banana. Anna."), so none.
PROMPT = None


class Speech:
    def __init__(self, on_result: Optional[Callable[[SpeechResult], None]] = None,
                 model: str = DEFAULT_MODEL, device=None):
        self.on_result = on_result
        self.device = device            # sounddevice input device (None = default mic)
        self.transcriber = Transcriber(model)
        self._lock = threading.Lock()
        self._chunks: list = []
        self._stream = None
        self._t0: Optional[float] = None
        self.recording = False
        self.busy = False               # transcription in progress
        self.last: Optional[SpeechResult] = None

    def load(self) -> "Speech":
        self.transcriber.load()
        return self

    def on_button(self, down: bool) -> None:
        self.begin() if down else self.end()

    def begin(self) -> None:
        with self._lock:
            if self.recording:
                return
            import sounddevice as sd
            self._chunks, self._t0 = [], None
            try:
                self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                              dtype="float32", device=self.device,
                                              callback=self._cb)
                self._stream.start()
            except Exception as e:
                self._stream = None
                self._deliver(SpeechResult("", None, error=f"mic: {e}"))
                return
            self.recording = True

    def _cb(self, indata, frames, time_info, status) -> None:
        if self._t0 is None:  # first block: back-date to its first sample
            self._t0 = time.monotonic() - frames / SAMPLE_RATE
        if sum(len(c) for c in self._chunks) < MAX_SECONDS * SAMPLE_RATE:
            self._chunks.append(indata[:, 0].copy())

    def end(self) -> None:
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            stream, self._stream = self._stream, None
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        import numpy as np
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        t0 = self._t0 or time.monotonic()
        threading.Thread(target=self._finish, args=(audio, t0), name="speech",
                         daemon=True).start()

    def _finish(self, audio, t0: float) -> None:
        secs = len(audio) / SAMPLE_RATE
        if secs < MIN_SECONDS:
            self._deliver(SpeechResult("", None, t0=t0, seconds=secs, error="too short"))
            return
        self.busy = True
        try:
            text, words = self.transcriber.transcribe(audio, t0, PROMPT)
            self._deliver(SpeechResult(text, match_target(text), words, t0, secs))
        except Exception as e:
            self._deliver(SpeechResult("", None, t0=t0, seconds=secs, error=str(e)))
        finally:
            self.busy = False

    def _deliver(self, r: SpeechResult) -> None:
        self.last = r
        if self.on_result:
            try:
                self.on_result(r)
            except Exception as e:
                print(f"[speech] callback error: {e}")


# --------------------------------------------------------------------------
# Self-test and CLI
# --------------------------------------------------------------------------

def _selftest(tts: bool = False) -> None:
    cases = {
        "Find the banana.": "banana",
        "find the bananas": "banana",
        "Where's my phone?": "cell phone",
        "uh can you find my cell phone": "cell phone",
        "look for the water bottle": "bottle",
        "find a cup": "cup",
        "find the teddy bear": "teddy bear",
        "Find the person.": "person",
        "hello there": None,
        "": None,
    }
    for text, want in cases.items():
        got = match_target(text)
        assert got == want, (text, got, want)
    print("speech selftest (matching): OK")
    if not tts:
        return
    import subprocess
    import tempfile
    tr = Transcriber()
    with tempfile.TemporaryDirectory() as d:
        for phrase, want in (("Find the banana.", "banana"), ("Find my phone.", "cell phone")):
            wav = os.path.join(d, "x.wav")
            subprocess.run(["say", "-o", wav, "--data-format=LEI16@16000", phrase], check=True)
            t = time.monotonic()
            text, words = tr.transcribe(load_audio(wav), t0=100.0, prompt=PROMPT)
            dt = time.monotonic() - t
            got = match_target(text)
            print(f"  say {phrase!r} -> {text!r} -> {got} ({dt:.2f}s); "
                  f"words={[(w.text, round(w.start, 2), round(w.end, 2)) for w in words]}")
            assert got == want, (phrase, text, got)
            assert words and all(100.0 <= w.start <= w.end for w in words), words
    print("speech selftest (tts round trip): OK")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="HDB hold-to-talk target selection")
    p.add_argument("--file", help="transcribe this audio file")
    p.add_argument("--hold", type=float, help="record N seconds from the mic")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--tts", action="store_true", help="with --selftest: macOS say round trip")
    a = p.parse_args(argv)
    if a.selftest:
        _selftest(a.tts)
        return

    def show(r: SpeechResult) -> None:
        if r.error:
            print(f"[speech] error: {r.error}")
        print(f"[speech] {r.seconds:.1f}s  text={r.text!r}  target={r.target!r}")
        for w in r.words:
            print(f"    {w.start:10.3f}-{w.end:10.3f}  {w.text}")

    if a.file:
        tr = Transcriber(a.model)
        text, words = tr.transcribe(load_audio(a.file), time.monotonic(), PROMPT)
        show(SpeechResult(text, match_target(text), words))
        return
    if a.hold:
        done = threading.Event()
        sp = Speech(on_result=lambda r: (show(r), done.set()), model=a.model).load()
        print(f"[speech] recording {a.hold:.1f}s: say 'find the banana'")
        sp.begin()
        time.sleep(a.hold)
        sp.end()
        done.wait(60)
        return
    p.print_help()


if __name__ == "__main__":
    main()
