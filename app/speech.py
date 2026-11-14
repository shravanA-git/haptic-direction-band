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
    python -m app.speech --listen Shravan,Jason    # always-on name spotting from the mic
    python -m app.speech --selftest                # target matching (no model)
    python -m app.speech --selftest --tts          # + macOS `say` round trip (needs model)

Name recognition (second demo)
------------------------------
    nl = NameListener(["Shravan"], on_hit=cb)   # cb(NameHit) from a worker thread
    nl.start(); nl.pause(); nl.resume(); nl.stop()   # pause while Action is held
    NameHit: name, heard, start, end (time.monotonic() s), start_ms, end_ms,
             score (1.0 exact), text, latency (s from name end to report)
Hand start_ms/end_ms to the direction estimator; it picks the MIC frames in that
window after clock alignment.

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
    prob: float = 1.0  # whisper's word probability


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
        self._run_lock = threading.Lock()

    def load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                self._model = WhisperModel(self.model_name, device="cpu",
                                           compute_type="int8",
                                           download_root=str(MODELS_DIR))
        return self._model

    def transcribe(self, audio, t0: float = 0.0, prompt: Optional[str] = None,
                   hotwords: Optional[str] = None, vad: bool = True):
        """audio: float32 mono at 16 kHz. Returns (text, [Word]) with word times
        shifted onto the monotonic clock by t0. ``hotwords`` biases toward unusual
        names ("Sophanit" is heard as "Soften it" without it)."""
        model = self.load()
        with self._run_lock:  # one decode at a time (hold-to-talk and name listener)
            segs, _ = model.transcribe(audio, language="en", beam_size=1,
                                       word_timestamps=True, vad_filter=vad,
                                       condition_on_previous_text=False,
                                       initial_prompt=prompt, hotwords=hotwords)
            text, words = [], []
            for s in segs:
                text.append(s.text.strip())
                for w in (s.words or []):
                    words.append(Word(w.word.strip(), t0 + float(w.start), t0 + float(w.end),
                                      float(getattr(w, "probability", 1.0))))
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
                 model: str = DEFAULT_MODEL, device=None, preload: bool = True):
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
        if preload:  # warm whisper in the background so the first Action press is fast
            threading.Thread(target=self._preload, name="speech-load", daemon=True).start()

    def _preload(self) -> None:
        try:
            self.transcriber.load()
        except Exception as e:
            print(f"[speech] model load failed: {e}")

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
# Name recognition: always-on listener -> NameHit with start/end on the
# laptop's monotonic clock (for the 4-mic direction estimator)
# --------------------------------------------------------------------------

def _squash(w: str) -> str:
    """Rough phonetic key: lowercase letters, 'ph'->'f', 'sh'->'s', drop h/w/y and
    vowels after the first letter, collapse repeats ("Shravin" ~ "Shravan")."""
    w = re.sub(r"[^a-z]", "", w.lower()).replace("ph", "f").replace("sh", "s")
    if not w:
        return ""
    head, tail = w[0], re.sub(r"[aeiouhwy]", "", w[1:])
    out = head
    for c in tail:
        if c != out[-1]:
            out += c
    return out


def match_name(words: list, names: list, threshold: float = 0.8):
    """Find a configured name in recognized words. Tries single words and pairs
    of adjacent words ("Shra van"). Returns (name, first_idx, last_idx, score)
    or None. Exact (case-insensitive) matches score 1.0."""
    import difflib
    toks = [re.sub(r"[^a-z]", "", (w.text if hasattr(w, "text") else w).lower())
            for w in words]
    best = None
    for name in names:
        key = re.sub(r"[^a-z]", "", name.lower())
        skey = _squash(name)
        for n in (1, 2):
            for i in range(len(toks) - n + 1):
                cand = "".join(toks[i:i + n])
                if not cand:
                    continue
                if cand == key:
                    score = 1.0
                else:
                    r1 = difflib.SequenceMatcher(None, cand, key).ratio()
                    r2 = difflib.SequenceMatcher(None, _squash(cand), skey).ratio()
                    score = 0.5 * r1 + 0.5 * r2 - (0.05 if n == 2 else 0.0)
                if score >= threshold and (best is None or score > best[3]):
                    best = (name, i, i + n - 1, round(score, 3))
    return best


@dataclass
class NameHit:
    name: str          # configured name that matched
    heard: str         # what whisper wrote, e.g. "Shravan," or "Soften it"
    start: float       # time.monotonic() seconds, start of the name
    end: float         # time.monotonic() seconds, end of the name
    score: float       # 1.0 exact, lower for fuzzy matches
    text: str          # the whole utterance
    latency: float     # seconds from end of the name to this report

    @property
    def start_ms(self) -> int:
        return int(round(self.start * 1000))

    @property
    def end_ms(self) -> int:
        return int(round(self.end * 1000))


class NameListener:
    """Always-on laptop-mic listener that reports when a configured name is said.

        nl = NameListener(["Shravan"], on_hit=cb)   # cb(NameHit), from a worker thread
        nl.start()          # opens the default mic
        nl.pause(); nl.resume()   # while Action is held (hold-to-talk owns speech)
        nl.stop()
        nl.feed(samples, t_first)  # or push audio yourself (tests, replay)

    Utterances are cut by an energy detector (speech, then SILENCE_S of quiet, or
    MAX_UTTER_S), transcribed with the names as whisper hotwords, then fuzzy-
    matched. Times are laptop time.monotonic(): each mic block is stamped on
    arrival and the sample clock is anchored with a min-filter, so callback
    scheduling jitter doesn't shift the word times.
    """

    FRAME = 480            # 30 ms at 16 kHz
    PRE_ROLL_S = 0.3
    SILENCE_S = 0.45
    MAX_UTTER_S = 5.0
    MIN_UTTER_S = 0.25
    MIN_START_RMS = 0.003  # absolute floor for "someone is talking"
    VERIFY_THRESHOLD = 0.6  # hotwords can make whisper invent the name in noise
                            # ("Shravan Shravan" from room chatter), so a candidate
                            # must also roughly match in an unbiased decode

    def __init__(self, names, on_hit: Optional[Callable[[NameHit], None]] = None,
                 transcriber: Optional[Transcriber] = None, device=None,
                 threshold: float = 0.8, cooldown_s: float = 1.5):
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        self.names = list(names)
        self.on_hit = on_hit
        self.on_text: Optional[Callable[[str, list], None]] = None  # debug: every utterance
        self.transcriber = transcriber or Transcriber()
        self.device = device
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.paused = False
        self.last: Optional[NameHit] = None
        self._stream = None
        self._q: "queue.Queue" = __import__("queue").Queue(maxsize=8)
        self._worker: Optional[threading.Thread] = None
        self._reset_seg()
        self._levels: "collections.deque" = __import__("collections").deque(maxlen=100)
        self.noise_floor = 0.0
        self._n = 0                # samples fed so far
        self._offset = None        # monotonic(sample 0), min-filtered
        self._last_hit_t = -1e9

    # ---- public ----
    def set_names(self, names) -> None:
        self.names = [n.strip() for n in (names.split(",") if isinstance(names, str)
                                          else names) if n.strip()]

    def start(self) -> "NameListener":
        import sounddevice as sd
        self._ensure_worker()
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                      blocksize=self.FRAME, device=self.device,
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
        self._q.put(None)

    def pause(self) -> None:
        self.paused = True
        self._reset_seg()

    def resume(self) -> None:
        self._reset_seg()
        self.paused = False

    def feed(self, samples, t_first: Optional[float] = None) -> None:
        """Push float32 mono 16 kHz audio. t_first = monotonic time of samples[0]
        (default: now minus the block length)."""
        import numpy as np
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if t_first is None:
            t_first = time.monotonic() - len(samples) / SAMPLE_RATE
        off = t_first - self._n / SAMPLE_RATE
        self._offset = off if self._offset is None else min(self._offset + 1e-6, off)
        self._ensure_worker()
        for i in range(0, len(samples), self.FRAME):
            self._frame(samples[i:i + self.FRAME])

    # ---- internals ----
    def _cb(self, indata, frames, time_info, status) -> None:
        lat = self._stream.latency if self._stream is not None else 0.0
        self.feed(indata[:, 0].copy(), time.monotonic() - frames / SAMPLE_RATE - lat)

    def _t(self, n: int) -> float:
        return (self._offset or 0.0) + n / SAMPLE_RATE

    def _reset_seg(self) -> None:
        self._seg: list = []
        self._seg_start = None     # sample index of the segment start
        self._quiet = 0            # samples of quiet since the last loud frame
        self._pre: list = []       # pre-roll frames

    def _frame(self, f) -> None:
        import numpy as np
        n0 = self._n
        self._n += len(f)
        if self.paused or len(f) == 0:
            return
        rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
        # Noise floor = 20th percentile of the last ~3 s of frames: robust to the
        # mic's start-up click and to speech itself.
        self._levels.append(rms)
        if len(self._levels) < 10:
            return
        floor = float(np.percentile(self._levels, 20))
        self.noise_floor = floor
        if self._seg_start is None:
            loud = rms > max(3.0 * floor, self.MIN_START_RMS)
        else:  # hysteresis: keep the utterance going on softer frames
            loud = rms > max(1.8 * floor, self.MIN_START_RMS * 0.6)
        if self._seg_start is None:
            self._pre.append((n0, f))
            pre_n = int(self.PRE_ROLL_S * SAMPLE_RATE / self.FRAME)
            del self._pre[:-pre_n]
            if loud:
                self._seg_start = self._pre[0][0]
                self._seg = [x for _, x in self._pre]
                self._pre = []
                self._quiet = 0
            return
        self._seg.append(f)
        self._quiet = 0 if loud else self._quiet + len(f)
        seg_len = sum(len(x) for x in self._seg)
        if (self._quiet >= self.SILENCE_S * SAMPLE_RATE
                or seg_len >= self.MAX_UTTER_S * SAMPLE_RATE):
            if seg_len - self._quiet >= self.MIN_UTTER_S * SAMPLE_RATE:
                audio = np.concatenate(self._seg)
                try:
                    self._q.put_nowait((audio, self._t(self._seg_start)))
                except Exception:
                    pass  # decoder is behind; drop rather than lag further
            self._reset_seg()

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="names", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            audio, t0 = item
            try:
                self.process(audio, t0)
            except Exception as e:
                print(f"[names] error: {e}")

    def process(self, audio, t0: float) -> Optional[NameHit]:
        """Transcribe one utterance that started at monotonic t0; report a hit."""
        if not self.names or self.paused:
            return None
        text, words = self.transcriber.transcribe(audio, t0, hotwords=" ".join(self.names),
                                                  vad=False)
        if self.on_text:
            self.on_text(text, words)
        m = match_name(words, self.names, self.threshold)
        if m is None:
            return None
        name, i, j, score = m
        _, plain = self.transcriber.transcribe(audio, t0, vad=False)
        if match_name(plain, [name], self.VERIFY_THRESHOLD) is None:
            if self.on_text:
                self.on_text(f"(rejected {name!r}: unbiased decode heard "
                             f"{' '.join(w.text for w in plain)!r})", plain)
            return None
        start, end = words[i].start, words[j].end
        if start - self._last_hit_t < self.cooldown_s:
            return None
        self._last_hit_t = start
        hit = NameHit(name, " ".join(w.text for w in words[i:j + 1]), start, end, score,
                      text, round(time.monotonic() - end, 3))
        self.last = hit
        if self.on_hit:
            try:
                self.on_hit(hit)
            except Exception as e:
                print(f"[names] callback error: {e}")
        return hit


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
    nm = [("Hey Shravan,".split(), ["Shravan"], "Shravan"), (["Sravan"], ["Shravan"], "Shravan"),
          (["Jayson"], ["Jason"], "Jason"), (["Mason"], ["Jason"], None),
          (["seven"], ["Shravan"], None), ("hello there".split(), ["Shravan", "Jason"], None)]
    for words, names, want in nm:
        got = match_name(words, names)
        assert (got[0] if got else None) == want, (words, names, got)
    print("speech selftest (name matching): OK")
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
    _selftest_names(tr)


def _selftest_names(tr) -> None:
    """Feed `say` audio through NameListener as if it came from the mic at a known
    monotonic time; check the hit and its timestamps."""
    import subprocess
    import tempfile
    import numpy as np
    T0 = 1000.0
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        for phrase, names, want in (("Hey Shravan, come here.", ["Shravan"], "Shravan"),
                                    ("Hey Sophanit, look left.", ["Sophanit", "Jason"],
                                     "Sophanit"),
                                    ("The weather is nice today.", ["Shravan"], None)):
            wav = os.path.join(d, "n.wav")
            subprocess.run(["say", "-o", wav, "--data-format=LEI16@16000", phrase], check=True)
            speech = load_audio(wav)
            pad = np.zeros(SAMPLE_RATE, np.float32)
            audio = np.concatenate([pad, speech, pad, pad])
            audio = audio + rng.normal(0, 0.001, len(audio)).astype(np.float32)
            done = threading.Event()
            hits = []
            nl = NameListener(names, on_hit=lambda h: (hits.append(h), done.set()),
                              transcriber=tr)
            seen = []
            nl.on_text = lambda text, words: (seen.append(text), done.set())
            for i in range(0, len(audio), 1600):  # 100 ms blocks, stamped like the mic
                nl.feed(audio[i:i + 1600], T0 + i / SAMPLE_RATE)
            done.wait(30)
            time.sleep(0.2)
            nl.stop()
            got = hits[0] if hits else None
            print(f"  {phrase!r} names={names} heard={seen} -> "
                  + (f"{got.name} '{got.heard}' {got.start - T0:.2f}-{got.end - T0:.2f}s "
                     f"score={got.score}" if got else "no hit"))
            assert (got.name if got else None) == want, (phrase, got, seen)
            if got:  # name sits inside the spoken part (1.0 s .. 1.0 s + clip)
                clip = len(speech) / SAMPLE_RATE
                assert T0 + 0.9 <= got.start < got.end <= T0 + 1.1 + clip, got
                assert got.end - got.start >= 0.15, got
    print("speech selftest (name listener): OK")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="HDB hold-to-talk target selection")
    p.add_argument("--file", help="transcribe this audio file")
    p.add_argument("--hold", type=float, help="record N seconds from the mic")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--listen", metavar="NAMES",
                   help="always-on name spotting, e.g. --listen Shravan,Jason (Ctrl-C quits)")
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

    if a.listen:
        nl = NameListener(a.listen, on_hit=lambda h: print(
            f"[names] HIT {h.name!r} heard={h.heard!r} {h.start_ms}-{h.end_ms} ms "
            f"(monotonic) score={h.score} latency={h.latency:.2f}s  | {h.text}"),
            transcriber=Transcriber(a.model))
        nl.on_text = lambda text, words: print(f"[names] heard: {text!r}")
        nl.transcriber.load()
        nl.start()
        print(f"[names] listening for {nl.names}; Ctrl-C to quit")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            nl.stop()
        return
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
