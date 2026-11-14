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
import collections
import os
import queue
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
               "cellphone": "cell phone", "cop": "cup",
               "coffee": "cup", "macbook": "laptop", "mac book": "laptop"})
# Things people ask for that the detector can't see. Matched before the
# singular forms, so "glasses" doesn't become "glass" -> cup.
_UNKNOWN = {"glasses", "eyeglasses", "eye glasses", "spectacles", "sunglasses",
            "sun glasses", "reading glasses", "specs"}


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
    blocked = set()   # positions inside an _UNKNOWN phrase; longer phrases are tried
    for n in (3, 2, 1):  # first, so "wine glasses" still finds "wine glass"
        for i in range(len(words) - n + 1):
            if blocked.intersection(range(i, i + n)):
                continue
            if " ".join(words[i:i + n]) in _UNKNOWN:
                blocked.update(range(i, i + n))
                continue
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


# Common first names: the sound-alike check rejects a hotword hit when an unbiased
# re-decode hears one of these instead (e.g. "Hey Sharon" -> "Shravan").
COMMON_NAMES = frozenset("""
aaron adam adrian aiden alan alex alexa alice alicia allen alyssa amanda amber amy
ana andre andrea andrew angela anna anne anthony april ariana arthur ashley austin ava
barbara ben benjamin beth betty bill blake bob brad brandon brenda brian brittany brooke
bruce bryan caleb cameron carl carla carlos carol caroline carter casey catherine chad charles
charlie chelsea cheryl chloe chris christian christina christine christopher cindy claire cody colin connor
craig crystal cynthia dan daniel danielle danny david dawn dean deborah denise dennis derek
diana diane donald donna doug dylan eddie edward elena eli elijah elizabeth ella ellen
emily emma eric erica erin ethan eva evan frank gabriel gary george grace greg
hailey hannah harry heather helen henry holly ian isaac isabella jack jackson jacob jake
james jamie jan jane janet jared jasmine jason jay jean jeff jen jenna jennifer
jenny jeremy jerry jesse jessica jill jim joan joe joel john johnny jon jonathan
jordan jose joseph josh joshua joy juan judy julia julie justin karen kate katie
kayla keith kelly ken kevin kim kyle laura lauren leah lee leo liam lily
linda lisa logan lori louis lucas lucy luis luke lynn madison maria mark martin
mary mason matt matthew megan melissa mia michael michelle mike molly morgan nancy natalie
nathan nicholas nick nicole noah olivia oscar owen pam pat patrick paul peter rachel
randy ray rebecca richard rick rob robert robin roger ron rose ruby russell ryan
sam samantha sandra sara sarah scott sean seth shane shannon sharon shawn sheila sherry
sophia sophie stacy stan stephanie stephen steve steven sue susan taylor terry thomas tim
timothy tina todd tom tony tracy travis tyler vanessa victor victoria vincent wayne wendy
will william zach zoe shaun shana shauna shayna sherman shirley sharron sharyn shan sheldon
""".split())


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
        nl.start()          # opens the default mic (loads + warms whisper first)
        nl.pause(); nl.resume()   # while Action is held (hold-to-talk owns speech)
        nl.stop()
        nl.feed(samples, t_first)  # or push audio yourself (tests, replay)

    Rolling window: the last WINDOW_S of audio is decoded every HOP_S while
    anything is audible (frame RMS above ACTIVE_RATIO x the noise floor within
    the last ACTIVE_TAIL_S), with the names as whisper hotwords, then
    fuzzy-matched. The floor is the 10th percentile over FLOOR_S (~30 s): a 3 s
    floor became the chatter itself and stopped decoding in a busy room.
    This doesn't wait for a pause in the conversation, so a name said over
    background chatter is reported about one hop + one decode after it ends.
    Hotwords can make whisper invent the name (in chatter it rewrites "the sun"
    as "Shravan"; on silence it loops "Shravan Shravan ..."), so a candidate must
    pass three checks: whisper's probability for the matched word >= MIN_PROB
    (measured over TTS chatter: real calls 0.08-0.34, invented ones <= 0.01),
    the name appears at most MAX_REPEATS times in the window, and the audio under
    the word is louder than the room (WORD_LOUD x noise floor). verify="decode"
    also re-decodes +-VERIFY_PAD_S around the word without hotwords; over
    chatter that rejected most real calls, so it is off by default. Known gap:
    a sound-alike ("Sharon") in loud chatter can still pass; requiring two
    windows to agree, or re-decoding confident candidates, did not fix it in
    tests (fewer real calls found, false alarms unchanged).
    Decodes run at temperature 0 with a token cap (whisper's temperature
    fallback took up to 9 s on near-silent windows). The same name seen in
    overlapping windows is reported once (starts within cooldown_s merge).

    Times are laptop time.monotonic(): each mic block is stamped on arrival and
    the sample clock is anchored with a min-filter, so callback scheduling
    jitter doesn't shift the word times. Live (start()), the decoder keeps only
    the newest windows so it never lags; with feed() every window is decoded.
    """

    FRAME = 480            # 30 ms at 16 kHz
    WINDOW_S = 2.0
    HOP_S = 0.5
    ACTIVE_TAIL_S = 1.0    # keep decoding this long after the last loud frame
    MIN_START_RMS = 0.003  # absolute floor for "someone is talking"
    FLOOR_S = 30.0         # noise floor = 10th percentile of frame RMS over this long
    ACTIVE_RATIO = 1.5     # "audible" = frame RMS above this x floor
    EDGE_S = 0.2           # a name ending this close to the window's end may be cut off:
                           # wait for the next window (one hop later) for full timing
    MIN_PROB = 0.05
    MAX_REPEATS = 2
    WORD_LOUD = 2.0
    MAX_TOKENS = 40        # ~2 s of speech is ~15 tokens; caps hallucination loops
    VERIFY_PAD_S = 0.5     # unbiased re-decode around a single sighting
    ALIKE_PAD_S = 0.3      # sound-alike check: unbiased re-decode of the word +- this
    VERIFY_THRESHOLD = 0.6
    LIVE_QUEUE = 2         # live: windows waiting for the decoder (older ones dropped)

    def __init__(self, names, on_hit: Optional[Callable[[NameHit], None]] = None,
                 transcriber: Optional[Transcriber] = None, device=None,
                 threshold: float = 0.8, cooldown_s: float = 1.5, verify: str = "prob"):
        if verify not in ("prob", "decode"):
            raise ValueError("verify must be 'prob' or 'decode'")
        self.verify = verify
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        self.names = list(names)
        self.on_hit = on_hit
        self.on_text: Optional[Callable[[str, list], None]] = None  # debug: every decode
        self.transcriber = transcriber or Transcriber()
        self.device = device
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.paused = False
        self.live = False          # True once start() opened the mic
        self.last: Optional[NameHit] = None
        self.decodes = 0
        self.dropped = 0
        self._stream = None
        self._q: "queue.Queue" = queue.Queue()
        self._qlock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._warm = False
        self._levels: "collections.deque" = collections.deque(
            maxlen=int(self.FLOOR_S * SAMPLE_RATE / self.FRAME))
        self.noise_floor = 0.0
        self._n = 0                # samples fed so far
        self._offset = None        # monotonic(sample 0), min-filtered
        self._last_hit_t = -1e9
        self._pending: Optional[NameHit] = None   # edge-deferred candidate
        self._vetoed: "collections.deque" = collections.deque(maxlen=20)  # (name, start)
        self.rejected: dict = {}   # reason -> count (diagnostics)
        self._reset_buf()

    # ---- public ----
    def set_names(self, names) -> None:
        self.names = [n.strip() for n in (names.split(",") if isinstance(names, str)
                                          else names) if n.strip()]

    def warmup(self) -> None:
        """Load whisper and decode 1 s of silence so the first real decode is fast."""
        if self._warm:
            return
        import numpy as np
        self._decode(np.zeros(SAMPLE_RATE, np.float32), 0.0, hotwords=None)
        self._warm = True

    def _decode(self, audio, t0: float, hotwords: Optional[str]):
        """Greedy, temperature-0, token-capped decode on the shared whisper model
        (same lock as hold-to-talk). -> (text, [Word]) on the monotonic clock."""
        model = self.transcriber.load()
        with self.transcriber._run_lock:
            segs, _ = model.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                       word_timestamps=True, vad_filter=False,
                                       condition_on_previous_text=False, hotwords=hotwords,
                                       max_new_tokens=self.MAX_TOKENS)
            text, words = [], []
            for sg in segs:
                text.append(sg.text.strip())
                for w in (sg.words or []):
                    words.append(Word(w.word.strip(), t0 + float(w.start), t0 + float(w.end),
                                      float(getattr(w, "probability", 1.0))))
        return " ".join(text).strip(), words

    def start(self) -> "NameListener":
        import sounddevice as sd
        self.live = True
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
        self._reset_buf()

    def resume(self) -> None:
        self._reset_buf()
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

    def drain(self, timeout: float = 30.0) -> bool:
        """Wait until every queued window has been fully processed (decodes and
        on_hit). True if drained within timeout."""
        end = time.monotonic() + timeout
        while self._q.unfinished_tasks:
            if time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True

    # ---- internals ----
    def _cb(self, indata, frames, time_info, status) -> None:
        lat = self._stream.latency if self._stream is not None else 0.0
        self.feed(indata[:, 0].copy(), time.monotonic() - frames / SAMPLE_RATE - lat)

    def _t(self, n: int) -> float:
        return (self._offset or 0.0) + n / SAMPLE_RATE

    def _reset_buf(self) -> None:
        self._buf: "collections.deque" = collections.deque()   # (sample index, frame)
        self._buf_len = 0
        self._last_loud = None     # sample index of the last loud frame
        self._next_hop = None      # sample index at which to decode next

    def _frame(self, f) -> None:
        import numpy as np
        n0 = self._n
        self._n += len(f)
        if self.paused or len(f) == 0:
            return
        rms = float(np.sqrt(np.mean(f * f)) + 1e-9)
        self._levels.append(rms)
        self._buf.append((n0, f))
        self._buf_len += len(f)
        while self._buf_len - len(self._buf[0][1]) >= self.WINDOW_S * SAMPLE_RATE:
            self._buf_len -= len(self._buf.popleft()[1])
        if len(self._levels) < 10:
            return
        floor = float(np.percentile(self._levels, 10))
        self.noise_floor = floor
        if rms > max(self.ACTIVE_RATIO * floor, self.MIN_START_RMS):
            if self._last_loud is None:          # talking starts: first decode one hop in
                self._next_hop = self._n + int(self.HOP_S * SAMPLE_RATE)
            self._last_loud = self._n
        if self._last_loud is None or self._n < self._next_hop:
            return
        self._next_hop = self._n + int(self.HOP_S * SAMPLE_RATE)
        if self._n - self._last_loud > self.ACTIVE_TAIL_S * SAMPLE_RATE:
            self._last_loud = None               # quiet again: stop decoding
            return
        audio = np.concatenate([x for _, x in self._buf])
        self._enqueue((audio, self._t(self._buf[0][0])))

    def _enqueue(self, item) -> None:
        with self._qlock:
            if self.live:   # never lag behind the room: keep only the newest windows
                while self._q.qsize() >= self.LIVE_QUEUE:
                    try:
                        self._q.get_nowait()
                        self._q.task_done()
                        self.dropped += 1
                    except queue.Empty:
                        break
            self._q.put(item)

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="names", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        try:
            self.warmup()
        except Exception as e:
            print(f"[names] warm-up failed: {e}")
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                audio, t0 = item
                self.process(audio, t0)
            except Exception as e:
                print(f"[names] error: {e}")
            finally:
                self._q.task_done()

    def _reject(self, audio, t0: float, words: list, name: str, i: int, j: int) -> Optional[str]:
        """Why this candidate is probably invented by the hotword, or None."""
        import numpy as np
        start, end = words[i].start, words[j].end
        prob = max(w.prob for w in words[i:j + 1])
        if prob < self.MIN_PROB:
            return f"word probability {prob:.3f} < {self.MIN_PROB}"
        reps = sum(1 for k in range(len(words)) if match_name(words[k:k + 1], [name], self.threshold))
        if reps > self.MAX_REPEATS:
            return f"name repeated {reps}x (hallucination loop)"
        a = max(0, int((start - t0) * SAMPLE_RATE))
        b = min(len(audio), max(a + 1, int((end - t0) * SAMPLE_RATE)))
        rms = float(np.sqrt(np.mean(audio[a:b] ** 2)) + 1e-9)
        if rms < self.WORD_LOUD * max(self.noise_floor, 1e-4):
            return f"word audio is quiet (rms {rms:.4f}, floor {self.noise_floor:.4f})"
        return None

    def _more_coming(self) -> bool:
        """Another window will be decoded after this one (still talking)."""
        return self._last_loud is not None or self._q.qsize() > 0

    def process(self, audio, t0: float) -> Optional[NameHit]:
        """Decode one window that starts at monotonic t0; report a new hit."""
        if not self.names or self.paused:
            return None
        self.decodes += 1
        text, words = self._decode(audio, t0, hotwords=" ".join(self.names))
        if self.on_text:
            self.on_text(text, words)
        # Every occurrence of a name is a candidate (a window can hold an invented
        # "Shravan" and the real one); try the most probable first.
        cands, seen = [], set()
        for n in (1, 2):              # single words first ("Hey Shravan" must not win
            for k in range(len(words) - n + 1):     # over "Shravan" and shift the start)
                m = match_name(words[k:k + n], self.names, self.threshold)
                if m and not seen.intersection(range(k, k + n)):
                    seen.update(range(k, k + n))
                    cands.append((m[0], k, k + n - 1, m[3]))
        cands.sort(key=lambda c: -max(w.prob for w in words[c[1]:c[2] + 1]))
        win_end = t0 + len(audio) / SAMPLE_RATE
        found = deferred = None
        for name, i, j, score in cands:
            start, end = words[i].start, words[j].end
            if start - self._last_hit_t < self.cooldown_s:
                continue               # same call already reported from an earlier window
            why = self._reject(audio, t0, words, name, i, j)
            if not why and self.verify == "decode":
                heard = self._unbiased(audio, t0, start, end)
                if match_name(heard, [name], self.VERIFY_THRESHOLD) is None:
                    why = f"unbiased decode heard {' '.join(w.text for w in heard)!r}"
            if not why and any(n == name and abs(st - start) < self.cooldown_s
                               for n, st in self._vetoed):
                why = "sound-alike: same moment as an earlier sound-alike window"
            if not why:
                why = self._soundalike(audio, t0, name, start, end)
                if why:
                    self._vetoed.append((name, start))
            if why:
                self._note(name, why, words)
                continue
            hit = NameHit(name, " ".join(w.text for w in words[i:j + 1]), start, end, score,
                          text, 0.0)
            if end > win_end - self.EDGE_S and self._more_coming():
                deferred = deferred or hit   # may be cut at the right edge: next window
                continue                     # usually has all of it (better timing)
            found = hit
            break
        pending, self._pending = self._pending, None
        if found is None and pending is not None:
            found = pending            # next window lost it: report the earlier sighting
        elif found is None and deferred is not None:
            self._pending = deferred
            return None
        if found is None:
            return None
        self._last_hit_t = found.start
        found.latency = round(time.monotonic() - found.end, 3)
        self.last = found
        if self.on_hit:
            try:
                self.on_hit(found)
            except Exception as e:
                print(f"[names] callback error: {e}")
        return found

    def _soundalike(self, audio, t0: float, name: str, start: float, end: float) -> Optional[str]:
        """The hotword turns similar names into the target ("Hey Sharon" -> "Shravan").
        Re-decode the word without hotwords: if whisper then hears a different common
        first name that sounds like the target, someone else was called. Garbled real
        calls ("Soften it", "KJ signs") aren't names, so they pass. Runs only on
        candidates that passed the other checks, so it rarely costs a decode."""
        key = re.sub(r"[^a-z]", "", name.lower())
        for w in self._unbiased(audio, t0, start, end, pad=self.ALIKE_PAD_S):
            tok = re.sub(r"[^a-z]", "", w.text.lower())
            if tok and tok != key and tok in COMMON_NAMES:
                m = match_name([tok], [name], 0.0)
                if m and m[3] >= 0.5:
                    return f"sound-alike: unbiased decode heard {w.text!r} (score {m[3]:.2f})"
        return None

    def _unbiased(self, audio, t0: float, start: float, end: float, pad: float = None) -> list:
        pad = self.VERIFY_PAD_S if pad is None else pad
        a = max(0, int((start - pad - t0) * SAMPLE_RATE))
        b = min(len(audio), int((end + pad - t0) * SAMPLE_RATE))
        return self._decode(audio[a:b], t0 + a / SAMPLE_RATE, hotwords=None)[1]

    def _note(self, name: str, why: str, words: list) -> None:
        key = why.split(" ")[0]
        self.rejected[key] = self.rejected.get(key, 0) + 1
        if self.on_text:
            self.on_text(f"(rejected {name!r}: {why})", words)


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
            hits, seen = [], []
            nl = NameListener(names, on_hit=hits.append, transcriber=tr)
            nl.on_text = lambda text, words: seen.append(text)
            for i in range(0, len(audio), 1600):  # 100 ms blocks, stamped like the mic
                nl.feed(audio[i:i + 1600], T0 + i / SAMPLE_RATE)
            t_fed = time.monotonic()
            assert nl.drain(60), "name listener did not finish"
            busy = time.monotonic() - t_fed
            nl.stop()
            got = hits[0] if hits else None
            print(f"  {phrase!r} names={names} heard={seen} -> "
                  + (f"{got.name} '{got.heard}' {got.start - T0:.2f}-{got.end - T0:.2f}s "
                     f"score={got.score}" if got else "no hit") + f"  (decode {busy:.2f}s)")
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
