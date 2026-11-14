"""NameListener rolling window, dedupe, hotword-hallucination gates, live drop
policy and warm-up, with a fake whisper model (no download)."""
import threading
import time
from types import SimpleNamespace as NS

import numpy as np

from app.speech import SAMPLE_RATE, NameListener

SR = SAMPLE_RATE


class FakeModel:
    """Stands in for faster_whisper.WhisperModel. Every loud run of 30 ms frames
    in the window becomes one word; script(audio_rms, start, end) -> (text, prob)
    decides what whisper "heard" for it."""

    def __init__(self, script=None, delay=0.0):
        self.calls, self.delay = [], delay
        self.script = script or (lambda rms, a, b: ("Shravan", 0.5) if rms > 0.05 else None)

    def transcribe(self, audio, hotwords=None, **kw):
        self.calls.append({"n": len(audio), "hotwords": hotwords, **kw})
        if self.delay:
            time.sleep(self.delay)
        f = 480
        rms = [float(np.sqrt(np.mean(audio[i:i + f] ** 2))) for i in range(0, len(audio) - f + 1, f)]
        words, start = [], None
        for k, r in enumerate(rms + [0.0]):
            if r > 0.02 and start is None:
                start = k
            elif r <= 0.02 and start is not None:
                a, b = start * f / SR, k * f / SR
                got = self.script(max(rms[start:k]), a, b) if hotwords else None
                if got:
                    for text, prob in (got if isinstance(got, list) else [got]):
                        words.append(NS(word=" " + text, start=a, end=b, probability=prob))
                start = None
        return [NS(text=" ".join(w.word for w in words), words=words)], None


class FakeTranscriber:
    def __init__(self, model):
        self.model, self._run_lock = model, threading.Lock()

    def load(self):
        return self.model


def audio_with(bursts, seconds=5.0, noise=0.002):
    rng = np.random.default_rng(0)
    x = (noise * rng.standard_normal(int(seconds * SR))).astype(np.float32)
    t = np.arange(int(seconds * SR)) / SR
    for a, b, amp in bursts:
        m = (t >= a) & (t < b)
        x[m] += (amp * np.sin(2 * np.pi * 300 * t[m])).astype(np.float32)
    return x


def listen(audio, model, T0=500.0, **kw):
    hits, texts = [], []
    nl = NameListener(["Shravan"], on_hit=hits.append, transcriber=FakeTranscriber(model), **kw)
    nl.on_text = lambda t, w: texts.append(t)
    for i in range(0, len(audio), 1600):
        nl.feed(audio[i:i + 1600], T0 + i / SR)
    assert nl.drain(10)
    nl.stop()
    return nl, hits, texts


def test_name_reported_once_with_monotonic_times_and_warmup():
    nl, hits, _ = listen(audio_with([(1.5, 2.0, 0.3)]), FakeModel())
    assert len(hits) == 1, hits                  # seen in several windows, reported once
    h = hits[0]
    assert abs(h.start - 501.5) < 0.05 and abs(h.end - 502.0) < 0.05
    calls = nl.transcriber.model.calls
    assert calls[0]["hotwords"] is None and calls[0]["n"] == SR      # warm-up: 1 s silence
    rolling = [c for c in calls[1:] if c["hotwords"]]
    assert len(rolling) >= 2 and all(c["n"] <= 2 * SR for c in rolling)
    assert all(c["temperature"] == 0.0 and c["max_new_tokens"] for c in calls)


def test_silence_is_not_decoded():
    nl, hits, _ = listen(audio_with([], seconds=3.0), FakeModel())
    assert hits == [] and nl.decodes == 0


def test_low_probability_name_is_rejected():
    model = FakeModel(lambda rms, a, b: ("Shravan", 0.01))      # chatter rewritten as the name
    _, hits, texts = listen(audio_with([(1.5, 2.0, 0.3)]), model)
    assert hits == [] and any("word probability" in t for t in texts)


def test_repetition_loop_is_rejected():
    model = FakeModel(lambda rms, a, b: [("Shravan", 0.9)] * 3)
    _, hits, texts = listen(audio_with([(1.5, 2.0, 0.3)]), model)
    assert hits == [] and any("repeated" in t for t in texts)


def test_name_over_quiet_audio_is_rejected():
    # whisper places a confident name on a region no louder than the room
    model = FakeModel(lambda rms, a, b: ("Shravan", 0.9))
    _, hits, texts = listen(audio_with([(1.5, 2.0, 0.025)], noise=0.012), model)
    assert hits == []


def test_best_candidate_wins_over_invented_one():
    seq = iter([0.0, 0.8] * 50)
    model = FakeModel(lambda rms, a, b: ("Shravan", next(seq)))
    _, hits, _ = listen(audio_with([(1.0, 1.3, 0.3), (2.0, 2.4, 0.3)]), model)
    assert len(hits) == 1


def test_two_calls_far_apart_are_both_reported():
    _, hits, _ = listen(audio_with([(1.0, 1.4, 0.3), (4.0, 4.4, 0.3)], seconds=6.0), FakeModel())
    assert [round(h.start - 500, 1) for h in hits] == [1.0, 4.0]


def test_live_mode_drops_old_windows_instead_of_lagging():
    model = FakeModel(delay=0.25)
    nl = NameListener(["Shravan"], transcriber=FakeTranscriber(model))
    nl.live = True
    audio = audio_with([(0.5, 4.5, 0.3)], seconds=5.0)
    for i in range(0, len(audio), 1600):         # 10x faster than real time
        nl.feed(audio[i:i + 1600], 100 + i / SR)
        assert nl._q.qsize() <= nl.LIVE_QUEUE
        time.sleep(0.01)
    nl.drain(10)
    nl.stop()
    assert nl.dropped > 0


def test_pause_resets_and_suppresses():
    model = FakeModel()
    nl = NameListener(["Shravan"], transcriber=FakeTranscriber(model))
    nl.pause()
    a = audio_with([(0.5, 1.0, 0.3)], seconds=2.0)
    nl.feed(a, 10.0)
    nl.drain(5)
    assert nl.decodes == 0
    nl.stop()


def test_continuous_chatter_keeps_decoding_through_a_louder_name():
    # Room chatter never pauses; the name is a bit louder than the chatter. A floor
    # from the last 3 s became the chatter itself and decoding stopped before the
    # name was complete; the 30 s 10th-percentile floor keeps decoding through it.
    rng = np.random.default_rng(1)
    n = int(14 * SR)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 2.5 * t))          # syllable-ish chatter
    x = (0.08 * env * rng.standard_normal(n)).astype(np.float32)
    m = (t >= 9.0) & (t < 10.0)
    x[m] += (0.15 * np.sin(2 * np.pi * 300 * t[m])).astype(np.float32)
    nl = NameListener(["Shravan"], transcriber=FakeTranscriber(FakeModel()))
    ends = []
    nl.process = lambda audio, t0: ends.append(t0 + len(audio) / SR - 100)
    for i in range(0, n, 480):
        nl.feed(x[i:i + 480], 100 + i / SR)
    nl.drain(5)
    nl.stop()
    assert any(e >= 10.0 for e in ends), ends          # a window holding the whole name


class SoundAlikeModel(FakeModel):
    """The hotword turns "Sharon" into "Shravan"; without hotwords whisper hears
    `plain` (a different name, or a garbled real call)."""

    def __init__(self, plain):
        super().__init__(lambda rms, a, b: ("Shravan", 0.5) if rms > 0.05 else None)
        self.plain = plain

    def transcribe(self, audio, hotwords=None, **kw):
        if hotwords:
            return super().transcribe(audio, hotwords=hotwords, **kw)
        self.calls.append({"n": len(audio), "hotwords": None, **kw})
        return [NS(text=" " + self.plain, words=[NS(word=" " + self.plain, start=0.1,
                                                   end=0.5, probability=0.9)])], None


def test_sound_alike_name_is_rejected():
    model = SoundAlikeModel("Sharon")
    _, hits, texts = listen(audio_with([(1.5, 2.0, 0.3)]), model)
    assert hits == [] and any("sound-alike" in t for t in texts)


def test_garbled_real_call_is_not_mistaken_for_a_sound_alike():
    model = SoundAlikeModel("Soften")   # not a name: the real call stands
    _, hits, _ = listen(audio_with([(1.5, 2.0, 0.3)]), model)
    assert len(hits) == 1 and hits[0].name == "Shravan"
