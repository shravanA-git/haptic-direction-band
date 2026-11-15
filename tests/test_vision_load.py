"""Vision must not starve the name pipeline: no inference without a target, capped rate with one,
no startup preload, thread priority, a decode duty cap, and no stale name windows. Silent: no speakers,
no camera (fake capture), fake whisper; demo name Max."""
import statistics
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app import direction, priority
from app import vision as V
from app.speech import SAMPLE_RATE, NameListener
from tests.test_name_listener import FakeModel, FakeTranscriber, audio_with

SR = SAMPLE_RATE


def in_thread(fn):
    """Run fn on a throwaway thread: QoS applies to the calling thread, and the test thread must not be
    pushed to background priority (its own sleeps would stretch)."""
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", fn()))
    t.start()
    t.join()
    return out["r"]


def test_thread_qos_never_raises_and_only_works_on_macos(monkeypatch):
    assert priority.set_thread_qos("nonsense") is False
    levels = in_thread(lambda: [priority.set_thread_qos(level) for level in priority.QOS])
    assert all(isinstance(x, bool) for x in levels)
    monkeypatch.setattr(sys, "platform", "linux")
    assert in_thread(lambda: priority.set_thread_qos("utility")) is False
    monkeypatch.setattr(priority, "_lib", None)
    monkeypatch.setattr(sys, "platform", "darwin")
    import ctypes
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("no libSystem")))
    assert in_thread(lambda: priority.set_thread_qos("utility")) is False   # a failing OS call is swallowed


class FakeCap:
    """A 30 fps camera: grab() waits for the next frame, read() = grab + retrieve."""

    def __init__(self):
        self.reads = 0
        self.grabs = 0

    def isOpened(self):
        return True

    def grab(self):
        time.sleep(1 / 30)
        self.grabs += 1
        return True

    def read(self):
        self.grab()
        self.reads += 1
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        pass


class CountingDetector:
    device, imgsz, open_vocab = "cpu", 640, None

    def __init__(self):
        self.calls = 0

    def detect(self, frame, target):
        self.calls += 1
        return []


def test_no_target_means_no_inference_and_a_slow_camera_then_a_capped_rate_with_one(monkeypatch):
    monkeypatch.delenv("HDB_OPEN_VOCAB_PRELOAD", raising=False)
    cap, det = FakeCap(), CountingDetector()
    v = V.Vision(camera=0)
    assert v.max_fps <= 10.0                                         # default cap, was 15
    v.detector = det
    v._open = lambda: cap
    assert v.start()
    try:
        time.sleep(2.0)
        idle_reads, idle_calls = cap.reads, det.calls
        assert idle_calls == 0                                       # nothing is detected without a target
        assert idle_reads <= 3.5 * 2.0 + 2                           # ~3 fps, not the camera's 30
        v.set_target("banana")
        r0, c0 = cap.reads, det.calls
        time.sleep(2.0)
        fps = (cap.reads - r0) / 2.0
        assert 5.0 <= fps <= 11.5 and (det.calls - c0) >= (cap.reads - r0) - 1     # capped at ~10 fps
        assert cap.grabs > cap.reads * 1.5                           # waiting drains stale frames, not sleeps
    finally:
        v.stop()


def test_open_vocab_preload_is_off_by_default_and_opt_in(monkeypatch):
    warmed = []
    monkeypatch.setattr(V, "open_vocab_ready", lambda: None)
    for env, expect in ((None, False), ("1", True)):
        warmed.clear()
        if env is None:
            monkeypatch.delenv("HDB_OPEN_VOCAB_PRELOAD", raising=False)
        else:
            monkeypatch.setenv("HDB_OPEN_VOCAB_PRELOAD", env)
        v = V.Vision(camera=0)
        v.detector = CountingDetector()
        v._open = lambda: FakeCap()
        v._warm_open_vocab = lambda: warmed.append(1)
        assert v.start()
        time.sleep(0.2)
        v.stop()
        assert bool(warmed) is expect


def max_model(delay=0.0):
    return FakeModel(lambda rms, a, b: ("Max", 0.5) if rms > 0.05 else None, delay=delay)


def test_decode_hop_stretches_with_decode_time_and_is_capped():
    nl = NameListener(["Max"], transcriber=FakeTranscriber(max_model()))
    assert nl.hop_s() == nl.HOP_S                                    # nothing measured yet
    nl._decode_s.extend([0.1] * 8)
    assert nl.hop_s() == nl.HOP_S                                    # fast decodes: the normal 0.5 s hop
    nl._decode_s.extend([0.46] * 8)
    assert nl.HOP_S < nl.hop_s() <= nl.HOP_MAX_S                     # 0.46 s decodes: duty capped at 40%
    assert nl.hop_s() == pytest.approx(0.46 / nl.DUTY_MAX, abs=0.01)
    nl._decode_s.extend([5.0] * 8)
    assert nl.hop_s() == nl.HOP_MAX_S                                # never more than 1.2 s apart
    assert nl.timing()["hop_s"] == nl.HOP_MAX_S


def test_duty_cap_cuts_decodes_while_a_name_is_still_found_fast():
    audio = audio_with([(1.0, 1.4, 0.3), (5.0, 5.4, 0.3)], seconds=8.0)
    counts, lat = {}, {}
    for label, cap in (("uncapped", 1.0), ("capped", None)):
        hits = []
        nl = NameListener(["Max"], on_hit=hits.append, transcriber=FakeTranscriber(max_model(delay=0.2)))
        if cap:
            nl.DUTY_MAX = 1.0                                        # old behaviour: hop stays 0.5 s
        nl.live = True
        T0 = time.monotonic()
        for i in range(0, len(audio), 1600):
            nl.feed(audio[i:i + 1600], T0 + i / SR)
            time.sleep(max(0, T0 + (i + 1600) / SR - time.monotonic()))
        assert nl.drain(10)
        nl.stop()
        counts[label], lat[label] = nl.decodes, [h.latency for h in hits]
        assert len(hits) == 2, (label, hits)
    assert counts["capped"] < counts["uncapped"] * 0.9               # fewer decodes = less CPU
    assert max(lat["capped"]) < 3.0                                  # and a name is still reported < 3 s


def test_name_latency_stays_bounded_when_decoding_is_slow_and_timing_is_exposed():
    audio = audio_with([(1.0, 1.4, 0.3), (4.5, 4.9, 0.3)], seconds=7.0)
    hits = []
    nl = NameListener(["Max"], on_hit=hits.append, transcriber=FakeTranscriber(max_model(delay=0.6)))
    nl.live = True                                                  # keep only the newest windows
    T0 = time.monotonic()
    for i in range(0, len(audio), 1600):
        nl.feed(audio[i:i + 1600], T0 + i / SR)
        time.sleep(max(0, T0 + (i + 1600) / SR - time.monotonic()))
    assert nl.drain(10)
    nl.stop()
    assert len(hits) == 2
    assert max(h.latency for h in hits) < 3.0                       # never lags behind the room
    t = nl.timing()
    assert t["hits"] == 2 and t["latency_p95"] < 3.0 and t["decode_p50"] >= 0.6 and t["decodes"] >= 2   # hop stretches to 1.2 s: fewer decodes
    assert t["wait_p95"] is not None and "dropped" in t and "noise_floor" in t


def test_a_late_name_window_is_still_inside_the_band_frame_buffer():
    live = direction.LiveEstimator(pad_ms=0)
    assert live.keep_ms >= 30000                                    # was 8 s: a 10 s-late hit read 'stale'
    t0 = 100.0
    for i in range(1000):                                           # 25 s of 25 ms band frames
        ms = i * 25
        lv = [20, 22, 18, 21]
        if 2000 <= ms <= 2450:
            lv = [20, 22, 18, 21 + 60]                              # a name from the left at 2.0-2.45 s
        live(NS(ms=ms, levels=lv, rx=t0 + ms / 1000))
    e, win = live.estimate(t0 + 2.0, t0 + 2.45, now=t0 + 25.1)      # reported ~23 s after it was said
    assert win is not None and e.reason != "no frames in window (stale?)"
    assert e.motor == 6
