"""Teach a Sound (app/listen.py) on synthetic laptop-mic audio."""
import numpy as np
import pytest

from app import listen

SR = listen.SAMPLE_RATE
rng = np.random.default_rng(0)


def quiet(s):
    return (0.001 * rng.standard_normal(int(s * SR))).astype(np.float32)


def tone(hz, s, amp=0.2, extra=()):
    t = np.arange(int(s * SR)) / SR
    x = np.sin(2 * np.pi * hz * t) + sum(0.5 * np.sin(2 * np.pi * h * t) for h in extra)
    env = np.minimum(1, np.minimum(t, t[-1] - t) / 0.01)     # 10 ms fade in/out
    return (amp * x * env).astype(np.float32) + quiet(s)


def burst(s, amp=0.2):
    return (amp * rng.standard_normal(int(s * SR))).astype(np.float32)


def taught(tmp_path, sound, **kw):
    sl = listen.SoundListener(template_file=tmp_path / "t.json", **kw)
    msgs = []
    sl.on_taught = lambda tpl, m: msgs.append(m)
    sl.feed(quiet(1.0), t_first=100.0)
    sl.teach(1.5)
    sl.feed(np.concatenate([quiet(0.3), sound, quiet(1.5)]))
    assert sl.template is not None and "taught" in msgs[0]
    return sl


def test_teach_then_detect_same_sound_at_other_levels(tmp_path):
    doorbell = lambda amp: tone(880, 0.4, amp, extra=(1320,))
    sl = taught(tmp_path, doorbell(0.2))
    hits = []
    sl.on_detect = hits.append
    sl.feed(np.concatenate([quiet(0.5), doorbell(0.05), quiet(2.0), doorbell(0.5), quiet(0.5)]))
    assert len(hits) == 2 and all(h.score >= 0.8 for h in hits)
    assert hits[0].end - hits[0].start == pytest.approx(0.4, abs=0.1)
    # saved: a fresh listener picks the template up
    assert listen.SoundListener(template_file=tmp_path / "t.json").template is not None


def test_other_sounds_do_not_fire(tmp_path):
    sl = taught(tmp_path, tone(880, 0.4, extra=(1320,)))
    hits, events = [], []
    sl.on_detect = hits.append
    sl.on_event = lambda s, d: events.append(s)
    sl.feed(np.concatenate([quiet(0.5), burst(0.3), quiet(1.0), tone(300, 0.4), quiet(1.0),
                            tone(3000, 0.4), quiet(1.0), tone(880, 1.4, extra=(1320,))[:int(0.08 * SR)],
                            quiet(1.0)]))
    assert hits == [] and len(events) >= 3


def test_cooldown_and_pause(tmp_path):
    clap = burst(0.1, 0.4)
    sl = taught(tmp_path, clap)
    hits = []
    sl.on_detect = hits.append
    sl.feed(np.concatenate([quiet(0.3), burst(0.1, 0.4), quiet(0.3), burst(0.1, 0.4), quiet(2.0)]))
    assert len(hits) == 1                       # second clap inside the 1.5 s cooldown
    sl.pause()
    sl.feed(np.concatenate([burst(0.1, 0.4), quiet(0.5)]))
    sl.resume()
    sl.feed(np.concatenate([quiet(0.3), burst(0.1, 0.4), quiet(0.5)]))
    assert len(hits) == 2


def test_teach_with_silence_reports_error(tmp_path):
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    msgs = []
    sl.on_taught = lambda tpl, m: msgs.append((tpl, m))
    sl.feed(quiet(1.0))
    sl.teach(1.0)
    sl.feed(quiet(1.5))
    assert msgs and msgs[0][0] is None and "no sound" in msgs[0][1] and not sl.teaching


def test_hit_times_on_monotonic_clock(tmp_path):
    taught(tmp_path, tone(1500, 0.3))
    hits = []
    sl = listen.SoundListener(hits.append, template_file=tmp_path / "t.json")
    sl.feed(quiet(1.0), t_first=500.0)
    sl.feed(np.concatenate([quiet(0.5), tone(1500, 0.3), quiet(0.5)]), t_first=501.0)
    assert hits[0].start == pytest.approx(501.5, abs=0.05)


def test_app_teach_detect_buzzes_pattern_and_logs(tmp_path):
    import time

    from app import haptics
    from app.link import Link, SimPort
    from app.server import App

    port = SimPort()
    link = Link(port).start()
    app = App(link)
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    app.attach_sound(sl)
    assert app.state["sound"] == "not taught yet"
    sl.feed(quiet(1.0))
    app.handle({"cmd": "teach", "seconds": 1.5})
    sl.feed(np.concatenate([quiet(0.3), tone(1200, 0.3), quiet(1.5)]))
    assert app.state["sound"] == "taught, listening"
    app._action(True)                                   # Action held: listener paused
    sl.feed(np.concatenate([tone(1200, 0.3), quiet(0.5)]))
    app._action(False)
    assert app.alerts.recent() == []
    sl.feed(np.concatenate([quiet(0.3), tone(1200, 0.3), quiet(0.5)]))
    time.sleep(0.6)                                     # pattern plays on a thread
    link.close()
    [a] = app.alerts.recent()
    assert a["source"] == "sound" and a["cue"] == "sound" and a["motor"] is None
    assert a["confidence"] >= 0.8 and a["duration_ms"] == pytest.approx(300, abs=80)
    want = [f"PULSE {m} {ms}" for _, m, ms in haptics.taught_sound()]
    assert port.sent[:len(want)] == want


# ---------------------------------------------------------------- pitch + pattern
def room(s, hum=0.004):
    """Quiet room with mains hum and its harmonic (stationary, so it is subtracted)."""
    t = np.arange(int(s * SR)) / SR
    return (quiet(s) * 3 + hum * np.sin(2 * np.pi * 120 * t)
            + 0.5 * hum * np.sin(2 * np.pi * 240 * t)).astype(np.float32)


def bell(hz, s, amp, decay=0.5):
    t = np.arange(int(s * SR)) / SR
    x = sum(a * np.sin(2 * np.pi * hz * m * t) for m, a in ((1, 1), (2.76, 0.4), (5.4, 0.2)))
    return (amp * x * np.exp(-t / decay) * np.minimum(1, t / 0.004)).astype(np.float32)


def ding_dong(amp=0.15, hi=659, lo=523):
    out = bell(hi, 1.4, amp)
    b = bell(lo, 1.0, amp, 0.6)
    k = int(0.45 * SR)
    out[k:k + len(b)] += b[:len(out) - k]
    return out


def beeps(hz=2000, n=3, on=0.15, off=0.15, amp=0.1):
    b = tone(hz, on, amp)
    return np.concatenate([np.concatenate([b, quiet(off)]) for _ in range(n)])


def clap(amp=0.3):
    n = int(0.06 * SR)
    return (amp * rng.standard_normal(n) * np.exp(-np.arange(n) / SR / 0.012)).astype(np.float32)


def in_room(x, before=1.0, after=1.5):
    return np.concatenate([room(before), room(len(x) / SR) + x, room(after)])


def teach_in_room(sl, sound, label=None, add=False):
    msgs = []
    sl.on_taught = lambda tpl, m: msgs.append(m)
    sl.feed(room(1.0))
    sl.teach(len(sound) / SR + 1.0, label, add)
    sl.feed(in_room(sound, 0.3, 2.0))
    assert msgs and "taught" in msgs[-1], msgs
    return msgs[-1]


def test_doorbell_is_two_tone_falling_despite_hum():
    [(t0, sig)] = listen.analyze(in_room(ding_dong()))
    assert sig.category == "two-tone" and "falling" in sig.description
    hz = listen._distinct_pitches(sig.notes)
    assert hz[0] == pytest.approx(659, rel=0.02) and hz[1] == pytest.approx(523, rel=0.02)
    assert sig.duration_s > 1.0                    # the decaying ring stays one event


def test_categories_by_frequency_and_pattern():
    cases = {"repeating": beeps(), "percussive": clap(), "tone": tone(1000, 0.6, 0.1)}
    for want, x in cases.items():
        evs = listen.analyze(in_room(x))
        assert [s.category for _, s in evs] == [want], (want, [s.description for _, s in evs])
    [(_, s)] = listen.analyze(in_room(beeps()))
    assert "x3" in s.description and s.pitch_hz == pytest.approx(2000, rel=0.02)


def test_clap_and_reversed_chime_do_not_fire_a_taught_doorbell(tmp_path):
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    teach_in_room(sl, ding_dong(), "doorbell")
    hits, scores = [], []
    sl.on_detect = hits.append
    sl.on_event = lambda s, d: scores.append(round(s, 2))
    for x in (clap(), clap(0.6), ding_dong(0.15, 523, 659), beeps(659)):
        sl.feed(in_room(x, 0.5, 2.0))
    assert hits == [], scores
    sl.feed(in_room(ding_dong(0.05), 0.5, 2.0))   # quieter ring still matches
    assert [h.label for h in hits] == ["doorbell"] and hits[0].category == "two-tone"


def test_several_labelled_sounds_saved_grouped_and_forgotten(tmp_path):
    f = tmp_path / "t.json"
    sl = listen.SoundListener(template_file=f)
    teach_in_room(sl, ding_dong(), "doorbell")
    teach_in_room(sl, beeps(), "microwave")
    teach_in_room(sl, ding_dong(0.08), "doorbell", add=True)
    assert len(sl.sounds["doorbell"]) == 2 and sl.template.label == "doorbell"
    hits = []
    sl.on_detect = hits.append
    sl.cooldown_s = 0.0            # fed faster than real time: wall-clock stamps bunch up
    sl.feed(in_room(beeps(), 0.5, 2.0))
    sl.feed(in_room(ding_dong(), 0.5, 2.0))
    assert [h.label for h in hits] == ["microwave", "doorbell"]
    cats = sl.categories()
    assert set(cats) == {"two-tone", "repeating"} and cats["repeating"][0][0] == "microwave"
    assert "doorbell" in sl.summary() and "[2 examples]" in sl.summary()
    again = listen.SoundListener(template_file=f)            # v2 file round trip
    assert set(again.sounds) == {"doorbell", "microwave"} and again.sounds["doorbell"][0].sig
    again.forget("microwave")
    assert set(listen.SoundListener(template_file=f).sounds) == {"doorbell"}
    again.forget()
    assert not f.exists() and listen.SoundListener(template_file=f).template is None


def test_v1_template_file_still_loads(tmp_path):
    import json
    v1 = {"fp": [0.0] * listen.NBANDS, "duration_s": 0.064, "level": 0.03, "taught_at": 1.0,
          "bands": listen.NBANDS, "frame": listen.FRAME, "rate": SR}
    (tmp_path / "t.json").write_text(json.dumps(v1))
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    assert sl.template is not None and sl.template.sig is None and sl.template.label == "sound"


def test_taught_sound_buzzes_its_direction_from_band_mics(tmp_path):
    """A recognised sound + band MIC frames from the back-left trace: the sound's
    sweep, then a directional pulse on motor 5; logged with the motor."""
    import time

    from app import haptics, notify, trace
    from app.link import Link, SimPort
    from app.server import App
    from tests.test_names import FIX, feed

    port = SimPort(frames=[])
    link = Link(port).start()
    app = App(link)
    t = trace.load(FIX / "back_left.json")
    base = time.monotonic() - t["frames"][-1]["ms"] / 1000 - 0.1
    feed(app.live, t, base)
    app.on_sound(listen.SoundHit(0.93, base + 1.0, base + 1.45, 0.1, "doorbell", "two-tone", 659.0))
    time.sleep(1.0)
    link.close()
    want = [f"PULSE {m} {ms}" for _, m, ms in haptics.taught_sound()] + \
        [f"PULSE 5 {haptics.DIRECTIONAL_MS}"]
    assert port.sent[:len(want)] == want
    [a] = app.alerts.recent()
    assert a["source"] == "sound" and a["motor"] == 5 and a["label"] == "doorbell"
    assert "back-left" in app.state["sound"] and "doorbell" in app.state["sound"]
    assert notify.pattern("sound")[-1] != (notify.SOUND_DIR_OFFSET_MS, 5, haptics.DIRECTIONAL_MS)
