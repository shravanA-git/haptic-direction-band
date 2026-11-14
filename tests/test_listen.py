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
    assert app.state["sound"] == "taught 'sound' [tone], listening"
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
    assert a["category"] == "tone"
    want = [f"PULSE {m} {ms}" for _, m, ms in sorted(haptics.taught_sound("tone"))]
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
    want = [f"PULSE {m} {ms}" for _, m, ms in sorted(haptics.taught_sound("two-tone"))] + \
        [f"PULSE 5 {haptics.DIRECTIONAL_MS}"]
    assert port.sent[:len(want)] == want
    [a] = app.alerts.recent()
    assert a["source"] == "sound" and a["motor"] == 5 and a["label"] == "doorbell"
    assert "back-left" in app.state["sound"] and "doorbell" in app.state["sound"]
    assert notify.pattern("sound")[-1] != (notify.SOUND_DIR_OFFSET_MS, 5, haptics.DIRECTIONAL_MS)


# ---------------------------------------------------------- several named sounds
def _app(tmp_path):
    from app.link import Link, SimPort
    from app.server import App

    port = SimPort()
    app = App(Link(port).start())
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    app.attach_sound(sl)
    sent = []
    app.emit = lambda m: sent.append(m) if m["type"] != "mic" else None
    return app, sl, sent


def test_app_teach_with_label_and_add_example(tmp_path):
    app, sl, sent = _app(tmp_path)
    sl.feed(room(1.0))
    app.handle({"cmd": "teach", "seconds": 2.4, "label": " doorbell "})
    assert "'doorbell'" in app.state["sound"]
    sl.feed(in_room(ding_dong(), 0.3, 2.0))
    app.handle({"cmd": "teach", "seconds": 2.0, "label": "microwave"})
    sl.feed(in_room(beeps(), 0.3, 2.0))
    app.handle({"cmd": "teach", "seconds": 2.4, "label": "doorbell", "add": True})
    sl.feed(in_room(ding_dong(0.08), 0.3, 2.0))
    app.link.close()
    assert len(sl.sounds["doorbell"]) == 2 and len(sl.sounds["microwave"]) == 1
    assert "doorbell" in app.state["sound"] and "two-tone" in app.state["sound"]
    items = {i["label"]: i for i in app.handle({"cmd": "sounds"})["items"]}
    assert items["doorbell"]["examples"] == 2 and items["doorbell"]["category"] == "two-tone"
    assert items["microwave"]["category"] == "repeating"
    assert [m for m in sent if m["type"] == "sounds"][-1]["items"] == list(items.values())
    logs = [m["text"] for m in sent if m["type"] == "log"]
    assert any("'doorbell'" in t for t in logs) and any("'microwave'" in t for t in logs)


def test_app_forget_by_label_keeps_the_others(tmp_path):
    app, sl, sent = _app(tmp_path)
    teach_in_room(sl, ding_dong(), "doorbell")
    teach_in_room(sl, beeps(), "microwave")
    r = app.handle({"cmd": "forget_sound", "label": "microwave"})
    assert set(sl.sounds) == {"doorbell"} and "microwave" in r["text"]
    assert set(listen.SoundListener(template_file=tmp_path / "t.json").sounds) == {"doorbell"}
    assert app.state["sound"] == "taught, listening"
    assert [i["label"] for i in [m for m in sent if m["type"] == "sounds"][-1]["items"]] == ["doorbell"]
    r = app.handle({"cmd": "forget_sound", "label": "kettle"})     # unknown: nothing forgotten
    assert "no taught sound" in r["text"] and set(sl.sounds) == {"doorbell"}
    app.handle({"cmd": "forget_sound"})                             # no label: all
    app.link.close()
    assert sl.sounds == {} and app.state["sound"] == "not taught yet"
    assert [m for m in sent if m["type"] == "sounds"][-1] == {"type": "sounds", "items": []}


def test_each_category_has_its_own_non_directional_pattern():
    from app import haptics, notify

    cats = ["tone", "two-tone", "repeating", "melody", "percussive", "noise", "voice"]
    pats = {c: haptics.taught_sound(c) for c in cats}
    assert len({tuple(sorted(p)) for p in pats.values()} | {tuple(haptics.SOUND_STEPS)}) == 8
    for c, p in pats.items():
        motors = {m for _, m, _ in p}
        # never readable as a side: opposite motors together, or the full circle
        assert motors == set(range(8)) or all((m + 4) % 8 in motors for m in motors), c
        for off in {o for o, _, _ in p}:
            assert sum(o == off for o, _, _ in p) <= 2, c        # at most two motors at once
        assert all(haptics.MIN_PULSE_MS <= ms <= haptics.GENERIC_MS for _, _, ms in p), c
        assert max(o + ms for o, _, ms in p) < notify.SOUND_DIR_OFFSET_MS, c
        # the pulse toward the sound still follows the category pattern
        assert notify.pattern("sound", 5, c) == p + [(notify.SOUND_DIR_OFFSET_MS, 5,
                                                      haptics.DIRECTIONAL_MS)]
    assert haptics.taught_sound(None) == haptics.taught_sound("unknown") == haptics.SOUND_STEPS
    assert notify.pattern("sound") == haptics.SOUND_STEPS


class FakeYamNet:
    """Stands in for YAMNet: 'Knock' for untuned sounds, 'Music' otherwise."""
    def __init__(self):
        self.calls = 0

    def load(self):
        return self

    def top(self, audio, k=5):
        self.calls += 1
        loud = float(np.sqrt(np.mean(audio ** 2)))
        return [("Knock", 0.6), ("Door", 0.4)] if loud > 0.02 else [("Music", 0.5)]


def wait_for(cond, s=2.0):
    import time
    t = time.monotonic()
    while not cond() and time.monotonic() - t < s:
        time.sleep(0.02)
    return cond()


def test_yamnet_names_untaught_sounds_and_taught_ones_win(tmp_path):
    y = FakeYamNet()
    sl = listen.SoundListener(template_file=tmp_path / "t.json", yamnet=y)
    hits, heard = [], []
    sl.on_detect, sl.on_heard = hits.append, heard.append
    sl.cooldown_s = 0.0
    sl.yamnet_watch = ("Knock",)
    sl.feed(room(1.0))
    sl.feed(in_room(burst(0.15, 0.3), 0.3, 1.0))             # untaught knock-like sound
    assert wait_for(lambda: hits)
    assert hits[0].label == "Knock" and hits[0].category == "percussive" and hits[0].score == 0.6
    assert heard and heard[0].startswith("YAMNet: Knock 0.60")
    teach_in_room(sl, ding_dong(), "doorbell")
    sl._last_hit = -1e9            # fed faster than real time: wall-clock stamps run backwards
    n = y.calls
    sl.feed(in_room(ding_dong(), 0.5, 2.0))                    # taught: no YAMNet call needed
    assert [h.label for h in hits] == ["Knock", "doorbell"]
    assert wait_for(lambda: True, 0.3) and y.calls == n


def test_yamnet_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HDB_YAMNET", raising=False)
    assert listen.SoundListener(template_file=tmp_path / "t.json").yamnet is None


class ScriptedYamNet(FakeYamNet):
    def __init__(self, top):
        super().__init__()
        self._top = top

    def top(self, audio, k=5):
        self.calls += 1
        return self._top


@pytest.mark.parametrize("top,want", [
    ([("Doorbell", 0.91), ("Ding-dong", 0.6)], [("Doorbell", "two-tone")]),
    ([("Ding-dong", 0.55)], [("Ding-dong", "two-tone")]),
    ([("Door", 0.41), ("Knock", 0.3)], []),                   # the live false alerts: off now
    ([("Doorbell", 0.45)], []),                                # below the 0.5 floor
])
def test_yamnet_default_is_doorbell_only_with_its_pattern(tmp_path, monkeypatch, top, want):
    from app import haptics
    for k in ("HDB_YAMNET_CLASSES", "HDB_YAMNET_MIN"):
        monkeypatch.delenv(k, raising=False)
    sl = listen.SoundListener(template_file=tmp_path / "t.json", yamnet=ScriptedYamNet(top))
    hits = []
    sl.on_detect = hits.append
    sl.feed(room(1.0))
    sl.feed(in_room(burst(0.15, 0.3), 0.3, 1.0))
    assert wait_for(lambda: sl.yamnet.calls)
    wait_for(lambda: bool(hits), 0.3)                          # let the worker finish
    assert [(h.label, h.category) for h in hits] == want
    if want:
        assert haptics.taught_sound(hits[0].category) == haptics.SOUND_PATTERNS["two-tone"]
