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


# ---------------------------------------------------------- one doorbell, many rings
def chime(amp=0.15, hi=728.0, lo=561.0, cents=0.0, gap=0.32):
    """Sophanit's doorbell: short 728 Hz ding, then a long-ringing 561 Hz dong."""
    f = 2 ** (cents / 1200)
    out = bell(hi * f, 1.6, amp)
    b = bell(lo * f, 1.3, amp, 0.6)
    k = int(gap * SR)
    out[k:k + len(b)] += b[:len(out) - k]
    return out


class Clock:
    """Feeds with a steady audio clock (tests run faster than real time)."""
    def __init__(self, sl, t0=1000.0):
        self.sl, self.t = sl, t0

    def feed(self, x):
        self.sl.feed(x, t_first=self.t)
        self.t += len(x) / SR


RINGS = [(0.15, 0), (0.05, 15), (0.3, -15), (0.08, -20), (0.2, 20), (0.03, 5)]   # (level, cents)


@pytest.mark.parametrize("with_yamnet", [False, True])
def test_doorbell_rung_six_times_gives_six_alerts_and_no_duplicates(tmp_path, with_yamnet):
    """One example taught at one level; rings at 0.2x-2x the level and +-20 cents.
    Before the fix the short ding was dropped as a 'fragment' on some rings
    (score 0.67) and YAMNet caught them instead. YAMNet here hears every event
    as a doorbell, so any double alert for one ring would show."""
    y = ScriptedYamNet([("Doorbell", 0.95)]) if with_yamnet else False
    sl = listen.SoundListener(template_file=tmp_path / "t.json", yamnet=y)
    hits, scores = [], []
    sl.on_classified = lambda sig, label, s: scores.append(round(s, 2))
    c = Clock(sl)
    c.feed(room(1.0))
    sl.teach(3.0, "doorbell")
    c.feed(in_room(chime(), 0.3, 2.0))
    assert sl.sounds["doorbell"][0].category == "two-tone"
    sl.on_detect = hits.append
    scores.clear()
    for amp, cents_ in RINGS:
        c.feed(in_room(chime(amp, cents=cents_), 1.0, 2.0))
    if with_yamnet:
        assert wait_for(lambda: sl._yq.empty(), 3.0)
        wait_for(lambda: len(hits) > 6, 0.5)
    assert [h.label for h in hits] == ["doorbell"] * 6, (scores, [h.label for h in hits])
    assert min(scores) >= sl.threshold, scores
    starts = [h.start for h in hits]
    assert all(b - a > 3.0 for a, b in zip(starts, starts[1:]))       # one alert per ring


def test_yamnet_backs_up_an_untaught_ring_once(tmp_path):
    """A chime the taught template doesn't match: YAMNet alerts once for it,
    after the ring is over, even if the ring is split into two events."""
    y = ScriptedYamNet([("Doorbell", 0.9)])
    sl = listen.SoundListener(template_file=tmp_path / "t.json", yamnet=y)
    hits = []
    c = Clock(sl)
    c.feed(room(1.0))
    sl.teach(3.0, "doorbell")
    c.feed(in_room(chime(), 0.3, 2.0))
    sl.on_detect = hits.append
    other = chime(0.15, hi=440.0, lo=880.0)                       # rising, other notes
    c.feed(np.concatenate([room(1.0), room(0.6) + other[:int(0.6 * SR)], room(0.3),
                           room(1.0) + other[int(0.6 * SR):int(1.6 * SR)], room(2.0)]))
    assert wait_for(lambda: hits, 3.0)
    wait_for(lambda: len(hits) > 1, 0.5)
    assert [(h.label, h.category) for h in hits] == [("Doorbell", "two-tone")]


# ---------------------------------------------------------- dips, weak dings, talk
def babble(s, level=0.006, seed=0):
    """Speech-like room talk: voiced syllables (gliding 100-250 Hz harmonics),
    some noisy consonants, short pauses."""
    r = np.random.default_rng(seed)
    n = int(s * SR)
    out, t = np.zeros(n, np.float32), 0
    while t < n:
        m = int(r.uniform(0.08, 0.3) * SR)
        if r.random() >= 0.2:
            tt = np.arange(m) / SR
            f0 = r.uniform(100, 250) * (1 + r.uniform(-0.15, 0.15) * tt / tt[-1])
            ph = 2 * np.pi * np.cumsum(f0) / SR
            x = sum(0.6 ** h * np.sin((h + 1) * ph) for h in range(8))
            if r.random() < 0.3:
                x = 0.3 * x + r.standard_normal(m)
            x = x * np.sin(np.pi * np.arange(m) / m) ** 0.5
            x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * level * r.uniform(0.5, 1.5)
            e = min(n, t + m)
            out[t:e] += x[:e - t].astype(np.float32)
        t += m
    return out


def dipped(x, at_s, depth=0.03, dur_s=0.12):
    """The speaker/room dips the level briefly (what split the dong live)."""
    env = np.ones(len(x), np.float32)
    i, n = int(at_s * SR), int(dur_s * SR)
    env[i:i + n] = depth
    return x * env


def test_a_dip_inside_one_ringing_note_is_not_a_second_strike():
    dt = listen.FRAME / SR
    rms = [0.2 * 0.93 ** i for i in range(30)]
    pitches = [561.0] * 30
    on = [True] * 30
    for i in (12, 13, 14):                       # ~100 ms dip, then back (lower: decaying)
        rms[i], pitches[i], on[i] = 0.004, None, False
    sig = listen.describe_frames(rms, pitches, on)
    assert len(sig.notes) == 1 and sig.onsets == [0.0], (sig.notes, sig.onsets)
    assert sig.category == "tone" and sig.notes[0][2] == pytest.approx(30 * dt)
    # real beeps (back to full level after each gap) are still separate strikes
    rms = ([0.1] * 5 + [0.002] * 5) * 3
    pitches = ([2000.0] * 5 + [None] * 5) * 3
    on = ([True] * 5 + [False] * 5) * 3
    sig = listen.describe_frames(rms, pitches, on)
    assert sig.category == "repeating" and len(sig.onsets) == 3


def test_doorbell_with_a_dip_in_the_dong_still_matches_as_two_tone(tmp_path):
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    seen = []
    sl.on_classified = lambda sig, label, s: seen.append((round(s, 2), sig.category))
    c = Clock(sl)
    c.feed(room(1.0))
    sl.teach(3.0, "doorbell")
    c.feed(in_room(chime(), 0.3, 2.0))
    seen.clear()
    for at in (0.6, 0.8, 1.0):
        c.feed(in_room(dipped(chime(), at), 1.0, 2.0))
    assert [cat for _, cat in seen] == ["two-tone"] * 3 and min(s for s, _ in seen) >= 0.95, seen


def test_quiet_rings_in_talk_match_and_talk_and_lookalikes_do_not(tmp_path):
    """Taught loud in a quiet room; rung at 25% volume while people talk nearby,
    with a weaker ding. Before: the weak ding was lost and the event read as one
    561 Hz tone (0.55-0.6). Single 561 Hz bells/beeps/tones and the reversed
    chime in the same talk must stay below the threshold."""
    sl = listen.SoundListener(template_file=tmp_path / "t.json")
    seen = []
    sl.on_classified = lambda sig, label, s: seen.append((round(s, 2), sig.description))
    c = Clock(sl)
    c.feed(room(1.0))
    sl.teach(3.0, "doorbell")
    c.feed(in_room(chime(), 0.3, 2.0))
    talk_level = 0.006
    c.feed(room(2.0) + babble(2.0, talk_level, 99))

    def ring_in_talk(x, seed):
        seg = np.concatenate([np.zeros(2 * SR, np.float32), x, np.zeros(int(2.5 * SR), np.float32)])
        seen.clear()
        c.feed(room(len(seg) / SR) + babble(len(seg) / SR, talk_level, seed) + seg)
        return max([s for s, _ in seen], default=0.0)

    rings = [ring_in_talk(chime(0.0375, cents=cn, gap=0.32) * 1.0, i) if i % 2 else
             ring_in_talk(chime(0.0375, cents=cn) - 0.5 * bell(728 * 2 ** (cn / 1200), 1.6, 0.0375), i)
             for i, cn in enumerate((0, 8, -8, 5, -5, 10))]
    assert min(rings) >= sl.threshold, rings
    lookalikes = [bell(561, 1.5, 0.04), bell(561, 1.5, 0.1), beeps(561, amp=0.05),
                  tone(561, 0.8, 0.04), chime(0.04, hi=561.0, lo=728.0), chime(0.04, hi=880.0, lo=660.0),
                  np.zeros(3 * SR, np.float32)]
    worst = [ring_in_talk(x, 50 + i) for i, x in enumerate(lookalikes)]
    assert max(worst) < sl.threshold, worst
