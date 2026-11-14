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
