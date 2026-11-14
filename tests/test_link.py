import json
import time
from pathlib import Path

import pytest

from app import direction, haptics, trace
from app.link import (ButtonEvent, Link, MicFrame, RawLine, SimPort, format_pulse,
                      format_stop, parse_line)

FIX = Path(__file__).parent / "fixtures"


def wait_for(cond, timeout=2.0):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------ parsing
def test_parse_mic():
    ev = parse_line("MIC 1234 20 18 90 101\r\n", rx=1.0)
    assert ev == MicFrame(1234, (20.0, 18.0, 90.0, 101.0), 1.0)


def test_parse_btn():
    assert parse_line("BTN ACTION DOWN", rx=0) == ButtonEvent("ACTION", True, 0)
    assert parse_line("BTN ACTION UP", rx=0) == ButtonEvent("ACTION", False, 0)


@pytest.mark.parametrize("bad", ["MIC 1 2 3", "MIC x 1 2 3 4", "MIC 5 1 2 -3 4",
                                 "ets Jun  8 2016 rst:0x1", "BTN ACTION SIDEWAYS", ""])
def test_parse_garbage_is_raw(bad):
    assert isinstance(parse_line(bad), RawLine)


def test_format():
    assert format_pulse(6, 150) == "PULSE 6 150"
    assert format_stop() == "STOP"
    for m, ms in [(8, 100), (-1, 100), (0, 0), (0, 100000)]:
        with pytest.raises(ValueError):
            format_pulse(m, ms)


# ------------------------------------------------------- link + sim
def test_link_pulse_and_stop_reach_port():
    port = SimPort()
    link = Link(port).start()
    sent = []
    link.on_sent(sent.append)
    link.pulse(0, 150)
    assert 0 in port.motors_on()
    link.stop()
    assert port.motors_on() == set()
    link.close()  # close sends a final STOP
    assert port.sent == ["PULSE 0 150", "STOP", "STOP"]
    assert sent == ["PULSE 0 150", "STOP", "STOP"]


def test_pulse_is_finite_in_sim():
    port = SimPort()
    link = Link(port).start()
    link.pulse(3, 50)
    assert 3 in port.motors_on()
    assert wait_for(lambda: port.motors_on() == set(), 1.0)
    link.close()


def test_sim_streams_quiet_mic_frames_and_buttons():
    port = SimPort()
    link = Link(port).start()
    got = []
    link.subscribe(got.append)
    port.inject("BTN ACTION DOWN")
    assert wait_for(lambda: sum(isinstance(e, MicFrame) for e in got) >= 3)
    assert any(isinstance(e, ButtonEvent) and e.down for e in got)
    assert link.connected()
    link.close()


def test_sim_replays_trace_frames_in_order():
    t = trace.load(FIX / "back_left.json")
    port = SimPort(t["frames"], realtime=False)
    link = Link(port)
    rec = trace.Recorder()
    rec.start()
    link.subscribe(rec)
    link.start()
    assert wait_for(lambda: len(rec.frames) == len(t["frames"]))
    link.close()
    assert [f["levels"] for f in rec.frames] == [f["levels"] for f in t["frames"]]


# -------------------------------------------------- record / replay
def test_record_save_load_roundtrip(tmp_path):
    rec = trace.Recorder()
    rec.start()
    for i in range(5):
        rec(MicFrame(i * 50, (1, 2, 3, 4)))
    rec.mark_alert(50, 150)
    p = trace.save(rec.stop(), tmp_path / "t.json")
    back = trace.load(p)
    assert len(back["frames"]) == 5 and back["alerts"] == [{"startMs": 50, "endMs": 150}]
    assert json.loads(p.read_text())["frames"][0] == {"ms": 0, "levels": [1, 2, 3, 4]}


def test_replayer_feeds_all_frames():
    t = trace.load(FIX / "unclear.json")
    got, done = [], []
    trace.Replayer(t, got.append, speed=0, on_done=lambda: done.append(1)).run()
    assert len(got) == len(t["frames"]) and done == [1]


# ------------------------------------------------- fixtures -> cues
@pytest.mark.parametrize("name", ["back_left", "unclear"])
def test_fixture_expected_cue(name):
    t = trace.load(FIX / f"{name}.json")
    a = t["alerts"][0]
    e = direction.estimate(t["frames"], a["startMs"], a["endMs"])
    assert e.motor == t["expected"]["motor"]
    cue = haptics.name_alert(e.motor)
    if e.motor is None:
        assert cue == haptics.generic_attention() == [(0, 0, 300)]
    else:
        assert cue == [(0, 5, 200)]


def test_stale_window_gives_generic():
    t = trace.load(FIX / "back_left.json")
    e = direction.estimate(t["frames"], 99_000, 99_500)
    assert e.motor is None


def test_cues_drive_link_through_sim():
    port = SimPort()
    link = Link(port).start()
    haptics.play(haptics.centered(), link.pulse).join()
    link.close()
    assert port.sent[:2] == ["PULSE 0 100", "PULSE 0 100"]


def test_clock_sync_uses_least_delayed_frame():
    cs = direction.ClockSync()
    cs.add(1000, 10.010)   # 10 ms delay
    cs.add(1050, 10.052)   # 2 ms delay
    assert cs.to_esp_ms(10.100) == pytest.approx(1098, abs=0.01)
