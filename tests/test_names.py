"""Live name alerts: speech timestamps (laptop clock) -> ESP32 window -> cue."""
import time
from pathlib import Path
from types import SimpleNamespace as Hit

from app import direction, trace
from app.link import Link, MicFrame, SimPort
from app.server import App

FIX = Path(__file__).parent / "fixtures"
DELAY = 0.004   # serial receipt lag, seconds


def feed(live, t, base):
    """Frames as if ESP32 ms 0 arrived at laptop time `base` (+ jitter)."""
    for i, f in enumerate(t["frames"]):
        live(MicFrame(f["ms"], tuple(f["levels"]), base + f["ms"] / 1000 + DELAY + (i % 3) * 0.003))


def test_live_window_maps_laptop_clock_to_esp_ms():
    t = trace.load(FIX / "back_left.json")
    live = direction.LiveEstimator(pad_ms=0)
    feed(live, t, base=500.0)
    win = live.window(501.0, 501.45)      # name spoken 1.00-1.45 s after ESP32 ms 0
    assert abs(win[0] - 1000) <= 5 and abs(win[1] - 1450) <= 5
    end = 500.0 + t["frames"][-1]["ms"] / 1000
    e, _ = live.estimate(501.0, 501.45, now=end + 0.2)
    assert e.motor == 5


def test_unclear_and_stale_and_empty_give_generic():
    live = direction.LiveEstimator()
    e, win = live.estimate(1.0, 1.4)
    assert e.motor is None and win is None and "not streaming" in e.reason
    t = trace.load(FIX / "unclear.json")
    feed(live, t, base=10.0)
    end = 10.0 + t["frames"][-1]["ms"] / 1000
    assert live.estimate(11.0, 11.45, now=end + 0.2)[0].motor is None      # conflicting
    e, _ = live.estimate(11.0, 11.45, now=end + 5)
    assert e.motor is None and "stale" in e.reason


def test_esp32_reboot_resets_clock():
    live = direction.LiveEstimator()
    feed(live, trace.load(FIX / "back_left.json"), base=0.0)
    live(MicFrame(0, (20, 18, 22, 19), 100.0))       # ms went backwards: board reset
    assert live.window(100.0, 100.0) == (-live.pad_ms, live.pad_ms)


class FakeNames:
    def __init__(self):
        self.on_hit, self.paused, self.names = None, False, ["Shravan"]

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_names(self, n):
        self.names = [x.strip() for x in n.split(",") if x.strip()]


def test_name_hit_buzzes_estimated_motor_and_action_pauses_listener():
    port = SimPort(frames=[])                      # no sim frames; we feed the buffer
    link = Link(port).start()
    app = App(link)
    names = FakeNames()
    app.attach_names(names)
    t = trace.load(FIX / "back_left.json")
    base = time.monotonic() - t["frames"][-1]["ms"] / 1000 - 0.1
    feed(app.live, t, base)
    app.recorder.start()
    names.on_hit(Hit(name="Shravan", start=base + 1.0, end=base + 1.45, latency=1.2))
    time.sleep(0.05)
    assert port.sent == ["PULSE 5 200"]
    assert "back-left" in app.state["name"]
    assert len(app.recorder.stop()["alerts"]) == 1

    port.inject("BTN ACTION DOWN")
    time.sleep(0.1)
    assert names.paused
    app.handle({"cmd": "action", "down": False})
    assert not names.paused
    app.handle({"cmd": "name", "name": "Sophanit, Shravan"})
    assert names.names == ["Sophanit", "Shravan"]

    time.sleep(1.2)                                # stream went quiet -> stale
    names.on_hit(Hit(name="Shravan", start=base + 1.0, end=base + 1.45, latency=1.2))
    time.sleep(0.05)
    assert port.sent[-1] == "PULSE 0 300"          # generic attention
    link.close()
