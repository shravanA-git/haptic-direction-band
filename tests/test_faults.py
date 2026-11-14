"""Fault tolerance: serial drop/return, camera stall, mic stall. The app keeps
running, STOP keeps working, and status() names what failed."""
import time
from types import SimpleNamespace as D

from app import haptics
from app.guide import ObjectGuide
from app.link import Link, NullPort, SimPort
from app.server import App


def wait(cond, t=3.0):
    end = time.monotonic() + t
    while time.monotonic() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_serial_drop_and_return_reconnects_and_never_raises():
    port = SimPort()
    link = Link(port, reopen=port.reopen, retry_s=0.05).start()
    app = App(link)
    assert wait(lambda: link.frames_rx > 2)
    port.broken = True                                # cable pulled
    assert wait(lambda: link.error is not None)
    app.handle({"cmd": "pulse", "motor": 2, "ms": 150})   # must not raise
    app.handle({"cmd": "stop"})
    app.notify.cue("name", "directional", 5)
    assert "serial" in app.check_health()
    assert app.status()["faults"]["serial"].startswith("band link")
    n = link.frames_rx
    port.broken = False                               # plugged back in
    assert wait(lambda: link.error is None and link.frames_rx > n + 2)
    assert link.reconnects == 1 and "serial" not in app.check_health()
    app.handle({"cmd": "stop"})
    assert port.sent[-1] == "STOP" and link.dropped_writes >= 2
    link.close()


def test_app_starts_without_a_band_and_connects_later():
    sim = SimPort()
    sim.broken = True
    link = Link(NullPort(), reopen=sim.reopen, retry_s=0.05).start()
    assert wait(lambda: link.error is not None and "waiting for band" in link.error)
    sim.broken = False
    assert wait(lambda: link.error is None and link.frames_rx > 0)
    assert link.device == "sim"
    link.close()


class StallVision:
    """Vision stand-in whose camera can 'freeze'."""

    def __init__(self):
        self.cb, self._paused, self._running, self.error = None, False, True, None
        self.t, self.starts, self.target = time.monotonic(), 0, "banana"

    def on_change(self, cb): self.cb = cb
    def latest(self): return D(t=self.t, target=self.target, state="left")
    def set_target(self, n): return n
    def pause(self): self._paused = True
    def resume(self): self._paused = False
    def stop(self): self._running = False

    def start(self):
        self.starts += 1
        self._running, self.t = True, time.monotonic()
        return True


def test_camera_stall_stops_directional_repeats_reports_and_restarts():
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    v = StallVision()
    guide = ObjectGuide(lambda cue: haptics.play(cue, link.pulse), repeat_s=0.2)
    app.attach_vision(v, guide)
    guide.start()
    v.cb(D(state="left", target="banana", t=time.monotonic(), changed=True, confidence=0.9))
    assert wait(lambda: port.sent.count("PULSE 6 200") >= 2)      # repeating "turn left"
    now = time.monotonic()
    v.t = now - 3.0                                                # frames stopped 3 s ago
    f = app.check_health(now)
    assert f["camera"].startswith("camera: no frames")
    time.sleep(0.3)
    n_left = port.sent.count("PULSE 6 200")
    time.sleep(0.6)
    assert port.sent.count("PULSE 6 200") == n_left                # repeats stopped
    app.check_health(now + app.VISION_RETRY_S + 0.1)                # retry window reached
    assert wait(lambda: v.starts == 1)                             # camera restarted
    assert "camera" not in app.check_health()                      # fresh frames: fault cleared
    guide.close()
    link.close()


def test_camera_not_flagged_while_action_held():
    link = Link(SimPort()).start()
    app = App(link)
    v = StallVision()
    app.attach_vision(v, ObjectGuide(lambda cue: None))
    app._action(True)
    v.t -= 10
    assert "camera" not in app.check_health()
    link.close()


class DeadMic:
    live, paused = True, False

    def __init__(self):
        self._n, self._stream, self.restarts, self.on_hit = 0, object(), 0, None

    def stop(self): pass
    def start(self): self.restarts += 1
    def pause(self): pass
    def resume(self): pass
    def set_names(self, n): pass


def test_mic_stall_is_reported_and_listener_restarted():
    link = Link(SimPort()).start()
    app = App(link)
    mic = DeadMic()
    app.attach_names(mic)
    t = time.monotonic()
    assert "mic" not in app.check_health(t)
    assert "mic" not in app.check_health(t + 1.0)
    f = app.check_health(t + app.MIC_STALL_S + 0.1)
    assert f["mic"].startswith("mic: no audio") and mic.restarts == 1
    mic._n += 4800                                                  # audio flowing again
    assert "mic" not in app.check_health(t + app.MIC_STALL_S + 0.6)
    link.close()


def test_hold_to_talk_mic_error_shows_until_next_good_result():
    link = Link(SimPort()).start()
    app = App(link)
    app._on_speech(D(text="", target=None, error="mic: PortAudioError -9996"))
    assert "speech" in app.check_health()
    app._on_speech(D(text="find the cup", target="cup", error=None))
    assert "speech" not in app.check_health()
    link.close()
