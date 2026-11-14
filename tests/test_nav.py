"""--nav (CoStar track): route intent, step cues relative to the tour camera,
name/sound alerts win over guidance, and nothing changes without --nav."""
import time

import pytest

from app import haptics
from app.link import Link, SimPort
from app.nav import NavGuide, route_request
from app.server import App
from app.speech import SpeechResult
from tests.test_costar_nav import model
from tools.costar_nav import bearing, build_twin, motor_for


@pytest.mark.parametrize("text,want", [
    ("Take me to the kitchen.", "kitchen"),
    ("take me to the living room please", "living room"),
    ("Go to the bathroom!", "bathroom"),
    ("Guide me to my bedroom", "bedroom"),
    ("I want to go to the kitchen", "kitchen"),
    ("Let's go back to the living room.", "living room"),
    ("find my keys", None),
    ("Find the cup.", None),
    ("where is my phone", None),
    ("", None),
])
def test_route_request(text, want):
    assert route_request(text) == want


class Rec:
    def __init__(self):
        self.cues = []

    def __call__(self, cue):
        self.cues.append(cue)


def guide(**kw):
    rec = Rec()
    return NavGuide(build_twin(model()), play=rec, **kw), rec


def yaw_for(heading):          # default YAW_SIGN -1: heading = -yaw
    return -heading


def test_step_is_relative_to_where_the_tour_faces_and_arrival_is_two_front_pulses():
    nav, rec = guide()
    t = nav.twin
    ok, msg = nav.set_goal("kitchen", now=100.0)
    assert ok and rec.cues == [haptics.target_ack()] and "waiting" in msg
    nav.update_pose("A", yaw=yaw_for(90), now=100.1)       # too soon after 'got it'
    assert len(rec.cues) == 1
    nav.tick(now=101.0)
    a, c = t.panos["A"], t.panos["C"]
    want = motor_for(bearing((a.x, a.y), (c.x, c.y)) - 90)  # A -> C (toward the door)
    assert rec.cues[-1] == haptics.directional(want)
    nav.tick(now=102.0)
    assert len(rec.cues) == 2                               # no change, not yet due
    nav.tick(now=101.0 + nav.repeat_s + 0.01)
    assert rec.cues[-1] == haptics.directional(want) and len(rec.cues) == 3   # repeat
    nav.update_pose("A", yaw=yaw_for(90 + 90), now=106.0)  # turned right 90 degrees
    assert rec.cues[-1] == haptics.directional((want - 2) % 8)
    nav.update_pose("C", yaw=yaw_for(90), now=107.0)       # walked to the door
    nav.update_pose("D", now=108.0)                        # in the kitchen
    assert rec.cues[-1] == haptics.centered() and nav.goal is None
    nav.tick(now=120.0)
    assert rec.cues[-1] == haptics.centered()               # guidance stopped


def test_unknown_room_and_position_matching():
    nav, rec = guide()
    ok, msg = nav.set_goal("garage")
    assert not ok and rec.cues == [haptics.target_unknown()] and "Kitchen" in msg
    # tour gives no known sweep id: nearest scan point by position, tour (x, y, z) -> model (x, -z)
    nav.update_pose("not-a-pano", {"x": 6.1, "y": 1.5, "z": -2.1})
    assert nav.pano == "D"
    nav.update_pose(None, {"x": 30.0, "y": 1.5, "z": -2.0})   # nowhere near: keep the last
    assert nav.pano == "D"


def make_app(nav_on=True):
    port = SimPort(frames=[])
    link = Link(port).start()
    app = App(link)
    nav = None
    if nav_on:
        nav = NavGuide(build_twin(model()), play=lambda cue: app.play_cue(cue, "nav"))
        app.attach_nav(nav, {"model": "test"})
    return app, nav, port, link


class FakeVision:
    def __init__(self):
        self.targets = []

    def set_target(self, name):
        self.targets.append(name)
        return name

    def pause(self):
        pass

    def resume(self):
        pass


def test_name_and_sound_alerts_win_then_guidance_resumes():
    app, nav, port, link = make_app()
    nav.set_goal("kitchen", now=0.0)
    nav.update_pose("A", yaw=yaw_for(90), now=0.0)
    now = time.monotonic()
    app.notify.cue("name", "directional", 3)                # a name alert starts
    assert nav._hold_until > now + 0.5                      # 200 ms pattern + margin
    n = len(nav_cues(port))
    nav.tick(now=now + 0.1)
    assert len(nav_cues(port)) == n                         # quiet during the alert
    nav.tick(now=now + 2.0)
    time.sleep(0.3)
    assert len(nav_cues(port)) > n                          # step given again after it
    # a taught sound's longer pattern holds longer; muted and dry alerts don't hold
    now = time.monotonic()
    app.notify.cue("sound", "sound", None, category="melody")
    assert nav._hold_until >= now + (8 * 60) / 1000
    nav._hold_until = 0.0
    app.settings.muted.add("name")
    app.notify.cue("name", "directional", 2)
    app.notify.cue("name", "directional", 2, dry=True)
    assert nav._hold_until == 0.0
    link.close()


def nav_cues(port):
    return [line for line in port.sent if line.startswith("PULSE")]


def test_nav_pulses_go_through_the_link():
    app, nav, port, link = make_app()
    nav.set_goal("kitchen", now=0.0)
    nav.update_pose("A", yaw=yaw_for(90), now=5.0)
    time.sleep(0.4)
    link.close()
    a, c = nav.twin.panos["A"], nav.twin.panos["C"]
    want = motor_for(bearing((a.x, a.y), (c.x, c.y)) - 90)
    assert f"PULSE {want} {haptics.DIRECTIONAL_MS}" in port.sent


def test_in_nav_mode_take_me_to_is_a_route_and_find_stays_the_webcam():
    app, nav, port, link = make_app()
    vision = app.vision = FakeVision()
    app._on_speech(SpeechResult("Take me to the kitchen.", None))
    assert nav.goal == "kitchen" and vision.targets == [None]      # find guidance stopped
    app._on_speech(SpeechResult("go to the toilet", "toilet"))     # a COCO word, still a route
    assert nav.goal == "kitchen"              # no 'toilet' room: the route stays
    assert "toilet" not in vision.targets
    app._on_speech(SpeechResult("find my cup", "cup"))
    assert vision.targets[-1] == "cup" and nav.goal is None        # webcam find, route off
    link.close()


def test_without_nav_nothing_changes():
    app, nav, port, link = make_app(nav_on=False)
    vision = app.vision = FakeVision()
    app._on_speech(SpeechResult("go to the toilet", "toilet"))     # as today: a find target
    assert vision.targets == ["toilet"] and app.nav is None
    assert "navigation is off" in app.handle({"cmd": "nav_goal", "to": "kitchen"})["text"]
    assert app.handle({"cmd": "nav_pose", "sweep": "A"}) is None
    link.close()


def test_action_held_pauses_guidance():
    app, nav, port, link = make_app()
    nav.set_goal("kitchen", now=0.0)
    app._action(True)
    assert nav.paused
    nav.update_pose("A", yaw=0, now=10.0)
    app._action(False)
    assert not nav.paused
    link.close()
