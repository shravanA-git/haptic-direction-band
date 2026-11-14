"""Obstacle avoidance (app/avoid.py): distances, sectors, steer, crash warning,
and the hooks into ObjectGuide and the dashboard. No camera or model."""
import asyncio
from types import SimpleNamespace

import numpy as np

from app import avoid as A
from app import haptics
from app.guide import ObjectGuide

W, H = 640, 480
F = A.focal_px(W)                      # ~512 px


def box(cls, cx, dist, score=0.9, cls_h=None):
    """A detection of `cls` centred at pixel x `cx`, `dist` metres away."""
    h = F * (cls_h or A.CLASS_HEIGHT_M[cls]) / dist
    y2 = min(H - 40, 240 + h / 2)
    y1 = max(0, y2 - h)
    w = h * 0.4
    return (cls, cx - w / 2, y1, cx + w / 2, y2, score)


def test_distance_from_box_height_and_clipping():
    d, clipped = A.estimate_distance("person", 100, 100 + F * 1.7 / 3.0, W, H)
    assert abs(d - 3.0) < 0.01 and not clipped
    d, clipped = A.estimate_distance("person", 0, H, W, H)       # fills the frame
    assert clipped and d <= 0.8
    assert A.obstacles([("cup", 300, 300, 340, 340, 0.9)], W, H) == []   # not an obstacle
    assert A.obstacles([("chair", 300, 100, 340, 300, 0.2)], W, H) == []  # low score


def test_far_obstacle_is_clear_and_close_centre_needs_two_frames_to_stop():
    av = A.Avoider()
    assert av.update([box("person", 320, 4.0)], W, H, 0.0).action == "clear"
    av.reset()
    close = [("person", 200, 0, 440, H, 0.9)]                    # right in front, cut off
    assert av.update(close, W, H, 0.0).action != "stop"           # one frame: not yet
    d = av.update(close, W, H, 0.1)
    assert d.action == "stop" and d.nearest.cls == "person" and d.changed
    assert av.update([], W, H, 0.2).action != "stop"              # gone: warning ends


def test_blocked_ahead_steers_to_the_open_side():
    av = A.Avoider()
    chair = [box("chair", 360, 1.1)]                              # just right of centre
    d = av.update(chair, W, H, 0.0)
    assert d.action == "steer" and d.motor == A.VEER_LEFT and d.bearing < 0, d
    states = [s for _, s, _ in d.sectors]
    assert states[2] == "blocked" and states[0] == "free"


def test_goal_off_to_the_side_is_clear_even_if_centre_blocked():
    av = A.Avoider()
    d = av.update([box("chair", 320, 1.1)], W, H, 0.0, goal_bearing=28.0)
    assert d.action == "clear"


def test_everything_blocked_turns_toward_the_roomier_edge():
    av = A.Avoider()
    wall = [box("couch", 100, 1.0), box("chair", 330, 1.1), ("chair", 450, 60, 640, 470, 0.9)]
    d = av.update(wall, W, H, 0.0)
    assert d.action == "turn" and d.motor in (A.TURN_LEFT, A.TURN_RIGHT)
    assert all(s == "blocked" for _, s, _ in d.sectors)


def test_one_missed_frame_does_not_clear_a_blocked_sector():
    av = A.Avoider()
    chair = [box("chair", 360, 1.1)]
    av.update(chair, W, H, 0.0)
    av.update(chair, W, H, 0.1)
    assert av.update([], W, H, 0.2).action == "steer"             # median of last 3


def test_mirror_swaps_sides():
    av = A.Avoider()
    d = av.update([box("chair", 280, 1.1)], W, H, 0.0, mirror=True)  # mirrored -> right of centre
    assert d.action == "steer" and d.motor == A.VEER_LEFT


class FakeAvoid:
    def __init__(self, d):
        self.d = d

    def override(self, now=None):
        return self.d


def test_guide_takes_steer_and_stop_from_avoid():
    played = []
    g = ObjectGuide(played.append)
    g.avoid = FakeAvoid(SimpleNamespace(action="steer", motor=A.VEER_RIGHT))
    g.on_decision(SimpleNamespace(state="left", t=0.0, confidence=0.9, target="cup"), now=0.0)
    assert played == [haptics.directional(A.VEER_RIGHT)]          # not motor 6
    g.avoid.d = SimpleNamespace(action="stop", motor=None)
    g.tick(now=1.0)
    assert len(played) == 1                                       # the loop plays the warning
    g.avoid.d = None
    g.tick(now=2.0)
    assert played[-1] == haptics.directional(6)                   # back to plain guidance


class FakeDetector:
    def __init__(self):
        self.dets = []

    def detect_all(self, frame, min_score=0.3):
        return self.dets


class FakeVision:
    mirror = False

    def __init__(self):
        self.detector = FakeDetector()
        self.d = SimpleNamespace(state="idle", offset=None)

    def latest(self):
        return self.d


def test_loop_warns_and_steers_without_a_target():
    v, played = FakeVision(), []
    loop = A.AvoidLoop(v, play=played.append)
    frame = np.zeros((H, W, 3), np.uint8)
    v.detector.dets = [("person", 200, 0, 440, H, 0.9)]
    loop.step(frame, now=0.0)
    loop.step(frame, now=0.1)
    assert played == [A.stop_cue()]
    loop.step(frame, now=0.3)
    assert len(played) == 1                                       # repeats every STOP_REPEAT_S
    loop.step(frame, now=0.1 + A.STOP_REPEAT_S + 0.01)
    assert len(played) == 2
    loop.avoider.reset()
    v.detector.dets = [box("chair", 360, 1.1)]
    loop.step(frame, now=5.0)
    assert played[-1] == haptics.directional(A.VEER_LEFT)
    loop.enabled = False
    v.detector.dets = [("person", 200, 0, 440, H, 0.9)]
    for k in range(3):
        loop.step(frame, now=10 + k)
    assert played[-1] == haptics.directional(A.VEER_LEFT)        # disabled: silent
    assert loop.override(10.5) is None


def test_loop_uses_target_bearing_and_ignores_stale_decisions():
    v = FakeVision()
    loop = A.AvoidLoop(v, play=lambda c: None)
    v.d = SimpleNamespace(state="right", offset=0.9)
    v.detector.dets = [box("chair", 320, 1.1)]
    d = loop.step(np.zeros((H, W, 3), np.uint8), now=0.0)
    assert d.action == "clear"                                    # target is off to the right
    loop.decision = A.AvoidDecision("steer", 7, -20.0, t=0.0)
    assert loop.override(0.3) is not None and loop.override(1.0) is None


def test_dashboard_gets_avoid_messages_and_toggle():
    from aiohttp.test_utils import TestClient, TestServer

    from app.link import Link, SimPort
    from app.server import App
    from tests.test_server import _collect

    async def run():
        link = Link(SimPort()).start()
        app = App(link)
        v = FakeVision()
        loop = A.AvoidLoop(v, play=lambda c: None)
        app.attach_avoid(loop)
        async with TestClient(TestServer(app.make_app())) as client:
            ws = await client.ws_connect("/ws")
            v.detector.dets = [box("chair", 360, 1.1)]
            loop.step(np.zeros((H, W, 3), np.uint8), now=0.0)
            m = await _collect(ws, lambda m: m["type"] == "avoid")
            assert m["action"] == "steer" and m["motor"] == A.VEER_LEFT and len(m["sectors"]) == 5
            await ws.send_json({"cmd": "avoid", "on": False})
            m = await _collect(ws, lambda m: m["type"] == "log")
            assert "off" in m["text"] and not loop.enabled
            r = await client.get("/")
            assert 'id="path"' in await r.text()
        link.close()

    asyncio.run(run())


def test_eval_avoid_scores_warn_time_and_false_warnings(tmp_path):
    import json

    from tools import eval_avoid as E
    from tools import eval_find as ef
    plan = ["free", "near", "blocked", "side_left", "side_right", "free"]
    labels = {"clips": [{"object": "person", "file": "p.mp4", "fps": 10,
                         "segments": [{"label": l, "f0": 50 * k, "f1": 50 * (k + 1)}
                                      for k, l in enumerate(plan)]}]}
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(labels))
    segs = ef.parse_labels(labels, tmp_path)[0]["segments"]

    class Frame:
        shape = (H, W, 3)

        def __init__(self, i):
            self.i = i

    def frames_for(path):
        return 10.0, ((i, W, Frame(i)) for i in range(300))

    def detect_all(f):
        lab, _ = ef.label_at(segs, f.i)
        return {"near": [box("person", 320, 2.0)],
                "blocked": [("person", 200, 0, 440, H, 0.9)],
                "side_left": [box("chair", 90, 1.1)],
                "side_right": [box("chair", 550, 1.1)]}.get(lab, [])

    res = E.evaluate(p, detect_all=detect_all, frames_for=frames_for)
    m = res["metrics"]
    assert m["missed"] == 0 and m["warn_s"] <= 0.2              # 2 frames to confirm
    assert m["false_warn"] == 0.0 and m["false_per_min"] == 0.0
    assert m["steer_wrong"] in (None, 0.0)
    assert res["verdict"]["warn_s"] == "PASS" and "RESULT: PASS" in E.report(res)
    assert "SKIP" in E.report(E.evaluate(tmp_path / "none.json"))
