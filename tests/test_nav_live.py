"""--nav-live: route playback drives the 3D tour; our own moves' pose echoes are
ignored, and a user taking over the tour pauses Play and hands over to
interactive guidance."""
import pytest

from app import haptics
from app.link import Link, SimPort
from app.nav import NavGuide, RoutePlayer, TourDriver
from app.server import App
from tests.test_nav_route import Log, long_twin, run_clock


def test_tour_driver_tells_echoes_from_overrides():
    d = TourDriver()
    assert d.classify("P1", 0, now=0.0) == "free"
    d.moved("P1", -90.0, now=10.0)
    assert d.classify("P0", 45.0, now=11.0) == "echo"             # mid-transition: anything goes
    assert d.classify("P1", -80.0, now=13.0) == "echo"            # settled where we sent it
    assert d.classify("P1", 20.0, now=13.0) == "override"         # user turned the view
    assert d.classify("P2", -90.0, now=13.0) == "override"        # user clicked elsewhere
    assert d.classify("P1", 175.0, now=13.0) == "override"
    d.moved("P1", 179.0, now=20.0)
    assert d.classify("P1", -179.0, now=23.0) == "echo"           # wraps around 180


def test_play_moves_through_each_scan_point_and_turns_at_the_leg_end():
    log = Log()
    moves = []
    rp = RoutePlayer(long_twin(), play=log.cue, speak=log.say, threaded=False)
    rp.on_move = lambda pano, heading: (moves.append((round(rp._last_t, 2), pano, round(heading))),
                                        log.items.append(("move", pano)))
    rp.plan("P0", "office")
    rp.play(now=0.0)
    run_clock(rp, log, 40)
    # 12 m north at 0.8 m/s (15 s), scan points every 4 m, then 3 m east into the office
    assert [(p, h) for _, p, h in moves] == [("P0", 0), ("P1", 0), ("P2", 0), ("P3", 90), ("P4", 90)]
    assert [t for t, _, _ in moves] == pytest.approx([0.0, 5.0, 10.0, 15.0, 18.75], abs=0.06)
    k = log.items.index(("move", "P3"))
    assert log.items[k - 1] == ("cue", haptics.directional(2))     # the turn PULSE, same moment


def make_live_app(monkeypatch, live=True):
    monkeypatch.setenv("MATTERPORT_SDK_KEY", "browser-key")
    port = SimPort(frames=[])
    link = Link(port).start()
    app = App(link)
    twin = long_twin()
    nav = NavGuide(twin, play=lambda c: app.play_cue(c, "nav"))
    app.attach_nav(nav, {"model": "M1", "live": live})
    log = Log()
    rp = RoutePlayer(twin, play=lambda c: app.play_cue(c, "nav"), speak=log.say, threaded=False)
    app.attach_route(rp, {"name": "corridor"})
    sent = []
    app.emit = lambda m: sent.append(m)
    return app, nav, rp, sent, link


def test_live_play_drives_the_tour_from_where_it_stands(monkeypatch):
    app, nav, rp, sent, link = make_live_app(monkeypatch)
    app.handle({"cmd": "nav_pose", "sweep": "P1", "yaw": 0})      # the tour stands at P1
    app.handle({"cmd": "route_plan", "start": "", "to": "office"})
    assert rp.start_pano == "P1" and round(rp.steps[0].metres) == 8
    app.handle({"cmd": "route_play"})
    moves = [m for m in sent if m["type"] == "tour_move"]
    assert moves[0]["pano"] == "P1" and moves[0]["yaw"] == 0       # facing north (heading 0)
    assert moves[0]["position"] == {"x": 0, "y": 0, "z": -4}       # z-up model -> y-up tour
    app.handle({"cmd": "route_next"})                              # Next: jump to the turn now
    moves = [m for m in sent if m["type"] == "tour_move"]
    assert moves[-1]["pano"] == "P3" and moves[-1]["yaw"] == -90   # heading 90 east, yaw -90
    link.close()


def test_echoes_of_our_own_moves_dont_replan_but_a_user_override_pauses_play(monkeypatch):
    app, nav, rp, sent, link = make_live_app(monkeypatch)
    app.handle({"cmd": "nav_pose", "sweep": "P0", "yaw": 0})
    app.handle({"cmd": "route_plan", "start": "", "to": "office"})
    app.handle({"cmd": "route_play"})
    for yaw in (0, 30, -20):                                      # the SDK echoing our move
        app.handle({"cmd": "nav_pose", "sweep": "P0", "yaw": yaw})
    assert rp.auto and rp.goal == "office" and nav.goal is None   # still driving, no re-plan
    app.driver.expect = (app.driver.expect[0], app.driver.expect[1], 0.0)   # transition over
    app.handle({"cmd": "nav_pose", "sweep": "P0", "yaw": 5})      # settled where we sent it
    assert rp.auto and nav.goal is None
    app.handle({"cmd": "nav_pose", "sweep": "P2", "yaw": 0})      # the user clicked ahead
    assert not rp.auto and rp.goal is None                        # Play paused
    assert nav.goal == "office" and nav.pano == "P2"              # live guidance from there
    assert any(m.get("event") == "override" for m in sent if m["type"] == "route")
    link.close()


def test_without_live_mode_play_never_moves_the_tour(monkeypatch):
    app, nav, rp, sent, link = make_live_app(monkeypatch, live=False)
    app.handle({"cmd": "route_plan", "start": "P0", "to": "office"})
    app.handle({"cmd": "route_play"})
    app.handle({"cmd": "route_next"})
    assert not [m for m in sent if m["type"] == "tour_move"]
    app.handle({"cmd": "nav_pose", "sweep": "P2", "yaw": 0})      # no driving: no override
    assert rp.goal == "office"
    link.close()


def test_driven_tour_routes_through_the_scan_points_in_between():
    """Matterport links far-apart scan points directly (one 12 m hop here): a
    driven tour would jump. Weighting by distance squared keeps the chain."""
    from tools.costar_nav import build_twin
    t = long_twin()
    m = {"name": "corridor", "floors": [], "rooms": [{"id": "R", "classifications": [{"label": "Office"}],
                                                       "boundary": {"edges": []}}],
         "panoLocations": [{"id": p.id, "label": p.id, "position": {"x": p.x, "y": p.y, "z": 0}}
                           for p in t.panos.values()],
         "locations": [{"id": "P4", "room": {"id": "R"}, "neighbors": ["P3", "P0"]},
                       {"id": "P3", "neighbors": ["P2", "P4"]}, {"id": "P2", "neighbors": ["P1", "P3"]},
                       {"id": "P1", "neighbors": ["P0", "P2"]}, {"id": "P0", "neighbors": ["P1", "P4"]}]}
    twin = build_twin(m)                                 # P0 -> P4 directly is 12.4 m
    rp = RoutePlayer(twin, play=lambda c: None, threaded=False)
    rp.plan("P0", "office")
    assert [L["subs"] for L in rp.legs] == [[]]          # one hop, nothing in between
    rp.short_hops = True
    rp.plan("P0", "office")
    assert [p for p, _, _ in rp.legs[0]["subs"]] == ["P1", "P2"]
