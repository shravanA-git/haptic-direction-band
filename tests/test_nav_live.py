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
    """Matterport links far-apart scan points directly (one 11.5 m hop here,
    a little shorter than the chain): a driven tour would jump. Weighting by
    distance squared keeps the chain of scan points."""
    from tools.costar_nav import build_twin
    pts = {"P0": (0, 0), "P1": (0.5, 4), "P2": (0.5, 8), "P3": (0, 11.5), "P4": (3, 11.5)}
    nb = {"P0": ["P1", "P3"], "P1": ["P0", "P2"], "P2": ["P1", "P3"], "P3": ["P2", "P4", "P0"],
          "P4": ["P3"]}
    twin = build_twin({
        "name": "corridor", "floors": [],
        "rooms": [{"id": "R", "classifications": [{"label": "Office"}], "boundary": {"edges": []}}],
        "panoLocations": [{"id": k, "label": k, "position": {"x": x, "y": y, "z": 0}}
                          for k, (x, y) in pts.items()],
        "locations": [{"id": k, "neighbors": v, **({"room": {"id": "R"}} if k == "P4" else {})}
                      for k, v in nb.items()]})
    rp = RoutePlayer(twin, play=lambda c: None, threaded=False)
    rp.plan("P0", "office")
    assert rp.legs[0]["end"] == "P3" and rp.legs[0]["subs"] == []     # one 11.5 m hop
    rp.short_hops = True
    rp.plan("P0", "office")
    assert [p for p, _, _ in rp.legs[0]["subs"]] == ["P1", "P2"]
    assert round(sum(s.metres for s in rp.steps)) == 15                # barely longer


def office_twin():
    """long_twin's corridor, with a lobby at P0 and a deeper office (P4, P5): the
    arrival point P4 sits at the office's edge, the room extends north to P5."""
    from tools.costar_nav import build_twin
    pts = {"P0": (0, 0), "P1": (0, 4), "P2": (0, 8), "P3": (0, 12), "P4": (3, 12), "P5": (3, 15)}
    nb = {"P0": ["P1"], "P1": ["P0", "P2"], "P2": ["P1", "P3"], "P3": ["P2", "P4"],
          "P4": ["P3", "P5"], "P5": ["P4"]}
    room = {"P0": "L", "P4": "O", "P5": "O"}
    return build_twin({
        "name": "corridor", "floors": [],
        "rooms": [{"id": "O", "classifications": [{"label": "Office"}], "boundary": {"edges": []}},
                  {"id": "L", "classifications": [{"label": "Lobby"}], "boundary": {"edges": []}}],
        "panoLocations": [{"id": k, "label": k, "position": {"x": x, "y": y, "z": 0}}
                          for k, (x, y) in pts.items()],
        "locations": [{"id": k, "neighbors": v, **({"room": {"id": room[k]}} if k in room else {})}
                      for k, v in nb.items()]})


def test_arrival_faces_into_the_room_not_the_wall():
    log, moves = Log(), []
    rp = RoutePlayer(office_twin(), play=log.cue, speak=log.say, threaded=False)
    rp.on_move = lambda pano, heading: moves.append((pano, round(heading)))
    rp.plan("P0", "office")
    rp.play(now=0.0)
    run_clock(rp, log, 60)
    assert moves[-1] == ("P4", 0)          # walked east into P4, then turned north into the room


def test_live_plans_from_the_tour_not_the_dropdown_and_demo_flies_to_its_start(monkeypatch):
    app, nav, rp, sent, link = make_live_app(monkeypatch)
    app.route.twin = nav.twin = office_twin()
    app.handle({"cmd": "nav_pose", "sweep": "P2", "yaw": 0})      # the tour stands at P2
    app.handle({"cmd": "route_plan", "start": "Lobby", "to": "office"})   # stale dropdown value
    assert rp.start_pano == "P2"
    assert "Starting at" not in rp.intro
    sent.clear()
    out = app.handle({"cmd": "route_demo"})
    assert rp.start_pano == "P0" and rp.goal == "Office"
    assert rp.intro.endswith("Starting at the lobby."), out
    demo = [m for m in sent if m["type"] == "route_demo"][0]
    assert (demo["start"], demo["to"], demo["legs"]) == ("Lobby", "Office", 2)
    link.close()


def test_drive_keeps_here_and_heading_fresh(monkeypatch):
    app, nav, rp, sent, link = make_live_app(monkeypatch)
    app.handle({"cmd": "nav_pose", "sweep": "P0", "yaw": 0})
    app.handle({"cmd": "route_plan", "start": "", "to": "office"})
    app.handle({"cmd": "route_play"})
    app.handle({"cmd": "route_next"})                              # jump to the turn at P3
    navs = [m for m in sent if m["type"] == "nav"]
    assert navs[-1]["pano"] == "P3" and navs[-1]["heading"] == 90 and navs[-1]["goal"] == "office"
    link.close()
