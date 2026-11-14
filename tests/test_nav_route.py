"""--nav route mode ("plan once, play the route"): play order, Next, alerts win
mid-route, nothing spoken while Action is held, and everything served offline
from mp_cache/ after a prefetch."""
import json
import time

import pytest

from app import haptics
from app.link import Link, SimPort
from app.nav import RoutePlayer, Speaker
from app.server import App
from app.speech import SpeechResult
from tests.test_costar_nav import model
from tools import costar_assets, costar_nav
from tools.costar_nav import build_twin


def wait_for(cond, s=3.0):
    t = time.monotonic()
    while not cond() and time.monotonic() - t < s:
        time.sleep(0.01)
    return cond()


class Log:
    def __init__(self):
        self.items = []

    def cue(self, c):
        self.items.append(("cue", c))

    def say(self, t):
        self.items.append(("say", t))


def player(twin=None, speed=0.8):
    log = Log()
    rp = RoutePlayer(twin or build_twin(model()), play=log.cue, speak=log.say, speed=speed,
                     threaded=False)
    return rp, log


def long_twin():
    """12 m straight north, then 3 m east: a long leg, then a right turn."""
    pts = {"P0": (0, 0), "P1": (0, 4), "P2": (0, 8), "P3": (0, 12), "P4": (3, 12)}
    return build_twin({"name": "corridor", "floors": [], "rooms": [
        {"id": "R", "classifications": [{"label": "Office"}],
         "boundary": {"edges": []}}],
        "panoLocations": [{"id": k, "label": k, "position": {"x": x, "y": y, "z": 0}}
                          for k, (x, y) in pts.items()],
        "locations": [{"id": "P4", "room": {"id": "R"}, "neighbors": ["P3"]},
                      {"id": "P3", "neighbors": ["P2", "P4"]}, {"id": "P2", "neighbors": ["P1", "P3"]},
                      {"id": "P1", "neighbors": ["P0", "P2"]}, {"id": "P0", "neighbors": ["P1"]}]})


def run_clock(rp, log, until, dt=0.05, t0=0.0):
    """Tick a threaded=False player; returns [(t, kind, payload)]."""
    out, t = [], t0
    seen = 0
    while t <= until and not rp.done:
        rp.tick(t)
        for item in log.items[seen:]:
            out.append((round(t, 2), *item))
        seen = len(log.items)
        t += dt
    return out


def test_plan_lists_steps_and_play_gives_them_in_order_then_arrival():
    rp, log = player()
    ok, msg = rp.plan("B", "kitchen")            # B -> C bends 26 deg into C -> D: two legs
    assert ok and len(rp.steps) == 2 and "2 steps" in msg
    assert rp.steps[0].motor == 0                # facing the first leg
    assert rp.texts[-1].endswith("into the kitchen")
    rp.intro = "Welcome to test house."
    rp.play(now=0.0)
    run_clock(rp, log, 30)
    assert rp.done
    cues = [c for k, c in log.items if k == "cue"]
    says = [t for k, t in log.items if k == "say"]
    assert cues == [haptics.directional(s.motor) for s in rp.steps] + [haptics.centered()]
    assert says[0].startswith("Welcome to test house. ahead")       # briefing once, with step 1
    assert says[-1] == "You've arrived at the kitchen." and rp.texts[1] in says
    assert [rp.starts()[0][0], rp.starts()[1][0]] == ["Living Room", "Kitchen"]


def test_leg_time_scales_with_metres_and_speed():
    rp, _ = player(long_twin())
    rp.plan("P0", "office")
    assert [round(s.metres) for s in rp.steps] == [12, 3]
    assert rp.leg_seconds(0) == pytest.approx(15.0) and rp.leg_seconds(1) == pytest.approx(3.75)
    rp.set_speed("demo")
    assert rp.leg_seconds(0) == pytest.approx(8.0) and rp.leg_seconds(1) == pytest.approx(2.5)  # min
    rp.set_speed("slow")
    assert rp.leg_seconds(0) == pytest.approx(20.0)


def test_turn_pulse_fires_at_the_end_of_the_leg_with_a_spoken_warning_before():
    rp, log = player(long_twin())
    rp.plan("P0", "office")
    rp.play(now=0.0)
    ev = run_clock(rp, log, 40)
    turn = [t for t, k, c in ev if k == "cue" and c == haptics.directional(rp.steps[1].motor)]
    assert rp.steps[1].motor == 2 and turn and turn[0] == pytest.approx(15.0, abs=0.06)
    warn = [(t, x) for t, k, x in ev if k == "say" and x.startswith("in ")]
    assert warn and warn[0][0] == pytest.approx(13.5, abs=0.06) and warn[0][1] == "in 1 metre, turn right"
    assert not [1 for t, k, c in ev if k == "cue" and abs(t - 13.5) < 0.06]   # warning: no buzz
    arrive = [t for t, k, c in ev if k == "cue" and c == haptics.centered()]
    assert arrive[0] == pytest.approx(15.0 + 3.75, abs=0.06)
    assert ev[0][:2] == (0.0, "cue")                                         # first leg: at start


def test_keep_going_pulses_on_long_legs_only():
    rp, log = player(long_twin())
    rp.plan("P0", "office")
    rp.play(now=0.0)
    ev = run_clock(rp, log, 40)
    keep = [t for t, k, c in ev if k == "cue" and c == [(0, 0, 80)]]
    assert keep == pytest.approx([4.0, 8.0, 12.0], abs=0.06)    # 15 s leg; none in the last 1.5 s
    assert all(t < 15.0 for t in keep)                           # the 3.75 s leg gets none


def test_next_skips_now_and_stop_stops():
    rp, log = player(long_twin())
    rp.plan("P0", "office")
    rp.play(now=0.0)
    run_clock(rp, log, 1.0)
    n = len(log.items)
    rp.next(now=1.0)                                             # skip the rest of the corridor
    assert log.items[n][1] == haptics.directional(rp.steps[1].motor) and rp.leg == 1
    ev = run_clock(rp, log, 10, t0=1.05)
    assert [t for t, k, c in ev if c == haptics.centered()][0] == pytest.approx(1.0 + 3.75, abs=0.06)
    # manual: Next only, no clock
    rp, log = player(long_twin())
    rp.plan("P0", "office")
    rp.next(now=0.0)
    run_clock(rp, log, 30)
    assert rp.leg == 0 and not rp.done                           # waits for the presenter
    rp.next(now=31.0)
    rp.next(now=32.0)
    assert rp.done and [c for k, c in log.items if k == "cue"][-1] == haptics.centered()
    rp, log = player(long_twin())
    rp.plan("P0", "office")
    rp.play(now=0.0)
    rp.stop()
    run_clock(rp, log, 30)
    assert rp.leg == 0 and not rp.done


def test_unknown_room_is_refused():
    rp, log = player()
    assert not rp.plan(None, "garage")[0]


def make_app():
    port = SimPort(frames=[])
    link = Link(port).start()
    app = App(link)
    log = Log()
    rp = RoutePlayer(long_twin(), play=lambda c: app.play_cue(c, "nav"), speak=log.say,
                     threaded=False)
    app.attach_route(rp, {"name": "test house", "address": "1 Test St"})
    return app, rp, port, link, log


def test_a_name_alert_mid_route_freezes_the_route_until_it_is_over():
    app, rp, port, link, log = make_app()
    t0 = time.monotonic()
    rp.plan("P0", "office")
    rp.play(now=t0)
    for k in range(1, 101):                               # 5 s into the corridor
        rp.tick(t0 + k * 0.05)
    app.notify.cue("name", "directional", 3)             # someone calls the wearer
    hold_end = rp._hold_until
    assert hold_end >= time.monotonic() + 0.5
    leg_t = rp.leg_t
    t = time.monotonic()
    while t < hold_end - 0.05:
        rp.tick(t)
        t += 0.05
    assert rp.leg_t == leg_t                              # frozen: no cue, no speech
    t = hold_end
    turned = None
    while not rp.done and t < hold_end + 30:
        rp.tick(t)
        if turned is None and rp.leg == 1:
            turned = t
        t += 0.05
    assert turned == pytest.approx(hold_end + (15.0 - leg_t), abs=0.1)   # turn cue shifted, not lost
    time.sleep(0.3)
    assert "PULSE 3 200" in port.sent
    link.close()


def test_nothing_is_spoken_or_played_while_action_is_held():
    app, rp, port, link, log = make_app()
    app.route_go("P0", "office")                          # plans with the briefing as intro
    app._action(True)
    rp.play(now=0.0)
    for k in range(100):
        rp.tick(k * 0.05)
    assert rp.leg == -1 and log.items == []               # paused: no step, no speech
    app._action(False)
    for k in range(100, 600):
        rp.tick(k * 0.05)
    assert rp.done and log.items[0][1].startswith("Welcome to test house")
    link.close()


def test_speaker_is_silent_while_blocked(monkeypatch):
    calls = []

    class FakeProc:
        def __init__(self, args, **kw):
            calls.append(args)

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminated")
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    held = [True]
    sp = Speaker(blocked=lambda: held[0])
    sp.available = True
    assert not sp.say("ahead 3 metres") and calls == []
    held[0] = False
    assert sp.say("ahead 3 metres") and calls == [["say", "ahead 3 metres"]]
    sp.cancel()
    assert calls[-1] == "terminated"


def test_voice_in_route_mode_plans_and_plays(monkeypatch):
    monkeypatch.delenv("MATTERPORT_SDK_KEY", raising=False)
    app, rp, port, link, log = make_app()
    app._on_speech(SpeechResult("Take me to the office.", None))
    assert rp.goal == "office" and rp.leg == 0 and rp.auto      # planned and playing
    assert log.items[0][1].startswith("Welcome to test house, 1 Test St.")
    time.sleep(0.3)
    assert port.sent[:2] == ["PULSE 2 120", "PULSE 6 120"]     # 'got it' first
    link.close()


# ---------------------------------------------------------------- offline cache
def fake_network(monkeypatch, calls):
    """Model API + downloads answered locally; counts calls."""
    def graphql(q, v):
        calls.append(q.split("{")[0].strip() or q[:20])
        if "skyboxes" in q:
            return {"model": {"panoLocations": [
                {"id": "A", "skyboxes": [{"resolution": "2k", "status": "available",
                                          "children": [f"https://x/A_{i}.jpg" for i in range(6)]}]}]}}
        if "image" in q:
            return {"model": {"image": {"url": "https://x/preview.jpg"}}}
        return {"model": dict(model(), address={"address": "1 Test St"},
                              geolocation={"lat": 34.05, "long": -118.25}, locations=[])}

    def get(url, headers=None, data=None, timeout=30):
        calls.append(url)
        return "image/jpeg", b"\xff\xd8jpeg"
    monkeypatch.setattr(costar_assets, "graphql", graphql)
    monkeypatch.setattr(costar_assets, "_get", get)
    monkeypatch.setattr(costar_nav, "fetch_model", lambda mid: graphql("query M", {})["model"])


def test_prefetch_caches_everything_then_works_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(costar_nav, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(costar_assets, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("VEXCEL_API_KEY", "test")
    monkeypatch.delenv("COSTAR_KIT_URL", raising=False)
    calls = []
    fake_network(monkeypatch, calls)
    rep = costar_assets.prefetch("M1")
    assert rep["360 photos"] == "1 scan points" and rep["aerial"] == "ok" and rep["preview"] == "ok"
    d = tmp_path / "M1"
    assert sorted(p.name for p in (d / "pano").iterdir()) == [f"A_{i}.jpg" for i in (1, 2, 3, 4)]
    assert json.loads((d / "brief.json").read_text())["address"] == "1 Test St"
    # wifi off: any network call fails, everything still loads from the cache
    def offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(costar_assets, "graphql", offline)
    monkeypatch.setattr(costar_assets, "_get", offline)
    monkeypatch.setattr(costar_nav, "fetch_model", offline)
    monkeypatch.delenv("MATTERPORT_API_KEY", raising=False)
    m = costar_nav.load_model("M1", None, False)
    b = costar_assets.brief("M1", m)
    assert b["aerial"] and b["preview"] and b["name"] == "test house"
    assert len(costar_assets.pano_faces("M1", "A")) == 4
    assert "Welcome to test house, 1 Test St." == costar_assets.briefing_text(b)


def test_nav_page_serves_cached_pictures(tmp_path, monkeypatch):
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer
    monkeypatch.setattr(costar_assets, "CACHE_DIR", tmp_path)
    (tmp_path / "M1" / "pano").mkdir(parents=True)
    for i in (1, 2, 3, 4):
        (tmp_path / "M1" / "pano" / f"A_{i}.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "M1" / "aerial.jpg").write_bytes(b"\xff\xd8")

    async def run():
        app, rp, port, link, log = make_app()
        app.nav_info = {"model": "M1"}
        async with TestClient(TestServer(app.make_app())) as client:
            cfg = await (await client.get("/nav/config")).json()
            assert cfg["mode"] == "route" and cfg["starts"] == ["Office"] and cfg["speed"] == 0.8
            faces = (await (await client.get("/nav/pano/A")).json())["faces"]
            assert faces == [f"/nav/pano/A/{i}" for i in (1, 2, 3, 4)]
            assert (await client.get(faces[0])).status == 200
            assert (await client.get("/nav/pano/Z/1")).status == 404
            assert (await client.get("/nav/asset/aerial.jpg")).status == 200
            assert (await client.get("/nav/asset/brief.json")).status == 404
            assert (await client.get("/nav/pano/..%2F..%2Fx")).status == 404
            ws = await client.ws_connect("/ws")
            await ws.send_json({"cmd": "route_plan", "start": "P0", "to": "office"})
            await ws.send_json({"cmd": "route_speed", "speed": "demo"})
            for _ in range(100):
                if rp.goal == "office" and rp.speed == 1.5:
                    break
                await asyncio.sleep(0.02)
            assert rp.goal == "office" and app.route_start == "P0" and rp.speed == 1.5
        link.close()
    asyncio.run(run())


def test_vexcel_box_is_about_70_m():
    import math
    w = costar_assets.lot_wkt(34.05, -118.25)
    nums = [float(x) for x in w.replace("POLYGON((", "").replace("))", "").replace(",", " ").split()]
    lons, lats = nums[0::2], nums[1::2]
    assert (max(lats) - min(lats)) * 111_320 == pytest.approx(70, abs=0.5)
    assert (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(34.05)) == pytest.approx(70, abs=0.5)
