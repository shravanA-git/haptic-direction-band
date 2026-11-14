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


def player(step_s=0.05):
    log = Log()
    rp = RoutePlayer(build_twin(model()), play=log.cue, speak=log.say, step_s=step_s)
    return rp, log


def test_plan_lists_steps_and_play_gives_them_in_order_then_arrival():
    rp, log = player()
    ok, msg = rp.plan("B", "kitchen")            # B -> C bends 26 deg into C -> D: two legs
    assert ok and len(rp.steps) == 2 and "2 steps" in msg
    assert rp.steps[0].motor == 0                # facing the first leg
    assert rp.texts[-1].endswith("into the kitchen")
    rp.intro = "Welcome to test house."
    rp.play()
    assert wait_for(lambda: rp.done)
    cues = [c for k, c in log.items if k == "cue"]
    says = [t for k, t in log.items if k == "say"]
    assert cues == [haptics.directional(s.motor) for s in rp.steps] + [haptics.centered()]
    assert says[0].startswith("Welcome to test house. ahead")       # briefing once, with step 1
    assert says[1:] == rp.texts[1:] + ["You've arrived at the kitchen."]
    assert [rp.starts()[0][0], rp.starts()[1][0]] == ["Living Room", "Kitchen"]


def test_next_gives_one_step_per_press_and_unknown_room_is_refused():
    rp, log = player()
    assert not rp.plan(None, "garage")[0]
    rp.plan("B", "kitchen")
    rp.next()
    assert wait_for(lambda: rp.index == 1)
    time.sleep(0.15)
    assert rp.index == 1 and not rp.done                    # waits for the presenter
    rp.next()
    rp.next()
    assert wait_for(lambda: rp.done)
    assert [c for k, c in log.items if k == "cue"][-1] == haptics.centered()


def make_app():
    port = SimPort(frames=[])
    link = Link(port).start()
    app = App(link)
    log = Log()
    rp = RoutePlayer(build_twin(model()), play=lambda c: app.play_cue(c, "nav"), speak=log.say,
                     step_s=0.4)
    app.attach_route(rp, {"name": "test house", "address": "1 Test St"})
    return app, rp, port, link, log


def test_a_name_alert_mid_route_pre_empts_the_next_step():
    app, rp, port, link, log = make_app()
    times = []
    status = rp.on_status
    rp.on_status = lambda st: (times.append((st["event"], time.monotonic())), status(st))
    rp.plan("B", "kitchen")
    rp.play()
    assert wait_for(lambda: rp.index == 1)
    t_alert = time.monotonic()
    app.notify.cue("name", "directional", 3)              # someone calls the wearer
    assert rp._hold_until >= t_alert + 0.6
    assert wait_for(lambda: rp.done, 5.0)
    after = [t for ev, t in times if ev in ("step", "arrived") and t > t_alert]
    assert after and min(after) >= rp._hold_until - 0.01   # nothing until the alert was over
    assert "PULSE 3 200" in port.sent
    link.close()


def test_nothing_is_spoken_or_played_while_action_is_held():
    app, rp, port, link, log = make_app()
    rp.plan("B", "kitchen")
    app._action(True)
    rp.play()
    time.sleep(0.3)
    assert rp.index == 0 and log.items == []               # paused: no step, no speech
    app._action(False)
    assert wait_for(lambda: rp.done, 5.0)
    assert len([1 for k, _ in log.items if k == "say"]) == 3
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
    app._on_speech(SpeechResult("Take me to the kitchen.", None))
    assert rp.goal == "kitchen"
    assert wait_for(lambda: rp.done, 5.0)
    assert log.items[0][1].startswith("Welcome to test house, 1 Test St.")
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
            assert cfg["mode"] == "route" and cfg["starts"] == ["Living Room", "Kitchen"]
            faces = (await (await client.get("/nav/pano/A")).json())["faces"]
            assert faces == [f"/nav/pano/A/{i}" for i in (1, 2, 3, 4)]
            assert (await client.get(faces[0])).status == 200
            assert (await client.get("/nav/pano/Z/1")).status == 404
            assert (await client.get("/nav/asset/aerial.jpg")).status == 200
            assert (await client.get("/nav/asset/brief.json")).status == 404
            assert (await client.get("/nav/pano/..%2F..%2Fx")).status == 404
            ws = await client.ws_connect("/ws")
            await ws.send_json({"cmd": "route_plan", "start": "Living Room", "to": "kitchen"})
            for _ in range(100):
                if rp.goal == "kitchen":
                    break
                await asyncio.sleep(0.02)
            assert rp.goal == "kitchen" and app.route_start == "Living Room"
        link.close()
    asyncio.run(run())


def test_vexcel_box_is_about_70_m():
    import math
    w = costar_assets.lot_wkt(34.05, -118.25)
    nums = [float(x) for x in w.replace("POLYGON((", "").replace("))", "").replace(",", " ").split()]
    lons, lats = nums[0::2], nums[1::2]
    assert (max(lats) - min(lats)) * 111_320 == pytest.approx(70, abs=0.5)
    assert (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(34.05)) == pytest.approx(70, abs=0.5)
