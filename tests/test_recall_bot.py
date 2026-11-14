"""Recall chatbot: tools, offline parser, Claude tool loop (fake client), mute."""
import asyncio
import json
import time
from types import SimpleNamespace as NS

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app import recall_bot
from app.link import Link, SimPort
from app.recall import AlertLog
from app.recall_bot import RecallBot, recent_alerts
from app.server import App
from app.settings import Settings


def sample_log():
    log = AlertLog()
    log.add("name", "directional", 5, 0.9, 400, heard="Shravan")
    log.add("find", "directional", 6, 0.8, 120, target="banana")
    log.add("find", "centered", 0, 0.9, 140, target="banana")
    log.add("replay", "directional", 5, dry=True)
    g = log.add("name", "generic", reason="conflicting directions")
    log.label(g["id"], "false")
    return log


def test_recent_alerts_filters_and_counts():
    log = sample_log()
    r = recent_alerts(log, "any", 60, 10)
    assert r["total"] == 4 and r["counts"] == {"name": 2, "find": 2}   # replay excluded
    r = recent_alerts(log, "name", 60, 1)
    assert r["total"] == 2 and len(r["alerts"]) == 1 and r["alerts"][0]["truth"] == "false"
    assert recent_alerts(log, "any", 1, 10, now=time.time() + 3600)["total"] == 0


def test_settings_validation():
    s = Settings()
    assert s.set("intensity", "Strong") == "intensity set to high"
    assert s.set("name_alerts", "off") == "name alerts off" and s.is_muted("name")
    assert s.set("names", "Shravan, Sophanit") == "listening for Shravan, Sophanit"
    for k, v in [("intensity", "max"), ("find_cues", "maybe"), ("sound_threshold", "2"),
                 ("names", " , "), ("volume", "3")]:
        with pytest.raises(ValueError):
            s.set(k, v)


@pytest.mark.parametrize("q,want", [
    ("who called me in the last 10 minutes?", "2 name calls in the last 10 minutes"),
    ("what was the last alert?", "Latest: "),
    ("anything about the banana?", "banana: centered"),
    ("any taught sound today?", "No taught-sound alerts today"),
])
def test_offline_questions(q, want):
    a = RecallBot(sample_log(), Settings(), "offline").ask(q)
    assert want in a.text and a.via == "offline" and a.actions[0][0] == "recent_alerts"


def test_offline_settings_commands():
    s = Settings()
    bot = RecallBot(sample_log(), s, "offline")
    assert "high" in bot.ask("make the buzz stronger").text and s.intensity == "high"
    bot.ask("set intensity low")
    assert s.intensity == "low"
    bot.ask("mute name alerts")
    assert s.is_muted("name")
    bot.ask("turn on name alerts")
    assert not s.is_muted("name")
    bot.ask("listen for Shravan and Jason")
    assert s.names == "Shravan, Jason"
    assert "STOP" in bot.ask("stop all motors right now").text
    assert "Which alerts" in bot.ask("mute it").text


# ------------------------------------------------------------- claude loop
class FakeClaude:
    """Scripted stand-in for anthropic.Anthropic(): returns queued responses
    and records each request."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_use(name, inp, id="tu1"):
    return NS(stop_reason="tool_use",
              content=[NS(type="tool_use", name=name, input=inp, id=id)])


def text(t):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=t)])


def test_claude_loop_runs_tools_and_keeps_history():
    s = Settings()
    fake = FakeClaude([tool_use("recent_alerts", {"source": "name", "minutes": 30, "limit": 5}),
                       text("Someone called you from the back-left."),
                       tool_use("change_setting", {"setting": "intensity", "value": "high"}, "tu2"),
                       text("Intensity is now high.")])
    bot = RecallBot(sample_log(), s, client=fake)
    assert bot.mode == "claude"
    a = bot.ask("who called me?")
    assert a.text == "Someone called you from the back-left." and a.via == "claude"
    req = fake.calls[0]
    assert req["model"] == "claude-opus-5-5" and {t["name"] for t in req["tools"]} == {
        "recent_alerts", "change_setting"}
    assert all(t["strict"] for t in req["tools"]) and "tool_choice" not in req
    tr = fake.calls[1]["messages"][-1]["content"][0]
    assert tr["type"] == "tool_result" and json.loads(tr["content"])["total"] == 2
    a = bot.ask("make it stronger")
    assert s.intensity == "high" and a.actions[0][0] == "change_setting"
    assert fake.calls[2]["messages"][0]["content"] == "who called me?"   # history kept


def test_claude_tool_error_is_reported_to_model():
    fake = FakeClaude([tool_use("change_setting", {"setting": "intensity", "value": "max"}),
                       text("Intensity can be low, medium or high.")])
    RecallBot(AlertLog(), Settings(), client=fake).ask("max power")
    tr = fake.calls[1]["messages"][-1]["content"][0]
    assert tr["is_error"] and "intensity must be one of" in tr["content"]


def test_claude_failure_falls_back_offline_and_history_stays_valid():
    fake = FakeClaude([ConnectionError("no network"), text("Fine.")])
    bot = RecallBot(sample_log(), Settings(), client=fake)
    a = bot.ask("who called me?")
    assert a.via == "offline" and "Claude unavailable" in a.text and bot.messages == []
    assert bot.ask("hi").text == "Fine."


def test_auto_mode_without_credentials_is_offline(monkeypatch):
    monkeypatch.setattr(recall_bot, "claude_available", lambda: False)
    assert RecallBot(AlertLog(), Settings()).mode == "offline"


# ------------------------------------------------------------- app wiring
def test_muted_source_is_logged_but_not_played():
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    app.settings.set("name_alerts", "off")
    a = app.notify.cue("name", "directional", 5)
    app.notify.cue("manual", "directional", 2)
    time.sleep(0.1)
    link.close()
    assert a["muted"] is True
    assert port.sent[0] == "PULSE 2 200" and "PULSE 5 200" not in port.sent


def test_ws_ask_and_setting():
    async def run():
        port = SimPort()
        link = Link(port).start()
        app = App(link)
        app.alerts.add("name", "directional", 6, 0.8, 300)
        async with TestClient(TestServer(app.make_app())) as client:
            ws = await client.ws_connect("/ws")
            st = None
            while not st:
                m = json.loads((await ws.receive()).data)
                st = m if m["type"] == "settings" else None
            assert st["intensity"] == "medium" and st["name_alerts"] == "on"
            await ws.send_json({"cmd": "ask", "text": "who called me?"})
            while True:
                m = json.loads((await asyncio.wait_for(ws.receive(), 3)).data)
                if m["type"] == "recall":
                    break
            assert "left" in m["a"] and m["via"] == "offline"
            await ws.send_json({"cmd": "ask", "text": "mute find cues"})
            while True:
                m = json.loads((await asyncio.wait_for(ws.receive(), 3)).data)
                if m["type"] == "settings":
                    break
            assert m["find_cues"] == "off" and app.settings.is_muted("find")
            await ws.send_json({"cmd": "setting", "setting": "intensity", "value": "loud"})
            while True:
                m = json.loads((await asyncio.wait_for(ws.receive(), 3)).data)
                if m["type"] == "log":
                    break
            assert m["text"].startswith("setting: intensity must be")
            await ws.close()
        link.close()
    asyncio.run(run())


# ------------------------------------------------------------- intensity
def test_intensity_scales_cues_within_cap_and_keeps_rhythm():
    from app import haptics
    assert haptics.scaled(haptics.directional(3), "low") == [(0, 3, 120)]
    assert haptics.scaled(haptics.directional(3), "high") == [(0, 3, 300)]
    assert haptics.scaled(haptics.lost(), "high")[0][2] == 900
    assert haptics.scaled([(0, 0, 900)], "high") == [(0, 0, 1000)]           # cap
    assert haptics.scaled([(0, 0, 80)], "low") == [(0, 0, 60)]               # floor
    c = haptics.scaled(haptics.centered(), "high")
    assert c == [(0, 0, 150), (300, 0, 150)]                                 # gap scales too
    assert haptics.scaled(haptics.centered(), "medium") == haptics.centered()


def test_intensity_applies_to_cues_not_tests_or_stop():
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    app.handle({"cmd": "setting", "setting": "intensity", "value": "high"})
    app.handle({"cmd": "cue", "name": "generic"})
    time.sleep(0.05)
    app.handle({"cmd": "pulse", "motor": 2, "ms": 150})                      # test pulse
    app.handle({"cmd": "stop"})
    app.play_cue([(0, 6, 200)], "find").join()
    app.settings.set("find_cues", "off")
    assert app.play_cue([(0, 6, 200)], "find") is None
    link.close()
    assert port.sent == ["PULSE 0 450", "PULSE 2 150", "STOP", "PULSE 6 300", "STOP"]
