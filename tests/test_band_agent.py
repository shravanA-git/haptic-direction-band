"""Band agent (app/band_agent.py): tools, router, allowlist, urgent texts, deep agent.
No network and no key. The deep-agent test needs langchain/deepagents and is skipped without them."""
import time
from types import SimpleNamespace

import pytest

from app import band_agent as B
from app import haptics
from app.recall import AlertLog


class Clock:
    def __init__(self, t=None):
        self.t = time.time() if t is None else t

    def __call__(self):
        return self.t


def seeded(clock=None):
    log = AlertLog()
    log.add("name", "directional", 6)
    log.add("sound", "sound", 0, label="doorbell")
    log.add("sound", "sound", 4, label="alarm")
    log.add("find", "centered", None, target="keys")
    log.add("find", "directional", 2, target="keys", repeat=True)          # a repeat: never listed
    log.add("manual", "generic")                                            # a test cue: never listed
    log.add("avoid", "generic", None, action="stop", obstacle="chair", metres=0.8)
    status = {"connected": True, "faults": {}, "target": "keys", "settings": {"intensity": "medium"}}
    return log, B.BandTools(log, lambda: status, lambda t: t.strip(), lambda m: (True, None),
                            clock=clock or Clock())


def test_plain_text_is_speakable():
    t = B.plain_text("**Status**\n- Band: ok \U0001F600\n- Camera: ok\n| a | b |\n# Heading")
    assert not any(c in t for c in "*#|\n") and "\U0001F600" not in t and t.startswith("Status")
    long = B.plain_text("One sentence here. " * 80)
    assert len(long) <= B.MAX_REPLY_CHARS and long.endswith(".")


def test_daily_summary_newest_first_with_meanings_and_no_noise():
    log, tools = seeded()
    s = tools.daily_summary("today")
    assert s.startswith("Today: 5 buzzes.")                      # repeat and manual test left out
    assert s.index("obstacle ahead") < s.index("found keys") < s.index("alarm") < s.index("doorbell")
    assert "your name was called from your left" in s and "alarm, from behind, urgent" in s
    assert "from the" not in s.replace("from your", "")          # direction phrases are the spoken ones
    assert B.plain_text(s) == s


def test_summary_merges_near_duplicates_and_respects_the_period():
    clock = Clock()
    log, tools = seeded(clock)
    log.add("name", "directional", 6)
    log.add("name", "directional", 6)
    s = tools.daily_summary("today")
    assert "times" in s
    clock.t += 7200
    assert tools.daily_summary("hour") == "In the last hour: no alerts."
    assert tools.recent_alerts(60).startswith("Nothing in the last 60 minutes")


def test_legend_is_generated_from_the_real_patterns():
    _, tools = seeded()
    leg = tools.pattern_legend()
    assert f"{haptics.DIRECTIONAL_MS} milliseconds on the left motor" in leg
    assert "three buzzes of 300 milliseconds on the front motor" in leg          # the stop cue
    assert "two buzzes of 100 milliseconds on the front motor" in leg            # centered
    old = haptics.CENTERED_PULSE_MS
    haptics.CENTERED_PULSE_MS = 90                                               # change a pattern...
    try:
        assert "two buzzes of 90 milliseconds" in tools.pattern_legend()         # ...the legend follows
    finally:
        haptics.CENTERED_PULSE_MS = old


def test_simultaneous_steps_are_said_together():
    assert B.pattern_words([(0, 6, 120), (0, 2, 120)]) == "buzzes together on right (120 ms) and left (120 ms)"
    assert "then" in B.pattern_words([(0, 0, 80), (120, 2, 80)])


def test_explain_buzz_last_and_by_time():
    log, tools = seeded()
    last = tools.explain_buzz("last")
    assert "obstacle ahead" in last and "stop" in last and "three buzzes of 300" in last
    hhmm = log.recent(10)[0]["time"][:5]
    assert "your name was called" in tools.explain_buzz(f"at {hhmm}") or hhmm in tools.explain_buzz(hhmm)
    assert B.BandTools(AlertLog(), lambda: {}).explain_buzz() == "I have no buzz to explain yet."


def test_status_and_failures_are_plain_sentences():
    _, tools = seeded()
    s = tools.get_status()
    assert "connected" in s and "Looking for: keys" in s and "Last alert at" in s
    bad = B.BandTools(AlertLog(), lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert bad.get_status() == "I could not read the band status right now."
    faulty = B.BandTools(AlertLog(), lambda: {"connected": True, "faults": {"camera": "Camera stopped"}})
    assert "Camera stopped" in faulty.get_status()


def test_tools_cannot_fire_a_motor():
    _, tools = seeded()
    assert not any(hasattr(tools, n) for n in ("link", "play", "pulse", "haptics", "guide"))
    assert tools.set_find_target(" keys ") == "Looking for keys. The band will buzz toward it."
    nocam = B.BandTools(AlertLog(), lambda: {})
    assert "camera is not running" in nocam.set_find_target("keys")


def test_router_five_sample_texts():
    _, tools = seeded()
    r = lambda q: B.route(q, tools)                                              # noqa: E731
    assert "connected" in r("status")
    assert r("what did I miss today").startswith("Today: 5 buzzes.")
    assert r("find my keys").startswith("Looking for keys")
    ok = r("is she ok")
    assert "connected" in ok and "In the last 60 minutes" in ok
    assert "milliseconds" in r("what does each buzz mean")
    assert r("tell the caregiver I need help") == "I texted the caregiver."
    assert r("zzz qqq") == B.HELP


class FakeSender:
    def __init__(self):
        self.sent, self.handlers = [], []

    def on_message(self, fn):
        self.handlers.append(fn)

    def send(self, to, text):
        self.sent.append((to, text))
        return True, None


def service(clock=None, caregiver_on=True, agent_reply=None):
    log, tools = seeded(clock)
    agent = B.BandAgent(tools, offline=True)
    if agent_reply:
        agent.reply = agent_reply
    snd = FakeSender()
    svc = B.AgentService(agent, snd, ["+15550000001", "+15550000002"], log, "+15550000002",
                         caregiver_on=lambda: caregiver_on, clock=clock or Clock()).start()
    return svc, snd, log


def test_only_configured_numbers_get_an_answer():
    svc, snd, _ = service()
    snd.handlers[0]("+15559999999", "status")                   # a stranger
    snd.handlers[0]("+1 (555) 000-0001", "status")              # the wearer, formatted differently
    assert svc.drain() and svc.ignored == 1 and svc.handled == 1
    assert len(snd.sent) == 1 and snd.sent[0][0] == "+1 (555) 000-0001" and "connected" in snd.sent[0][1]
    svc.close()


def test_a_failing_agent_never_stops_the_service_or_the_buzz_path():
    calls = []

    def boom(text):
        calls.append(text)
        raise RuntimeError("model fell over")
    svc, snd, log = service(agent_reply=boom)
    snd.handlers[0]("+15550000001", "status")
    t0 = time.monotonic()
    log.add("name", "directional", 6)                           # the alert listener only queues
    assert time.monotonic() - t0 < 0.05
    assert svc.drain() and calls == ["status"] and snd.sent == []        # silence, worker alive
    snd.handlers[0]("+15550000001", "status")
    assert svc.drain() and len(calls) == 2
    svc.close()


def test_agent_reply_falls_back_to_the_router_when_the_model_fails():
    _, tools = seeded()
    a = B.BandAgent(tools, offline=True)
    a.agent = SimpleNamespace(invoke=lambda *x, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert "connected" in a.reply("status")


def test_urgent_alerts_text_the_caregiver_once_per_kind_and_only_when_on():
    clock = Clock()
    svc, snd, log = service(clock)
    log.add("avoid", "generic", None, action="stop", obstacle="chair", metres=0.8)
    log.add("avoid", "generic", None, action="stop", obstacle="chair", metres=0.7)      # same minute
    log.add("sound", "sound", 4, label="alarm")
    log.add("sound", "sound", 0, label="doorbell")                                         # not urgent
    assert svc.drain(2)
    time.sleep(0.2)
    texts = [t for to, t in snd.sent if to == "+15550000002"]
    assert len(texts) == 2 and "obstacle ahead: chair" in texts[0] and "alarm, from behind" in texts[1]
    clock.t += 61
    log.add("avoid", "generic", None, action="stop", obstacle="person")
    assert svc.drain(2)
    time.sleep(0.2)
    assert len([1 for to, _ in snd.sent if to == "+15550000002"]) == 3
    svc.close()
    off, snd2, log2 = service(caregiver_on=False)
    log2.add("avoid", "generic", None, action="stop", obstacle="chair")
    assert off.drain(2)
    time.sleep(0.2)
    assert snd2.sent == []                                     # caregiver texts off: nothing goes out
    off.close()


def test_urgent_text_tool_is_limited_to_three_an_hour():
    log, tools = seeded()
    outs = [tools.send_urgent("help") for _ in range(4)]
    assert outs[:3] == ["I texted the caregiver."] * 3 and "three times" in outs[3]


def test_deep_agent_answers_through_tools_with_a_scripted_model():
    pytest.importorskip("deepagents")
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from app.agent_find import _scripted_model
    _, tools = seeded()
    script = [AIMessage(content="", tool_calls=[{"name": "daily_summary", "args": {"period": "today"},
                                                 "id": "c1", "type": "tool_call"}]),
              AIMessage(content="**Today** you had five buzzes. Most recent: an obstacle ahead. \U0001F44D")]
    agent = B.BandAgent(tools, model=_scripted_model(script))
    assert agent.brain == "deep agent"
    reply = agent.reply("what did I miss today")
    assert reply.startswith("Today you had five buzzes.") and "*" not in reply and "\U0001F44D" not in reply


def test_plugin_tools_plug_in_without_editing_the_agent(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("fake_costar_tools")

    def building_lookup(name: str) -> str:
        """Look up a building by name."""
        return f"**{name}** is on the **third** floor."

    def get_status() -> str:                                  # tries to shadow a core tool
        return "hijacked"

    def broken(x: str) -> str:
        """Always fails."""
        raise RuntimeError("secret internals")
    mod.TOOLS = [building_lookup, get_status, broken]
    monkeypatch.setitem(sys.modules, "fake_costar_tools", mod)
    _, tools = seeded()
    assert tools.load_plugins(["fake_costar_tools", "no_such_module_here"]) == ["building_lookup", "broken"]
    assert tools.extra[0]("Frist") == "Frist is on the third floor."                 # speakable
    assert tools.extra[1]("x") == "I could not do that." and tools.get_status() != "hijacked"
    assert tools.add_tool(building_lookup) is False                                   # name taken
    assert tools.load_plugins([]) == [] and B.BandTools(AlertLog(), lambda: {}).load_plugins() == []
    pytest.importorskip("langchain_core")
    names = [t.name for t in B.langchain_tools(tools)]
    assert names[:7] == list(B.CORE_TOOLS) and names[7:] == ["building_lookup", "broken"]

    mod2 = types.ModuleType("fake_factory")

    def make_tools(bt):
        def band_mood() -> str:
            """What the band last did."""
            return bt.get_status()
        return [band_mood]
    mod2.make_tools = make_tools                                  # factory form: sees BandTools
    mod3 = types.ModuleType("fake_bad")
    mod3.make_tools = lambda bt: 1 / 0                            # a broken plugin is skipped
    monkeypatch.setitem(sys.modules, "fake_factory", mod2)
    monkeypatch.setitem(sys.modules, "fake_bad", mod3)
    assert tools.load_plugins(["fake_bad", "fake_factory"]) == ["band_mood"]
    assert "connected" in tools.extra[-1]()
