"""Alert log (app/recall.py) and the single cue path (app/notify.py)."""
import time
from types import SimpleNamespace as D

import pytest

from app import haptics, recall, trace
from app.guide import ObjectGuide
from app.link import Link, MicFrame, SimPort
from app.notify import Notifier
from app.server import App
from tests.test_names import FIX, feed


def test_log_writes_jsonl_and_labels(tmp_path):
    p = tmp_path / "alerts.jsonl"
    log = recall.AlertLog(p)
    seen = []
    log.subscribe(seen.append)
    a = log.add("name", "directional", 5, confidence=0.91234, latency_ms=412.6, reason="ok")
    b = log.add("name", "generic", reason="too few loud frames")
    assert a["direction"] == "back-left" and a["confidence"] == 0.912 and a["latency_ms"] == 413
    assert log.latest()["id"] == b["id"]
    log.label(b["id"], "false")
    assert log.latest()["id"] == a["id"]           # newest *unlabelled*
    log.label(None, 2, missed=True)
    alerts, labels = recall.load(p)
    assert [x["id"] for x in alerts] == [1, 2]
    assert labels[0] == {"type": "label", "ref": 2, "truth": "false", "t": labels[0]["t"]}
    assert labels[1]["missed"] and labels[1]["ref"] is None
    assert log.recent()[1]["truth"] == "false" and len(seen) == 4
    with pytest.raises(ValueError):
        log.label(1, 9)
    with pytest.raises(ValueError):
        log.add("tv", "generic")


def test_notifier_plays_and_logs():
    port = SimPort()
    link = Link(port).start()
    dry = []
    n = Notifier(link.pulse, dry_send=lambda m, ms: dry.append((m, ms)))
    n.cue("name", "directional", 6)
    n.cue("manual", "sound")
    n.cue("replay", "directional", 5, dry=True)
    time.sleep(0.5)
    link.close()
    assert port.sent[:5] == ["PULSE 6 200", "PULSE 0 80", "PULSE 2 80", "PULSE 4 80", "PULSE 6 80"]
    assert dry == [(5, 200)]
    assert [a["source"] for a in n.log.recent()] == ["name", "manual", "replay"]


def test_guide_reports_every_cue_with_latency():
    got = []
    g = ObjectGuide(lambda cue: None, repeat_s=0.5, on_cue=lambda k, m, i: got.append((k, m, i)))
    g.on_decision(D(state="left", confidence=0.8, target="banana", t=10.0), now=10.05)
    g.tick(now=10.6)
    g.on_decision(D(state="centered", confidence=0.9, target="banana", t=11.0), now=11.02)
    g.on_decision(D(state="lost", confidence=0.0, target="banana", t=12.0), now=12.0)
    assert [(k, m, i["repeat"]) for k, m, i in got] == [
        ("directional", 6, False), ("directional", 6, True), ("centered", None, False), ("lost", None, False)]
    assert got[0][2]["latency_ms"] == pytest.approx(50) and got[1][2]["latency_ms"] is None
    assert got[2][2]["confidence"] == 0.9 and got[2][2]["target"] == "banana"


def test_app_logs_name_find_manual_and_labels(tmp_path):
    port = SimPort(frames=[])
    link = Link(port).start()
    log = recall.AlertLog(tmp_path / "a.jsonl")
    app = App(link, log)
    # name alert from a back-left trace
    t = trace.load(FIX / "back_left.json")
    end = time.monotonic()
    base = end - t["frames"][-1]["ms"] / 1000 - 0.1
    feed(app.live, t, base)
    app.on_name_hit(D(name="Sam", heard="Sam", start=base + 1.0, end=base + 1.45, latency=0.3))
    # find cues from the guide
    from tests.test_server import FakeVision
    fv = FakeVision()
    guide = ObjectGuide(lambda cue: haptics.play(cue, link.pulse))
    app.attach_vision(fv, guide)
    fv.cb(D(state="right", target="banana", confidence=0.7, t=time.monotonic()))
    app.handle({"cmd": "cue", "name": "generic"})
    time.sleep(0.1)
    items = log.recent()
    assert [(a["source"], a["cue"], a["motor"]) for a in items] == [
        ("name", "directional", 5), ("find", "directional", 2), ("manual", "generic", None)]
    assert items[0]["window"] and items[0]["latency_ms"] >= 0 and items[0]["confidence"] > 0.5
    assert items[1]["target"] == "banana"
    assert "PULSE 5 200" in port.sent and "PULSE 2 200" in port.sent
    r = app.handle({"cmd": "label", "truth": "5"})       # newest unlabelled non-manual = find
    assert "#2" in r["text"]
    app.handle({"cmd": "label", "truth": "5", "id": 1})
    app.handle({"cmd": "missed", "truth": "4"})
    _, labels = recall.load(tmp_path / "a.jsonl")
    assert [(l["ref"], l["truth"]) for l in labels] == [(2, 5), (1, 5), (None, 4)]
    link.close()
