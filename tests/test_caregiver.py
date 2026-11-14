"""Caregiver texting: filtering, rate limit/digest, Photon bridge protocol
(fake bridge, no credentials), settings toggle and Recall command."""
import os
import shutil
import time

import pytest

from app import notify
from app.notify import Caregiver, DryRunSender, PhotonSender
from app.recall import AlertLog
from app.recall_bot import RecallBot
from app.settings import Settings

PHONE = "+15550001111"          # fictional test number


def setup(on=True, **kw):
    s = Settings()
    if on:
        s.set("caregiver_texts", "on")
    log = AlertLog()
    sender = DryRunSender()
    cg = Caregiver(s, sender, PHONE, **kw)
    log.subscribe(cg)
    return s, log, sender, cg


def wait(cond, t=2.0):
    end = time.monotonic() + t
    while time.monotonic() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_off_by_default_sends_nothing():
    s, log, sender, cg = setup(on=False)
    log.add("name", "directional", 5)
    time.sleep(0.1)
    assert sender.sent == [] and s.snapshot()["caregiver_texts"] == "off"


def test_name_found_and_sound_text_and_noise_is_filtered():
    s, log, sender, cg = setup(min_interval_s=0.0)
    log.add("name", "directional", 5)
    assert wait(lambda: len(sender.sent) == 1)
    assert sender.sent[0][0] == PHONE and "Name called from the back-left." in sender.sent[0][1]
    for args, kw in [(("replay", "directional", 5), {"dry": True}), (("manual", "generic"), {}),
                     (("find", "directional", 6), {"target": "banana"}), (("find", "lost"), {}),
                     (("find", "directional", 6), {"repeat": True}), (("find", "ack"), {})]:
        log.add(*args, **kw)
    log.add("find", "centered", 0, target="banana")
    log.add("find", "centered", 0, target="banana")      # same target again: once per 5 min
    assert wait(lambda: len(sender.sent) == 2)
    log.add("sound", "sound")
    log.add("name", "generic")
    assert wait(lambda: len(sender.sent) >= 3)
    time.sleep(0.2)
    texts = " | ".join(t for _, t in sender.sent)
    assert "found the banana" in texts.lower() and texts.lower().count("banana") == 1
    assert "taught sound heard" in texts.lower() and "direction unclear" in texts


def test_rate_limit_merges_into_digest_and_caps_per_hour():
    s, log, sender, cg = setup(min_interval_s=0.4, max_per_hour=2)
    log.add("name", "directional", 2)
    assert wait(lambda: len(sender.sent) == 1)
    log.add("name", "directional", 6)
    log.add("sound", "sound")
    time.sleep(0.15)
    assert len(sender.sent) == 1                          # held for the interval
    assert wait(lambda: len(sender.sent) == 2, 1.5)
    assert sender.sent[1][1].startswith("[Band] 2 alerts") and "from the left" in sender.sent[1][1]
    log.add("name", "directional", 0)
    time.sleep(0.8)
    assert len(sender.sent) == 2 and cg.status()["suppressed"] == 1


def test_unconfigured_phone_fails_clearly():
    s = Settings()
    cg = Caregiver(s, DryRunSender(), phone="5551234")   # not E.164
    r = cg.send_test()
    assert not r["ok"] and "CAREGIVER_PHONE" in r["error"] and cg.status()["to"] != "5551234"


def test_status_masks_phone():
    _, _, _, cg = setup()
    st = cg.status()
    assert PHONE not in str(st) and st["to"].endswith("11")


FAKE_BRIDGE = """
import readline from "node:readline";
process.stdout.write(JSON.stringify({ready: true}) + "\\n");
const rl = readline.createInterface({input: process.stdin});
rl.on("line", (l) => { const m = JSON.parse(l);
  const ok = !m.text.includes("FAIL");
  process.stdout.write(JSON.stringify(ok ? {id: m.id, ok: true} : {id: m.id, ok: false, error: "nope"}) + "\\n"); });
"""


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_photon_sender_speaks_the_bridge_protocol(tmp_path):
    b = tmp_path / "bridge.mjs"
    b.write_text(FAKE_BRIDGE)
    ps = PhotonSender(bridge=b, timeout_s=5)
    assert ps.send(PHONE, "hello") == (True, None)
    assert ps.send(PHONE, "please FAIL") == (False, "nope")
    ps.close()
    bad = tmp_path / "bad.mjs"
    bad.write_text('process.stdout.write(JSON.stringify({ready:false,error:"no creds"})+"\\n");')
    ok, err = PhotonSender(bridge=bad, timeout_s=5).send(PHONE, "x")
    assert not ok and "no creds" in err


def test_load_env_never_overrides(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# secrets\nCAREGIVER_PHONE=+15550002222\nHDB_TEST_KEEP='from-file'\n")
    monkeypatch.setenv("HDB_TEST_KEEP", "from-env")
    monkeypatch.delenv("CAREGIVER_PHONE", raising=False)
    notify.load_env(p)
    assert os.environ["CAREGIVER_PHONE"] == "+15550002222" and os.environ["HDB_TEST_KEEP"] == "from-env"


def test_recall_and_settings_toggle():
    s = Settings()
    bot = RecallBot(AlertLog(), s, "offline")
    assert "caregiver texts on" in bot.ask("turn on caregiver texts").text.lower()
    assert s.caregiver_texts
    bot.ask("stop texting my caregiver")
    assert not s.caregiver_texts


def test_app_wiring_and_test_text():
    from app.link import Link, SimPort
    from app.server import App
    link = Link(SimPort()).start()
    app = App(link)
    sender = DryRunSender()
    app.attach_caregiver(Caregiver(app.settings, sender, PHONE, min_interval_s=0.0))
    app.handle({"cmd": "setting", "setting": "caregiver_texts", "value": "on"})
    app.notify.cue("name", "directional", 7)
    assert wait(lambda: len(sender.sent) == 1)
    app.handle({"cmd": "caregiver_test"})
    assert wait(lambda: len(sender.sent) == 2) and "Test" in sender.sent[1][1]
    link.close()
