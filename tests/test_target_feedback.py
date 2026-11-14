"""Hold-to-talk target phrasing and wearer feedback cues."""
import time
from types import SimpleNamespace as R

import pytest

from app import haptics
from app.guide import ObjectGuide
from app.link import Link, SimPort
from app.server import App
from app.speech import match_target
from tests.test_server import FakeVision


@pytest.mark.parametrize("text,want", [
    ("find my glasses", "glasses"), ("where are my reading glasses", "reading glasses"),
    ("find my sunglasses", "sunglasses"), ("find the wine glasses", "wine glass"),
    ("where are my headphones?", "headphones"), ("hello there", None), ("find", None),
    ("find a glass of water", "cup"), ("find my coffee", "cup"),
    ("find my coffee mug", "cup"), ("find my MacBook", "laptop"),
    ("find my glasses and my phone", "cell phone"), ("find the banana", "banana"),
])
def test_match_target_phrasing(text, want):
    assert match_target(text) == want


class KnownVision(FakeVision):
    def set_target(self, name):
        self.target = name
        return name if name in ("banana", "cup", "cell phone") else None


def run(results, typed=()):
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    app.attach_vision(KnownVision(), ObjectGuide(lambda cue: None))
    for r in results:
        app._on_speech(r)
        time.sleep(0.45)                    # let each pattern finish
    for t in typed:
        app.handle({"cmd": "target", "name": t})
        time.sleep(0.45)
    link.close()
    return port.sent[:-1], [(a["cue"], a.get("reason")) for a in app.alerts.recent()]


ACK = [f"PULSE {m} {ms}" for _, m, ms in sorted(haptics.target_ack())]
UNK = [f"PULSE {m} {ms}" for _, m, ms in sorted(haptics.target_unknown())]


def test_spoken_target_gets_ack_and_unknowns_get_distinct_cue():
    sent, log = run([R(text="Find the banana.", target="banana", error=None),
                     R(text="Find my glasses.", target=None, error=None),
                     R(text="", target=None, error="too short")])
    assert sent == ACK + UNK + UNK
    assert log == [("ack", None), ("unknown", "no known object"), ("unknown", "too short")]
    assert set(ACK).isdisjoint(UNK)


def test_typed_target_feedback_and_clear_is_silent():
    sent, log = run([], typed=["cup", "spaceship", ""])
    assert sent == ACK + UNK and [c for c, _ in log] == ["ack", "unknown"]


def test_feedback_respects_find_mute_and_intensity():
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    app.attach_vision(KnownVision(), ObjectGuide(lambda cue: None))
    app.settings.set("intensity", "high")
    app._on_speech(R(text="find the cup", target="cup", error=None))
    time.sleep(0.2)
    app.settings.set("find_cues", "off")
    app._on_speech(R(text="find the cup", target="cup", error=None))
    time.sleep(0.2)
    link.close()
    assert sorted(port.sent[:-1]) == ["PULSE 2 180", "PULSE 6 180"]
    assert app.alerts.recent()[-1]["muted"] is True
