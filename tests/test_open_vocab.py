"""Open-vocabulary find without loading any model: phrase extraction, the
COCO-first routing in Vision.set_target, and the non-blocking dashboard lookup."""
import time

import pytest

from app import vision as V
from app.vision import Vision, clean_phrase, resolve_target


@pytest.mark.parametrize("text,want", [
    ("where are my headphones?", "headphones"),
    ("find the red umbrella please", "red umbrella"),
    ("Find my glasses.", "glasses"),
    ("where's my water bottle", "water bottle"),
    ("can you find my AirPods", "airpods"),
    ("look for my keys and my phone", "keys"),
    ("find", None),
])
def test_clean_phrase(text, want):
    assert clean_phrase(text) == want


def test_speech_needs_a_trigger_but_typed_text_does_not():
    assert clean_phrase("hello there", require_trigger=True) is None
    assert clean_phrase("headphones") == "headphones"


def test_glasses_are_not_a_cup():
    assert resolve_target("my glasses") is None and resolve_target("wine glass") == "wine glass"


class StubOpenVocab:
    def __init__(self, *a, **k):
        self.phrase, self.calls = None, []

    def set_phrase(self, p):
        self.calls.append(p)
        self.phrase = p

    def detect(self, frame):
        return []


class StubDetector:
    device, imgsz, open_vocab = "cpu", 640, None


def make_vision(monkeypatch, ready=True):
    monkeypatch.setattr(V, "OpenVocab", StubOpenVocab)
    monkeypatch.setattr(V, "open_vocab_ready", lambda: None if ready else "no clip")
    v = Vision(camera=0)
    v.detector = StubDetector()
    return v


def test_coco_first_then_open_phrase(monkeypatch):
    v = make_vision(monkeypatch)
    assert v.set_target("my phone") == "cell phone" and v.tracker.min_score is None
    assert v.detector.open_vocab is None                       # COCO never touches YOLO-World
    assert v.set_target("where are my headphones") == "headphones"
    assert v.detector.open_vocab.calls == ["headphones"]
    assert v.tracker.target == "headphones" and v.tracker.min_score == V.OPEN_MIN_SCORE
    assert v.set_target(None) is None and v.tracker.target is None


def test_open_vocab_unavailable_means_cant_look(monkeypatch):
    v = make_vision(monkeypatch, ready=False)
    assert v.set_target("headphones") is None and v.tracker.target is None


def test_dashboard_open_lookup_does_not_block_and_buzzes_ack():
    from app.guide import ObjectGuide
    from app.link import Link, SimPort
    from app.server import App
    from tests.test_server import FakeVision

    class SlowOpen(FakeVision):
        def set_target(self, name):
            time.sleep(0.5)                       # CLIP text encoding takes seconds
            self.target = name
            return name

    link = Link(SimPort()).start()
    app = App(link)
    app.attach_vision(SlowOpen(), ObjectGuide(lambda cue: None))
    t = time.monotonic()
    reply = app.handle({"cmd": "target", "name": "headphones"})
    assert time.monotonic() - t < 0.2 and "looking for" in reply["text"]
    time.sleep(1.0)
    link.close()
    assert [a["cue"] for a in app.alerts.recent()] == ["ack"]


def test_expand_phrase_adds_working_prompts():
    assert V.expand_phrase("cap")[:2] == ["cap", "hat"]          # "cap" alone ~never fires
    assert "snack pack" in V.expand_phrase("fruit snacks")
    assert V.expand_phrase("Headphones") == ["headphones", "headset", "headphone"]
    assert V.expand_phrase("umbrella stand") == ["umbrella stand", "umbrella stands"]
    assert V.expand_phrase("glass") == ["glass"]                 # not "glasses" (eyewear)
    assert len(V.expand_phrase("charger")) <= 4


def test_resume_keeps_open_floor_and_confirm(monkeypatch):
    v = make_vision(monkeypatch)
    monkeypatch.setattr(V, "OPEN_CONFIRM", 2)
    v.set_target("headphones")
    v.pause()
    v.resume()                                                   # Action released
    assert v.tracker.min_score == V.OPEN_MIN_SCORE and v.tracker.confirm == 2


def test_confirm_needs_hits_in_a_row():
    tr = V.TargetTracker()
    tr.set_target("cap", 0.1, confirm=2)
    box = [(300, 100, 340, 200, 0.5)]
    assert tr.update(box, 640, 0.0).state == "searching"        # one frame: not yet
    assert tr.update([], 640, 0.1).state == "searching"         # streak broken
    assert tr.update(box, 640, 0.2).state == "searching"
    assert tr.update(box, 640, 0.3).state == "centered"         # two in a row


@pytest.mark.parametrize("text,want", [
    ("where did I put my headphones", "headphones"),     # was "did put headphones"
    ("er where the heck is my wallet", "wallet"),         # was "heck wallet"
    ("I lost my AirPods", "airpods"), ("have you seen my keys", "keys"),
    ("show me the glasses", "glasses"), ("take me to my hat", "hat"),
    ("get my charger", "charger"), ("hmm I need my pen", "pen"),
    ("keys please", "keys"), ("hat", "hat"), ("where'd my keys go", "keys"),
    ("I misplaced my wallet", "wallet"), ("do you see my glasses anywhere", "glasses"),
    ("hello there", None), ("what time is it", None), ("I see you", None),
    ("um okay", None), ("show me", None), ("I need a break", None),
])
def test_spoken_requests_without_find(text, want):
    from app.speech import match_target
    assert match_target(text) == want


def test_top_edge_sliver_is_not_a_target():
    tr = V.TargetTracker()
    tr.set_target("cap", 0.1)
    hair = [(200, 0, 330, 60, 0.3)]                              # 60 px of 480 at the top edge
    assert tr.update(hair, 640, 0.0, frame_height=480).state == "searching"
    held = [(200, 0, 330, 200, 0.3)]                             # a real cap held high
    assert tr.update(held, 640, 0.1, frame_height=480).state != "searching"
