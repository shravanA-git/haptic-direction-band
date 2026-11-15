"""tools/label_places.py with a mocked Claude client (no network): the request
shape, the agentic pass (retry once, smoothing, grouping), manual fixes that
persist, label overrides in routing, signage search and the other-floor rule."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import costar_assets, costar_nav, label_places as lp

MODEL = "VENUEtest01"
POS = {"A": (0, 0, 1.5), "B": (0, 3, 1.5), "C": (0, 6, 1.5), "H2": (0, 9, 1.5), "W": (3, 9, 1.5),
       "D": (3, 6, 1.5), "S": (-3, 9, 1.5), "E": (0, 0, 5.5)}
NB = {"A": ["B", "W"], "B": ["A", "C"], "C": ["B", "H2", "D"], "H2": ["C", "W", "S"],
      "W": ["H2", "A"], "D": ["C"], "S": ["H2"], "E": []}
FLOOR = {k: "F2" for k in POS} | {"E": "F3"}
# what the fake Claude "sees": first look, then the closer look
ANSWERS = {
    "A": [("hallway", 0.9)], "B": [("hallway", 0.3), ("hallway", 0.8)],
    "C": [("classroom", 0.6), ("hallway", 0.85)], "H2": [("hallway", 0.9)],
    "W": [("other", 0.2), ("other", 0.2)], "D": [("classroom", 0.9)],
    "S": [("stairs", 0.95)], "E": [("lecture hall", 0.9)],
}


@pytest.fixture
def venue(tmp_path, monkeypatch):
    for mod in (costar_nav, costar_assets, lp):
        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
    model = {"id": MODEL, "name": "Test Hall", "floors": [
        {"id": "F2", "label": "Floor 2", "vertices": [], "edges": []},
        {"id": "F3", "label": "Floor 3", "vertices": [], "edges": []}],
        "rooms": [{"id": "V", "classifications": [{"label": "Void"}], "boundary": {"edges": []}}],
        "panoLocations": [{"id": k, "label": k, "position": {"x": x, "y": y, "z": z}}
                          for k, (x, y, z) in POS.items()],
        "locations": [{"id": k, "neighbors": v, "floor": {"id": FLOOR[k]}} for k, v in NB.items()]}
    (tmp_path / f"{MODEL}.json").write_text(json.dumps(model))
    import cv2
    d = tmp_path / MODEL / "pano"
    d.mkdir(parents=True)
    for k in POS:
        for f in (1, 2, 3, 4):
            cv2.imwrite(str(d / f"{k}_{f}.jpg"), np.full((16, 16, 3), 128, np.uint8))
    return model


class FakeMessages:
    def __init__(self):
        self.requests = []
        self.seen: dict = {}

    def create(self, **kw):
        self.requests.append(kw)
        content = kw["messages"][0]["content"]
        text = content[-1]["text"]
        pano = self.current
        n = 1 if "Look again" in text else 0
        place, conf = ANSWERS[pano][min(n, len(ANSWERS[pano]) - 1)]
        out = {"place": place, "name": f"{place} near {pano}", "confidence": conf,
               "signs": ["Room 101"] if pano == "D" else []}
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(out))])


class FakeClient:
    def __init__(self):
        self.beta = SimpleNamespace(messages=FakeMessages())


class FakeLabeller(lp.Labeller):
    """Tells the fake which scan point the photos belong to (from the file name)."""
    def ask(self, paths, text):
        self.client.beta.messages.current = paths[0].name.rsplit("_", 1)[0]
        return super().ask(paths, text)


def test_one_request_per_look_with_photos_schema_and_fallback(venue):
    client = FakeClient()
    lab = FakeLabeller(client, model="claude-sonnet-5-5")
    out = lab.ask(costar_assets.pano_faces(MODEL, "D"), "what is this")
    assert out == {"place": "classroom", "name": "classroom near D", "signs": ["Room 101"],
                   "confidence": 0.9}
    req = client.beta.messages.requests[0]
    assert req["model"] == "claude-sonnet-5-5"
    assert [b["type"] for b in req["messages"][0]["content"]] == ["image"] * 4 + ["text"]
    assert req["output_config"]["format"]["schema"]["properties"]["place"]["enum"] == lp.PLACES
    assert req["fallbacks"] == "default" and req["betas"] == ["server-side-fallback-2026-07-01"]
    haiku = FakeLabeller(FakeClient(), model="claude-haiku-5-5")
    haiku.ask(costar_assets.pano_faces(MODEL, "D"), "x")
    assert "fallbacks" not in haiku.client.beta.messages.requests[0]


def test_refusals_and_bad_json_are_skipped(venue):
    client = FakeClient()
    client.beta.messages.create = lambda **kw: SimpleNamespace(stop_reason="refusal", content=[])
    assert lp.Labeller(client, "claude-sonnet-5-5").ask([], "x") is None
    client.beta.messages.create = lambda **kw: SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"place": "spaceship"}')])
    assert lp.Labeller(client, "claude-sonnet-5-5").ask([], "x") is None


def test_agentic_pass_retries_once_smooths_and_groups(venue):
    lab = FakeLabeller(FakeClient(), model="claude-haiku-5-5")
    data = lp.label_model(MODEL, lab, log=lambda s: None)
    p = data["panos"]
    assert lab.calls == 8 + 3 and data["retries"] == 3            # B (low), C (conflict), W (low)
    assert lab.calls <= 2 * len(POS)                              # cost cap: <= 1 retry per pano
    assert p["B"]["place"] == "hallway" and p["B"]["retried"]
    assert p["C"]["place"] == "hallway" and p["C"]["confidence"] == 0.85      # closer look won
    retry_prompt = lab.client.beta.messages.requests[-2]["messages"][0]["content"][-1]["text"]
    assert "Look again" in retry_prompt
    assert p["W"]["place"] == "hallway" and p["W"]["smoothed"]               # neighbours decided
    assert p["E"]["floor"] == 3 and p["A"]["floor"] == 2
    names = {d["name"]: d for d in data["destinations"]}
    assert set(names) == {"Hallway (floor 2)", "Classroom (floor 2)", "Stairs (floor 2)",
                          "Lecture hall (floor 3)"}
    assert sorted(names["Hallway (floor 2)"]["panos"]) == ["A", "B", "C", "H2", "W"]
    assert names["Classroom (floor 2)"]["signs"] == ["Room 101"]
    assert "Hallway (floor 2)" in lp.summary(data)


def test_manual_fix_persists_and_wins(venue):
    lp.label_model(MODEL, FakeLabeller(FakeClient(), "claude-haiku-5-5"), log=lambda s: None)
    lp.set_manual(MODEL, "D", "lab", "robotics lab")
    data = lp.label_model(MODEL, FakeLabeller(FakeClient(), "claude-haiku-5-5"), log=lambda s: None)
    assert data["manual"]["D"] == {"place": "lab", "name": "robotics lab"}
    assert lp.effective(data)["D"]["place"] == "lab"
    assert "robotics lab" in {d["name"] for d in data["destinations"]}
    with pytest.raises(ValueError):
        lp.set_manual(MODEL, "D", "spaceship")


def test_labels_make_rooms_routable_and_signs_searchable(venue):
    data = lp.label_model(MODEL, FakeLabeller(FakeClient(), "claude-haiku-5-5"), log=lambda s: None)
    twin = costar_nav.apply_labels(costar_nav.build_twin(venue), data)
    assert costar_nav.goal_panos(twin, "classroom") == {"D"}
    assert costar_nav.goal_panos(twin, "room 101") == {"D"} == costar_nav.goal_panos(twin, "101")
    assert costar_nav.goal_panos(twin, "lecture hall") == {"E"}
    assert costar_nav.shortest_path(twin, "A", costar_nav.goal_panos(twin, "room 101"))[-1] == "D"


def test_goal_on_another_floor_goes_to_the_stairs_and_says_so(venue):
    from app.nav import RoutePlayer
    data = lp.label_model(MODEL, FakeLabeller(FakeClient(), "claude-haiku-5-5"), log=lambda s: None)
    twin = costar_nav.apply_labels(costar_nav.build_twin(venue), data)
    said = []
    rp = RoutePlayer(twin, play=lambda c: None, speak=said.append, threaded=False)
    ok, msg = rp.plan("A", "lecture hall")
    assert ok and msg.startswith("That's on floor 3; take the stairs")
    assert rp.legs[-1]["end"] == "S"
    rp.play(now=0.0)
    t = 0.0
    while not rp.done and t < 60:
        t += 0.1
        rp.tick(t)
    assert said[0].startswith("That's on floor 3; take the stairs.")
    assert said[-1] == "You're at the stairs (floor 2). That's on floor 3; take the stairs."


def test_no_stairs_labelled_gives_the_honest_answer_only(venue):
    from app.link import Link, SimPort
    from app.nav import RoutePlayer
    from app.server import App
    lp.label_model(MODEL, FakeLabeller(FakeClient(), "claude-haiku-5-5"), log=lambda s: None)
    data = lp.set_manual(MODEL, "S", "office", "storage")                  # no stairs any more
    link = Link(SimPort(frames=[])).start()
    app = App(link)
    said = []
    rp = RoutePlayer(costar_nav.apply_labels(costar_nav.build_twin(venue), data),
                     play=lambda c: None, speak=said.append, threaded=False)
    app.attach_route(rp, {"name": "Test Hall"})
    out = app.route_go("A", "lecture hall")
    assert rp.goal is None and said == ["That's on floor 3; take the stairs."], out
    link.close()


def test_fix_label_from_the_page_rebuilds_the_rooms(venue):
    from app.link import Link, SimPort
    from app.nav import RoutePlayer
    from app.server import App
    data = lp.label_model(MODEL, FakeLabeller(FakeClient(), "claude-haiku-5-5"), log=lambda s: None)
    link = Link(SimPort(frames=[])).start()
    app = App(link)
    app.nav_info, app.nav_model = {"model": MODEL}, venue
    rp = RoutePlayer(costar_nav.apply_labels(costar_nav.build_twin(venue), data),
                     play=lambda c: None, threaded=False)
    app.attach_route(rp, {"name": "Test Hall"})
    sent = []
    app.emit = sent.append
    out = app.handle({"cmd": "label_fix", "pano": "D", "place": "office", "name": "dean's office"})
    assert "dean's office" in out["text"] and {"type": "labels_changed"} in sent
    assert costar_nav.goal_panos(rp.twin, "dean's office") == {"D"}
    assert lp.load_labels(MODEL)["manual"]["D"]["name"] == "dean's office"
    link.close()
