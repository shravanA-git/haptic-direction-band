"""tools/costar_kit.py on a tiny synthetic kit zip (made-up tenants only; the
real kit's tenant names are real businesses and never go in the repo), plus
tenant routing in --nav route mode."""
import json
import zipfile

import pytest

from tools import costar_kit, costar_nav

PID = 777001


def make_kit(path):
    props = [{"id": PID, "buildingName": "Test Plaza", "numStoriesAboveGrade": 22,
              "parkingRatio": 2.5,
              "primaryAddress": {"streetName": "1 Test Plaza", "city": "Los Angeles", "state": "CA",
                                 "latitude": 34.05, "longitude": -118.25},
              "floors": [{"propertyFloorId": 9001, "floorName": 35}]},
             {"id": 777002, "buildingName": "Other Tower"}]
    occ = [
        {"id": 1, "propertyId": PID, "toDate": None, "tenantLocationId": 11,
         "spaces": [{"suite": "301", "propertyFloorId": 9001}]},
        {"id": 2, "propertyId": PID, "toDate": None, "tenantLocationId": 12,
         "spaces": [{"suite": "Suite 1008", "propertyFloorId": 9001}]},
        {"id": 3, "propertyId": PID, "toDate": None, "tenantLocationId": 13,
         "spaces": [{"suite": "PH", "propertyFloorId": 9001}]},          # floor from floorName
        {"id": 4, "propertyId": PID, "toDate": "2019-01-01", "tenantLocationId": 14,
         "spaces": [{"suite": "500"}]},                                  # moved out
        {"id": 5, "propertyId": 777002, "toDate": None, "tenantLocationId": 15,
         "spaces": [{"suite": "1008"}]},                                 # other building
    ]
    comp = [{"id": 11, "name": "Acme Dental Group"}, {"id": 12, "name": "Blue Harbor Insurance, Inc."},
            {"id": 13, "name": "Northwind Legal LLP"}, {"id": 14, "name": "Old Tenant Co"},
            {"id": 15, "name": "Elsewhere Corp"}]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("kit/commercial-properties.csv",
                   "PropertyId,MatterportModelId,Address,City,State\n"
                   f"777002,OTHERmodel1,2 Other St,LA,CA\n{PID},TESTmodel01,1 Test Plaza,Los Angeles,CA\n")
        z.writestr("kit/data/commercial-property.jsonl", "\n".join(json.dumps(p) for p in props) + "\n")
        z.writestr("kit/data/location-occupancy.jsonl", "\n".join(json.dumps(o) for o in occ) + "\n")
        z.writestr("kit/data/company-detail.jsonl", "\n".join(json.dumps(c) for c in comp) + "\n")
    return path


@pytest.fixture
def kit(tmp_path, monkeypatch):
    monkeypatch.setattr(costar_kit, "CACHE_DIR", tmp_path / "cache")
    zp = make_kit(tmp_path / "kit.zip")
    monkeypatch.setenv("COSTAR_KIT_ZIP", str(zp))
    monkeypatch.setenv("COSTAR_ZIP_PASSWORD", "unused-for-a-plain-zip")
    return zp


def test_build_reads_the_property_and_current_tenants(kit):
    d = costar_kit.build("TESTmodel01", "Test Plaza, Suite 301")
    assert d["building"] == "Test Plaza" and d["stories"] == 22 and d["scan_suite"] == "301"
    names = {t["name"]: t for t in d["tenants"]}
    assert set(names) == {"Acme Dental Group", "Blue Harbor Insurance, Inc.", "Northwind Legal LLP"}
    assert names["Blue Harbor Insurance, Inc."]["suite"] == "1008"
    assert names["Blue Harbor Insurance, Inc."]["floor"] == 10          # from the suite number
    assert names["Northwind Legal LLP"]["floor"] == 35                  # no number: floorName
    assert costar_kit.briefing(d) == "Test Plaza, 22 floors, 3 tenants. This scan is Suite 301."
    with pytest.raises(costar_kit.KitError):
        costar_kit.build("NOTaCoStar1", "")


def test_cache_works_offline(kit):
    costar_kit.build("TESTmodel01", "Test Plaza, Suite 301")
    kit.unlink()                                             # kit gone, wifi off
    d = costar_kit.load("TESTmodel01", "Test Plaza, Suite 301")
    assert d and d["tenant_count"] == 3
    assert costar_kit.load("UNKNOWNmod1", "") is None        # no cache, no kit: silent


@pytest.mark.parametrize("phrase,want", [
    ("suite 1008", ("Blue Harbor Insurance, Inc.", "1008", False)),
    ("1008", ("Blue Harbor Insurance, Inc.", "1008", False)),
    ("blue harbor", ("Blue Harbor Insurance, Inc.", "1008", False)),
    ("blue harbour insurance", ("Blue Harbor Insurance, Inc.", "1008", False)),
    ("acme dental", ("Acme Dental Group", "301", True)),
    ("suite 301", ("Acme Dental Group", "301", True)),
    ("suite 999", (None, "999", False)),
    ("old tenant", None),                                    # moved out
    ("elsewhere corp", None),                                # other building
    ("kitchen", None),
])
def test_resolve(kit, phrase, want):
    d = costar_kit.build("TESTmodel01", "Test Plaza, Suite 301")
    t = costar_kit.resolve(phrase, d)
    assert (t if t is None else (t["name"], t["suite"], t["in_scan"])) == want


def test_honest_answer_text(kit):
    d = costar_kit.build("TESTmodel01", "Test Plaza, Suite 301")
    t = costar_kit.resolve("blue harbor", d)
    assert costar_kit.answer(t, d) == ("Blue Harbor Insurance, Inc. is in suite 1008 on floor 10; "
                                       "this scan covers suite 301 only.")
    assert costar_kit.answer(costar_kit.resolve("suite 999", d), d) == \
        "Suite 999 on floor 9 has no tenant listed; this scan covers suite 301 only."


# ---------------------------------------------------------------- in the app
def app_with_costar(kit):
    from app import haptics
    from app.link import Link, SimPort
    from app.nav import RoutePlayer
    from app.server import App
    from tests.test_costar_nav import model
    port = SimPort(frames=[])
    link = Link(port).start()
    app = App(link)
    said = []
    m = model()
    m["rooms"][0]["classifications"][0]["label"] = "Office"            # Living Room -> Office
    rp = RoutePlayer(costar_nav.build_twin(m), play=lambda c: app.play_cue(c, "nav"),
                     speak=said.append, threaded=False)
    app.attach_route(rp, {"name": "x"}, costar=costar_kit.build("TESTmodel01", "Test Plaza, Suite 301"))
    return app, rp, said, link, haptics


def test_tenant_elsewhere_gets_the_honest_answer_and_no_route(kit):
    from app.speech import SpeechResult
    app, rp, said, link, haptics = app_with_costar(kit)
    app._on_speech(SpeechResult("Take me to Blue Harbor Insurance.", None))
    assert rp.goal is None and rp.steps == []                       # never an invented route
    assert said == ["Blue Harbor Insurance, Inc. is in suite 1008 on floor 10; "
                    "this scan covers suite 301 only."]
    app._on_speech(SpeechResult("take me to suite 1008", None))
    assert rp.goal is None and said[-1].endswith("this scan covers suite 301 only.")
    link.close()


def test_the_scanned_tenant_routes_inside_the_scan_and_rooms_still_win(kit):
    app, rp, said, link, haptics = app_with_costar(kit)
    out = app.route_go("D", "acme dental")                          # start in the kitchen
    assert rp.goal == "office" and rp.steps, out
    out = app.route_go("A", "kitchen")                              # a room is still a room
    assert rp.goal == "kitchen"
    rp.play(now=0.0)
    assert said[0].startswith("Test Plaza, 22 floors, 3 tenants. This scan is Suite 301.")
    assert app.briefing_line() == "Test Plaza, 22 floors, 3 tenants. This scan is Suite 301."
    link.close()


def test_route_request_takes_suite_numbers():
    from app.nav import route_request
    assert route_request("take me to suite 1008") == "suite 1008"
    assert route_request("Go to Blue Harbor Insurance, Inc.") == "blue harbor insurance, inc"
