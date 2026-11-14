"""Offline checks for tools/costar_nav.py on a hand-made two-room floor plan:

    y=4  +-----------------+-----------------+
         |  B              |                 |
         |  Living Room    :door  Kitchen    |   door = opening at x=5, y 1.5..2.5
         |  A         C    :  D        E     |
    y=0  +-----------------+-----------------+
        x=0               x=5              x=10
"""
import pytest

from tools.costar_nav import (NavError, blocked, build_twin, find_pano, goal_panos, motor_for,
                              route_steps, segments_cross, shortest_path)

V = {"a": (0, 0), "b": (5, 0), "c": (10, 0), "d": (10, 4), "e": (5, 4),
     "f": (0, 4), "g": (5, 1.5), "h": (5, 2.5)}
EDGES = {  # id: (type, v1, v2)
    "bot_l": ("wall", "a", "b"), "bot_r": ("wall", "b", "c"),
    "right": ("wall", "c", "d"), "top_r": ("wall", "d", "e"),
    "top_l": ("wall", "e", "f"), "left": ("wall", "f", "a"),
    "div_lo": ("wall", "b", "g"), "div_hi": ("wall", "h", "e"),
    "door": ("opening", "g", "h"),
}


def model(door_type="opening"):
    edges = dict(EDGES, door=(door_type, "g", "h"))
    return {
        "name": "test house",
        "floors": [{
            "id": "F1",
            "vertices": [{"id": k, "position": {"x": x, "y": y}} for k, (x, y) in V.items()],
            "edges": [{"id": k, "type": t, "vertices": [{"id": u}, {"id": v}]}
                      for k, (t, u, v) in edges.items()],
        }],
        "rooms": [
            {"id": "R1", "floor": {"id": "F1"},
             "classifications": [{"label": "Living Room", "confidence": 1.0}],
             "boundary": {"edges": [{"id": e} for e in
                                    ("bot_l", "div_lo", "door", "div_hi", "top_l", "left")]}},
            {"id": "R2", "floor": {"id": "F1"},
             "classifications": [{"label": "Kitchen", "confidence": 0.9}],
             "boundary": {"edges": [{"id": e} for e in
                                    ("bot_r", "right", "top_r", "div_hi", "door", "div_lo")]}},
        ],
        "panoLocations": [
            {"id": "A", "label": "1", "position": {"x": 1, "y": 1, "z": 1.5}},
            {"id": "B", "label": "2", "position": {"x": 1, "y": 3.5, "z": 1.5}},
            {"id": "C", "label": "3", "position": {"x": 4, "y": 2, "z": 1.5}},
            {"id": "D", "label": "4", "position": {"x": 6, "y": 2, "z": 1.5}},
            {"id": "E", "label": "5", "position": {"x": 8.5, "y": 1, "z": 1.5}},
            {"id": "U", "label": "9", "position": {"x": 8.5, "y": 3, "z": 4.5}},  # upstairs
        ],
    }


def test_panos_land_in_their_rooms():
    t = build_twin(model())
    assert t.panos["A"].room == "R1" and t.panos["C"].room == "R1"
    assert t.panos["D"].room == "R2" and t.panos["E"].room == "R2"
    assert goal_panos(t, "kitchen") >= {"D", "E"}


def test_walls_block_but_doorway_does_not():
    t = build_twin(model())
    linked = {v for v, _ in t.links["A"]}
    assert "D" not in linked                      # A->D crosses the divider wall
    assert "D" in {v for v, _ in t.links["C"]}    # C->D goes through the door
    assert not t.links["U"]                       # other floor: never linked


def test_route_goes_through_the_door():
    t = build_twin(model())
    path = shortest_path(t, "A", goal_panos(t, "kitchen"))
    assert path[0] == "A" and path[-1] == "D"
    assert "C" in path                            # had to use the doorway


def test_closed_door_means_no_route():
    t = build_twin(model(door_type="wall"))
    assert shortest_path(t, "A", goal_panos(t, "kitchen")) is None


def test_steps_are_relative_to_the_wearer():
    t = build_twin(model())
    path = ["B", "C", "D"]          # B->C heads down-right, C->D heads straight right
    steps = route_steps(t, path, heading=90)   # wearer faces +x ("east")
    assert steps[0].motor == motor_for(t_bearing(t, "B", "C") - 90)
    assert steps[-1].room == "Kitchen"
    assert all(s.pulse.startswith("PULSE ") and s.pulse.endswith(" 200") for s in steps)
    # Default: wearer faces the first leg, so the first cue is "ahead".
    assert route_steps(t, path)[0].motor == 0


def test_motor_rounding():
    assert [motor_for(d) for d in (0, 44, 46, 90, 180, 270, -45, 359)] == [0, 1, 1, 2, 4, 6, 7, 0]


def test_segment_crossing_ignores_touching():
    assert segments_cross((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_cross((0, 0), (1, 1), (1, 1), (2, 0))   # shares an endpoint


def t_bearing(t, a, b):
    from tools.costar_nav import bearing
    pa, pb = t.panos[a], t.panos[b]
    return bearing((pa.x, pa.y), (pb.x, pb.y))


# --- review fixes -------------------------------------------------------------
def test_a_wall_stored_as_two_segments_blocks_at_their_shared_vertex():
    wall = [((5, 0), (5, 2)), ((5, 2), (5, 4))]          # one straight wall, split at (5, 2)
    assert blocked((4, 2), (6, 2), wall)                 # through the split point
    assert blocked((4, 1), (6, 3), wall)
    jamb = [((5, 0), (5, 1.5))]                          # wall end at a doorway
    assert not blocked((4, 1.5), (6, 1.5), jamb)         # grazing the jamb is fine
    corner = [((0, 0), (5, 0)), ((5, 0), (5, 4))]        # L corner, room inside x<5, y>0
    assert blocked((4, 1), (6, -1), corner)              # cuts through the corner
    assert not blocked((4, -1), (6, 1), corner)          # passes outside it


def test_unknown_and_railing_edges_block_openings_pass():
    for kind in ("railing", "window", "mystery"):
        assert shortest_path(build_twin(model(kind)), "A", goal_panos(build_twin(model(kind)), "kitchen")) is None
    for kind in ("opening", "OPENING", "door", "invisible"):
        t = build_twin(model(kind))
        assert shortest_path(t, "A", goal_panos(t, "kitchen")), kind
    assert build_twin(model()).edge_types == {"wall": 8, "opening": 1}


def test_a_merged_leg_cues_where_to_go_now_not_where_it_ends():
    """Sub-legs bending 12 degrees each (0, 12, 24, 36). Before: each compared with
    the previous one, so all merged and the cue was the LAST bearing (36 -> motor 1)
    while the wearer should walk straight ahead. Now bends are measured from the
    leg's first bearing: 0 and 12 merge (cue 0), 24 starts a new leg."""
    import math
    pts, x, y, b = [], 0.0, 0.0, 0.0
    for _ in range(5):
        pts.append((x, y))
        x += 1.5 * math.sin(math.radians(b))
        y += 1.5 * math.cos(math.radians(b))
        b += 12
    mdl = {"name": "arc", "floors": [], "rooms": [],
           "panoLocations": [{"id": f"P{i}", "label": str(i), "position": {"x": px, "y": py, "z": 0}}
                             for i, (px, py) in enumerate(pts)]}
    t = build_twin(mdl)
    path = ["P0", "P1", "P2", "P3", "P4"]
    steps = route_steps(t, path, heading=0)
    assert [s.motor for s in steps] == [0, 0] and len(steps) == 2
    assert route_steps(t, path, heading=-60)[0].motor == motor_for(60)


def test_room_matching_and_errors_dont_exit():
    t = build_twin(model())
    assert goal_panos(t, "kitchens") == goal_panos(t, "kitchen") and goal_panos(t, "") == set()
    with pytest.raises(NavError):
        find_pano(t, "garage")


def test_doorway_opening_on_a_wall_edge_is_a_gap():
    """Real models: one wall edge across the divider with a doorway opening on it."""
    m = model()
    fl = m["floors"][0]
    fl["edges"] = [e for e in fl["edges"] if e["id"] not in ("div_lo", "div_hi", "door")]
    fl["edges"].append({"id": "div", "type": "wall", "vertices": [{"id": "b"}, {"id": "e"}],
                        "openings": [{"type": "doorway", "width": 1.0, "relativeCenter": 0.5}]})
    for r in m["rooms"]:
        r["boundary"]["edges"] = [x for x in r["boundary"]["edges"]
                                  if x["id"] not in ("div_lo", "div_hi", "door")] + [{"id": "div"}]
    t = build_twin(m)
    assert shortest_path(t, "A", goal_panos(t, "kitchen")) is not None
    fl["edges"][-1]["openings"][0]["type"] = "window"     # a window is not a way through
    t = build_twin(m)
    assert shortest_path(t, "A", goal_panos(t, "kitchen")) is None


def test_matterport_neighbours_are_the_candidate_links():
    m = model()
    m["locations"] = [{"id": "A", "neighbors": ["B"]}, {"id": "B", "neighbors": ["A"]}]
    t = build_twin(m)
    assert {v for v, _ in t.links["A"]} == {"B"}          # only listed neighbours
    assert not t.links["C"]
