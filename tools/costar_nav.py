"""Route planner over a Matterport digital twin (CoStar track prototype).

Standalone: it does not import or change anything in `app/`. It prints the
route and the PULSE commands the band would get; it never opens the serial port.

    # live (keys from /costar-keys in the track Discord, as env vars):
    export MATTERPORT_API_KEY=... MATTERPORT_API_TOKEN=...
    python -m tools.costar_nav zvKHR9DHNjJ --to kitchen

    # offline, from a cached or hand-made model file:
    python -m tools.costar_nav --file mp_cache/zvKHR9DHNjJ.json --to kitchen

How it works:
  1. Fetch the model's scan positions (panos), wall graph and labelled rooms.
  2. Link two panos when they're on the same floor, close enough, and the
     straight line between them crosses no wall. Doorways are gaps in the
     wall graph, so routes go through them.
  3. Shortest path (Dijkstra) from the start pano to any pano inside a room
     whose label matches --to.
  4. Each leg becomes a direction relative to where the wearer faces, rounded
     to one of the 8 motors (0 = ahead, 2 = right, 4 = behind, 6 = left).

Model frame: metres, z up. We take +y as "north" in the model frame; only
relative turns matter, so the choice doesn't affect the motor numbers.
Limits: single floor per route (stairs aren't in the data), no live position;
the wearer's position comes from the 3D tour (Showcase SDK) in the demo
(app/nav.py, `python -m app.main --nav <model>`).

Edge types: only types in PASSABLE (opening, door, ...) let a route through;
anything else, including unknown types, blocks. NOT verified against a real
Model API response yet: run --rooms on a real model and check the printed
edge-type counts and link count before trusting routes.
"""
from __future__ import annotations

import argparse
import base64
import heapq
import json
import math
import os
import sys
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def _ssl_context() -> ssl.SSLContext:
    """python.org Python on macOS ships without a CA store, so HTTPS fails with
    CERTIFICATE_VERIFY_FAILED. Use certifi's bundle when it is installed."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

API_URL = "https://api.matterport.com/api/models/graph"
CACHE_DIR = Path(__file__).resolve().parent.parent / "mp_cache"   # gitignored

MAX_LINK_M = 4.5      # don't link panos further apart than this (no sweep graph)
NEIGHBOUR_MAX_M = 12  # cap for Matterport's own neighbour links
SAME_FLOOR_DZ = 1.0   # panos within this height difference are on one floor
STEP_PULSE_MS = 200   # direction cue length (protocol: 200 ms)
MERGE_DEG = 20        # merge consecutive legs that bend less than this
VERTEX_EPS_M = 0.02   # a link passing this close to a wall vertex "touches" it
# Edge types a route may pass through (lower-cased). Everything else blocks,
# including types we don't know (a railing or window must not become a route).
PASSABLE = {"opening", "door", "doorway", "invisible", "virtual"}

DIRS = ["ahead", "ahead-right", "right", "behind-right",
        "behind", "behind-left", "left", "ahead-left"]

QUERY = """
query Nav($id: ID!) { model(id: $id) {
  id name
  address { address }
  geolocation { lat long }
  floors { id label
           vertices { id position { x y } }
           edges { id type vertices { id } openings { type width relativeCenter } } }
  rooms { id floor { id } classifications { label confidence }
          boundary { edges { id } } }
  panoLocations { id label position { x y z } }
  locations { id neighbors floor { id } room { id } pano { id } }
} }
"""
# Real models store doors as openings ON a wall edge, not as separate edges.
# Only doorways become gaps; windows keep the wall solid.
DOOR_OPENINGS = {"doorway", "door", "opening"}
DOOR_MARGIN_M = 0.05   # widen each doorway gap a little on both sides


def _cut_doorways(seg: Seg, openings: list[dict]) -> list[Seg]:
    """Split a wall segment into the solid pieces left after its doorways."""
    (x1, y1), (x2, y2) = seg
    length = math.dist(seg[0], seg[1])
    if length == 0:
        return [seg]
    gaps = []
    for o in openings or []:
        if str(o.get("type") or "").lower() not in DOOR_OPENINGS:
            continue
        rc, w = o.get("relativeCenter"), o.get("width")
        c = (0.5 if rc is None else float(rc)) * length     # 0..1 along the edge
        half = (0.8 if w is None else float(w)) / 2 + DOOR_MARGIN_M
        gaps.append((max(0.0, c - half), min(length, c + half)))
    if not gaps:
        return [seg]
    pieces, pos = [], 0.0
    for g0, g1 in sorted(gaps):
        if g0 > pos:
            pieces.append((pos, g0))
        pos = max(pos, g1)
    if pos < length:
        pieces.append((pos, length))

    def at(d):
        t = d / length
        return (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    return [(at(a), at(b)) for a, b in pieces if b - a > 1e-3]


class NavError(Exception):
    """Bad keys, API error, unknown room or pano (the CLI prints it and exits)."""


# --- fetching --------------------------------------------------------------

def fetch_model(model_id: str) -> dict:
    """GraphQL Model API; HTTP basic auth, key as user, token as password."""
    key = os.environ.get("MATTERPORT_API_KEY")
    token = os.environ.get("MATTERPORT_API_TOKEN")
    if not key or not token:
        raise NavError("Set MATTERPORT_API_KEY and MATTERPORT_API_TOKEN "
                       "(from /costar-keys), or use --file.")
    body = json.dumps({"query": QUERY, "variables": {"id": model_id}}).encode()
    auth = base64.b64encode(f"{key}:{token}".encode()).decode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Basic {auth}"})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        raise NavError(f"Model API HTTP {e.code} (check the key/token)") from None
    if out.get("errors"):
        raise NavError(f"Model API error: {out['errors'][0].get('message')}")
    model = (out.get("data") or {}).get("model")
    if not model:
        raise NavError(f"Model API returned no model for {model_id!r}")
    return model


def load_model(model_id: str | None, file: str | None, refresh: bool) -> dict:
    if file:
        data = json.loads(Path(file).read_text())
        return data.get("data", {}).get("model", data)  # raw response or model
    cache = CACHE_DIR / f"{model_id}.json"
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text())
        if "locations" in cached:          # older caches lack the sweep graph
            return cached
    model = fetch_model(model_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(model))
    return model


# --- geometry ----------------------------------------------------------------

Seg = tuple[tuple[float, float], tuple[float, float]]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_cross(p1, p2, q1, q2) -> bool:
    """Proper crossing (touching an endpoint doesn't count, so a path that
    grazes a door jamb isn't blocked)."""
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _vertex_blocks(p1, p2, segs: list[Seg]) -> bool:
    """A link that passes exactly through a wall vertex (not crossing either wall
    segment properly) is blocked if walls meeting there lie on both sides of it:
    a straight wall stored as two segments, or a wall corner. A door jamb (one
    wall end, the opening on the other side) still lets it through."""
    lx, ly = p2[0] - p1[0], p2[1] - p1[1]
    n = math.hypot(lx, ly)
    if n == 0:
        return False
    sides: dict = {}
    for a, b in segs:
        for v, other in ((a, b), (b, a)):
            t = ((v[0] - p1[0]) * lx + (v[1] - p1[1]) * ly) / (n * n)
            if not 0 < t < 1:
                continue
            if abs(_cross(p1, p2, v)) / n > VERTEX_EPS_M:
                continue
            side = _cross(p1, p2, other)
            if abs(side) / n > VERTEX_EPS_M:
                key = (round(v[0], 3), round(v[1], 3))
                sides.setdefault(key, set()).add(side > 0)
    return any(len(s) == 2 for s in sides.values())


def blocked(p1, p2, segs: list[Seg]) -> bool:
    return any(segments_cross(p1, p2, a, b) for a, b in segs) or _vertex_blocks(p1, p2, segs)


def point_in_segments(pt, segs: list[Seg]) -> bool:
    """Even-odd test against an unordered set of boundary segments."""
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in segs:
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def bearing(a, b) -> float:
    """Clockwise degrees from +y (model "north") going a -> b."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def motor_for(relative_deg: float) -> int:
    return int(round((relative_deg % 360) / 45)) % 8


# --- the twin as a graph -----------------------------------------------------

@dataclass
class Pano:
    id: str
    label: str
    x: float
    y: float
    z: float
    floor: str | None = None
    room: str | None = None


@dataclass
class Room:
    id: str
    label: str
    floor: str | None
    segs: list[Seg] = field(default_factory=list)


@dataclass
class Twin:
    name: str
    panos: dict[str, Pano]
    rooms: dict[str, Room]
    walls: dict[str | None, list[Seg]]   # floor id -> wall segments
    links: dict[str, list[tuple[str, float]]]
    edge_types: dict[str, int] = field(default_factory=dict)   # type -> count (diagnostics)


def build_twin(model: dict) -> Twin:
    walls: dict[str | None, list[Seg]] = {}
    edge_seg: dict[str, Seg] = {}
    edge_types: dict[str, int] = {}
    for fl in model.get("floors") or []:
        pos = {v["id"]: (v["position"]["x"], v["position"]["y"])
               for v in fl.get("vertices") or []}
        walls[fl["id"]] = []
        for e in fl.get("edges") or []:
            vs = [v["id"] for v in e.get("vertices") or []]
            if len(vs) != 2 or vs[0] not in pos or vs[1] not in pos:
                continue
            seg = (pos[vs[0]], pos[vs[1]])
            edge_seg[e["id"]] = seg
            kind = str(e.get("type") or "wall").lower()
            edge_types[kind] = edge_types.get(kind, 0) + 1
            if kind not in PASSABLE:             # walls, railings, unknown: block
                walls[fl["id"]].extend(_cut_doorways(seg, e.get("openings")))

    rooms: dict[str, Room] = {}
    for r in model.get("rooms") or []:
        cls = sorted(r.get("classifications") or [],
                     key=lambda c: -(c.get("confidence") or 0))
        label = cls[0]["label"] if cls else "Unknown"
        segs = [edge_seg[e["id"]] for e in (r.get("boundary") or {}).get("edges") or []
                if e["id"] in edge_seg]
        rooms[r["id"]] = Room(r["id"], label, (r.get("floor") or {}).get("id"), segs)

    panos: dict[str, Pano] = {}
    for p in model.get("panoLocations") or []:
        ps = p["position"]
        pano = Pano(p["id"], str(p.get("label") or p["id"]), ps["x"], ps["y"], ps.get("z", 0.0))
        for room in rooms.values():
            if room.segs and point_in_segments((pano.x, pano.y), room.segs):
                # Prefer a labelled room over a "Void" one if both contain it.
                if pano.room is None or rooms[pano.room].label == "Void":
                    pano.room, pano.floor = room.id, room.floor
        panos[pano.id] = pano

    # Matterport's own sweep graph: each location (= pano) lists the panos you
    # can move to from it, plus its floor and room. Trust its floor/room over
    # our polygon test, and use its neighbours as the candidate links.
    neighbours: dict[str, set[str]] = {}
    locs = model.get("locations") or []
    # neighbours are location ids; equal to pano ids in the models seen so far,
    # but map them explicitly in case a model differs
    loc_pano = {loc.get("id"): (loc.get("pano") or {}).get("id") or loc.get("id") for loc in locs}
    for loc in locs:
        pid = loc_pano.get(loc.get("id"))
        if pid not in panos:
            continue
        p = panos[pid]
        fid = (loc.get("floor") or {}).get("id")
        rid = (loc.get("room") or {}).get("id")
        if fid:
            p.floor = fid
        if rid in rooms:
            p.room = rid
        ns = {loc_pano.get(n, n) for n in loc.get("neighbors") or []}
        neighbours[pid] = {n for n in ns if n in panos and n != pid}

    all_walls = [s for segs in walls.values() for s in segs]
    links: dict[str, list[tuple[str, float]]] = {pid: [] for pid in panos}

    def try_link(a: Pano, b: Pano, max_d: float) -> None:
        d = math.dist((a.x, a.y), (b.x, b.y))
        if d > max_d or abs(a.z - b.z) > SAME_FLOOR_DZ:
            return
        if a.floor and b.floor and a.floor != b.floor:
            return
        floor = a.floor or b.floor
        segs = walls.get(floor, all_walls) if floor else all_walls
        if blocked((a.x, a.y), (b.x, b.y), segs):
            return
        links[a.id].append((b.id, d))
        links[b.id].append((a.id, d))

    if any(neighbours.values()):
        # Neighbours can see each other, sometimes through glass or across a
        # railing, so they still have to pass the wall test (with doorway
        # gaps). They may be further apart than MAX_LINK_M (long corridors).
        seen = set()
        for a_id, ns in neighbours.items():
            for b_id in ns:
                key = tuple(sorted((a_id, b_id)))
                if key not in seen:
                    seen.add(key)
                    try_link(panos[a_id], panos[b_id], NEIGHBOUR_MAX_M)
    else:
        plist = list(panos.values())
        for i, a in enumerate(plist):
            for b in plist[i + 1:]:
                try_link(a, b, MAX_LINK_M)
    return Twin(model.get("name") or model.get("id", "?"), panos, rooms, walls, links, edge_types)


def find_pano(twin: Twin, ref: str) -> Pano:
    """A pano by id or label, or the pano inside the first room matching ref."""
    for p in twin.panos.values():
        if ref in (p.id, p.label):
            return p
    goal = goal_panos(twin, ref)
    if goal:
        return twin.panos[sorted(goal)[0]]
    raise NavError(f"No pano or room matches {ref!r}.")


def goal_panos(twin: Twin, target: str) -> set[str]:
    """Panos in rooms whose label contains the target ("kitchen" -> "Kitchen",
    "bath" -> "Bathroom"); a trailing plural s is ignored ("bedrooms")."""
    t = target.lower().strip()
    if not t:
        return set()
    keys = {t, t[:-1]} if len(t) > 3 and t.endswith("s") else {t}
    room_ids = {r.id for r in twin.rooms.values() if any(k in r.label.lower() for k in keys)}
    return {p.id for p in twin.panos.values() if p.room in room_ids}


def shortest_path(twin: Twin, start: str, goals: set[str]) -> list[str] | None:
    dist = {start: 0.0}
    prev: dict[str, str] = {}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in goals:
            path = [u]
            while path[-1] != start:
                path.append(prev[path[-1]])
            return path[::-1]
        if d > dist.get(u, math.inf):
            continue
        for v, w in twin.links[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return None


# --- route -> haptic steps ---------------------------------------------------

@dataclass
class Step:
    to_pano: str
    room: str
    metres: float
    motor: int
    turn: str

    @property
    def pulse(self) -> str:
        return f"PULSE {self.motor} {STEP_PULSE_MS}"


def route_steps(twin: Twin, path: list[str], heading: float | None = None) -> list[Step]:
    """Legs relative to the wearer's heading. If heading is None the wearer is
    assumed to face the first leg. After each leg they face the way they walked.

    Nearly straight sub-legs (each within MERGE_DEG of the leg's FIRST bearing,
    so small bends can't add up) merge into one leg; its cue is that first
    bearing (where to go from here), and the wearer ends up facing its last."""
    pts = [(twin.panos[p].x, twin.panos[p].y) for p in path]
    legs: list[list] = []   # [end index, first bearing, last bearing, metres]
    for i in range(1, len(pts)):
        b, m = bearing(pts[i - 1], pts[i]), math.dist(pts[i - 1], pts[i])
        if legs and abs((b - legs[-1][1] + 180) % 360 - 180) < MERGE_DEG:
            legs[-1][0], legs[-1][2] = i, b
            legs[-1][3] += m
            continue
        legs.append([i, b, b, m])

    steps = []
    facing = heading if heading is not None else (legs[0][1] if legs else 0.0)
    for i, first, last, m in legs:
        motor = motor_for(first - facing)
        p = twin.panos[path[i]]
        room = twin.rooms[p.room].label if p.room else "-"
        steps.append(Step(path[i], room, m, motor, DIRS[motor]))
        facing = last
    return steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", help="Matterport model id (sid)")
    ap.add_argument("--file", help="read the model from a JSON file instead")
    ap.add_argument("--to", help="room label to reach, e.g. kitchen (not needed with --rooms)")
    ap.add_argument("--start", help="start pano id/label or room (default: first pano)")
    ap.add_argument("--heading", type=float,
                    help="wearer's heading in model degrees (default: facing the first leg)")
    ap.add_argument("--refresh", action="store_true", help="ignore the local cache")
    ap.add_argument("--rooms", action="store_true", help="list rooms and exit")
    a = ap.parse_args(argv)
    if not a.model and not a.file:
        ap.error("give a model id or --file")
    if not a.to and not a.rooms:
        ap.error("give --to <room> (or --rooms to list the rooms)")

    try:
        return _run(a)
    except NavError as e:
        print(e, file=sys.stderr)
        return 2


def _run(a) -> int:
    twin = build_twin(load_model(a.model, a.file, a.refresh))
    n_links = sum(len(v) for v in twin.links.values()) // 2
    print(f"{twin.name}: {len(twin.panos)} panos, {len(twin.rooms)} rooms, {n_links} walkable links")
    print("  edge types: " + (", ".join(f"{k} {n}{'' if k in PASSABLE else ' (blocks)'}"
                                        for k, n in sorted(twin.edge_types.items())) or "none"))

    if a.rooms:
        counts: dict[str, int] = {}
        for p in twin.panos.values():
            lab = twin.rooms[p.room].label if p.room else "(no room)"
            counts[lab] = counts.get(lab, 0) + 1
        for lab, n in sorted(counts.items()):
            print(f"  {lab:24s} {n} panos")
        return 0

    start = find_pano(twin, a.start) if a.start else next(iter(twin.panos.values()))
    goals = goal_panos(twin, a.to)
    if not goals:
        labels = sorted({r.label for r in twin.rooms.values()})
        print(f"No room labelled like {a.to!r}. Rooms: {', '.join(labels)}")
        return 1
    if start.id in goals:
        print("Already there.  -> PULSE 0 120 x2 (arrived)")
        return 0
    path = shortest_path(twin, start.id, goals)
    if not path:
        print(f"No walkable route from pano {start.label} to {a.to!r} "
              "(different floor, or the wall graph closes it off).  -> PULSE 0 300 (unclear)")
        return 1

    steps = route_steps(twin, path, a.heading)
    total = sum(s.metres for s in steps)
    print(f"Route from pano {start.label} to {a.to}: {total:.1f} m, {len(steps)} legs")
    for k, s in enumerate(steps, 1):
        print(f"  {k}. {s.turn:12s} {s.metres:4.1f} m  -> {s.room:16s} {s.pulse}")
    print("  arrived      -> PULSE 0 120, PULSE 0 120")
    return 0


if __name__ == "__main__":
    sys.exit(main())
