"""AI room labelling for a Matterport model: Claude looks at each scan point's
360 photo and names the place, then an agentic pass makes the labels agree
over the walkable graph and groups them into routable destinations.

    python -m app.main --nav P3S6VUTP3qB --nav-prefetch   # cache the 360 photos first
    python -m tools.label_places P3S6VUTP3qB             # label (one call per pano, <= 1 retry)
    python -m tools.label_places P3S6VUTP3qB --summary   # show the saved labels

Model: HDB_LABEL_MODEL (default claude-sonnet-5-5; claude-haiku-5-5 is faster
and cheaper). Key: ANTHROPIC_API_KEY (from .env). Only the photos and the
prompt go to the Anthropic API; nothing else leaves the laptop.

Output: mp_cache/<model>_labels.json (gitignored), used by tools/costar_nav
(apply_labels) and the /nav room buttons. Manual fixes live in its "manual"
section and survive re-runs.

Pass 1   one call per scan point: faces 1-4 of its skybox -> strict JSON
         {place, name, signs, confidence}.
Pass 2   agentic: each pano is checked against its walkable neighbours and the
         model's own room polygon label (when it isn't Void/Other). Low
         confidence or conflicting panos get ONE retry with a closer prompt
         that includes that context. Still-weak labels take their
         neighbours' majority. Connected panos with the same place on the same
         floor become one destination, e.g. "Lecture hall (floor 2)".
"""
from __future__ import annotations

import base64
import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from tools.costar_nav import CACHE_DIR, build_twin, load_model

PLACES = ["lobby", "entrance", "hallway", "lecture hall", "classroom", "office",
          "conference room", "lab", "kitchen", "restroom", "stairs", "elevator",
          "lounge", "study area", "other"]
DEFAULT_MODEL = "claude-sonnet-5-5"
FACES = (1, 2, 3, 4)
IMG_PX = 768                 # downscale each 2048 px face: plenty to read a room
LOW_CONF = 0.55              # below this, a pano is re-asked (once)
WEAK_CONF = 0.4              # still below this after the retry: neighbours decide
POLYGON_HINTS = {"hallway": "hallway", "kitchen": "kitchen", "office": "office",
                 "bathroom": "restroom", "restroom": "restroom", "stairs": "stairs",
                 "staircase": "stairs", "elevator": "elevator", "lobby": "lobby",
                 "conference room": "conference room", "classroom": "classroom"}

SCHEMA = {
    "type": "object",
    "properties": {
        "place": {"type": "string", "enum": PLACES},
        "name": {"type": "string"},
        "signs": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["place", "name", "signs", "confidence"],
    "additionalProperties": False,
}

PROMPT = """These four photos are the front, right, back and left views from one spot \
inside {building}, a university building. Say what kind of place this spot is, for a \
blind visitor who will ask to be guided there by name.

place: the single best category from the list. "hallway" for corridors and open \
circulation; "lobby" for a large entry hall or atrium; "stairs" or "elevator" only if \
you are standing at or in them.
name: a short, specific description a sighted guide would use, e.g. "atrium with glass \
staircase", "corridor outside room 201".
signs: any room numbers or signage you can actually read; [] if none.
confidence: 0 to 1, how sure you are of place."""

RETRY = """Look again, more closely, at this spot in {building}. A first look said \
"{first}" (confidence {conf:.2f}). Context from the 3D scan:
- scan points you can walk to from here were labelled: {neighbours}
- the scan's floor plan calls this area: {polygon}
Use the context only as a hint: trust what you see. Check for signage, furniture \
(rows of seats, desks, counters, sinks), doors and stairs before answering."""


# ------------------------------------------------------------------ the model call
def _image_blocks(paths: list) -> list:
    import cv2
    blocks = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > IMG_PX:
            s = IMG_PX / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(buf.tobytes()).decode()}})
    return blocks


class Labeller:
    """One structured-output vision call. client: anthropic.Anthropic (or a fake
    with the same .beta.messages.create in tests)."""

    def __init__(self, client=None, model: Optional[str] = None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("HDB_LABEL_MODEL") or DEFAULT_MODEL
        self.calls = 0

    def ask(self, paths: list, text: str) -> Optional[dict]:
        content = _image_blocks(paths) + [{"type": "text", "text": text}]
        kw = {}
        if not self.model.startswith("claude-haiku"):
            # a policy decline re-runs on a fallback model instead of losing the pano
            kw = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        self.calls += 1
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=2048,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": content}], **kw)
        if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
            return None
        txt = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            out = json.loads(txt)
        except ValueError:
            return None
        if out.get("place") not in PLACES:
            return None
        out["confidence"] = max(0.0, min(1.0, float(out.get("confidence") or 0)))
        out["signs"] = [str(s).strip() for s in out.get("signs") or [] if str(s).strip()][:6]
        out["name"] = str(out.get("name") or out["place"]).strip()[:80]
        return out


# ------------------------------------------------------------------ the agentic pass
def labels_path(model_id: str) -> Path:
    return CACHE_DIR / f"{model_id}_labels.json"


def load_labels(model_id: str) -> Optional[dict]:
    p = labels_path(model_id)
    return json.loads(p.read_text()) if p.exists() else None


def _polygon_hint(twin, pid) -> Optional[str]:
    p = twin.panos[pid]
    if not p.room:
        return None
    label = twin.rooms[p.room].label.lower()
    return POLYGON_HINTS.get(label)


def _floor_names(model: dict) -> dict:
    out = {}
    for k, f in enumerate(model.get("floors") or [], 1):
        label = str(f.get("label") or "")
        digits = "".join(ch for ch in label if ch.isdigit())
        out[f["id"]] = int(digits) if digits else k
    return out


def label_model(model_id: str, labeller: Optional[Labeller] = None,
                log: Callable[[str], None] = print, only: Optional[set] = None) -> dict:
    """Run both passes and save mp_cache/<model>_labels.json."""
    from tools.costar_assets import pano_faces
    model = load_model(model_id, None, False)
    twin = build_twin(model)
    building = model.get("name") or "this building"
    old = load_labels(model_id) or {}
    manual = old.get("manual") or {}
    labeller = labeller or Labeller()
    panos: dict = {}
    todo = [pid for pid in twin.panos if (only is None or pid in only)]
    t0 = time.time()
    # pass 1
    for k, pid in enumerate(todo, 1):
        faces = pano_faces(model_id, pid)
        if not faces:
            continue
        out = labeller.ask(faces, PROMPT.format(building=building))
        if out:
            panos[pid] = dict(out, retried=False, smoothed=False)
        if k % 10 == 0:
            log(f"  {k}/{len(todo)} scan points labelled ({time.time() - t0:.0f} s)")
    # pass 2: conflicts with the graph / floor plan -> one closer look
    retries = 0
    for pid, lab in list(panos.items()):
        nb = [panos[v]["place"] for v, _ in twin.links.get(pid, []) if v in panos]
        common = collections.Counter(nb).most_common(1)
        hint = _polygon_hint(twin, pid)
        conflict = (common and common[0][1] >= 2 and common[0][0] != lab["place"]
                    and lab["confidence"] < 0.8) or (hint and hint != lab["place"]
                                                     and lab["confidence"] < 0.8)
        if lab["confidence"] >= LOW_CONF and not conflict:
            continue
        faces = pano_faces(model_id, pid)
        again = labeller.ask(faces, PROMPT.format(building=building) + "\n\n" + RETRY.format(
            building=building, first=f"{lab['place']}: {lab['name']}", conf=lab["confidence"],
            neighbours=", ".join(f"{p} x{n}" for p, n in collections.Counter(nb).most_common()) or "none",
            polygon=hint or "unlabelled"))
        retries += 1
        if again and (again["confidence"] >= lab["confidence"] or again["place"] in nb):
            panos[pid] = dict(again, retried=True, smoothed=False)
        else:
            lab["retried"] = True
    # still weak: the neighbours' majority
    for pid, lab in panos.items():
        if lab["confidence"] >= WEAK_CONF:
            continue
        nb = [panos[v] for v, _ in twin.links.get(pid, []) if v in panos
              and panos[v]["confidence"] >= LOW_CONF]
        votes = collections.Counter(n["place"] for n in nb).most_common(1)
        if votes and votes[0][1] >= 2:
            lab.update(place=votes[0][0], smoothed=True,
                       confidence=round(sum(n["confidence"] for n in nb
                                            if n["place"] == votes[0][0]) / votes[0][1] * 0.7, 2))
    floors = _floor_names(model)
    for pid in panos:
        panos[pid]["floor"] = floors.get(twin.panos[pid].floor)
    data = {"model": model_id, "building": building, "labeller": labeller.model,
            "created": time.strftime("%Y-%m-%d %H:%M"), "calls": labeller.calls,
            "retries": retries, "panos": panos, "manual": manual}
    data["destinations"] = destinations(twin, data, floors)
    path = labels_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return data


def effective(data: dict) -> dict:
    """pano -> label, manual corrections winning over the AI."""
    out = {pid: dict(v) for pid, v in (data.get("panos") or {}).items()}
    for pid, fix in (data.get("manual") or {}).items():
        out[pid] = dict(out.get(pid, {}), **fix, confidence=1.0, manual=True)
    return out


def destinations(twin, data: dict, floors: Optional[dict] = None) -> list:
    """Connected panos with the same place on one floor -> one named destination."""
    labs = effective(data)
    floors = floors or {}
    seen, groups = set(), []
    for pid in twin.panos:
        if pid in seen or pid not in labs:
            continue
        place, floor = labs[pid]["place"], labs[pid].get("floor")
        comp, stack = [], [pid]
        seen.add(pid)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in twin.links.get(u, []):
                if v not in seen and v in labs and labs[v]["place"] == place \
                        and labs[v].get("floor") == floor:
                    seen.add(v)
                    stack.append(v)
        groups.append((place, floor, comp))
    count = collections.Counter((p, f) for p, f, _ in groups)
    index: dict = collections.Counter()
    out = []
    for place, floor, comp in sorted(groups, key=lambda g: (g[1] or 0, g[0], -len(g[2]))):
        index[(place, floor)] += 1
        n = f" {index[(place, floor)]}" if count[(place, floor)] > 1 else ""
        fl = f" (floor {floor})" if floor else ""
        names = collections.Counter(labs[p].get("name") for p in comp if labs[p].get("name"))
        manual_name = next((labs[p]["name"] for p in comp if labs[p].get("manual")
                            and labs[p].get("name")), None)
        signs = sorted({s for p in comp for s in labs[p].get("signs") or []})
        out.append({"name": manual_name or f"{place.capitalize()}{n}{fl}", "place": place,
                    "floor": floor, "panos": comp,
                    "about": names.most_common(1)[0][0] if names else place,
                    "signs": signs,
                    "confidence": round(sum(labs[p]["confidence"] for p in comp) / len(comp), 2)})
    return out


def set_manual(model_id: str, pano: str, place: str, name: str = "") -> dict:
    """A correction from the /nav page; persists across re-runs."""
    if place not in PLACES:
        raise ValueError(f"place must be one of: {', '.join(PLACES)}")
    data = load_labels(model_id) or {"model": model_id, "panos": {}, "manual": {}}
    data.setdefault("manual", {})[pano] = {"place": place, "name": name.strip()[:80] or place}
    twin = build_twin(load_model(model_id, None, False))
    floors = _floor_names(load_model(model_id, None, False))
    data["destinations"] = destinations(twin, data, floors)
    labels_path(model_id).write_text(json.dumps(data, indent=1))
    return data


def summary(data: dict) -> str:
    labs = effective(data)
    by = collections.Counter(v["place"] for v in labs.values())
    conf = [v["confidence"] for v in labs.values()]
    lines = [f"{data.get('building')}: {len(labs)} scan points labelled by {data.get('labeller')} "
             f"({data.get('calls', 0)} calls, {data.get('retries', 0)} retries, "
             f"{sum(1 for v in labs.values() if v.get('smoothed'))} smoothed, "
             f"{len(data.get('manual') or {})} manual); mean confidence "
             f"{sum(conf) / max(1, len(conf)):.2f}",
             "  places: " + ", ".join(f"{p} {n}" for p, n in by.most_common())]
    for d in data.get("destinations") or []:
        lines.append(f"  - {d['name']}: {len(d['panos'])} scan points, conf {d['confidence']}"
                     + (f", signs {', '.join(d['signs'][:4])}" if d["signs"] else "")
                     + f"  ({d['about']})")
    return "\n".join(lines)


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.label_places <model id> [--summary]")
    _load_dotenv()
    mid = sys.argv[1]
    if "--summary" in sys.argv:
        d = load_labels(mid)
        print(summary(d) if d else "no labels yet")
        sys.exit(0)
    d = label_model(mid)
    print(summary(d))
