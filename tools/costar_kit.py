"""CoStar kit lookups for the --nav route player (CoStar track). Standalone:
imports nothing from app/.

    python -m tools.costar_kit MFhskYSBMMe              # build + show the property cache
    python -m tools.costar_kit MFhskYSBMMe "suite 1008" # resolve a phrase

The kit is an encrypted zip (COSTAR_KIT_ZIP, password COSTAR_ZIP_PASSWORD, both
from the environment / .env). It is streamed once per model and a small JSON is
cached in mp_cache/<model>/costar.json (gitignored) so the demo runs offline:

    commercial-properties.csv        PropertyId -> MatterportModelId
    data/commercial-property.jsonl   the property (buildingName, floors, address...)
    data/location-occupancy.jsonl    tenancies (toDate null = current), spaces[]{suite,...}
    data/company-detail.jsonl        tenantLocationId -> tenant name

Commercial scans cover ONE SUITE (named in the model's name, e.g. "..., Suite 301").
Tenant names are real businesses: they live in the cache and on the /nav page
only, never in committed files or logs.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

from tools.costar_nav import CACHE_DIR

MEMBERS = {"index": "commercial-properties.csv", "property": "commercial-property.jsonl",
           "occupancy": "location-occupancy.jsonl", "company": "company-detail.jsonl"}
_SUITE = re.compile(r"\b(?:suite|ste\.?|unit|#)\s*([0-9]+[a-z]?|[a-z]?[0-9]+)\b", re.I)


class KitError(Exception):
    pass


def cache_path(model_id: str) -> Path:
    return CACHE_DIR / model_id / "costar.json"


def _member(z: zipfile.ZipFile, key: str) -> str:
    for n in z.namelist():
        if n.endswith(MEMBERS[key]):
            return n
    raise KitError(f"kit has no {MEMBERS[key]}")


def scan_suite(model_name: str) -> Optional[str]:
    """'One Park Plaza, Suite 301' -> '301'."""
    m = _SUITE.search(model_name or "")
    return m.group(1).upper() if m else None


def suite_floor(suite: Optional[str]) -> Optional[int]:
    """Floor from a US-style suite number: 301 -> 3, 1008 -> 10. None if unclear."""
    digits = re.sub(r"\D", "", suite or "")
    if len(digits) == 3:
        return int(digits[0]) or None
    if len(digits) == 4:
        return int(digits[:2]) or None
    return None


def _norm_suite(s) -> str:
    return re.sub(r"^(?:suite|ste\.?|unit|#)\s*", "", str(s or "").strip(), flags=re.I).upper()


def build(model_id: str, model_name: str = "", zip_path: Optional[str] = None,
          password: Optional[str] = None) -> dict:
    """Stream the kit once for this model; returns (and caches) the small JSON."""
    zip_path = zip_path or os.environ.get("COSTAR_KIT_ZIP")
    password = password if password is not None else os.environ.get("COSTAR_ZIP_PASSWORD")
    if not zip_path or not Path(zip_path).exists():
        raise KitError("COSTAR_KIT_ZIP is not set or missing")
    pwd = password.encode() if password else None
    with zipfile.ZipFile(zip_path) as z:
        pid = None
        with z.open(_member(z, "index"), pwd=pwd) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, "utf-8-sig")):
                if (row.get("MatterportModelId") or "").strip() == model_id:
                    pid = int(row["PropertyId"])
                    break
        if pid is None:
            raise KitError(f"no CoStar property for model {model_id}")
        prop = None
        keys = (f'"id": {pid},'.encode(), f'"id":{pid},'.encode())
        with z.open(_member(z, "property"), pwd=pwd) as f:
            for raw in f:
                if raw.lstrip().startswith((b'{' + keys[0], b'{' + keys[1])):
                    prop = json.loads(raw)
                    break
        if prop is None:
            raise KitError(f"property {pid} not in the kit")
        tenancies = []
        needle = (f'"propertyId": {pid},'.encode(), f'"propertyId":{pid},'.encode(),
                  f'"propertyId": {pid}}}'.encode(), f'"propertyId":{pid}}}'.encode())
        with z.open(_member(z, "occupancy"), pwd=pwd) as f:
            for raw in f:
                if any(n in raw for n in needle):
                    r = json.loads(raw)
                    if r.get("propertyId") == pid and r.get("toDate") is None:
                        tenancies.append(r)
        ids = {r.get("tenantLocationId") for r in tenancies}
        names = {}
        with z.open(_member(z, "company"), pwd=pwd) as f:
            for raw in f:
                m = re.match(rb'\s*\{\s*"id"\s*:\s*(\d+)', raw)
                if m and int(m.group(1)) in ids:
                    c = json.loads(raw)
                    names[c["id"]] = (c.get("name") or "").strip()
    floor_name = {fl.get("propertyFloorId"): fl.get("floorName") for fl in prop.get("floors") or []}
    tenants = []
    for r in tenancies:
        name = names.get(r.get("tenantLocationId"))
        if not name:
            continue
        for sp in r.get("spaces") or [{}]:
            suite = _norm_suite(sp.get("suite"))
            floor = suite_floor(suite) or floor_name.get(sp.get("propertyFloorId"))
            tenants.append({"name": name, "suite": suite or None, "floor": floor})
    addr = prop.get("primaryAddress") or {}
    data = {"model": model_id, "property_id": pid,
            "building": prop.get("buildingName") or addr.get("streetName"),
            "stories": prop.get("numStoriesAboveGrade"),
            "address": ", ".join(str(x) for x in (addr.get("streetName"), addr.get("city"),
                                                  addr.get("state")) if x),
            "lat": addr.get("latitude"), "lon": addr.get("longitude"),
            "parking_ratio": prop.get("parkingRatio"),
            "tenant_count": len({t["name"] for t in tenants}),
            "tenants": tenants, "scan_suite": scan_suite(model_name)}
    path = cache_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def load(model_id: str, model_name: str = "", build_if_missing: bool = True) -> Optional[dict]:
    """The cached property (offline), else built from the kit; None if neither."""
    path = cache_path(model_id)
    if path.exists():
        data = json.loads(path.read_text())
        if model_name and not data.get("scan_suite"):
            data["scan_suite"] = scan_suite(model_name)
        return data
    if not build_if_missing:
        return None
    try:
        return build(model_id, model_name)
    except (KitError, OSError, zipfile.BadZipFile, RuntimeError, ValueError, KeyError):
        return None          # no kit / wrong password / not a CoStar model: degrade silently


def briefing(data: Optional[dict]) -> str:
    """'One Park Plaza, 22 floors, 63 tenants. This scan is Suite 301.'"""
    if not data:
        return ""
    parts = [data.get("building") or "This building"]
    if data.get("stories"):
        parts.append(f"{data['stories']} floors")
    if data.get("tenant_count"):
        parts.append(f"{data['tenant_count']} tenants")
    text = ", ".join(parts) + "."
    if data.get("scan_suite"):
        text += f" This scan is Suite {data['scan_suite']}."
    return text


def resolve(phrase: str, data: Optional[dict]) -> Optional[dict]:
    """A tenant by suite number ('suite 1008', '1008') or fuzzy name ('prime nursing').
    -> {"name", "suite", "floor", "in_scan"} or None."""
    if not data or not phrase:
        return None
    tenants = data.get("tenants") or []
    p = phrase.strip().lower()
    m = _SUITE.search(p) or re.fullmatch(r"\s*([0-9]{2,5}[a-z]?)\s*", p)
    if m:
        want = m.group(1).upper()
        hits = [t for t in tenants if (t.get("suite") or "").upper() == want]
        hit = hits[0] if hits else {"name": None, "suite": want, "floor": suite_floor(want)}
        return dict(hit, in_scan=want == (data.get("scan_suite") or "").upper())

    def norm(s):
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        s = re.sub(r"\b(inc|llc|ltd|corp|co|the|of|and)\b", " ", s)
        return " ".join(s.split())
    q = norm(p)
    if len(q) < 3:
        return None
    best, best_s = None, 0.0
    for t in tenants:
        n = norm(t["name"])
        s = difflib.SequenceMatcher(None, q, n).ratio()
        if q in n or all(w in n.split() for w in q.split()):
            s = max(s, 0.9)
        if s > best_s:
            best, best_s = t, s
    if best is None or best_s < 0.75:
        return None
    return dict(best, in_scan=(best.get("suite") or "").upper() == (data.get("scan_suite") or "").upper())


def answer(t: dict, data: dict) -> str:
    """What to say when a tenant isn't inside the scan (never a made-up route)."""
    who = t.get("name") or f"Suite {t.get('suite')}"
    where = f"suite {t['suite']}" if t.get("suite") else "this building"
    floor = f" on floor {t['floor']}" if t.get("floor") else ""
    scan = f"; this scan covers suite {data['scan_suite']} only" if data.get("scan_suite") else \
        "; it isn't in this scan"
    if not t.get("name"):
        return f"Suite {t.get('suite')}{floor} has no tenant listed{scan}."
    return f"{who} is in {where}{floor}{scan}."


def _load_dotenv() -> None:
    """KEY=value lines of the repo's .env into os.environ (CLI only, no overrides)."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.costar_kit <model id> [phrase]")
    _load_dotenv()
    mid = sys.argv[1]
    name = ""
    model_file = CACHE_DIR / f"{mid}.json"
    if model_file.exists():
        name = json.loads(model_file.read_text()).get("name") or ""
    try:
        d = load(mid, name, build_if_missing=False) or build(mid, name)
    except KitError as e:
        sys.exit(str(e))
    print(briefing(d), f"({len(d['tenants'])} tenant spaces cached in {cache_path(mid)})")
    if len(sys.argv) > 2:
        t = resolve(" ".join(sys.argv[2:]), d)
        print("no match" if t is None else
              ("inside the scan" if t["in_scan"] else answer(t, d)))
