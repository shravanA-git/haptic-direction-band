"""Pictures and facts for the --nav route player, fetched once and cached so the
demo runs with wifi off (CoStar track). Standalone: imports nothing from app/.

    python -m tools.costar_assets zvKHR9DHNjJ        # = python -m app.main --nav-prefetch ...

Cache layout (gitignored): mp_cache/<model>.json (the model, tools/costar_nav)
and mp_cache/<model>/:
    brief.json            name, address, lat/long, CoStar facts, what failed and why
    pano/<pano>_<1..4>.jpg  the four side faces of each scan point's 360 skybox
    preview.jpg           the model's preview image (fallback view)
    aerial.jpg            Vexcel aerial photo of the lot (~70 m box)

Everything degrades quietly: a missing key, a 403 or an unknown API field just
leaves that picture or fact out (the reason is kept in brief.json "errors").
The Model API field names for skyboxes and the preview, and the Vexcel request
format, are UNVERIFIED (no keys on this machine): prefetch prints what worked.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from tools.costar_nav import CACHE_DIR, NavError, _ssl_context, graphql, load_model

FACES = (1, 2, 3, 4)            # skybox children: 0 up, 1-4 around, 5 down
VEXCEL_URL = "https://api.vexcelgroup.com/v2/ortho/extract"
VEXCEL_LAYERS = "urban,urban-r,wide-area,urban-g"
AERIAL_HALF_M = 35.0            # ~70 m box around the building
THUMB_URL = "https://my.matterport.com/api/v2/player/models/{id}/thumb"   # public models

# Candidate Model API queries, tried in order (the first that answers wins).
SKYBOX_QUERIES = [
    'query S($id: ID!) { model(id: $id) { panoLocations { id '
    'skyboxes { resolution status children } } } }',
    'query S($id: ID!) { model(id: $id) { panoLocations { id '
    'skybox(resolution: "2k") { status children } } } }',
]
PREVIEW_QUERIES = [
    'query P($id: ID!) { model(id: $id) { image(resolution: "1024x576") { url } } }',
    'query P($id: ID!) { model(id: $id) { assets { images { url } } } }',
]


def asset_dir(model_id: str) -> Path:
    return CACHE_DIR / model_id


def _get(url: str, headers: Optional[dict] = None, data: Optional[bytes] = None,
         timeout: float = 30) -> tuple:
    """(content-type, bytes)."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return r.headers.get("Content-Type", ""), r.read()


def _save(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


# ---------------------------------------------------------------- 360 photos
def _skybox_urls(entry: dict) -> list:
    """Children URLs of the lowest resolution skybox that has them."""
    boxes = entry.get("skyboxes")
    if boxes is None and entry.get("skybox"):
        boxes = [entry["skybox"]]
    good = [b for b in boxes or [] if len(b.get("children") or []) >= 5]

    def res(b):
        digits = "".join(ch for ch in str(b.get("resolution") or "9999") if ch.isdigit())
        return int(digits or 9999)
    return sorted(good, key=res)[0]["children"] if good else []


def fetch_panos(model_id: str, only: Optional[set] = None, refresh: bool = False) -> dict:
    """pano id -> [face paths]; downloads the side faces of every scan point."""
    out_dir = asset_dir(model_id) / "pano"
    last = None
    for q in SKYBOX_QUERIES:
        try:
            panos = (graphql(q, {"id": model_id}).get("model") or {}).get("panoLocations") or []
            break
        except NavError as e:
            last = e
    else:
        raise NavError(f"skyboxes: {last}")
    got = {}
    for entry in panos:
        pid = entry.get("id")
        if not pid or (only and pid not in only):
            continue
        urls = _skybox_urls(entry)
        paths = []
        for face in FACES:
            path = out_dir / f"{pid}_{face}.jpg"
            if path.exists() and not refresh:
                paths.append(path)
            elif face < len(urls):
                _, data = _get(urls[face])
                paths.append(_save(path, data))
        if paths:
            got[pid] = paths
    return got


def pano_faces(model_id: str, pano_id: str) -> list:
    """Cached face images of a scan point (offline)."""
    d = asset_dir(model_id) / "pano"
    return [p for p in (d / f"{pano_id}_{f}.jpg" for f in FACES) if p.exists()]


def fetch_preview(model_id: str, refresh: bool = False) -> Path:
    path = asset_dir(model_id) / "preview.jpg"
    if path.exists() and not refresh:
        return path
    urls = []
    for q in PREVIEW_QUERIES:
        try:
            m = graphql(q, {"id": model_id}).get("model") or {}
        except NavError:
            continue
        img = m.get("image") or {}
        urls += [img.get("url")] if isinstance(img, dict) else []
        urls += [i.get("url") for i in ((m.get("assets") or {}).get("images") or [])]
    urls = [u for u in urls if u] + [THUMB_URL.format(id=urllib.parse.quote(model_id))]
    err = None
    for u in urls:
        try:
            ctype, data = _get(u)
            if ctype.startswith("image/") and data:
                return _save(path, data)
        except (urllib.error.URLError, OSError) as e:
            err = e
    raise NavError(f"preview image: {err or 'none found'}")


# ---------------------------------------------------------------- aerial
def lot_wkt(lat: float, lon: float, half_m: float = AERIAL_HALF_M) -> str:
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    w, e, s, n = lon - dlon, lon + dlon, lat - dlat, lat + dlat
    return f"POLYGON(({w:.7f} {s:.7f}, {e:.7f} {s:.7f}, {e:.7f} {n:.7f}, {w:.7f} {n:.7f}, {w:.7f} {s:.7f}))"


def fetch_aerial(model_id: str, lat: float, lon: float, refresh: bool = False) -> Path:
    path = asset_dir(model_id) / "aerial.jpg"
    if path.exists() and not refresh:
        return path
    key = os.environ.get("VEXCEL_API_KEY")
    if not key:
        raise NavError("VEXCEL_API_KEY not set")
    form = urllib.parse.urlencode({"wkt": lot_wkt(lat, lon), "layer": VEXCEL_LAYERS,
                                   "srid": "4326", "image-format": "jpeg"}).encode()
    try:
        ctype, data = _get(VEXCEL_URL, {"X-API-KEY": key,
                                        "Content-Type": "application/x-www-form-urlencoded"}, form, 60)
    except urllib.error.HTTPError as e:
        raise NavError(f"Vexcel HTTP {e.code}: {e.read()[:160]!r}") from None
    if ctype.startswith("image/"):
        return _save(path, data)
    try:
        j = json.loads(data)
    except ValueError:
        raise NavError(f"Vexcel answered {ctype or 'no type'}, not an image") from None
    url = j.get("url") or j.get("image_url") or (j.get("data") or {}).get("url")
    if not url:
        raise NavError(f"Vexcel JSON without an image url: {str(j)[:160]}")
    _, img = _get(url)
    return _save(path, img)


# ---------------------------------------------------------------- CoStar facts
def fetch_costar(address: str) -> dict:
    """A few property facts from the CoStar kit, if it answers (it 403s today)."""
    base = os.environ.get("COSTAR_KIT_URL")
    if not base or not address:
        return {}
    sep = "&" if "?" in base else "?"
    try:
        ctype, data = _get(f"{base}{sep}address={urllib.parse.quote(address)}", timeout=15)
        j = json.loads(data)
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    j = j[0] if isinstance(j, list) and j else j
    if not isinstance(j, dict):
        return {}
    facts = {k: v for k, v in j.items() if isinstance(v, (str, int, float)) and str(v).strip()}
    return dict(list(facts.items())[:6])


# ---------------------------------------------------------------- briefing
def brief(model_id: str, model: dict, refresh: bool = False, online: bool = True) -> dict:
    """Name, address, location, aerial photo, CoStar facts; cached in brief.json.
    online=False: only what's cached (the demo path)."""
    path = asset_dir(model_id) / "brief.json"
    if path.exists() and not refresh:
        b = json.loads(path.read_text())
        b["aerial"] = (asset_dir(model_id) / "aerial.jpg").exists()
        b["preview"] = (asset_dir(model_id) / "preview.jpg").exists()
        return b
    addr = (model.get("address") or {}).get("address") if isinstance(model.get("address"), dict) \
        else model.get("address")
    geo = model.get("geolocation") or {}
    b = {"name": model.get("name") or model_id, "address": addr or None,
         "lat": geo.get("lat"), "lon": geo.get("long"), "facts": {}, "errors": {}}
    if online:
        if b["lat"] is not None and b["lon"] is not None:
            try:
                fetch_aerial(model_id, float(b["lat"]), float(b["lon"]), refresh)
            except (NavError, urllib.error.URLError, OSError) as e:
                b["errors"]["aerial"] = str(e)
        try:
            fetch_preview(model_id, refresh)
        except (NavError, urllib.error.URLError, OSError) as e:
            b["errors"]["preview"] = str(e)
        b["facts"] = fetch_costar(addr or "")
        _save(path, json.dumps(b, indent=1).encode())
    b["aerial"] = (asset_dir(model_id) / "aerial.jpg").exists()
    b["preview"] = (asset_dir(model_id) / "preview.jpg").exists()
    return b


def briefing_text(b: dict) -> str:
    """What is spoken once when a route starts."""
    parts = [f"Welcome to {b.get('name') or 'the building'}"]
    if b.get("address"):
        parts[0] += f", {b['address']}"
    facts = b.get("facts") or {}
    if facts:
        parts.append(", ".join(f"{k.replace('_', ' ')} {v}" for k, v in list(facts.items())[:3]))
    return ". ".join(parts) + "."


def prefetch(model_id: str, refresh: bool = False) -> dict:
    """Fetch and cache everything a route demo needs; print what worked."""
    model = load_model(model_id, None, refresh)
    report = {"model": f"{model.get('name')}: {len(model.get('panoLocations') or [])} scan points"}
    try:
        got = fetch_panos(model_id, refresh=refresh)
        report["360 photos"] = f"{len(got)} scan points"
    except (NavError, urllib.error.URLError, OSError) as e:
        report["360 photos"] = f"FAILED: {e}"
    b = brief(model_id, model, refresh=True)
    report["preview"] = "ok" if b["preview"] else f"FAILED: {b['errors'].get('preview')}"
    report["aerial"] = "ok" if b["aerial"] else f"FAILED: {b['errors'].get('aerial', 'no location')}"
    report["CoStar"] = f"{len(b['facts'])} facts" if b["facts"] else "none (kit unavailable)"
    report["address"] = b.get("address") or "none"
    for k, v in report.items():
        print(f"  {k:12s} {v}")
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.costar_assets <model id> [--refresh]")
    try:
        prefetch(sys.argv[1], "--refresh" in sys.argv)
    except NavError as e:
        sys.exit(str(e))
