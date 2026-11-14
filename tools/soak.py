"""Soak test: run the app for a long time while poking it, and watch for
crashes, thread growth and memory growth.

  python -m tools.soak --minutes 30                       # app.main --sim
  python -m tools.soak --minutes 30 -- --vision --name Shravan   # extra app.main args

The app runs as a child process (python -m app.main --sim --http PORT ...).
The soak connects to the dashboard WebSocket and, every couple of seconds,
sends a random mix of what an operator does: test pulses, cues, STOP, bench
sequences, replays, record start/stop, typed targets, Action press/release,
setting changes and Recall questions. Every 30 s it samples the child's RSS,
thread count and open files, plus the app's own status (MIC frames, faults).

Pass: the app is alive at the end, MIC frames kept arriving, and after the
first 2 minutes (warm-up: models load, caches fill) RSS grew < 60 MB and the
thread count grew < 5. Samples go to recordings/soak-HHMMSS.csv.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS = ROOT / "recordings"
WARMUP_S = 120
MAX_RSS_GROWTH_MB = 60
MAX_THREAD_GROWTH = 5


def proc_stats(pid: int) -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return ""
    rss = run(["ps", "-o", "rss=", "-p", str(pid)]).strip()
    if sys.platform == "darwin":
        threads = max(0, len(run(["ps", "-M", "-p", str(pid)]).strip().splitlines()) - 1)
    else:
        threads = int((run(["ps", "-o", "nlwp=", "-p", str(pid)]).strip() or 0))
    files = max(0, len(run(["lsof", "-p", str(pid)]).strip().splitlines()) - 1)
    return {"rss_mb": round(int(rss) / 1024, 1) if rss.isdigit() else None,
            "threads": threads, "files": files}


ACTIONS = [
    (6, lambda r: {"cmd": "pulse", "motor": r.randrange(8), "ms": 150}),
    (4, lambda r: {"cmd": "cue", "name": r.choice(["generic", "centered", "lost", "sound",
                                                    "ack", "unknown", "directional"]),
                   "motor": r.randrange(8)}),
    (3, lambda r: {"cmd": "stop"}),
    (1, lambda r: {"cmd": "bench", "name": "cardinals"}),
    (2, lambda r: {"cmd": "replay", "name": r.choice(["fixtures/back_left.json",
                                                       "fixtures/unclear.json"]), "speed": 4}),
    (1, lambda r: {"cmd": "record_start"}),
    (1, lambda r: {"cmd": "record_stop"}),
    (2, lambda r: {"cmd": "target", "name": r.choice(["banana", "cup", "spaceship", ""])}),
    (2, lambda r: {"cmd": "action", "down": True}),
    (2, lambda r: {"cmd": "action", "down": False}),
    (2, lambda r: {"cmd": "setting", "setting": "intensity",
                   "value": r.choice(["low", "medium", "high"])}),
    (1, lambda r: {"cmd": "ask", "text": r.choice(["who called me?", "what was the last alert?",
                                                    "make it stronger", "settings"])}),
    (1, lambda r: {"cmd": "label", "truth": r.choice([0, 5, "false", None])}),
]


async def drive(url: str, seconds: float, sample_every: float, pid: int, rows: list,
                seed: int) -> dict:
    rnd = random.Random(seed)
    weights = [w for w, _ in ACTIONS]
    status: dict = {}
    sent = errors = 0
    t0 = time.monotonic()
    next_sample = t0
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(url, heartbeat=10) as ws:
            async def reader():
                nonlocal errors
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    m = json.loads(msg.data)
                    if m.get("type") == "status":
                        status.update(m)
                    elif m.get("type") == "log" and str(m.get("text", "")).startswith("error"):
                        errors += 1
            task = asyncio.ensure_future(reader())
            while time.monotonic() - t0 < seconds:
                now = time.monotonic()
                if now >= next_sample:
                    st = proc_stats(pid)
                    st.update(t=round(now - t0), frames=status.get("frames"),
                              faults=";".join((status.get("faults") or {}).keys()))
                    rows.append(st)
                    print(f"[soak] {st['t']:5d}s rss {st['rss_mb']} MB  threads {st['threads']}  "
                          f"files {st['files']}  frames {st['frames']}  faults {st['faults'] or '-'}",
                          flush=True)
                    next_sample += sample_every
                if ws.closed:
                    break
                _, make = rnd.choices(ACTIONS, weights)[0]
                await ws.send_json(make(rnd))
                sent += 1
                await asyncio.sleep(rnd.uniform(0.5, 3.0))
            await ws.send_json({"cmd": "action", "down": False})
            await ws.send_json({"cmd": "stop"})
            await asyncio.sleep(1)
            task.cancel()
    return {"sent": sent, "errors": errors}


def verdict(rows: list, alive: bool) -> tuple:
    problems = []
    if not alive:
        problems.append("app process died")
    if len(rows) < 2:
        return False, problems + ["too few samples"]
    base = next((r for r in rows if r["t"] >= WARMUP_S), rows[0])
    last = rows[-1]
    if base["rss_mb"] and last["rss_mb"] and last["rss_mb"] - base["rss_mb"] > MAX_RSS_GROWTH_MB:
        problems.append(f"RSS grew {last['rss_mb'] - base['rss_mb']:.0f} MB after warm-up")
    if last["threads"] - base["threads"] >= MAX_THREAD_GROWTH:
        problems.append(f"threads grew {base['threads']} -> {last['threads']}")
    if base["frames"] is not None and last["frames"] is not None and last["frames"] <= base["frames"]:
        problems.append("MIC frames stopped arriving")
    return not problems, problems


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--http", type=int, default=8790)
    ap.add_argument("--sample-s", type=float, default=30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    cmd = [sys.executable, "-u", "-m", "app.main", "--sim", "--http", str(a.http), "--dry-texts", *extra]
    RECORDINGS.mkdir(exist_ok=True)
    log_path = RECORDINGS / time.strftime("soak-%H%M%S.log")
    print(f"[soak] {' '.join(cmd[2:])}  ({a.minutes:g} min; app output -> {log_path.name})", flush=True)
    with open(log_path, "w") as log:
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        rows: list = []
        try:
            for _ in range(120):                       # wait for the dashboard (models may load)
                time.sleep(1)
                if child.poll() is not None:
                    break
                try:
                    import urllib.request
                    urllib.request.urlopen(f"http://127.0.0.1:{a.http}/", timeout=1)
                    break
                except Exception:
                    continue
            res = asyncio.run(drive(f"http://127.0.0.1:{a.http}/ws", a.minutes * 60, a.sample_s,
                                    child.pid, rows, a.seed)) if child.poll() is None else {}
        except Exception as e:
            print(f"[soak] driver error: {type(e).__name__}: {e}")
            res = {}
        alive = child.poll() is None
        if alive:
            rows.append({**proc_stats(child.pid), "t": rows[-1]["t"] + 1 if rows else 0,
                         "frames": rows[-1]["frames"] if rows else None, "faults": ""})
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
    csv = RECORDINGS / time.strftime("soak-%H%M%S.csv")
    with open(csv, "w") as f:
        f.write("t,rss_mb,threads,files,frames,faults\n")
        for r in rows:
            f.write(f"{r['t']},{r['rss_mb']},{r['threads']},{r['files']},{r['frames']},{r['faults']}\n")
    ok, problems = verdict(rows, alive)
    print(f"[soak] commands sent {res.get('sent', 0)}, error replies {res.get('errors', 0)}; "
          f"samples -> {csv.name}")
    print(f"[soak] {'PASS' if ok else 'FAIL'}" + ("" if ok else ": " + "; ".join(problems)))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
