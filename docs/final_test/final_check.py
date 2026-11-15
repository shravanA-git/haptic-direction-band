"""Automated end-to-end checks for the running Haptic Direction Band app. No AI needed.

Talks to the app's dashboard WebSocket (same one the web page uses). Run it AFTER the app is
launched (see RELAUNCH_AND_TEST.md). Uses the laptop SPEAKERS to speak test phrases (macOS `say`,
`afplay`) so the laptop mic hears them; keep the room quiet and the volume up.

  cd <path-to-repo>
  .venv/bin/python -I docs/final_test/final_check.py                  # link, settings, name, sound, hold-to-talk
  .venv/bin/python -I docs/final_test/final_check.py --touch          # also: 15 s window, touch the pad
  .venv/bin/python -I docs/final_test/final_check.py --pulse          # also: buzz motors 0,2,4,6 (battery ON)
  .venv/bin/python -I docs/final_test/final_check.py --caregiver      # also: send ONE real test text
  .venv/bin/python -I docs/final_test/final_check.py --no-speak       # skip everything that uses speakers
  --url http://localhost:8765   --name Max

Each line is PASS / FAIL / WARN / SKIP with what to do. Exit code 1 if any FAIL.
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time

import aiohttp

RESULTS = []


def report(status, what, detail=""):
    RESULTS.append(status)
    print(f"{status:<5} {what}" + (f"  -- {detail}" if detail else ""), flush=True)


class Feed:
    """Connects once, stores every message with a timestamp, lets us send commands."""

    def __init__(self, url):
        self.url, self.msgs, self.ws, self.session = url, [], None, None
        self._task = None

    async def start(self):
        self.session = aiohttp.ClientSession()
        self.ws = await self.session.ws_connect(self.url.rstrip("/") + "/ws")
        self._task = asyncio.create_task(self._pump())

    async def _pump(self):
        async for m in self.ws:
            if m.type == aiohttp.WSMsgType.TEXT:
                d = json.loads(m.data)
                d["_t"] = time.time()
                self.msgs.append(d)

    async def send(self, **cmd):
        await self.ws.send_str(json.dumps(cmd))

    def since(self, t0, kind=None):
        return [m for m in self.msgs if m["_t"] >= t0 and (kind is None or m.get("type") == kind)]

    def last(self, kind):
        for m in reversed(self.msgs):
            if m.get("type") == kind:
                return m
        return None

    async def close(self):
        if self._task:
            self._task.cancel()
        if self.ws:
            await self.ws.close()
        if self.session:
            await self.session.close()


def speak(phrase, voice="Samantha"):
    return subprocess.Popen(["say", "-v", voice, phrase])


async def check_link(f):
    await asyncio.sleep(2.5)
    st = f.last("status")
    if not st:
        return report("FAIL", "link status", "no status message from the app")
    h = st.get("health", {})
    if not st.get("connected") or h.get("state") != "streaming":
        return report("FAIL", "band link", f"state={h.get('state')} error={st.get('error')}; "
                      "check USB cable, close Serial Monitor, relaunch (RELAUNCH_AND_TEST.md, section 2)")
    gap = h.get("mic_gap_ms")
    ok_gap = gap is not None and 20 <= gap <= 30
    report("PASS" if ok_gap else "WARN", "band link streaming",
           f"mic gap {gap} ms (want ~25), reconnects {h.get('reconnects')}, mic lines {h.get('mic')}")
    if st.get("faults"):
        report("WARN", "app faults", json.dumps(st["faults"])[:200])


async def check_settings(f, name):
    s = f.last("settings")
    if not s:
        return report("FAIL", "settings", "none received")
    bad = []
    if (s.get("names") or "").lower() != name.lower():
        bad.append(f"names={s.get('names')} (want {name}; relaunch with --name {name})")
    for k in ("name_alerts", "find_cues", "sound_alerts"):
        if s.get(k) != "on":
            bad.append(f"{k}=off")
    report("PASS" if not bad else "FAIL", "settings", "; ".join(bad) or f"names={s.get('names')}, alerts on")
    cg = f.last("caregiver")
    if cg:
        report("PASS" if cg.get("configured") else "WARN", "caregiver configured",
               f"sender={cg.get('sender')} to={cg.get('to')} texts={cg.get('on')} "
               "(check the digits are the NEW number)")


async def check_name(f, name):
    t0 = time.time()
    for _ in range(3):
        p = speak(f"hey {name}, over here")
        while p.poll() is None:
            await asyncio.sleep(0.2)
        await asyncio.sleep(2.5)
    hits = [a for a in f.since(t0, "alert") if a.get("source") == "name"]
    if not hits:
        return report("FAIL", "name recognition", f"no name alerts for '{name}'. Mic permission? "
                      "volume up? Run: python -m app.speech --listen " + name)
    lat = [a["latency_ms"] for a in hits if a.get("latency_ms")]
    med = statistics.median(lat) if lat else None
    directional = sum(1 for a in hits if a.get("cue") == "directional")
    report("PASS" if len(hits) >= 2 else "WARN", "name recognition", f"{len(hits)}/3 heard, median latency {med} ms")
    status = "PASS" if med is not None and med < 3000 else "WARN" if med is not None and med < 5000 else "FAIL"
    report(status, "name latency", f"median {med} ms (PASS < 3000, WARN < 5000). If ~10000: the audio-clock fix is "
           "missing or the CPU is starved (camera + models + other processes)")
    report("PASS" if directional else "WARN", "name direction", f"{directional} directional, "
           f"{len(hits) - directional} generic ({hits[0].get('reason')}). Laptop speaker is not on the head: "
           "generic is fine here; real direction needs a real source near the hat")


async def check_sound(f):
    """Teach a short system tone, then replay it. The app teaches from the LOUDEST event in a 2.5 s window,
    so a noisy room teaches noise instead; retry until a short clean tone was taught (<1 s)."""
    import re
    label = "check-glass"
    glass = "/System/Library/Sounds/Glass.aiff"
    taught = False
    for attempt in range(1, 4):
        t_teach = time.time()
        await f.send(cmd="teach", seconds=2.5, label=label)
        await asyncio.sleep(0.8)
        subprocess.Popen(["afplay", "-v", "2", glass])
        await asyncio.sleep(3.2)
        logs = [m["text"] for m in f.since(t_teach, "log") if "taught a" in m["text"]]
        ms = int(re.search(r"taught a (\d+) ms", logs[0]).group(1)) if logs and re.search(r"taught a (\d+) ms", logs[0]) else None
        if ms is not None and ms < 1000:
            taught = True
            break
        await f.send(cmd="forget_sound", label=label)
        await asyncio.sleep(1.0)
    if not taught:
        return report("WARN", "taught sound", "could not teach a clean short tone in 3 tries: the room is too noisy "
                      "(typing, talking, fans, agents running speaker tests). Quiet the room or teach by hand on the dashboard")
    t0 = time.time()
    for _ in range(2):
        subprocess.Popen(["afplay", "-v", "2", glass])
        await asyncio.sleep(3.5)
    hits = [a for a in f.since(t0, "alert") if a.get("source") == "sound"]
    pulses = f.since(t0, "pulse")
    await f.send(cmd="forget_sound", label=label)
    report("PASS" if hits else "FAIL", "taught sound recognised",
           f"{len(hits)}/2 replays matched (taught on try {attempt})" if hits else
           "taught a clean tone but the replay did not match: volume too low? retry in a quieter room")
    motors = sorted({p["motor"] for p in pulses})
    report("PASS" if pulses else "FAIL", "taught sound buzzes", f"motors used {motors}, "
           f"last pulse {pulses[-1]['ms'] if pulses else None} ms (the direction pulse should be ~450 ms)")


async def check_hold_to_talk(f):
    t0 = time.time()
    await f.send(cmd="action", down=True)               # same effect as holding the pad
    await asyncio.sleep(0.6)
    p = speak("find the cup")
    while p.poll() is None:
        await asyncio.sleep(0.2)
    await asyncio.sleep(0.4)
    await f.send(cmd="action", down=False)
    await asyncio.sleep(6)
    logs = [m["text"] for m in f.since(t0, "log")]
    got = [t for t in logs if t.startswith("heard") and "target" in t]
    heard = [t for t in logs if t.startswith("heard")]
    report("PASS" if got else "FAIL", "hold-to-talk 'find the cup'", got[0][:120] if got else
           ("heard but no target: " + heard[0][:100] if heard else "nothing heard: mic permission / start with --vision"))
    await f.send(cmd="target", name="")                  # clear the search so nothing keeps buzzing
    await f.send(cmd="stop")


async def check_touch(f):
    print("      ...touch the pad now: hold 2 s, release, repeat 3 times (15 s window)", flush=True)
    t0 = time.time()
    await asyncio.sleep(15)
    ev = [(round(m["_t"] - t0, 1), "DOWN" if m["down"] else "UP") for m in f.since(t0, "btn")]
    downs = sum(1 for _, k in ev if k == "DOWN")
    ups = sum(1 for _, k in ev if k == "UP")
    report("PASS" if downs >= 1 and ups >= 1 else "FAIL", "touch pad events", f"{downs} down / {ups} up {ev[:6]}"
           if ev else "no events: reflash band_link.ino; IO wire on GPIO27 or GPIO4; module powered from 3V3")
    if downs and abs(downs - ups) > 1:
        report("WARN", "touch pad stuck", f"{downs} down vs {ups} up")


async def check_pulse(f):
    t0 = time.time()
    for motor, where in ((0, "front"), (2, "right"), (4, "back"), (6, "left")):
        await f.send(cmd="pulse", motor=motor, ms=400)
        await asyncio.sleep(1.4)
    sent = [p["motor"] for p in f.since(t0, "pulse")]
    report("PASS" if sent == [0, 2, 4, 6] else "WARN", "motor test pulses sent", f"{sent}; confirm BY FEEL: front, right, "
           "back, left in that order. Weak buzz or board reset = battery sag")
    await f.send(cmd="stop")


async def check_caregiver(f):
    t0 = time.time()
    await f.send(cmd="caregiver_test")
    await asyncio.sleep(6)
    cg = f.last("caregiver") or {}
    last = cg.get("last") or {}
    ok = bool(last.get("ok")) and (last.get("time") or "") != ""
    report("PASS" if ok else "FAIL", "caregiver test text sent", "accepted by Photon (check the phone got it)" if ok
           else f"error: {last.get('error')}. 'no conversation yet' = that number must text the agent first; "
                "wrong number = a stale CAREGIVER_PHONE in the terminal: relaunch with env -u CAREGIVER_PHONE")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--name", default="Max")
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--touch", action="store_true")
    ap.add_argument("--pulse", action="store_true")
    ap.add_argument("--caregiver", action="store_true")
    a = ap.parse_args()
    f = Feed(a.url)
    try:
        await f.start()
    except Exception as e:
        await f.close()
        print(f"FAIL  cannot reach the app at {a.url}: {e}\n      Launch it first (RELAUNCH_AND_TEST.md, section 3).")
        return 1
    await check_link(f)
    await check_settings(f, a.name)
    if a.no_speak:
        report("SKIP", "name / sound / hold-to-talk", "--no-speak")
    else:
        await check_name(f, a.name)
        await check_sound(f)
        await check_hold_to_talk(f)
    if a.touch:
        await check_touch(f)
    else:
        report("SKIP", "touch pad", "add --touch")
    if a.pulse:
        await check_pulse(f)
    else:
        report("SKIP", "motors", "add --pulse (battery ON)")
    if a.caregiver:
        await check_caregiver(f)
    else:
        report("SKIP", "caregiver text", "add --caregiver (sends ONE real text)")
    await f.close()
    fails, warns = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n{RESULTS.count('PASS')} pass, {warns} warn, {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
