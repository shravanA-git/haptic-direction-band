"""Bench test over USB serial, no browser needed. Built so the gates take
minutes once a board is plugged in.

  python -m tools.bench --check         # link health: lines, MIC rate, dead mics
  python -m tools.bench --gate          # H3 gate: motor 0 pulse + STOP (asks y/n)
  python -m tools.bench --map           # H9 gate: walk all 8 channels, you say where you felt each
  python -m tools.bench --map --motors 0,2,4,6   # fallback: 4 cardinal motors
  python -m tools.bench                 # interactive: `p <motor> [ms]`, `s` STOP, `q`
Add --port /dev/cu.usbserial-XXXX (default: auto-detect) or --sim (no hardware).
Close the Arduino Serial Monitor first: only one program can own the port.

--map: each physical channel buzzes twice; type where you felt it
(f fr r br b bl l fl, or 0-7; n = nothing; a = again). If every channel
is felt at a distinct position but some are swapped, it offers to save
app/motor_map.json so the app fixes the swap in software (no rewiring).
Results are appended to recordings/bench.log.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import median
from typing import Callable, Optional

from app.link import (MAX_PULSE_MS, NUM_MOTORS, Link, MicFrame, SerialPort, SimPort,
                      find_esp32_port, load_motor_map, save_motor_map)

POS = ["f", "fr", "r", "br", "b", "bl", "l", "fl"]
POS_NAMES = ["front", "front-right", "right", "back-right",
             "back", "back-left", "left", "front-left"]
LOG = Path(__file__).resolve().parent.parent / "recordings" / "bench.log"
Ask = Callable[[str], str]


def _say(out, *lines):
    for ln in lines:
        out(ln)


# ------------------------------------------------------------------- check
def check_link(link: Link, seconds: float = 3.0, out=print) -> bool:
    """Watch the link; report line counts, MIC frame rate and dead channels.
    Returns True when MIC frames stream at the 20-50 ms the protocol asks for.
    Early firmware (pulses only, no MIC yet) reports a warning, not a crash.
    """
    frames: list[MicFrame] = []
    link.subscribe(lambda ev: frames.append(ev) if isinstance(ev, MicFrame) else None)
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        time.sleep(0.05)
    h = link.health()
    _say(out, f"[check] {link.device}: state={h['state']}  lines={h['lines']}  "
              f"MIC={h['mic']}  BTN={h['btn']}  other={h['raw']}")
    if h["state"] == "error":
        _say(out, f"[check] FAIL port error: {link.error}")
        return False
    if h["last_raw"]:
        _say(out, f"[check] last non-protocol line: {h['last_raw']!r} "
                  "(boot/debug prints are fine; they are ignored)")
    if not frames:
        _say(out, "[check] WARN no MIC frames. Fine for H3 (pulses only); needed for H9.")
        return False
    ok = True
    gap = h["mic_gap_ms"]
    if gap is None or not 20 <= gap <= 50:
        _say(out, f"[check] WARN median MIC interval {gap} ms (protocol: 20-50 ms)")
        ok = False
    else:
        _say(out, f"[check] MIC every {gap} ms (max gap {h['mic_gap_max_ms']} ms)")
    if h["mic_gap_max_ms"] and h["mic_gap_max_ms"] > 250:
        _say(out, f"[check] WARN a {h['mic_gap_max_ms']} ms gap between frames (stalls?)")
    names = ["front", "right", "back", "left"]
    for i, name in enumerate(names):
        ch = [f.levels[i] for f in frames]
        lo, hi, mid = min(ch), max(ch), median(ch)
        flag = ""
        if hi == lo:
            flag, ok = "  <- constant: unplugged or dead mic?", False
        elif hi >= 4095:
            flag = "  <- hits 4095: raw ADC or clipping?"
        _say(out, f"[check]   {name:6s} min {lo:7.1f}  median {mid:7.1f}  max {hi:7.1f}{flag}")
    _say(out, "[check] tip: clap near one mic; that bar should jump the most "
              "(dashboard shows it live).")
    return ok


# -------------------------------------------------------------------- gate
def run_gate(link: Link, ask: Ask, out=print) -> bool:
    """H3 gate: the laptop pulses motor 0 on command and STOP works."""
    _say(out, "[gate] step 1: PULSE 0 150 (short buzz on the front motor)")
    link.pulse_channel(0, 150)
    time.sleep(0.6)
    a = _yes(ask("[gate] did motor 0 buzz briefly? [y/n] "))
    _say(out, "[gate] step 2: PULSE 0 1000, then STOP after 300 ms")
    link.pulse_channel(0, MAX_PULSE_MS)
    time.sleep(0.3)
    link.stop()
    time.sleep(0.9)
    b = _yes(ask("[gate] did it stop early (~0.3 s, not a full second)? [y/n] "))
    _say(out, "[gate] step 3: PULSE 0 300, then nothing: firmware must end it by itself")
    link.pulse_channel(0, 300)
    time.sleep(0.8)
    c = _yes(ask("[gate] did that buzz end on its own after ~0.3 s? [y/n] "))
    ok = a and b and c
    _say(out, f"[gate] H3 {'PASS' if ok else 'FAIL'}  (pulse={a} stop={b} finite={c})")
    if not ok:
        _say(out, "[gate] if nothing buzzes: rocker on? pack - tied to ESP32 GND? "
                  "transistor pinout (E-B-C)? try `python -m tools.bench` then `p 0 500`.")
    return ok


# --------------------------------------------------------------------- map
def parse_position(ans: str) -> Optional[object]:
    """'fr' / '1' -> 1; 'n' -> None (felt nothing); 'a' -> 'again'."""
    a = ans.strip().lower().replace("-", "").replace(" ", "")
    if a in ("a", "again", ""):
        return "again"
    if a in ("n", "none", "nothing"):
        return None
    if a in POS:
        return POS.index(a)
    full = [n.replace("-", "") for n in POS_NAMES]
    if a in full:
        return full.index(a)
    if a.isdigit() and 0 <= int(a) < NUM_MOTORS:
        return int(a)
    return "?"


def run_map(link: Link, ask: Ask, channels=range(NUM_MOTORS), out=print,
            pulse_ms: int = 250) -> dict:
    """Buzz each physical channel twice; ask where it was felt.
    Returns {channel: felt position or None}."""
    felt: dict = {}
    for ch in channels:
        while True:
            _say(out, f"[map] channel {ch} (pin map says: {POS_NAMES[ch]})")
            link.pulse_channel(ch, pulse_ms)
            time.sleep(2 * pulse_ms / 1000)       # buzz, gap, buzz: easy to feel
            link.pulse_channel(ch, pulse_ms)
            time.sleep(1.4 * pulse_ms / 1000)
            r = parse_position(ask("[map]   felt where? (f fr r br b bl l fl / 0-7, n=nothing, a=again) "))
            if r == "?":
                _say(out, "[map]   didn't understand; try again")
                continue
            if r != "again":
                felt[ch] = r
                break
    return felt


def summarize_map(felt: dict, out=print) -> Optional[list]:
    """Report mismatches. Returns a full logical->channel map when all 8
    channels were felt at 8 distinct positions and at least one is swapped;
    otherwise None (identity is already right, or it can't be fixed in software).
    """
    bad = {ch: p for ch, p in felt.items() if p != ch}
    dead = [ch for ch, p in felt.items() if p is None]
    for ch, p in sorted(felt.items()):
        mark = "ok" if p == ch else ("DEAD" if p is None else f"felt at {POS_NAMES[p]}")
        _say(out, f"[map]   ch {ch} ({POS_NAMES[ch]:11s}) -> {mark}")
    if not bad:
        _say(out, f"[map] PASS: all {len(felt)} channels match the pin map")
        return None
    if dead:
        _say(out, f"[map] FAIL: no buzz on channel(s) {dead}: check that channel's "
                  "transistor, base resistor and motor leads (or fall back to 4 cardinals)")
    positions = [p for p in felt.values() if p is not None]
    if len(felt) == NUM_MOTORS and not dead and len(set(positions)) == NUM_MOTORS:
        m = [0] * NUM_MOTORS
        for ch, p in felt.items():
            m[p] = ch
        _say(out, f"[map] channels are swapped, fixable in software: map = {m}")
        return m
    if not dead:
        _say(out, "[map] FAIL: two channels felt at the same spot; re-seat those motors "
                  "or re-run and answer more carefully")
    return None


def _yes(a: str) -> bool:
    return a.strip().lower().startswith("y")


def _log(lines: list) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a") as f:
            f.write(time.strftime("--- %Y-%m-%d %H:%M:%S\n") + "\n".join(lines) + "\n")
    except OSError:
        pass


# -------------------------------------------------------------------- main
def open_link(a) -> Optional[Link]:
    if a.sim:
        return Link(SimPort())
    dev = a.port or find_esp32_port()
    if not dev:
        print("No ESP32 serial port found. Plug it in (data cable, not charge-only), "
              "pass --port, or use --sim.")
        return None
    try:
        port = SerialPort(dev)
    except Exception as e:
        print(f"Could not open {dev}: {e}\nIs the Arduino Serial Monitor holding the port? Close it.")
        return None
    return Link(port, motor_map=load_motor_map())


def main(argv=None, ask: Ask = input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--sim", action="store_true", help="simulated band (no hardware)")
    ap.add_argument("--check", action="store_true", help="link health only")
    ap.add_argument("--gate", action="store_true", help="H3 gate (motor 0 + STOP)")
    ap.add_argument("--map", action="store_true", help="H9 gate: physical motor mapping")
    ap.add_argument("--motors", default="0,1,2,3,4,5,6,7", help="channels for --map")
    ap.add_argument("--seconds", type=float, default=3.0, help="--check duration")
    ap.add_argument("--quiet", action="store_true", help="interactive: hide MIC lines")
    a = ap.parse_args(argv)
    link = open_link(a)
    if link is None:
        return 1
    report: list = []

    def out(s):
        print(s)
        report.append(s)

    if not (a.check or a.gate or a.map):
        link.subscribe(lambda ev: None if (a.quiet and isinstance(ev, MicFrame)) else print("  <", ev))
    link.on_sent(lambda line: print("  >", line))
    link.start()
    time.sleep(0.1 if a.sim else 2.0)  # opening the port resets the ESP32; let it boot
    rc = 0
    try:
        if a.check or a.gate or a.map:
            mic_ok = check_link(link, a.seconds, out)
            if a.gate:
                rc |= 0 if run_gate(link, ask, out) else 2
            if a.map:
                chans = [int(c) for c in a.motors.split(",") if c.strip()]
                felt = run_map(link, ask, chans, out)
                m = summarize_map(felt, out)
                if any(p != ch for ch, p in felt.items()):
                    rc |= 4
                if m and _yes(ask("[map] save app/motor_map.json so the app uses it? [y/n] ")):
                    out(f"[map] saved {save_motor_map(m)}; re-run --map to confirm "
                        "(--map always pulses raw channels)")
            if a.check and not mic_ok:
                rc |= 8
            _log(report)
            return rc
        while True:
            cmd = ask("> ").split()
            if not cmd:
                continue
            if cmd[0] == "q":
                break
            try:
                if cmd[0] == "s":
                    link.stop()
                elif cmd[0] == "p":
                    link.pulse_channel(int(cmd[1]), int(cmd[2]) if len(cmd) > 2 else 150)
                elif cmd[0] == "h":
                    print(link.health())
                else:
                    raise ValueError
            except (ValueError, IndexError):
                print("usage: p <channel> [ms] | s (STOP) | h (link health) | q")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        link.close()  # sends STOP
    return rc


if __name__ == "__main__":
    sys.exit(main())
