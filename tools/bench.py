"""Bench test over USB serial without the dashboard (H3 gate: pulse motor 0, STOP).

  python -m tools.bench                 # auto-detect port, interactive
  python -m tools.bench --port /dev/cu.usbserial-0001 --gate
Interactive commands: `p <motor> [ms]`, `s` (STOP), `q`. Incoming lines are printed.
--gate runs: PULSE 0 150, wait 1 s, PULSE 0 1500, STOP after 300 ms, then asks you.
"""
from __future__ import annotations

import argparse
import sys
import time

from app.link import Link, MicFrame, SerialPort, SimPort, find_esp32_port


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="hide MIC lines")
    a = ap.parse_args(argv)
    if a.sim:
        port = SimPort()
    else:
        dev = a.port or find_esp32_port()
        if not dev:
            print("No ESP32 port found; pass --port. (Close the Arduino Serial Monitor.)")
            return 1
        port = SerialPort(dev)
    link = Link(port)
    link.subscribe(lambda ev: None if (a.quiet and isinstance(ev, MicFrame)) else print("  <", ev))
    link.on_sent(lambda line: print("  >", line))
    link.start()
    time.sleep(2.0 if not a.sim else 0.1)  # opening the port resets the ESP32
    try:
        if a.gate:
            link.pulse(0, 150)
            time.sleep(1.0)
            link.pulse(0, 1500)
            time.sleep(0.3)
            link.stop()
            ok = input("Did motor 0 buzz briefly, then a long buzz cut short by STOP? [y/n] ")
            print("GATE PASS" if ok.strip().lower().startswith("y") else "GATE FAIL")
            return 0
        while True:
            cmd = input("> ").split()
            if not cmd:
                continue
            if cmd[0] == "q":
                break
            if cmd[0] == "s":
                link.stop()
            elif cmd[0] == "p":
                link.pulse(int(cmd[1]), int(cmd[2]) if len(cmd) > 2 else 150)
    except (KeyboardInterrupt, EOFError, ValueError, IndexError) as e:
        if not isinstance(e, (KeyboardInterrupt, EOFError)):
            print("usage: p <motor> [ms] | s | q")
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
