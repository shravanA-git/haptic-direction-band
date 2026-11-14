"""Laptop app entry point: owns the serial port and serves the dashboard.

  python -m app.main                      # auto-detect the ESP32 USB port
  python -m app.main --port /dev/cu.usbserial-0001
  python -m app.main --sim                # simulated band (quiet room)
  python -m app.main --sim --trace tests/fixtures/back_left.json --loop
  python -m app.main --target banana      # webcam find-an-object cues on the band
Then open http://localhost:8765
"""
from __future__ import annotations

import argparse
import sys

from aiohttp import web

from . import trace
from .link import Link, SerialPort, SimPort, find_esp32_port, load_motor_map
from .server import App


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial device (default: auto-detect)")
    ap.add_argument("--sim", action="store_true", help="use the simulated band")
    ap.add_argument("--trace", help="with --sim: stream this trace's MIC frames")
    ap.add_argument("--loop", action="store_true", help="with --trace: loop forever")
    ap.add_argument("--http", type=int, default=8765, help="dashboard port")
    ap.add_argument("--vision", action="store_true", help="start the webcam (no target yet)")
    ap.add_argument("--target", help="start the webcam and find this object, e.g. banana")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    a = ap.parse_args(argv)

    if a.sim:
        frames = trace.load(a.trace)["frames"] if a.trace else None
        port = SimPort(frames, loop=a.loop)
    else:
        dev = a.port or find_esp32_port()
        if not dev:
            print("No ESP32 serial port found. Plug it in, pass --port, or use --sim.")
            return 1
        try:
            port = SerialPort(dev)
        except Exception as e:
            print(f"Could not open {dev}: {e}\n"
                  "Is the Arduino Serial Monitor (or another app) holding the port? Close it.")
            return 1

    motor_map = None if a.sim else load_motor_map()
    if motor_map:
        print(f"[hdb] motor map from app/motor_map.json: {motor_map}")
    link = Link(port, motor_map=motor_map).start()
    app = App(link)
    vision = guide = None
    if a.target is not None or a.vision:
        from . import haptics
        from .guide import ObjectGuide
        from .vision import Vision  # heavy import (YOLO); only when asked
        guide = ObjectGuide(lambda cue: haptics.play(cue, link.pulse)).start()
        vision = Vision(camera=a.camera)
        app.attach_vision(vision, guide)
        vision.start()
        if a.target:
            vision.set_target(a.target)
    print(f"[hdb] serial: {link.device}  dashboard: http://localhost:{a.http}")
    try:
        web.run_app(app.make_app(), host="127.0.0.1", port=a.http, print=None)
    finally:
        if vision:
            vision.stop()
            guide.close()
        link.close()  # sends STOP before releasing the port
    return 0


if __name__ == "__main__":
    sys.exit(main())
