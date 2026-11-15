"""Pre-demo checklist: is this laptop + band ready? One line per item, with a fix.

  python -m tools.preflight                 # everything, real band
  python -m tools.preflight --sim           # no band (skips the serial checks)
  python -m tools.preflight --pulse         # also buzz motor 0 once (150 ms) and STOP
  python -m tools.preflight --no-vision     # skip camera/detector (faster)
Exit code 0 = no FAIL (WARNs are allowed). `python -m tools.demo` runs this first.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MANUAL = """Manual checklist (tick each one out loud):
  [ ] Band: rocker switch ON, 2xAA pack seated, pack - tied to ESP32 GND
  [ ] USB data cable to the laptop; Arduino Serial Monitor CLOSED
  [ ] Laptop on charger; Do Not Disturb on; screen sleep off
  [ ] Webcam above the forehead, facing forward; banana (or target) on the table
  [ ] Dashboard open full-screen on the projector: http://localhost:{port}
  [ ] Caller knows their spot and the configured name; room as quiet as it gets
  [ ] Rehearse STOP: dashboard STOP button / Esc key; unplug USB = motors stop
  [ ] Backup: typed Find box; recorded video ready"""


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level: str, what: str, fix: str = "") -> None:
        self.rows.append((level, what, fix))
        print(f"  [{level:4s}] {what}" + (f"\n         fix: {fix}" if fix and level != "ok" else ""),
              flush=True)

    @property
    def failed(self) -> bool:
        return any(lv == "FAIL" for lv, _, _ in self.rows)


def check_python(r: Report) -> None:
    r.add("ok" if sys.version_info >= (3, 10) else "FAIL", f"python {sys.version.split()[0]}",
          "use Python 3.10+ (.venv)")
    need = [("aiohttp", True), ("serial", True), ("numpy", True), ("sounddevice", False),
            ("faster_whisper", False), ("cv2", False), ("ultralytics", False)]
    for mod, required in need:
        try:
            importlib.import_module(mod)
            r.add("ok", f"import {mod}")
        except Exception as e:
            r.add("FAIL" if required else "WARN", f"import {mod}: {type(e).__name__}",
                  ".venv/bin/pip install -r requirements.txt")


def check_torch(r: Report) -> None:
    """torch 2.12+ on macOS 14 aborts the whole app on MPS (native crash, not
    catchable); the soak hit it on the first YOLO detect."""
    try:
        import platform
        import torch
        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        mac = platform.mac_ver()[0]
        bad = mac.startswith("14.") and ver >= (2, 12)
        r.add("FAIL" if bad else "ok", f"torch {torch.__version__} on macOS {mac or '-'}",
              ".venv/bin/pip install 'torch<2.12' 'torchvision<0.27' (MPS crash on macOS 14)")
    except Exception as e:
        r.add("WARN", f"torch: {type(e).__name__}", "vision needs torch (pip install -r requirements.txt)")


def check_serial(r: Report, pulse: bool) -> None:
    from app.link import Link, SerialPort, find_esp32_port, load_motor_map
    from tools.bench import check_link
    dev = find_esp32_port()
    if not dev:
        r.add("FAIL", "ESP32 serial port", "plug the band in with a data cable (not charge-only)")
        return
    try:
        link = Link(SerialPort(dev), motor_map=load_motor_map()).start()
    except Exception as e:
        r.add("FAIL", f"open {dev}: {e}", "close the Arduino Serial Monitor / other apps on the port")
        return
    try:
        time.sleep(2.0)                     # opening the port resets the ESP32
        lines = []
        ok = check_link(link, 3.0, out=lines.append)
        h = link.health()
        if ok:
            r.add("ok", f"band streaming on {dev}: MIC every {h['mic_gap_ms']} ms")
        elif h["lines"]:
            r.add("WARN", f"band talks on {dev} but MIC frames are off or irregular "
                          f"({h['mic']} MIC / {h['lines']} lines)", "name alerts need MIC frames")
        else:
            r.add("FAIL", f"{dev} is silent for 3 s", "firmware flashed? baud 115200? press EN/reset")
        for ln in lines:
            if "WARN" in ln and "constant" in ln:
                r.add("WARN", ln.replace("[check]", "").strip(), "check that mic's wiring")
        link.stop()
        r.add("ok", "STOP written to the band")
        if pulse:
            link.pulse_channel(0, 150)
            time.sleep(0.4)
            link.stop()
            r.add("ok", "pulsed motor 0 for 150 ms (did it buzz? if not: rocker, pack, GND)")
    finally:
        link.close()
    mm = ROOT / "app" / "motor_map.json"
    r.add("ok", "motor map: " + ("app/motor_map.json (remapped)" if mm.exists() else "pin map (no remap)"))
    from app.direction import calibration_path
    cal = calibration_path() or ROOT / "app" / "calibration.json"
    r.add("ok" if cal.exists() else "WARN", "mic calibration: " + (f"app/{cal.name}" if cal.exists()
                                                                  else "none"),
          "optional: python -m tools.calibrate (claps at 8 positions)")


def check_mic(r: Report) -> None:
    try:
        import numpy as np
        import sounddevice as sd
        dev = sd.query_devices(kind="input")
        x = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype="float32", blocking=True)
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms == 0.0:
            r.add("FAIL", f"mic '{dev['name']}' returns pure silence",
                  "macOS: System Settings > Privacy & Security > Microphone: allow your terminal")
        else:
            r.add("ok", f"mic '{dev['name']}' (level {rms:.4f})")
    except Exception as e:
        r.add("FAIL", f"mic: {type(e).__name__}: {e}", "plug in / select an input device; allow mic access")
    whisper = ROOT / "models" / "whisper"
    has = whisper.exists() and any(whisper.glob("models--*"))
    r.add("ok" if has else "WARN", "whisper model cached" if has else "whisper model not downloaded",
          "be online once: python -m app.speech --selftest --tts")


def check_vision(r: Report, camera: int) -> None:
    try:
        from app import vision
    except Exception as e:
        r.add("FAIL", f"vision import: {e}", "pip install ultralytics opencv-python")
        return
    print("  -- vision --check (camera, detector) --", flush=True)
    try:
        ok = vision.check(camera)
    except Exception as e:
        ok = False
        print(f"  vision check crashed: {e}")
    r.add("ok" if ok else "FAIL", "vision check" + ("" if ok else " (see lines above)"),
          "fix the vision items above, or demo with the typed Find box")


def check_host(r: Report, port: int) -> None:
    with socket.socket() as s:
        busy = s.connect_ex(("127.0.0.1", port)) == 0
    r.add("FAIL" if busy else "ok", f"dashboard port {port} " + ("is in use" if busy else "free"),
          f"another app.main is running: quit it, or pass --http")
    free = shutil.disk_usage(ROOT).free / 1e9
    r.add("ok" if free > 1 else "WARN", f"disk free {free:.1f} GB", "free some space for recordings")
    try:
        from app.notify import load_env
        import os
        load_env()
        cg = bool(os.environ.get("CAREGIVER_PHONE"))
        r.add("ok", "caregiver texts: " + ("phone set (off until toggled)" if cg else "not configured"))
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", action="store_true", help="no band: skip serial checks")
    ap.add_argument("--pulse", action="store_true", help="buzz motor 0 once")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-mic", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--http", type=int, default=8765)
    a = ap.parse_args(argv)
    r = Report()
    print("== software ==")
    check_python(r)
    if not a.no_vision:
        check_torch(r)
    check_host(r, a.http)
    if not a.sim:
        print("== band ==")
        check_serial(r, a.pulse)
    if not a.no_mic:
        print("== microphone ==")
        check_mic(r)
    if not a.no_vision:
        print("== camera ==")
        check_vision(r, a.camera)
    n_fail = sum(lv == "FAIL" for lv, _, _ in r.rows)
    n_warn = sum(lv == "WARN" for lv, _, _ in r.rows)
    print(f"\nPREFLIGHT {'FAIL' if n_fail else 'OK'}: {n_fail} fail, {n_warn} warn\n")
    print(MANUAL.format(port=a.http))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
