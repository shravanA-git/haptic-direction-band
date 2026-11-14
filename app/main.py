"""Laptop app entry point: owns the serial port and serves the dashboard.

  python -m app.main                      # auto-detect the ESP32 USB port
  python -m app.main --port /dev/cu.usbserial-0001
  python -m app.main --sim                # simulated band (quiet room)
  python -m app.main --sim --trace tests/fixtures/back_left.json --loop
  python -m app.main --target banana      # webcam find-an-object cues on the band
  python -m app.main --sim --sound        # Teach a Sound on the laptop mic (dashboard button)
  python -m app.main --vision             # webcam + hold Action, say "find the banana"
  python -m app.main --sim --vision --mirror   # laptop's own camera facing you
  python -m app.main --name Shravan       # buzz toward whoever calls the name
Then open http://localhost:8765
"""
from __future__ import annotations

import argparse
import sys
import time

from aiohttp import web

from . import direction, trace
from .link import Link, NullPort, SerialPort, SimPort, find_esp32_port, load_motor_map
from .recall import AlertLog
from .server import App, RECORDINGS


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
    ap.add_argument("--mirror", action="store_true",
                    help="camera faces you (laptop webcam): swap left/right")
    ap.add_argument("--no-speech", action="store_true", help="with vision: typed target only")
    ap.add_argument("--avoid", action="store_true",
                    help="with vision: obstacle avoidance (steer around, crash warning)")
    ap.add_argument("--name", help="listen for this name (comma-separated ok) and buzz its direction")
    ap.add_argument("--recall", choices=("auto", "claude", "offline"), default="auto",
                    help="Recall chatbot backend (auto: Claude if credentials exist, else offline)")
    ap.add_argument("--dry-texts", action="store_true",
                    help="caregiver texts are logged, never sent (even if Photon is configured)")
    ap.add_argument("--sound", action="store_true",
                    help="Teach a Sound: laptop mic; teach from the dashboard, then it buzzes its own pattern")
    a = ap.parse_args(argv)

    reopen = None
    if a.sim:
        frames = trace.load(a.trace)["frames"] if a.trace else None
        port = SimPort(frames, loop=a.loop)
    else:
        def reopen():   # also used after an unplug: the device name can change
            dev = a.port or find_esp32_port()
            if not dev:
                raise OSError(2, "no ESP32 serial port found (plugged in? data cable?)")
            return SerialPort(dev)
        try:
            port = reopen()
        except Exception as e:
            print(f"[hdb] band not connected yet ({e}).\n"
                  "      Starting anyway; it connects when the band is plugged in. "
                  "(Close the Arduino Serial Monitor if it holds the port.)")
            port = NullPort()

    motor_map = None if a.sim else load_motor_map()
    if motor_map:
        print(f"[hdb] motor map from app/motor_map.json: {motor_map}")
    link = Link(port, motor_map=motor_map, reopen=reopen).start()
    alert_path = RECORDINGS / time.strftime("alerts-%Y%m%d-%H%M%S.jsonl")
    app = App(link, AlertLog(alert_path), recall_mode=a.recall)
    try:
        app.live.gains = direction.load_calibration()
        if app.live.gains:
            print(f"[hdb] mic calibration from app/calibration.json: gains {app.live.gains}")
    except (ValueError, KeyError) as e:
        print(f"[hdb] ignoring app/calibration.json: {e}")
    vision = guide = None
    if a.target is not None or a.vision:
        from .guide import ObjectGuide
        from .vision import Vision  # heavy import (YOLO); only when asked
        guide = ObjectGuide(lambda cue: app.play_cue(cue, "find")).start()
        vision = Vision(camera=a.camera, mirror=a.mirror)
        app.attach_vision(vision, guide)
        vision.start()
        if a.avoid:
            from .avoid import AvoidLoop
            app.attach_avoid(AvoidLoop(vision, play=lambda cue: None, guide=guide).start())
        if a.target:
            vision.set_target(a.target)
        if not a.no_speech:
            try:
                from .speech import Speech
                app.attach_speech(Speech())
                print("[hdb] hold-to-talk on Action (whisper loads on first use)")
            except Exception as e:  # faster-whisper/sounddevice not installed
                print(f"[hdb] speech off ({e}); typed Find box only")
    names = None
    if a.name:
        try:
            from .speech import NameListener
            shared = getattr(app.speech, "transcriber", None)  # one whisper model, not two
            names = NameListener(a.name, transcriber=shared)
            app.attach_names(names)
            names.start()
            print(f"[hdb] listening for name(s): {', '.join(names.names)}")
        except Exception as e:  # no faster-whisper/sounddevice, or no mic permission
            names = None
            print(f"[hdb] name listener off: {e}")
    sound = None
    if a.sound:
        try:
            from .listen import SoundListener
            sound = SoundListener()
            app.attach_sound(sound)
            sound.start()
            print(f"[hdb] Teach a Sound: {'taught sound loaded' if sound.template else 'press Teach a sound on the dashboard'}")
        except Exception as e:  # no numpy/sounddevice, or no mic permission
            sound = None
            print(f"[hdb] sound listener off: {e}")
    from .notify import caregiver_from_env
    caregiver = caregiver_from_env(app.settings, dry=a.dry_texts)
    app.attach_caregiver(caregiver)
    st = caregiver.status()
    print(f"[hdb] caregiver texts: off until toggled on the dashboard; via {st['sender']}, "
          f"to {st['to']}")
    print(f"[hdb] serial: {link.device}  dashboard: http://localhost:{a.http}")
    print(f"[hdb] alert log: {alert_path.relative_to(RECORDINGS.parent)}")
    try:
        web.run_app(app.make_app(), host="127.0.0.1", port=a.http, print=None)
    finally:
        if names:
            names.stop()
        if sound:
            sound.stop()
        if vision:
            vision.stop()
            guide.close()
        caregiver.close()
        link.close()  # sends STOP before releasing the port
    return 0


if __name__ == "__main__":
    sys.exit(main())
