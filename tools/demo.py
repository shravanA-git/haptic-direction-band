"""The one demo command: preflight, then the app with demo settings.

  python -m tools.demo                      # band + webcam + name "Shravan"
  python -m tools.demo --name Sophanit      # the configured name to listen for
  python -m tools.demo --mirror --camera 1  # passed through to app.main
  python -m tools.demo --sim                # rehearse without the band
  python -m tools.demo --force              # start even if preflight FAILs
  python -m tools.demo --skip-preflight

Starts: python -m app.main --vision --name NAME [--sim] [extra args], with vision
on CPU (HDB_VISION_DEVICE=cpu) unless --mps.
  find an object: hold Action (or dashboard "Hold to talk"), say "find the banana";
  hear your name: the name listener is always on; dashboard shows faults in red.
Then open http://localhost:8765 full-screen on the projector.
"""
from __future__ import annotations

import argparse
import os
import sys

from tools import preflight


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=os.environ.get("HDB_NAME", "Shravan"))
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--force", action="store_true", help="start even if preflight fails")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--mps", action="store_true",
                    help="let vision use the Apple GPU (default: CPU; a bad torch/macOS "
                         "pair aborts the whole app on MPS)")
    a, extra = ap.parse_known_args(argv)
    if not a.mps:
        os.environ.setdefault("HDB_VISION_DEVICE", "cpu")
    if not a.skip_preflight:
        pf = ["--camera", str(a.camera)] + (["--sim"] if a.sim else [])
        if preflight.main(pf) != 0 and not a.force:
            print("\nPreflight failed. Fix the FAIL items, or start anyway with --force "
                  "(the app keeps running with whatever works).")
            return 1
    cmd = [sys.executable, "-m", "app.main", "--vision", "--name", a.name,
           "--camera", str(a.camera)] + (["--sim"] if a.sim else []) + extra
    print("\n[demo] " + " ".join(cmd[1:]), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    sys.exit(main())
