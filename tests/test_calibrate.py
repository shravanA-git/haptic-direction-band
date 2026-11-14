"""tools/calibrate.py against the clap simulator (mismatched mics)."""
import json

import pytest

from app import direction
from app.link import Link, SimPort
from tools import calibrate


def run(positions, answers=None, **kw):
    port = calibrate.ClapSimPort(seed=3)
    link = Link(port).start()
    it = iter(answers or [])
    ask = (lambda prompt: next(it)) if answers else (lambda prompt: "")
    try:
        return calibrate.run_calibration(link, ask, positions, out=lambda s: None,
                                         quiet_s=0.4, clap_timeout=2.0,
                                         on_ready=port.clap, **kw)
    finally:
        link.close()


def test_eight_claps_recover_mic_mismatch():
    res = run(range(8))
    assert [p for p, _, _ in res["claps"]] == list(range(8))
    g = res["gains"]
    # effective sensitivity in baseline units = SENS / FLOOR; corrected it is ~equal
    eff = [s / f * k for s, f, k in zip(calibrate.ClapSimPort.SENS, calibrate.ClapSimPort.FLOORS, g)]
    assert max(eff) / min(eff) < 1.3
    assert res["correct_calibrated"] == 8 and res["correct_calibrated"] >= res["correct_raw"]
    t = res["trace"]
    assert [a["expected"] for a in t["alerts"]] == list(range(8))


def test_cardinals_and_skip():
    res = run([0, 2, 4, 6])
    assert res["gains"] is not None and len(res["claps"]) == 4
    res = run([0, 2], answers=["", "s"])          # skipped one: can't compute fair gains
    assert len(res["claps"]) == 1 and res["gains"] is None


def test_no_mic_stream_fails_cleanly():
    link = Link(SimPort(frames=[])).start()
    out = []
    assert calibrate.run_calibration(link, lambda p: "", [0], out=out.append, quiet_s=0.2) is None
    link.close()
    assert "no MIC frames" in out[-1]


def test_cli_saves_file_the_app_loads(tmp_path):
    out = tmp_path / "calibration.json"
    rc = calibrate.main(["--sim", "--cardinals", "--quiet-s", "0.4", "--save", "--out", str(out),
                         "--trace-dir", str(tmp_path)], ask=lambda p: "")
    assert rc == 0
    g = direction.load_calibration(out)
    assert g == json.loads(out.read_text())["gains"] and len(g) == 4
    assert list(tmp_path.glob("calib-*.json"))


def test_gains_change_the_estimate():
    frames = [{"ms": i * 50, "levels": [20, 20, 20, 20]} for i in range(20)]
    for f in frames[8:12]:
        f["levels"] = [60, 30, 20, 20]       # front strong, right weak mic
    assert direction.estimate(frames, 400, 550).motor == 0
    assert direction.estimate(frames, 400, 550, gains=[1, 2, 1, 1]).motor == 1
    with pytest.raises(ValueError):
        p = __import__("pathlib").Path("/tmp/hdb_badcal.json")
        p.write_text('{"gains": [1, 1, 1]}')
        direction.load_calibration(p)
