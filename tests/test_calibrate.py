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


class LoudFirstPort(calibrate.ClapSimPort):
    """A hat whose mics clip on the first clap at each position (levels capped at the 12-bit ADC)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.attempts = {}
        self.loud = 0

    def clap(self, position):
        n = self.attempts[position] = self.attempts.get(position, 0) + 1
        super().clap(position)
        i0, gains, amp = self._clap
        if n == 1:
            self.loud += 1
            self._clap = (i0, gains, 9000.0)             # far past the ADC ceiling

    def _next_frame(self):
        line, t = super()._next_frame()
        parts = line.split()
        return " ".join(parts[:2] + [str(min(int(v), 4095)) for v in parts[2:]]), t


def test_clipped_clap_is_not_counted_and_the_position_is_asked_again():
    port = LoudFirstPort(seed=3)
    link = Link(port).start()
    said = []
    try:
        res = calibrate.run_calibration(link, lambda p: "", [0, 4], out=said.append, quiet_s=0.4,
                                        clap_timeout=2.0, on_ready=port.clap)
    finally:
        link.close()
    assert port.attempts == {0: 2, 4: 2} and port.loud == 2            # each position needed a second, softer clap
    assert [p for p, _, _ in res["claps"]] == [0, 4]                   # only the unclipped ones count
    text = "\n".join(said)
    assert text.count("CLIPPED") == 2 and "trimmer" in text and "snap softer" in text
    assert "peaks F" in text                                           # per-position peaks are printed
    assert all(max(p) < calibrate.CLIP_ADC for p in
               (calibrate.clap_peaks(res["trace"]["frames"], a, b) for _, a, b in res["claps"]))


def test_clap_peak_helpers():
    fr = [{"ms": 0, "levels": [100, 100, 100, 100]}, {"ms": 25, "levels": [351, 149, 175, 348]},
          {"ms": 50, "levels": [4095, 4095, 4095, 4095]}, {"ms": 400, "levels": [900, 100, 100, 100]}]
    assert calibrate.clap_peaks(fr, 25, 25) == [4095, 4095, 4095, 4095]      # window is padded by 40 ms
    assert calibrate.clap_peaks(fr, 0, 0) == [351, 149, 175, 348]
    assert calibrate.is_clipped([4095, 100, 100, 100]) and not calibrate.is_clipped([708, 149, 100, 100])
    assert calibrate.fmt_peaks([351, 149, 175, 348]) == "F351 R149 B175 L348"


def test_local_calibration_beats_the_shared_file_and_save_writes_the_local_one(tmp_path, monkeypatch):
    shared, local = tmp_path / "calibration.json", tmp_path / "calibration.local.json"
    monkeypatch.setattr(direction, "CALIBRATION_FILE", shared)
    monkeypatch.setattr(direction, "CALIBRATION_LOCAL_FILE", local)
    assert direction.calibration_path() is None and direction.load_calibration() is None
    shared.write_text(json.dumps({"gains": [1.388, 1.376, 1.0, 1.708]}))
    assert direction.calibration_path() == shared
    assert direction.load_calibration() == [1.388, 1.376, 1.0, 1.708]       # the approved defaults
    local.write_text(json.dumps({"gains": [0.97, 0.98, 0.99, 1.06]}))
    assert direction.calibration_path() == local
    assert direction.load_calibration() == [0.97, 0.98, 0.99, 1.06]         # this hat's own gains win
    assert direction.load_calibration(shared) == [1.388, 1.376, 1.0, 1.708]  # an explicit path still works
    assert json.loads(shared.read_text())["gains"][0] == 1.388                # never edited
    # calibrate --save and --out default to the local file, not the shared one
    import argparse  # noqa: F401
    assert calibrate.CALIBRATION_LOCAL_FILE.name == "calibration.local.json"
    assert calibrate.save_calibration.__defaults__[0].name == "calibration.local.json"
