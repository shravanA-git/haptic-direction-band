"""Soak verdict, preflight and the one demo command (no hardware)."""
import os

from tools import demo, preflight, soak


def row(t, rss, threads, frames):
    return {"t": t, "rss_mb": rss, "threads": threads, "files": 50, "frames": frames, "faults": ""}


def test_soak_verdict():
    ok, _ = soak.verdict([row(0, 100, 10, 0), row(150, 180, 12, 500), row(1800, 200, 13, 9000)], True)
    assert ok                                         # growth measured after warm-up only
    ok, why = soak.verdict([row(0, 100, 10, 0), row(150, 120, 10, 5), row(1800, 300, 20, 5)], True)
    assert not ok and any("RSS" in w for w in why) and any("threads" in w for w in why) \
        and any("MIC" in w for w in why)
    assert not soak.verdict([row(0, 1, 1, 1), row(200, 1, 1, 2)], False)[0]


def test_preflight_sim_without_devices_passes(capsys):
    rc = preflight.main(["--sim", "--no-vision", "--no-mic", "--http", "8711"])
    out = capsys.readouterr().out
    assert rc == 0 and "PREFLIGHT OK" in out and "Manual checklist" in out


def test_demo_runs_preflight_then_app_on_cpu(monkeypatch):
    calls = {}
    monkeypatch.setattr(preflight, "main", lambda argv: calls.setdefault("pf", argv) and 0 or 0)
    monkeypatch.setattr(os, "execv", lambda exe, cmd: calls.setdefault("cmd", cmd))
    monkeypatch.delenv("HDB_VISION_DEVICE", raising=False)
    demo.main(["--sim", "--name", "Sophanit", "--mirror"])
    assert calls["pf"][-1] == "--sim"
    assert calls["cmd"][1:] == ["-m", "app.main", "--vision", "--name", "Sophanit", "--camera", "0",
                                "--sim", "--mirror"]
    assert os.environ["HDB_VISION_DEVICE"] == "cpu"


def test_demo_stops_on_preflight_fail(monkeypatch):
    monkeypatch.setattr(preflight, "main", lambda argv: 1)
    monkeypatch.setattr(os, "execv", lambda *a: (_ for _ in ()).throw(AssertionError("started")))
    assert demo.main(["--sim"]) == 1
