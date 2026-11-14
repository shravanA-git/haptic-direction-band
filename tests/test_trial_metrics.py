"""tools/trial_metrics.py: pitch table from alert logs and labelled traces."""
import json
from pathlib import Path

from app import recall
from tools import trial_metrics as tm

FIX = Path(__file__).parent / "fixtures"


def make_log(path):
    log = recall.AlertLog(path)
    ids = {}
    ids["ok"] = log.add("name", "directional", 5, 0.9, 400)["id"]
    ids["near"] = log.add("name", "directional", 4, 0.7, 600)["id"]
    ids["wrong"] = log.add("name", "directional", 1, 0.6, 500)["id"]
    ids["gen"] = log.add("name", "generic", latency_ms=450, reason="conflicting")["id"]
    ids["false"] = log.add("name", "generic", latency_ms=300)["id"]
    log.add("name", "directional", 2, 0.8, 350)                 # unlabelled
    log.add("replay", "directional", 5, dry=True)               # ignored
    log.add("manual", "generic")                                # ignored
    log.add("find", "directional", 6, 0.8, 120, target="banana")
    log.add("find", "directional", 6, 0.8, None, repeat=True)   # repeat: not a first cue
    log.add("find", "centered", 0, 0.9, 140)
    log.add("find", "lost", latency_ms=100)
    s = log.add("sound", "sound", latency_ms=250)["id"]
    log.label(ids["ok"], 5)
    log.label(ids["near"], 5)
    log.label(ids["wrong"], 5)
    log.label(ids["gen"], 3)
    log.label(ids["false"], "false")
    log.label(None, 6, missed=True)
    log.label(s, None)
    return path


def test_score_log_counts_and_latency(tmp_path):
    r = tm.report([make_log(tmp_path / "a.jsonl")], [])["live"]
    n = r["name"]
    assert {k: n[k] for k in tm.NAME_KEYS} == {"correct": 1, "near": 1, "wrong": 1, "generic": 1,
                                              "false": 1, "missed": 1, "unlabelled": 1}
    assert n["calls"] == 5 and n["score_pct"] == 40.0 and n["coarse_pct"] == 60.0
    assert n["wrong_pct"] == 20.0 and n["gate"] is False
    assert r["find"] == {"directional": 1, "centered": 1, "lost": 1, "repeats": 1}
    assert r["sound"] == {"detections": 1, "false": 0, "unlabelled": 0}
    assert r["latency"]["find"] == {"n": 3, "median_ms": 120, "p90_ms": 140, "max_ms": 140}
    assert r["latency"]["name"]["n"] == 6 and r["latency"]["name"]["median_ms"] == 425


def test_sessions_keep_ids_apart(tmp_path):
    # two sessions both have alert id 1; labels must stay with their own session
    a = recall.AlertLog(tmp_path / "s1.jsonl")
    a.label(a.add("name", "directional", 2)["id"], 2)
    b = recall.AlertLog(tmp_path / "s2.jsonl")
    b.add("name", "directional", 6)
    r = tm.report([tmp_path / "s1.jsonl", tmp_path / "s2.jsonl"], [])["live"]["name"]
    assert r["correct"] == 1 and r["unlabelled"] == 1 and r["wrong"] == 0


def test_traces_scored_with_estimator():
    r = tm.report([], [FIX / "back_left.json", FIX / "unclear.json"])["replay"]
    assert r["correct"] == 1 and r["generic"] == 1 and r["calls"] == 2 and r["gate"]


def test_cli_text_md_json_and_gate(tmp_path, capsys):
    p = make_log(tmp_path / "alerts-1.jsonl")
    assert tm.main([str(p), str(FIX / "back_left.json")]) == 0
    out = capsys.readouterr().out
    assert "live" in out and "traces" in out and "Latency" in out and "1 name alert(s) not labelled" in out
    tm.main([str(p), "--md"])
    md = capsys.readouterr().out
    assert md.startswith("| Hear your name") and "| Live trials | 5 | 1 | 1 | 1 | 1 | 1 | 1 | 40.0% |" in md
    tm.main([str(p), "--json"])
    assert json.loads(capsys.readouterr().out)["live"]["name"]["calls"] == 5
    assert tm.main([str(p), "--gate"]) == 1
    assert tm.main([str(FIX / "back_left.json"), "--gate"]) == 0


def test_cli_defaults_to_newest_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tm, "RECORDINGS", tmp_path)
    assert tm.main([]) == 1
    make_log(tmp_path / "alerts-20261114-010000.jsonl")
    assert tm.main([]) == 0
    assert "live" in capsys.readouterr().out
