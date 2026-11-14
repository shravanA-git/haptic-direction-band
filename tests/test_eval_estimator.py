"""Estimator eval harness (tools/eval_estimator.py): simulation and scoring."""
import json

from app import direction, trace
from tools import eval_estimator as ev


def test_classify():
    assert ev.classify(5, 5) == "correct"
    assert ev.classify(4, 5) == ev.classify(6, 5) == "near"
    assert ev.classify(0, 7) == "near"            # wraps around the ring
    assert ev.classify(1, 5) == "wrong"
    assert ev.classify(None, 3) == ev.classify(None, None) == "generic"
    assert ev.classify(2, None) == "wrong"        # buzzed a side with no real direction


def test_simulated_clean_directions_are_all_correct():
    for m in range(8):
        t = ev.simulate(m, ev.CONDITIONS["clean"], seed=m)
        a = t["alerts"][0]
        assert direction.estimate(t["frames"], a["startMs"], a["endMs"]).motor == m


def test_gate_holds_on_realistic_conditions():
    res = ev.evaluate(["clean", "noisy", "reverb", "motor", "clock"], trials=5)
    for name, (total, _) in res.items():
        assert total.passes(), (name, total.c)
        assert total.c["wrong"] == 0, (name, total.c)


def test_evaluate_is_deterministic_and_subset_independent():
    a = ev.evaluate(["hard"], trials=3)["hard"][0].c
    b = ev.evaluate(["clean", "hard"], trials=3)["hard"][0].c
    assert a == b


def test_cli_scores_fixtures_and_saves_replayable_traces(tmp_path, capsys):
    fx = [str(p) for p in sorted((ev.Path(__file__).parent / "fixtures").glob("*.json"))]
    rc = ev.main(["--conditions", "clean", "--trials", "1", "--save", str(tmp_path), "--gate", *fx])
    out = capsys.readouterr().out
    assert rc == 0
    assert "expected back-left, got back-left [correct]" in out
    assert "expected generic, got generic [generic]" in out
    saved = sorted(tmp_path.glob("clean_*.json"))
    assert len(saved) == 8
    t = trace.load(saved[0])                      # loads like any recorded trace
    assert t["alerts"][0]["expected"] in range(8)


def test_unlabelled_trace_is_skipped_unless_expect_given(tmp_path, capsys):
    t = ev.simulate(2, ev.CONDITIONS["clean"], seed=1)
    t.pop("expected")
    t["alerts"][0].pop("expected")
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(t))
    ev.main(["--conditions", "none", str(p)])
    assert "no labelled alerts" in capsys.readouterr().out
    ev.main(["--conditions", "none", "--expect", "2", str(p)])
    assert "expected right, got right [correct]" in capsys.readouterr().out
