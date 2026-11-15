"""Robust name-direction estimator (v2), centroid classifier and learner. No hardware."""
import json
import random

import pytest

from app import direction, trace
from app.direction import Estimate, estimate, estimate_v1
from tools import eval_estimator as E
from tools import learn_centroids as L

QUIET = [20, 22, 18, 21]


def frames(n=80, ms0=0, step=50, floor=QUIET, jitter=1.0, seed=1):
    r = random.Random(seed)
    return [{"ms": ms0 + i * step, "levels": [round(f + r.uniform(-jitter, jitter), 1) for f in floor]}
            for i in range(n)]


def add(fs, a, b, extra):
    """Add excess (per channel) to frames with a <= ms <= b."""
    for f in fs:
        if a <= f["ms"] <= b:
            f["levels"] = [round(v + x, 1) for v, x in zip(f["levels"], extra)]
    return fs


def test_v1_and_v2_agree_on_a_clean_call_and_v2_is_the_default():
    fs = add(frames(), 2000, 2450, [0, 60, 0, 6])                   # from the right
    assert estimate(fs, 1900, 2550).motor == 2 == estimate_v1(fs, 1900, 2550).motor
    assert estimate(fs, 1900, 2550, algo="v1").reason == estimate_v1(fs, 1900, 2550).reason
    e = estimate(fs, 1900, 2550)
    assert e.reason == "ok" and e.consistency > 0.9 and e.strength > 5 and e.vector[0] > 0.8


def test_front_noise_no_longer_biases_the_call_front():
    # The live bug: the front mic has a small, jumpy baseline. v1 measures excess relative to each
    # channel's OWN baseline, so a few counts of front noise look like a loud front signal.
    v1 = E.confusion(E.sim_pairs("frontnoise", 20, 0, "v1"))
    v2 = E.confusion(E.sim_pairs("frontnoise", 20, 0, "v2"))
    assert v1["front_bias"] > 0.15, v1["front_bias"]                # reproduces the report
    assert v2["front_bias"] <= 0.02 and v2["wrong"] == 0
    clean = E.confusion(E.sim_pairs("clean", 10, 0, "v2"))
    assert clean["wrong"] == 0 and clean["generic"] == 0 and clean["front_bias"] == 0


def test_noise_floor_is_median_and_mad_of_the_preroll_not_polluted_by_earlier_speech():
    fs = add(frames(floor=[8, 22, 22, 22], jitter=1.5), 500, 700, [0, 0, 0, 70])   # an earlier call, 1.4 s before
    med, sig = direction.noise_floor(fs, 1900, 2400)
    assert med[0] == pytest.approx(8, abs=1) and med[3] == pytest.approx(22, abs=1.5)   # median ignores the burst
    fs = add(fs, 2000, 2450, [0, 50, 0, 5])
    assert estimate(fs, 1900, 2550).motor == 2


def test_single_frame_outliers_and_a_motor_buzz_do_not_move_the_answer():
    fs = add(frames(), 2000, 2450, [0, 0, 0, 60])                    # from the left
    fs = add(fs, 2150, 2150, [90, 0, 0, 0])                          # one front spike inside the call
    assert estimate(fs, 1900, 2550).motor == 6
    # a long buzz of the front motor during the call bleeds into the mics; telling the estimator skips it
    fs = add(frames(), 2000, 2450, [0, 0, 0, 40])
    fs = add(fs, 2000, 2450, [30, 0, 0, 0])
    biased = estimate(fs, 1900, 2550)
    skipped = estimate(fs, 1900, 2550, skip_ms=[(1950, 2500)])
    assert biased.motor in (7, 6)                                    # the buzz pulled it toward front
    assert skipped.motor is None and skipped.reason == "too few loud frames"       # nothing trustworthy left


def test_rotating_peaks_and_diffuse_noise_stay_generic():
    fixture = trace.load("tests/fixtures/unclear.json")
    a = fixture["alerts"][0]
    assert estimate(fixture["frames"], a["startMs"], a["endMs"]).motor is None
    pairs = E.sim_pairs("diffuse", 10, 0, "v2")
    assert sum(1 for _, got in pairs if got is None) / len(pairs) >= 0.75


def test_weak_evidence_is_generic():
    fs = add(frames(), 2000, 2450, [0, 3, 0, 0])                     # barely above the noise
    e = estimate(fs, 1900, 2550)
    assert e.motor is None and e.reason == "too few loud frames"
    assert estimate(frames(), 9000, 9500).reason == "no frames in window (stale?)"


def test_live_estimator_skips_frames_around_motor_pulses():
    live = direction.LiveEstimator(pad_ms=0)
    t0 = 100.0
    for i in range(100):                                             # 50 ms frames, laptop clock = band clock + 100 s
        ms = i * 50
        lv = list(QUIET)
        if 2000 <= ms <= 2450:
            lv = [v + x for v, x in zip(lv, [0, 0, 0, 40])]
        if 2000 <= ms <= 2450:
            lv[0] += 30                                              # motor 0 buzzing in the same window
        live(type("Ev", (), {"ms": ms, "levels": lv, "rx": t0 + ms / 1000})())
    now = t0 + 5.0
    before, _ = live.estimate(t0 + 2.0, t0 + 2.45, now=now)
    live.note_motor(t0 + 1.99, 480)                                  # the app commanded PULSE 0 480
    after, _ = live.estimate(t0 + 2.0, t0 + 2.45, now=now)
    assert before.motor is not None and after.motor is None          # evidence during the buzz is not trusted
    assert len(live._motor_spans) == 1


def test_centroid_classifier_margin_and_generic_fallback():
    cents = {0: (0.0, 1.0), 2: (1.0, 0.0), 1: (0.45, 0.8)}           # front-right sits close to front
    assert direction.classify_centroid((0.0, 0.97), cents)[0] == 0
    assert direction.classify_centroid((0.44, 0.82), cents)[0] == 1
    motor, rel = direction.classify_centroid((0.22, 0.9), cents)      # on the midline between 0 and 1
    assert motor is None and rel < direction.CENTROID_MARGIN
    assert direction.classify_centroid((0.5, 0.5), {3: (0.1, 0.1)})[0] == 3        # one class: no margin


def test_centroids_fix_the_head_shadow_warp_that_analytic_gets_wrong(tmp_path):
    paths = []
    for lap in range(3):                                              # 3 laps x 8 positions, warped head
        for m in range(8):
            t = E.simulate(m, E.CONDITIONS["asym"], 700 + lap * 8 + m)
            paths.append(trace.save(t, tmp_path / f"calib-{lap}-{m}.json"))
    trials = L.gather(paths, None)
    loo = L.leave_one_out(trials)
    ana = L.summarize([(t, a) for t, a, _ in loo])
    cen = L.summarize([(t, c) for t, _, c in loo])
    assert ana["near"] >= 6 and cen["exact"] > ana["exact"] + 0.25 and cen["wrong"] == 0
    front = E.confusion([(t, a) for t, a, _ in loo])["front_bias"]
    assert front > 0.15 and E.confusion([(t, c) for t, _, c in loo])["front_bias"] == 0

    out = tmp_path / "centroids.json"
    assert L.main([*map(str, paths), "--no-gains", "--save", "--out", str(out)]) == 0
    cents = direction.load_centroids(out)
    assert sorted(cents) == list(range(8)) and json.loads(out.read_text())["trials"]["3"] == 3
    # the saved centroids drive the live estimate
    t = E.simulate(1, E.CONDITIONS["asym"], 999)
    a = t["alerts"][0]
    assert estimate(t["frames"], a["startMs"], a["endMs"], algo="v2").motor in (0, 1)
    assert estimate(t["frames"], a["startMs"], a["endMs"], centroids=cents).motor == 1


def test_learner_refuses_centroids_that_do_not_beat_analytic(tmp_path):
    paths = [trace.save(E.simulate(m, E.CONDITIONS["clean"], 40 + lap * 8 + m), tmp_path / f"c{lap}{m}.json")
             for lap in range(3) for m in range(8)]
    out = tmp_path / "centroids.json"
    assert L.main([*map(str, paths), "--no-gains", "--save", "--out", str(out)]) in (0, 2)   # equal: allowed
    assert L.main([str(tmp_path / "missing.json"), "--no-gains"]) == 1               # clear message, no crash


def test_calibrate_gains_accept_repeated_laps():
    from tools import calibrate
    base = [20.0, 26.0, 18.0, 23.0]
    fr = []
    claps = []
    for lap in range(3):
        for p in (0, 2, 4, 6):
            a = 1000 + (lap * 4 + p // 2) * 500
            fr += [{"ms": a + k * 20, "levels": [b + (60 if k in (1, 2, 3) else 0) for b in base]}
                   for k in range(6)]
            claps.append((p, a + 20, a + 80))
    fr.sort(key=lambda f: f["ms"])
    assert calibrate.compute_gains(fr, claps, base) is not None      # duplicate positions are fine now


def test_confusion_report_cli_shows_front_bias_before_and_after(capsys):
    assert E.main(["--confusion", "--trials", "6", "--conditions", "frontnoise,clean"]) == 0
    out = capsys.readouterr().out
    rows = {(r.split()[0], r.split()[1]): r for r in out.splitlines() if r.split()[:1] in (["frontnoise"], ["clean"])}
    assert ("frontnoise", "v1") in rows and ("frontnoise", "v2") in rows
    assert float(rows[("frontnoise", "v1")].split()[2].rstrip("%")) > 10
    assert float(rows[("frontnoise", "v2")].split()[2].rstrip("%")) <= 2
