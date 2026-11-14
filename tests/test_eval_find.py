"""Voice-to-find scoreboard (tools/eval_find.py): no camera, model or whisper."""
import json

from tools import eval_find as ef

FPS = 10.0
W = 640
# Recorder format: half-open f0/f1 ranges, 6 s per segment at 10 fps.
LABELS = {"camera": 0, "mirror": False, "clips": [{
    "object": "banana", "file": "banana.mp4", "fps": FPS,
    "segments": [{"label": lab, "f0": 60 * k, "f1": 60 * (k + 1)}
                 for k, lab in enumerate(["out", "left", "center", "right", "sweep", "out"])]}]}
X = {"left": 100, "center": 320, "right": 540}


def frames_for(path):
    return FPS, ((i, W, i) for i in range(360))   # the "frame" is just its index


class Truth:
    """Detector that sees the object exactly where the labels say (or mirrored)."""

    def __init__(self, segments, swap=False):
        self.segments, self.swap, self.min_score = segments, swap, None

    def target(self, obj):
        return obj

    def __call__(self, i, target):
        lab, a = ef.label_at(self.segments, i)
        if lab == "sweep":
            cx = 100 + (i - a) * 7
        elif lab in X:
            cx = X[lab]
        else:
            return []
        if self.swap:
            cx = W - cx
        return [(cx - 30, 100, cx + 30, 200, 0.9)]


def write(tmp_path, data=LABELS):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(data))
    return p


def test_parse_recorder_labels_half_open(tmp_path):
    clips = ef.parse_labels(LABELS, tmp_path)
    assert clips[0]["video"] == tmp_path / "banana.mp4" and clips[0]["mirror"] is False
    segs = clips[0]["segments"]
    assert segs[0] == ("out", 0, 59) and segs[1] == ("left", 60, 119) and segs[-1] == ("out", 300, 360)
    assert ef.label_at(segs, 60) == ("left", 60)


def test_perfect_detector_passes_clip_metrics(tmp_path):
    segs = ef.parse_labels(LABELS, tmp_path)[0]["segments"]
    res = ef.score_clips(write(tmp_path), detector=Truth(segs), frames_for=frames_for)
    m = res["all"]
    assert m["seen"] == 100.0 and m["false_out"] == 0.0 and m["detect_s"] == 0.0
    assert m["side_acc"] == 100.0 and m["wrong_cue"] == 0.0 and m["flips_min"] == 0.0
    assert 1.9 <= m["lost_s"] <= 2.2          # TrackerConfig.lost_after_s = 2.0
    v = ef.verdict(m)
    assert all(v[k] == "PASS" for k in m if k in v), v


def test_swapped_sides_fail_side_and_cues(tmp_path):
    segs = ef.parse_labels(LABELS, tmp_path)[0]["segments"]
    m = ef.score_clips(write(tmp_path), detector=Truth(segs, swap=True),
                       frames_for=frames_for)["all"]
    assert m["side_acc"] < 50 and m["wrong_cue"] == 100.0
    v = ef.verdict(m)
    assert v["side_acc"] == "FAIL" and v["wrong_cue"] == "FAIL" and v["seen"] == "PASS"


def test_text_scoring_and_match():
    r = ef.score_text(["find the {x}", "{x}"], ["banana", "headphones", "phone"],
                      match=lambda t: "banana" if "banana" in t else None)
    assert r["n"] == 6 and r["ok"] == 2 and r["acc"] == 33.3
    assert ef.is_match("air pods", "airpods") and ef.is_match("bottle", "pill bottle", "pill bottle")
    assert not ef.is_match("roger", "charger") and not ef.is_match(None, "cup")
    real = ef.score_text(["find the {x}", "where is my {x}"], ["banana", "phone", "headphones"])
    assert real["acc"] == 100.0, real["fails"]


def test_no_clips_skips_part_b(tmp_path, capsys):
    res = ef.evaluate(spoken=0, labels=tmp_path / "missing.json")
    assert "skip" in res["clips"] and res["verdict"]["seen"] == "SKIP"
    assert res["verdict"]["spoken_acc"] == "SKIP" and res["metrics"]["text_acc"] is not None
    assert "clips: SKIP" in ef.report(res)
    assert len(ef.TEMPLATES) >= 30 and len(ef.OBJECTS) >= 40
