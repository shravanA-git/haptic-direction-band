"""app/sound_classes.py streaming logic, with a stub instead of YAMNet (no mediapipe needed)."""
import numpy as np
import pytest

from app import sound_classes as sc

SR = sc.SAMPLE_RATE


class StubYamnet:
    """Scores 'Knock' by window loudness, 'Bark' when the window holds a 300 Hz tone."""
    names = ["Silence", "Knock", "Bark"]
    index = {n: i for i, n in enumerate(names)}

    def scores(self, x, sr=SR):
        loud = float(np.sqrt(np.mean(x ** 2)))
        spec = np.abs(np.fft.rfft(x))
        k = int(round(300 * len(x) / sr))
        bark = float(spec[k - 2:k + 3].max() > 20 * np.median(spec)) if loud > 0.01 else 0.0
        return np.array([1 - min(1, loud * 10), min(1, loud * 10) * (1 - bark), bark], np.float32)


SOUNDS = {"knock": (["Knock"], "percussive", 0.3), "dog": (["Bark"], "noise", 0.3)}


def make(hits, **kw):
    c = sc.SoundClassifier(on_detect=hits.append, sounds=SOUNDS, model=StubYamnet(), **kw)
    return c


def quiet(s):
    return np.zeros(int(s * SR), np.float32)


def thump(s=0.12, amp=0.5):
    return (amp * np.random.default_rng(1).standard_normal(int(s * SR))).astype(np.float32)


def test_hit_has_soundhit_shape_and_tight_times():
    hits = []
    c = make(hits)
    c.feed(np.concatenate([quiet(1.0), thump(), quiet(2.0)]), t_first=100.0)
    [h] = hits                                       # half-overlapping windows: one alert
    assert h.label == "knock" and h.category == "percussive" and h.score >= 0.3
    # YAMNet says "somewhere in this 0.975 s window"; times are tightened to the thump
    assert h.start == pytest.approx(101.0, abs=0.04) and h.end == pytest.approx(101.12, abs=0.04)


def test_best_sound_per_window_and_cooldown():
    hits = []
    c = make(hits)
    t = np.arange(int(0.5 * SR)) / SR
    bark = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    c.feed(np.concatenate([quiet(1.0), bark, quiet(0.5), bark, quiet(3.5), bark, quiet(1.0)]),
           t_first=0.0)
    assert [h.label for h in hits] == ["dog", "dog"]          # 2nd bark within cooldown
    assert hits[1].start - hits[0].start >= sc.COOLDOWN_S


def test_paused_while_action_held():
    hits = []
    c = make(hits)
    c.feed(quiet(1.0), t_first=0.0)
    c.pause()
    c.feed(np.concatenate([thump(), quiet(1.0)]))
    c.resume()
    c.feed(quiet(1.5))
    assert hits == []
    c.feed(np.concatenate([thump(), quiet(1.5)]))
    [h] = hits
    assert h.start == pytest.approx(3.62, abs=0.05)          # clock kept across the pause


def test_pick_is_relative_to_each_threshold():
    th = {"a": 0.5, "b": 0.2}
    assert sc.pick({"a": 0.55, "b": 0.3}, th) == "b"          # 0.3/0.2 beats 0.55/0.5
    assert sc.pick({"a": 0.4, "b": 0.1}, th) is None


def test_common_sounds_use_known_haptic_categories():
    from app import haptics
    for label, (classes, cat, thr) in sc.COMMON.items():
        assert classes and 0 < thr < 1, label
        assert cat in haptics.SOUND_PATTERNS, (label, cat)


def test_unknown_class_names_are_rejected():
    with pytest.raises(ValueError):
        sc.SoundClassifier(sounds={"x": (["Not a class"], "tone", 0.3)}, model=StubYamnet())
