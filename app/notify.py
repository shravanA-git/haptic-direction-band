"""One path for every cue: play the haptic pattern on the band and log it.

    n = Notifier(link.pulse, AlertLog(path))
    n.cue("name", "directional", motor=5, confidence=0.9, latency_ms=410)
    n.cue("name", "generic", reason="too few loud frames")
    n.cue("replay", "directional", motor=5, dry=True)    # dashboard only, no motors

Listeners on the AlertLog see each record (dashboard panel; caregiver texting
is backlog and would subscribe here too).
"""
from __future__ import annotations

from typing import Callable, Optional

from . import haptics
from .recall import AlertLog
from .settings import Settings


def pattern(cue: str, motor: Optional[int] = None) -> list:
    if cue == "directional":
        if motor is None:
            raise ValueError("directional cue needs a motor")
        return haptics.directional(motor)
    return {"generic": haptics.generic_attention, "centered": haptics.centered,
            "lost": haptics.lost, "sound": haptics.taught_sound,
            "ack": haptics.target_ack, "unknown": haptics.target_unknown}[cue]()


class Notifier:
    def __init__(self, send: Callable[[int, int], object], log: Optional[AlertLog] = None,
                 dry_send: Optional[Callable[[int, int], object]] = None,
                 settings: Optional[Settings] = None):
        self.send = send                      # Link.pulse
        self.settings = settings or Settings()
        self.dry_send = dry_send or (lambda m, ms: None)
        self.log = log or AlertLog()

    def cue(self, source: str, cue: str, motor: Optional[int] = None,
            confidence: Optional[float] = None, latency_ms: Optional[float] = None,
            dry: bool = False, play: bool = True, **detail) -> dict:
        """play=False logs a cue someone else already played (e.g. ObjectGuide).
        A muted source is still logged (muted=True) but never buzzes."""
        if not dry and self.settings.is_muted(source):
            play, detail["muted"] = False, True
        if play:
            steps = haptics.scaled(pattern(cue, motor), self.settings.intensity)
            haptics.play(steps, self.dry_send if dry else self.send)
        return self.log.add(source, cue, motor, confidence, latency_ms, dry, **detail)
