"""Alert settings the wearer (or Recall chatbot, or dashboard) can change at
run time. One instance lives on the App; cue paths read it on every cue.

  intensity      low | medium | high      (scales pulse length; see haptics)
  name_alerts    on | off                 (mute = still logged, no buzz)
  find_cues      on | off
  sound_alerts   on | off
  names          comma list               (who the name listener waits for)
  sound_threshold 0.5..0.95               (Teach a Sound match score)
  caregiver_texts on | off                (text a caregiver about alerts; default off)
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

INTENSITIES = ("low", "medium", "high")
MUTABLE = {"name_alerts": "name", "find_cues": "find", "sound_alerts": "sound"}
SETTINGS = ("intensity", *MUTABLE, "names", "sound_threshold", "caregiver_texts")
_ON = {"on", "true", "yes", "1", "enable", "enabled", "unmute"}
_OFF = {"off", "false", "no", "0", "disable", "disabled", "mute"}


class Settings:
    def __init__(self):
        self.intensity = "medium"
        self.muted: set = set()             # sources: name | find | sound
        self.names: Optional[str] = None    # mirrored from the name listener
        self.sound_threshold: Optional[float] = None
        self.caregiver_texts = False        # off until someone turns it on
        self._lock = threading.Lock()
        self._listeners: list[Callable[[str, object], None]] = []

    def on_change(self, fn: Callable[[str, object], None]) -> None:
        """fn(setting, value) after every successful change (apply side effects)."""
        self._listeners.append(fn)

    def is_muted(self, source: str) -> bool:
        return source in self.muted

    def snapshot(self) -> dict:
        with self._lock:
            d = {"intensity": self.intensity, "names": self.names,
                 "sound_threshold": self.sound_threshold,
                 "caregiver_texts": "on" if self.caregiver_texts else "off"}
            d.update({k: ("off" if src in self.muted else "on") for k, src in MUTABLE.items()})
            return d

    def set(self, setting: str, value) -> str:
        """Validate and apply; returns a short human-readable confirmation.
        Raises ValueError with a message the chatbot can relay."""
        v = str(value).strip()
        with self._lock:
            if setting == "intensity":
                v = v.lower()
                v = {"med": "medium", "mid": "medium", "normal": "medium", "strong": "high",
                     "stronger": "high", "weak": "low", "soft": "low", "gentle": "low"}.get(v, v)
                if v not in INTENSITIES:
                    raise ValueError(f"intensity must be one of {', '.join(INTENSITIES)}")
                self.intensity = v
                out, val = f"intensity set to {v}", v
            elif setting in MUTABLE:
                src = MUTABLE[setting]
                if v.lower() in _ON:
                    self.muted.discard(src)
                    val = "on"
                elif v.lower() in _OFF:
                    self.muted.add(src)
                    val = "off"
                else:
                    raise ValueError(f"{setting} must be on or off")
                out = f"{setting.replace('_', ' ')} {val}"
            elif setting == "caregiver_texts":
                if v.lower() in _ON:
                    self.caregiver_texts, val = True, "on"
                elif v.lower() in _OFF:
                    self.caregiver_texts, val = False, "off"
                else:
                    raise ValueError("caregiver_texts must be on or off")
                out = f"caregiver texts {val}"
            elif setting == "names":
                names = [n.strip() for n in v.split(",") if n.strip()]
                if not names:
                    raise ValueError("give at least one name")
                self.names = ", ".join(names)
                out, val = f"listening for {self.names}", self.names
            elif setting == "sound_threshold":
                try:
                    t = float(v)
                except ValueError:
                    raise ValueError("sound_threshold must be a number like 0.8") from None
                if not 0.5 <= t <= 0.95:
                    raise ValueError("sound_threshold must be between 0.5 and 0.95")
                self.sound_threshold = t
                out, val = f"taught-sound match threshold {t:.2f}", t
            else:
                raise ValueError(f"unknown setting {setting!r}; one of {', '.join(SETTINGS)}")
        for fn in self._listeners:
            try:
                fn(setting, val)
            except Exception as e:
                print(f"[settings] listener error: {e}")
        return out
