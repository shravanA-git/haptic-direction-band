"""Recall: a small chatbot over the alert log, with two tools.

  recent_alerts(source, minutes, limit)   look up what the band buzzed and why
  change_setting(setting, value)          intensity, mute a cue type, names, ...

    bot = RecallBot(alert_log, settings)          # mode="auto"
    bot.ask("who called me in the last ten minutes?")  -> Answer(text, via, actions)

Backends:
  claude   Claude (claude-opus-5-5) with the two tools, via the Anthropic SDK.
           Credentials come from the environment (ANTHROPIC_API_KEY or an
           `ant auth login` profile); nothing is stored in the repo.
  offline  a keyword parser with the same two tools. No network, no key:
           used in tests, with --sim, and whenever Claude is unavailable.
  auto     claude if the SDK and credentials are present, else offline; a
           failed Claude call falls back to offline for that question.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .recall import AlertLog
from .settings import INTENSITIES, MUTABLE, SETTINGS, Settings

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 5
MAX_TURNS = 10            # start a fresh conversation after this many questions
SOURCES = ("any", "name", "find", "sound", "manual")

SYSTEM = """You are Recall, the assistant inside a haptic headband prototype. The band \
buzzes one of eight motors around the head: when someone calls the wearer's name (from that \
direction, or a generic front buzz if the direction is unclear), to guide the head toward an \
object the wearer asked to find (left/right, two front pulses when centered, a distinct cue \
when lost), and when a taught sound happens. Wearers may be Deaf, hard of hearing, blind or \
low vision, so answer in plain words, at most three short sentences, no markdown tables.

Use recent_alerts to answer questions about what happened; never invent alerts. Times are \
the wearer's local clock. Direction names are relative to the head (front, front-right, \
right, back-right, back, back-left, left, front-left). "confidence" is the estimator's \
agreement from 0 to 1; "truth" is a label a helper added afterwards, if any.

Use change_setting when the wearer asks to change how the band behaves. Settings: \
intensity (low, medium, high), name_alerts / find_cues / sound_alerts (on or off; off still \
logs alerts but does not buzz), names (comma-separated names to listen for), \
sound_threshold (0.5-0.95, higher means fewer taught-sound alerts), caregiver_texts (on or \
off: text the caregiver about name calls, found objects and taught sounds). Confirm what you \
changed in one sentence. If a request is unclear, ask a short question instead of guessing. \
The STOP control is not a setting: if the wearer wants the motors off right now, tell them \
to press STOP or Escape on the dashboard."""

TOOLS = [
    {
        "name": "recent_alerts",
        "description": "Look up recent band alerts from the alert log, newest last. Returns "
                       "counts per source plus one entry per alert with time, source "
                       "(name/find/sound/manual), cue, direction, confidence, and the "
                       "helper's ground-truth label when present.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": list(SOURCES),
                           "description": "Filter by alert source; 'any' for all."},
                "minutes": {"type": "integer",
                            "description": "Only alerts from the last N minutes (1-1440)."},
                "limit": {"type": "integer", "description": "Most recent N alerts (1-50)."},
            },
            "required": ["source", "minutes", "limit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "change_setting",
        "description": "Change one alert setting on the band. Returns a confirmation or an "
                       "error message explaining the allowed values.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "setting": {"type": "string", "enum": list(SETTINGS)},
                "value": {"type": "string",
                          "description": "low|medium|high for intensity; on|off for "
                                         "name_alerts/find_cues/sound_alerts/caregiver_texts; comma list "
                                         "for names; 0.5-0.95 for sound_threshold."},
            },
            "required": ["setting", "value"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class Answer:
    text: str
    via: str                                   # "claude" | "offline"
    actions: list = field(default_factory=list)  # [(tool, input, result)]


# ------------------------------------------------------------------ tools
def recent_alerts(log: AlertLog, source: str = "any", minutes: int = 60, limit: int = 10,
                  now: Optional[float] = None) -> dict:
    minutes = max(1, min(int(minutes), 1440))
    limit = max(1, min(int(limit), 50))
    now = time.time() if now is None else now
    items = [a for a in log.recent(500)
             if not a.get("dry") and a.get("source") != "replay"
             and (source in (None, "any") or a.get("source") == source)
             and now - a.get("t", now) <= minutes * 60]
    counts: dict = {}
    for a in items:
        counts[a["source"]] = counts.get(a["source"], 0) + 1
    keep = ("id", "time", "source", "cue", "direction", "confidence", "truth", "target",
            "heard", "reason", "repeat")
    return {"minutes": minutes, "source": source, "total": len(items), "counts": counts,
            "alerts": [{k: a[k] for k in keep if a.get(k) is not None} for a in items[-limit:]]}


def run_tool(name: str, args: dict, log: AlertLog, settings: Settings) -> tuple[str, bool]:
    """-> (result text, is_error)"""
    try:
        if name == "recent_alerts":
            return json.dumps(recent_alerts(log, args.get("source", "any"),
                                            args.get("minutes", 60), args.get("limit", 10))), False
        if name == "change_setting":
            return settings.set(args["setting"], args["value"]), False
        return f"unknown tool {name}", True
    except (ValueError, KeyError, TypeError) as e:
        return f"error: {e}", True


# ---------------------------------------------------------------- offline
_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
        "twenty": 20, "thirty": 30, "few": 5, "couple": 2}


def _minutes(q: str) -> int:
    m = re.search(r"(\d+|an?|one|two|three|four|five|ten|fifteen|twenty|thirty|few|couple)"
                  r"\s*(?:of\s+)?(min|minute|minutes|hour|hours|hr|hrs)\b", q)
    if not m:
        if "today" in q:
            return 1440
        if "hour" in q:
            return 60
        return 60
    n = m.group(1)
    n = 1 if n in ("a", "an") else _NUM.get(n, None) or int(n)
    return n * 60 if m.group(2).startswith(("h",)) else n


def _source(q: str) -> str:
    if re.search(r"\b(name|call|called|calling|who)\b", q):
        return "name"
    if re.search(r"\b(find|found|object|banana|phone|cup|bottle|guide|target|lost)\b", q):
        return "find"
    if re.search(r"\bsound|doorbell|beep|alarm|knock\b", q):
        return "sound"
    return "any"


def _describe(a: dict) -> str:
    when = a.get("time", "?")[:5]
    if a["source"] == "name":
        what = f"name call from the {a['direction']}" if a.get("direction") else \
            "name call, direction unclear (generic buzz)"
    elif a["source"] == "find":
        what = {"directional": f"turn {a.get('direction', '?')}", "centered": "centered",
                "lost": "target lost"}.get(a.get("cue"), a.get("cue"))
        if a.get("target"):
            what = f"{a['target']}: {what}"
    elif a["source"] == "sound":
        what = "taught sound"
    else:
        what = f"test {a.get('cue')}"
    if a.get("truth") == "false":
        what += " (marked false alert)"
    return f"{when} {what}"


def offline_answer(q: str, log: AlertLog, settings: Settings) -> Answer:
    ql = q.lower().strip()
    acts = []

    def do(name, args):
        res, err = run_tool(name, args, log, settings)
        acts.append((name, args, res))
        return res, err

    # --- settings
    m = re.search(r"\b(intensity|stronger|weaker|softer|gentler|harder|louder|quieter)\b", ql)
    lvl = next((x for x in ("low", "medium", "high") if re.search(rf"\b{x}\b", ql)), None)
    if m or (lvl and re.search(r"\b(set|make|turn|buzz|vibration)\b", ql)):
        if not lvl:
            cur = INTENSITIES.index(settings.intensity)
            up = m and m.group(1) in ("stronger", "harder", "louder")
            lvl = INTENSITIES[max(0, min(2, cur + (1 if up else -1)))]
        res, err = do("change_setting", {"setting": "intensity", "value": lvl})
        return Answer(res[0].upper() + res[1:] + ("." if not err else ""), "offline", acts)
    if re.search(r"\b(caregiver|carer|text|texts|texting|sms)\b", ql) and \
            re.search(r"\b(on|off|stop|start|enable|disable|don't|dont|no more|resume)\b", ql):
        val = "off" if re.search(r"\b(off|stop|disable|don't|dont|no more)\b", ql) else "on"
        res, err = do("change_setting", {"setting": "caregiver_texts", "value": val})
        return Answer(res[0].upper() + res[1:] + ".", "offline", acts)
    off = re.search(r"\b(mute|silence|turn off|disable|stop buzzing for|no more)\b", ql)
    on = re.search(r"\b(unmute|turn on|enable|resume)\b", ql)
    if (off or on) and not re.search(r"\b(all motors|right now)\b", ql):
        src = _source(ql)
        key = {"name": "name_alerts", "find": "find_cues", "sound": "sound_alerts"}.get(src)
        if key:
            res, err = do("change_setting", {"setting": key, "value": "on" if on else "off"})
            return Answer(res[0].upper() + res[1:] + ".", "offline", acts)
        return Answer("Which alerts: name calls, object finding, or the taught sound?",
                      "offline", acts)
    m = re.search(r"(?:listen for|respond to|my name is|names? (?:to|is|are))\s+(.+)", q, re.I)
    if m:
        names = re.sub(r"[.!?]+$", "", m.group(1)).replace(" and ", ",")
        res, err = do("change_setting", {"setting": "names", "value": names})
        return Answer(res[0].upper() + res[1:] + ".", "offline", acts)
    m = re.search(r"threshold\D*(\d?\.\d+|\d+)", ql)
    if m:
        res, err = do("change_setting", {"setting": "sound_threshold", "value": m.group(1)})
        return Answer(res[0].upper() + res[1:] + ".", "offline", acts)
    if re.search(r"\b(stop|off)\b.*\b(now|motors?|buzzing)\b", ql):
        return Answer("Press STOP (or Escape) on the dashboard to turn every motor off now.",
                      "offline", acts)
    if re.search(r"\bsettings?\b", ql):
        s = settings.snapshot()
        muted = [k.replace("_", " ") for k in MUTABLE if s[k] == "off"]
        return Answer(f"Intensity {s['intensity']}; "
                      + (f"muted: {', '.join(muted)}." if muted else "nothing muted."),
                      "offline", acts)

    # --- questions about alerts
    src, mins = _source(ql), _minutes(ql)
    res, _ = do("recent_alerts", {"source": src, "minutes": mins, "limit": 5})
    r = json.loads(res)
    span = f"in the last {mins} minutes" if mins < 120 else (
        "today" if mins >= 1440 else f"in the last {mins // 60} hours")
    kind = {"name": "name calls", "find": "find cues", "sound": "taught-sound alerts"}.get(
        src, "alerts")
    if not r["total"]:
        return Answer(f"No {kind} {span}.", "offline", acts)
    lines = "; ".join(_describe(a) for a in r["alerts"])
    if re.search(r"\b(latest|most recent)\b|\blast (alert|one|call|buzz|cue|time|thing)\b", ql) \
            and not re.search(r"\bhow many\b", ql):
        return Answer(f"Latest: {_describe(r['alerts'][-1])}.", "offline", acts)
    return Answer(f"{r['total']} {kind} {span}. Most recent: {lines}.", "offline", acts)


# ----------------------------------------------------------------- claude
def claude_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()   # `ant auth login` profile


class RecallBot:
    def __init__(self, log: AlertLog, settings: Settings, mode: str = "auto", client=None,
                 model: str = MODEL):
        if mode not in ("auto", "claude", "offline"):
            raise ValueError("mode must be auto, claude or offline")
        self.log, self.settings, self.model = log, settings, model
        self.client = client
        if mode == "auto":
            mode = "claude" if (client is not None or claude_available()) else "offline"
        self.mode = mode
        self.messages: list = []          # Claude conversation, append-only
        self.turns = 0

    def reset(self) -> None:
        self.messages, self.turns = [], 0

    def ask(self, question: str) -> Answer:
        question = (question or "").strip()[:500]
        if not question:
            return Answer("Ask me about recent alerts, or to change a setting.", self.mode)
        if self.mode == "claude":
            try:
                return self._ask_claude(question)
            except Exception as e:     # no network, bad credentials, refusal...
                a = offline_answer(question, self.log, self.settings)
                a.text += f" (offline: Claude unavailable, {type(e).__name__})"
                return a
        return offline_answer(question, self.log, self.settings)

    def _client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        return self.client

    def _ask_claude(self, question: str) -> Answer:
        if self.turns >= MAX_TURNS:
            self.reset()
        client = self._client()
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        actions = []
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = client.beta.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    system=SYSTEM,
                    tools=TOOLS,
                    output_config={"effort": "low"},      # short chat answers
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    messages=self.messages,
                )
                if resp.stop_reason == "refusal":
                    raise RuntimeError("refusal")
                self.messages.append({"role": "assistant", "content": resp.content})
                uses = [b for b in resp.content if b.type == "tool_use"]
                if resp.stop_reason != "tool_use" or not uses:
                    text = " ".join(b.text for b in resp.content if b.type == "text").strip()
                    self.turns += 1
                    return Answer(text or "Done.", "claude", actions)
                results = []
                for b in uses:
                    args = b.input if isinstance(b.input, dict) else json.loads(b.input)
                    res, err = run_tool(b.name, args, self.log, self.settings)
                    actions.append((b.name, args, res))
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": res, "is_error": err})
                self.messages.append({"role": "user", "content": results})
            raise RuntimeError("too many tool rounds")
        except Exception:
            del self.messages[start:]      # drop the failed exchange; history stays valid
            raise


def main(argv=None) -> int:
    """python -m app.recall_bot [alerts.jsonl] [--offline]: chat in the terminal."""
    import argparse
    from . import recall
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("log", nargs="?", help="alert log JSONL to load (default: empty)")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    log = AlertLog()
    if a.log:
        alerts, labels = recall.load(a.log)
        for r in alerts:
            log._recent.append(r)
        for lb in labels:
            if lb.get("ref") is not None:
                log._labels[lb["ref"]] = lb["truth"]
    bot = RecallBot(log, Settings(), "offline" if a.offline else "auto")
    print(f"[recall] backend: {bot.mode}. Ctrl-D to quit.")
    try:
        while True:
            ans = bot.ask(input("you> "))
            print(f"recall> {ans.text}")
            for name, args, res in ans.actions:
                print(f"   [{name} {json.dumps(args)}]")
    except (EOFError, KeyboardInterrupt):
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
