"""The band agent: text the band, get a short plain answer.

The wearer or the caregiver texts "status", "what did I miss today", "find my
keys", "is she ok" to the band's iMessage number (Photon). A LangChain deep agent
answers using a few tools that wrap existing code, or, with no model or key, a
keyword router answers from the same tools. Replies are written to be read out
by text-to-speech: short plain sentences, no markdown, tables or emoji.

    tools   get_status, recent_alerts, daily_summary, pattern_legend, explain_buzz,
            set_find_target, send_urgent
    brain   deepagents (HDB_BAND_AGENT_MODEL, default anthropic:claude-haiku-5-5;
            key from the environment/.env, never the repo) or the offline router
    voice   Photon bridge (tools/photon/bridge.mjs) both ways

Safety, by construction:
  * only the configured numbers (WEARER_PHONE, CAREGIVER_PHONE) get an answer;
    anyone else is ignored without a reply
  * BandTools holds no link and no haptics: a text can change the find target
    and text the caregiver, never fire a motor
  * the buzz path never waits on this: texts run on their own worker thread, the
    alert listener only queues, and every failure ends in silence or a short sorry

Adding tools without editing this file
--------------------------------------
Put a module on the path (default: app/costar_tools.py, or any names in
HDB_AGENT_PLUGINS, comma-separated) that exposes either
    TOOLS = [fn, ...]                 plain functions: typed arguments, a docstring
                                      that says when to use it, return a short string
or  def make_tools(band_tools): return [fn, ...]    (to use alerts/status/clock)
Each function becomes a tool for the deep agent; its text is made speakable and
an exception becomes "I could not do that". Plugin tools get no link or haptics
from here. register_tool(fn) does the same from code. A missing or broken plugin
is skipped with one printed line; the agent still starts.

Running it
----------
    python -m app.main --vision --agent          # the band app + the texting agent
    python -m app.band_agent --chat              # talk to it in the terminal (no texts)
    python -m app.band_agent --chat --offline    # same, keyword router only
Env (.env, gitignored): PHOTON_PROJECT_ID, PHOTON_API_KEY (Photon 2.0 beta),
WEARER_PHONE, CAREGIVER_PHONE (a +E.164 number or the iMessage email), ANTHROPIC_API_KEY
(optional). Shared line: the person must text the agent first; see tools/photon/README.md.
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import threading
import time
from collections import deque
from typing import Callable, Optional

from . import haptics

DEFAULT_MODEL = "anthropic:claude-haiku-5-5"
MAX_REPLY_CHARS = 600
AGENT_TIMEOUT_S = 25.0
URGENT_SOUNDS = {"alarm", "siren", "car horn", "glass breaking", "shout", "scream", "yell",
                 "fire alarm", "smoke alarm"}
SKIP_SOURCES = {"manual", "replay"}
DEFAULT_PLUGINS = ("app.costar_tools",)
CORE_TOOLS = ("get_status", "recent_alerts", "daily_summary", "pattern_legend", "explain_buzz",
              "set_find_target", "send_urgent")

# ---------------------------------------------------------------------------
# Words: alerts and patterns in plain speech
# ---------------------------------------------------------------------------

_WHERE = {"front": "in front of you", "front-right": "from your front right",
          "right": "from your right", "back-right": "from behind on your right",
          "back": "from behind", "back-left": "from behind on your left",
          "left": "from your left", "front-left": "from your front left"}
_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def where(rec: dict) -> str:
    d = rec.get("direction")
    return _WHERE.get(d, "") if d else ""


def plain_text(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    """Speakable: no markdown, tables, bullets or emoji; sentences in one paragraph."""
    t = re.sub(r"[\U00010000-\U0010ffff☀-➿←-⇿⬀-⯿️]", "", text or "")
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", t, flags=re.M)        # bullets, numbering
    t = re.sub(r"[*_`#>|~]+", "", t)
    t = re.sub(r"\s*\n+\s*", ". ", t)
    t = re.sub(r"\.\s*\.+", ".", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    if len(t) > limit:
        cut = t[:limit].rsplit(". ", 1)[0]
        t = (cut if len(cut) > limit // 2 else t[:limit].rsplit(" ", 1)[0]).rstrip(".,;: ") + "."
    return t


def pattern_words(steps: list) -> str:
    """A buzz pattern, in words, from its real (offset, motor, ms) steps."""
    steps = sorted(steps)
    if not steps:
        return "no buzz"
    names = haptics.MOTOR_NAMES

    def m(k):
        return names[int(k) % 8].lower()
    if len({(mo, ms) for _, mo, ms in steps}) == 1:
        n, (_, mo, ms) = len(steps), steps[0]
        return (f"{_NUM.get(n, n)} {'buzz' if n == 1 else 'buzzes'} of {ms} milliseconds "
                f"on the {m(mo)} motor")
    if len({o for o, _, _ in steps}) == 1:
        return "buzzes together on " + " and ".join(f"{m(mo)} ({ms} ms)" for _, mo, ms in steps[:4])
    parts = [f"{m(mo)} {ms} ms" for _, mo, ms in steps[:6]]
    more = ", and so on" if len(steps) > 6 else ""
    return "buzzes in order: " + ", then ".join(parts) + more


def describe_alert(rec: dict) -> tuple:
    """(what happened, what it meant, urgent) for one alert record, or None to skip."""
    src, cue = rec.get("source"), rec.get("cue")
    if rec.get("dry") or rec.get("muted") or src in SKIP_SOURCES or rec.get("repeat"):
        return None
    w = where(rec)
    if src == "name":
        return ((f"your name was called {w}" if w else "your name was called, direction unclear"),
                "one short buzz on the side the caller was" if w else "one buzz at the front, direction unclear",
                False)
    if src == "sound":
        label = rec.get("label") or "a taught sound"
        urgent = (rec.get("label") or "").lower() in URGENT_SOUNDS
        return (f"{label}{', ' + w if w else ''}",
                "a sound pattern, then one buzz toward where it came from", urgent)
    if src == "avoid":
        what = rec.get("obstacle") or "something"
        return (f"obstacle ahead: {what}" + (f", about {rec['metres']} metres" if rec.get("metres") else ""),
                "three long front buzzes: stop", True)
    if src == "find":
        target = rec.get("target") or "the object"
        if cue == "centered":
            return (f"found {target}, straight ahead", "two quick front buzzes", False)
        if cue == "lost":
            return (f"lost {target}", "one long front buzz: turn slowly to find it again", False)
        if cue == "unknown":
            return ("a find request I could not look for", "three quick back buzzes", False)
    return None


def _clock(rec: dict) -> str:
    return rec.get("time", "")[:5]


# ---------------------------------------------------------------------------
# Tools: plain Python over existing objects. No link, no haptics player.
# ---------------------------------------------------------------------------

def _safe_tool(fn: Callable) -> Callable:
    """Plugin tools answer in speakable text and never raise into the agent."""
    import functools

    @functools.wraps(fn)
    def run(*a, **k):
        try:
            return plain_text(str(fn(*a, **k)))
        except Exception as e:
            print(f"[band-agent] tool {fn.__name__} failed ({type(e).__name__}: {str(e)[:80]})")
            return "I could not do that."
    return run


class BandTools:
    def __init__(self, alerts, status_fn: Callable[[], dict],
                 set_target_fn: Optional[Callable[[str], Optional[str]]] = None,
                 urgent_fn: Optional[Callable[[str], tuple]] = None,
                 clock: Callable[[], float] = time.time):
        self.alerts, self.status_fn = alerts, status_fn
        self.set_target_fn, self.urgent_fn, self.clock = set_target_fn, urgent_fn, clock
        self._urgent_sent: deque = deque()
        self.extra: list = []                  # plugin tools: see add_tool / load_plugins

    def add_tool(self, fn: Callable) -> bool:
        """Add a tool: a function with typed arguments and a docstring, returning a string."""
        name = getattr(fn, "__name__", "")
        if not name or name in CORE_TOOLS or any(t.__name__ == name for t in self.extra):
            print(f"[band-agent] tool {name!r} skipped: name missing or already taken")
            return False
        self.extra.append(_safe_tool(fn))
        return True

    def load_plugins(self, names: Optional[list] = None) -> list:
        """Import optional tool modules; returns the names of the tools added."""
        import importlib
        names = names if names is not None else (
            [n.strip() for n in os.environ.get("HDB_AGENT_PLUGINS", "").split(",") if n.strip()]
            or list(DEFAULT_PLUGINS))
        added = []
        for mod in names:
            try:
                try:
                    m = importlib.import_module(mod)
                except ModuleNotFoundError as e:
                    if e.name == mod:
                        continue                    # no such plugin installed: not an error
                    raise
                fns = list(m.make_tools(self)) if hasattr(m, "make_tools") else list(getattr(m, "TOOLS", []))
                added += [f.__name__ for f in fns if self.add_tool(f)]
            except Exception as e:                  # a broken plugin never stops the agent
                print(f"[band-agent] plugin {mod} skipped ({type(e).__name__}: {str(e)[:80]})")
        if added:
            print(f"[band-agent] plugin tools: {', '.join(added)}")
        return added

    def get_status(self) -> str:
        """Is the band connected, are the camera and microphones working, what is it looking for."""
        try:
            s = self.status_fn() or {}
        except Exception:
            return "I could not read the band status right now."
        out = ["The band is connected." if s.get("connected") else "The band is not connected."]
        faults = list((s.get("faults") or {}).values())
        out.append("Everything else looks fine." if not faults and s.get("connected")
                   else ("Problems: " + "; ".join(faults) + ".") if faults else "")
        if s.get("target"):
            out.append(f"Looking for: {s['target']}.")
        st = s.get("settings") or {}
        if st.get("intensity"):
            muted = [k.replace("_", " ") for k in ("name_alerts", "find_cues", "sound_alerts") if st.get(k) == "off"]
            out.append(f"Buzz strength is {st['intensity']}." + (f" Muted: {', '.join(muted)}." if muted else ""))
        last = self._meaningful(1)
        if last:
            what, _, _ = last[0][1]
            out.append(f"Last alert at {_clock(last[0][0])}: {what}.")
        return plain_text(" ".join(x for x in out if x))

    def _meaningful(self, n: int, since: float = 0.0, source: Optional[str] = None) -> list:
        """[(record, (what, meant, urgent))], newest first, repeats and tests left out."""
        out = []
        for rec in reversed(self.alerts.recent(500)):
            if rec.get("t", 0) < since:
                break
            if source and rec.get("source") != source:
                continue
            d = describe_alert(rec)
            if d:
                out.append((rec, d))
                if len(out) >= n:
                    break
        return out

    def recent_alerts(self, minutes: int = 60, limit: int = 8, source: str = "any") -> str:
        """What the band buzzed recently, newest first. source: any, name, find, sound or avoid."""
        since = self.clock() - max(1, int(minutes)) * 60
        items = self._meaningful(max(1, min(int(limit), 15)), since, None if source == "any" else source)
        if not items:
            return f"Nothing in the last {int(minutes)} minutes."
        return plain_text(f"In the last {int(minutes)} minutes: " + " ".join(
            f"{_clock(r)} {d[0]}{', urgent' if d[2] else ''}." for r, d in items))

    def daily_summary(self, period: str = "today") -> str:
        """Count and list the alerts for a period: today, hour, or day. Newest first, with meanings."""
        p = (period or "today").lower()
        now = self.clock()
        if "hour" in p:
            since, label = now - 3600, "In the last hour"
        elif "24" in p or p in ("day", "yesterday"):
            since, label = now - 86400, "In the last 24 hours"
        else:
            lt = time.localtime(now)
            since = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
            label = "Today"
        items = self._meaningful(200, since)
        if not items:
            return f"{label}: no alerts."
        lines, prev = [], None
        for rec, (what, meant, urgent) in items:                      # newest first, merge near-duplicates
            key = (rec.get("source"), what)
            if prev and prev[0] == key and abs(prev[1] - rec["t"]) < 60:
                lines[-1][1] += 1
                prev = (key, rec["t"])
                continue
            lines.append([f"{_clock(rec)} {what}{', urgent' if urgent else ''}", 1])
            prev = (key, rec["t"])
        n = sum(c for _, c in lines)
        body = " ".join(f"{t}{' times ' + str(c) if c > 1 else ''}." for t, c in lines[:10])
        more = f" And {len(lines) - 10} more." if len(lines) > 10 else ""
        return plain_text(f"{label}: {n} {'buzz' if n == 1 else 'buzzes'}. {body}{more}")

    def pattern_legend(self) -> str:
        """What each buzz pattern means. Built from the real patterns, so it cannot drift."""
        from . import avoid, guide, notify
        L = [
            ("Your name called", notify.pattern("directional", 6)),
            ("Name called, direction unclear", notify.pattern("generic")),
            ("Object found, straight ahead", notify.pattern("centered")),
            ("Object lost", notify.pattern("lost")),
            ("Got it, looking", notify.pattern("ack")),
            ("Cannot look for that", notify.pattern("unknown")),
            ("Taught sound", notify.pattern("sound")),
            ("Obstacle ahead, stop", avoid.stop_cue()),
        ]
        parts = [f"{what}: {pattern_words(steps)}." for what, steps in L]
        parts.append(f"While finding an object the band keeps buzzing toward it, from "
                     f"every {guide.SLOW_S:g} seconds when it is far off to every "
                     f"{guide.FAST_S:g} seconds when it is straight ahead and close.")
        return plain_text(" ".join(parts), limit=900)

    def explain_buzz(self, when: str = "last") -> str:
        """Explain one buzz: the last one, or the one nearest a time like 11:05."""
        items = self._meaningful(200)
        if not items:
            return "I have no buzz to explain yet."
        pick = items[0]
        m = re.search(r"(\d{1,2}):(\d{2})", when or "")
        if m:
            target = int(m.group(1)) * 60 + int(m.group(2))

            def mins(rec):
                h, mi = _clock(rec).split(":")
                return int(h) * 60 + int(mi)
            pick = min(items, key=lambda it: abs(mins(it[0]) - target))
        rec, (what, meant, urgent) = pick
        from . import avoid, notify
        try:
            if rec.get("source") == "avoid" and rec.get("action") == "stop":
                steps = avoid.stop_cue()        # logged as a generic cue, but it buzzed the stop pattern
            else:
                steps = notify.pattern(rec["cue"], rec.get("motor"), rec.get("category"))
            felt = f" The pattern was {pattern_words(steps)}."
        except Exception:
            felt = ""
        return plain_text(f"At {_clock(rec)}: {what}. That buzz meant {meant}.{felt}")

    def set_find_target(self, target: str) -> str:
        """Tell the band to look for an object, for example keys or a banana."""
        target = (target or "").strip()
        if not target:
            return "Tell me what to look for."
        if not self.set_target_fn:
            return "The camera is not running, so I cannot look for things."
        try:
            got = self.set_target_fn(target)
        except Exception:
            return "I could not start that search."
        return (f"Looking for {got}. The band will buzz toward it." if got
                else f"I cannot look for {target}.")

    def send_urgent(self, message: str) -> str:
        """Text the caregiver one short urgent line. Limited to three an hour."""
        now = self.clock()
        while self._urgent_sent and now - self._urgent_sent[0] > 3600:
            self._urgent_sent.popleft()
        if len(self._urgent_sent) >= 3:
            return "I already texted the caregiver three times this hour."
        if not self.urgent_fn:
            return "Caregiver texts are not set up."
        ok, err = self.urgent_fn("[Band] " + plain_text(message, 200))
        if ok:
            self._urgent_sent.append(now)
        return "I texted the caregiver." if ok else "I could not reach the caregiver."


# ---------------------------------------------------------------------------
# Brain 1: offline keyword router (no model, no key)
# ---------------------------------------------------------------------------

HELP = ("You can text: status, what did I miss today, what happened in the last hour, "
        "what does each buzz mean, what was that buzz, find my keys, or tell the caregiver.")


def route(text: str, tools: BandTools) -> str:
    q = " ".join((text or "").lower().replace("?", " ").replace("'", "").split())
    if not q:
        return HELP
    m = re.search(r"\b(?:find|look for|where (?:is|are|did i (?:put|leave)))\s+(?:my |the |a |an )?(.+)$", q)
    if m and not re.search(r"\b(?:what|status|buzz)\b", q.split(m.group(0))[0]):
        return tools.set_find_target(m.group(1).strip())
    if re.search(r"\b(?:tell|text|message|alert)\b.*\b(?:caregiver|mom|dad|care)\b|\bneed help\b|\bhelp me\b", q):
        msg = re.sub(r"^(?:please )?(?:tell|text|message|alert) (?:the )?(?:caregiver|care ?giver)\s*", "", q).strip()
        return tools.send_urgent(msg or "The wearer asked for help.")
    if re.search(r"\b(?:mean|legend|patterns?|guide|how do)\b", q) and "buzz" in q + " pattern":
        return tools.pattern_legend()
    if re.search(r"\b(?:what was that|explain|why did)\b", q):
        return tools.explain_buzz(q)
    if re.search(r"\b(?:miss|summary|today|day|happened|alerts?)\b", q):
        if re.search(r"\bhour\b|\blast (?:few|\d+) min", q):
            return tools.recent_alerts(60) if "hour" in q else tools.recent_alerts(30)
        return tools.daily_summary("today")
    if re.search(r"\b(?:ok|okay|safe|fine|alright)\b", q) or re.search(r"\b(?:status|working|connected|how)\b", q):
        s = tools.get_status()
        if re.search(r"\b(?:ok|okay|safe|alright)\b|\bis (?:she|he|they)\b", q):
            return plain_text(f"{s} {tools.recent_alerts(60, 3)}")
        return s
    return HELP


# ---------------------------------------------------------------------------
# Brain 2: LangChain deep agent (same tools)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the band agent for a haptic headband worn by a blind or Deaf wearer. \
The wearer or their caregiver is texting you. Your answers are read aloud by text-to-speech, \
so write two or three short plain sentences: no markdown, tables, lists, bullets or emoji.

Use your tools for every fact: get_status for how the band is doing, daily_summary and \
recent_alerts for what happened, explain_buzz and pattern_legend for what a buzz meant, \
set_find_target when asked to find something, send_urgent only when the wearer asks you to \
alert the caregiver or an alert is clearly an emergency. Never invent alerts or times. Say \
what a buzz meant in plain words. You cannot make the band buzz: it buzzes by itself. \
If you are unsure what is asked, ask one short question. You may also have extra tools \
(for example building lookup): use them when the question is about what they cover."""


def load_model(name: Optional[str] = None):
    from .agent_find import load_model as lm
    return lm(name or os.environ.get("HDB_BAND_AGENT_MODEL", DEFAULT_MODEL))


def langchain_tools(tools: BandTools) -> list:
    from langchain_core.tools import StructuredTool
    fns = [getattr(tools, n) for n in CORE_TOOLS] + list(tools.extra)
    return [StructuredTool.from_function(f, name=f.__name__, description=f.__doc__ or f.__name__)
            for f in fns]


def build_deep_agent(tools: BandTools, model=None):
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver
    return create_deep_agent(model=model if model is not None else load_model(),
                             tools=langchain_tools(tools), system_prompt=SYSTEM_PROMPT,
                             checkpointer=MemorySaver())


class BandAgent:
    """reply(text) -> a short speakable answer. Never raises."""

    def __init__(self, tools: BandTools, model=None, offline: bool = False, thread_id: str = "band"):
        self.tools, self.thread_id = tools, thread_id
        self.agent, self.error = None, None
        self.offline = offline
        if not offline:
            try:
                self.agent = build_deep_agent(tools, model)
            except Exception as e:      # missing package or key: the router still works
                self.error = f"{type(e).__name__}: {str(e)[:120]}"

    @property
    def brain(self) -> str:
        return "deep agent" if self.agent is not None else "keyword router"

    def reply(self, text: str) -> str:
        if self.agent is not None:
            out: dict = {}

            def run():
                try:
                    res = self.agent.invoke({"messages": [{"role": "user", "content": text}]},
                                            config={"configurable": {"thread_id": self.thread_id}})
                    last = res["messages"][-1]
                    c = getattr(last, "content", last)
                    out["text"] = c if isinstance(c, str) else " ".join(
                        b.get("text", "") for b in c if isinstance(b, dict))
                except Exception as e:
                    out["error"] = e
            t = threading.Thread(target=run, daemon=True)
            t.start()
            t.join(AGENT_TIMEOUT_S)
            if out.get("text", "").strip():
                return plain_text(out["text"])
            print(f"[band-agent] agent failed ({out.get('error') or 'timeout'}); using the router")
        try:
            return plain_text(route(text, self.tools))
        except Exception:
            return "Sorry, I could not work that out. Try status."


# ---------------------------------------------------------------------------
# Service: texts in, replies out, urgent alerts out
# ---------------------------------------------------------------------------

from .notify import norm_handle        # noqa: E402  (phones by digits, emails lowercased)


class AgentService:
    """Photon in and out. allow: the configured numbers; anyone else is ignored."""

    def __init__(self, agent: BandAgent, sender, allow: list, alerts=None, caregiver_phone: Optional[str] = None,
                 caregiver_on: Callable[[], bool] = lambda: True,
                 clock: Callable[[], float] = time.time, urgent_gap_s: float = 60.0, max_urgent_per_hour: int = 6):
        self.agent, self.sender = agent, sender
        self.allow = {norm_handle(p) for p in allow if norm_handle(p)}
        self.caregiver_phone, self.caregiver_on, self.clock = caregiver_phone, caregiver_on, clock
        self.urgent_gap_s, self.max_urgent = urgent_gap_s, max_urgent_per_hour
        self._q: queue.Queue = queue.Queue(maxsize=20)
        self._last_urgent: dict = {}
        self._urgent_times: deque = deque()
        self.ignored = self.handled = self.proactive = 0
        self._stop = threading.Event()
        if alerts is not None:
            alerts.subscribe(self.on_alert)          # only enqueues: never blocks the buzz path
        if hasattr(sender, "on_message"):
            sender.on_message(self.on_text)
        self.agent.tools.urgent_fn = self.urgent_text

    # -- inbound
    def on_text(self, sender: str, text: str) -> None:
        if norm_handle(sender) not in self.allow:
            self.ignored += 1                         # not on the list: no reply, not even "no"
            return
        try:
            self._q.put_nowait(("text", sender, text))
        except queue.Full:
            pass

    # -- proactive
    def urgent_kind(self, rec: dict) -> Optional[str]:
        if rec.get("dry") or rec.get("repeat") or rec.get("muted") or rec.get("type") != "alert":
            return None
        if rec.get("source") == "avoid" and rec.get("action") == "stop":
            return "avoid"
        if rec.get("source") == "sound" and (rec.get("label") or "").lower() in URGENT_SOUNDS:
            return (rec.get("label") or "sound").lower()
        return None

    def on_alert(self, rec: dict) -> None:
        kind = self.urgent_kind(rec)
        if kind:
            try:
                self._q.put_nowait(("urgent", kind, rec))
            except queue.Full:
                pass

    def urgent_text(self, text: str) -> tuple:
        if not (self.caregiver_phone and self.caregiver_on()):
            return False, "caregiver texts are off or not set"
        return self.sender.send(self.caregiver_phone, text)

    def _proactive(self, kind: str, rec: dict) -> None:
        now = self.clock()
        if not (self.caregiver_phone and self.caregiver_on()):
            return
        while self._urgent_times and now - self._urgent_times[0] > 3600:
            self._urgent_times.popleft()
        if now - self._last_urgent.get(kind, -1e18) < self.urgent_gap_s or len(self._urgent_times) >= self.max_urgent:
            return
        d = describe_alert(rec)
        if not d:
            return
        self._last_urgent[kind] = now
        self._urgent_times.append(now)
        ok, _ = self.sender.send(self.caregiver_phone, plain_text(f"[Band] Urgent at {_clock(rec)}: {d[0]}."))
        self.proactive += bool(ok)

    # -- worker
    def start(self) -> "AgentService":
        def run():
            while not self._stop.is_set():
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    if item[0] == "text":
                        _, sender, text = item
                        reply = self.agent.reply(text)
                        self.sender.send(sender, reply)
                        self.handled += 1
                    else:
                        self._proactive(item[1], item[2])
                except Exception as e:                # never kill the worker, never touch the buzz path
                    print(f"[band-agent] {type(e).__name__}: {e}")
        threading.Thread(target=run, name="band-agent", daemon=True).start()
        return self

    def drain(self, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self._q.empty():
                time.sleep(0.05)
                if self._q.empty():
                    return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        self._stop.set()


def from_app(app, caregiver, vision=None, offline: bool = False) -> Optional[AgentService]:
    """Wire the agent to the running band app. Uses the caregiver's Photon sender."""
    from .notify import load_env
    load_env()
    allow = [p for p in (os.environ.get("WEARER_PHONE"), os.environ.get("CAREGIVER_PHONE")) if p]

    def status():
        d = dict(app.status())
        d["target"] = (app.state.get("object") or "").split(":")[0].strip("– ") or None
        d["settings"] = app.settings.snapshot()
        return d
    tools = BandTools(app.alerts, status,
                      (lambda t: vision.set_target(t)) if vision is not None else None)
    tools.load_plugins()
    agent = BandAgent(tools, offline=offline)
    svc = AgentService(agent, caregiver.sender, allow, app.alerts, caregiver.phone,
                       caregiver_on=lambda: app.settings.caregiver_texts == "on")
    print(f"[hdb] band agent: {agent.brain}" + (f" ({agent.error})" if agent.error else "")
          + f"; answers {len(svc.allow)} configured number(s)")
    if not hasattr(caregiver.sender, "on_message"):
        print("[hdb] band agent: texting is a dry run (no Photon credentials): no texts in or out")
    return svc.start()


# ---------------------------------------------------------------------------
# CLI: talk to it in the terminal
# ---------------------------------------------------------------------------

def _demo_tools() -> BandTools:
    from .recall import AlertLog
    log = AlertLog()
    for src, cue, motor, kw in [("name", "directional", 6, {}), ("sound", "sound", 0, {"label": "doorbell"}),
                                ("sound", "sound", 4, {"label": "alarm"}),
                                ("find", "centered", None, {"target": "keys"}),
                                ("avoid", "generic", None, {"action": "stop", "obstacle": "chair", "metres": 0.8})]:
        log.add(src, cue, motor, **kw)
    return BandTools(log, lambda: {"connected": True, "faults": {}, "target": "keys",
                                   "settings": {"intensity": "medium"}},
                     lambda t: t, lambda m: (True, None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", action="store_true", help="talk to the agent here (sample alerts, no texts)")
    ap.add_argument("--offline", action="store_true", help="keyword router only, no model")
    a = ap.parse_args(argv)
    if not a.chat:
        ap.print_help()
        return 0
    from .notify import load_env
    load_env()
    demo = _demo_tools()
    demo.load_plugins()
    agent = BandAgent(demo, offline=a.offline)
    print(f"[band-agent] {agent.brain}" + (f" ({agent.error})" if agent.error else "")
          + "; sample alerts loaded; Ctrl-D to quit")
    try:
        while True:
            print("band:", agent.reply(input("you: ")))
    except (EOFError, KeyboardInterrupt):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
