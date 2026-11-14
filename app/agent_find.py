"""Find-an-object as a LangChain Deep Agent: the agent picks the target and the
search strategy; the fast left/right/centered/lost loop stays in plain Python
(app/vision.py -> app/guide.py -> app/haptics.py).

Owner: HDB-Vision (branch ``vision``). Pattern follows
langchain-samples/hack-princeton-intro-to-deep-agents (agent.py).

    main agent  plans with write_todos (TodoListMiddleware), delegates with task(),
                and calls start_guiding(target), which pauses for the wearer to confirm
                (interrupt_on) before the band starts guiding
    scout       tools look() and sweep(), wrapping the head camera
    memory-keeper  keeps /seen_objects.md in the agent filesystem (built-in file
                tools); the file is loaded from and saved to recordings/seen_objects.md
                (gitignored) so it survives restarts

Running it
----------
    pip install deepagents langchain-anthropic python-dotenv
    export ANTHROPIC_API_KEY=...        # or put it in .env (gitignored); never commit it
    python -m app.agent_find "find the banana"                 # webcam 0, simulated band
    python -m app.agent_find "find my cup" --mirror --speak    # laptop camera, spoken prompts
    python -m app.agent_find "find the phone" --source clip.mp4
    python -m app.agent_find "find the banana" --port /dev/cu.usbserial-0001   # real band
    python -m app.agent_find --selftest                        # scripted model: no key, no camera

The CLI owns the serial port, so don't run it alongside `python -m app.main`.
Model: HDB_AGENT_MODEL (default anthropic:claude-opus-5-5; for lower latency try
anthropic:claude-haiku-5-5).

Library use (e.g. from the app after hold-to-talk):
    fa = FindAgent(vision)                       # vision: a started app.vision.Vision
    out = fa.run("find the banana", confirm=lambda target, why: True)
    out.target, out.reply                        # guiding target (or None), spoken reply
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .vision import COCO_CLASSES, REPO_ROOT, resolve_target

DEFAULT_MODEL = "anthropic:claude-opus-5-5"
MEMORY_PATH = "/seen_objects.md"
MEMORY_DISK = REPO_ROOT / "recordings" / "seen_objects.md"
MEMORY_HEADER = ("# Seen objects\n\n| object | where | when | confidence |\n"
                 "|---|---|---|---|\n")


def side_of(offset: float) -> str:
    """Wearer's-view side for an offset in -1..1."""
    if offset < -0.6:
        return "far left"
    if offset < -0.15:
        return "left"
    if offset <= 0.15:
        return "straight ahead"
    if offset <= 0.6:
        return "right"
    return "far right"


# --------------------------------------------------------------------------
# Scene tools: plain Python around Vision (testable without an LLM)
# --------------------------------------------------------------------------

class SceneTools:
    """look()/sweep()/start_guiding() over a running Vision.

    vision needs: latest_raw(), detector.detect_all(frame), mirror, set_target(name).
    say(text) tells the wearer something (print, `say`, or a buzz)."""

    def __init__(self, vision, say: Callable[[str], None] = print):
        self.vision = vision
        self.say = say
        self.guiding: Optional[str] = None

    def _frame_objects(self, frame, min_score: float = 0.3) -> list:
        w = frame.shape[1]
        out = []
        for name, x1, y1, x2, y2, score in self.vision.detector.detect_all(frame, min_score):
            off = (x1 + x2) / w - 1.0
            if getattr(self.vision, "mirror", False):
                off = -off
            area = (x2 - x1) * (y2 - y1) / (w * frame.shape[0])
            out.append((name, round(off, 2), round(score, 2), round(area, 3)))
        return out

    def look(self) -> str:
        frame = self.vision.latest_raw()
        if frame is None:
            return "camera has no frame yet"
        objs = self._frame_objects(frame)
        if not objs:
            return "nothing recognizable in view"
        parts = [f"{n} {side_of(o)} (offset {o:+.2f}, score {s:.2f}"
                 + (", close" if a > 0.15 else "") + ")" for n, o, s, a in objs[:8]]
        return f"at {time.strftime('%H:%M:%S')}: " + "; ".join(parts)

    def sweep(self, target: str = "", seconds: float = 6.0) -> str:
        want = resolve_target(target) if target else None
        if target and want is None:
            return f"'{target}' is not something the camera can recognize"
        seconds = max(2.0, min(float(seconds), 12.0))
        self.say("Turn your head slowly from left to right.")
        t0 = time.monotonic()
        seen: dict = {}       # class -> (best score, offset, seconds into sweep)
        frames = 0
        while time.monotonic() - t0 < seconds:
            frame = self.vision.latest_raw()
            if frame is not None:
                frames += 1
                for n, off, s, _ in self._frame_objects(frame, 0.25):
                    if n not in seen or s > seen[n][0]:
                        seen[n] = (s, off, round(time.monotonic() - t0, 1))
            time.sleep(0.15)
        if want:
            if want in seen:
                s, off, at = seen[want]
                return (f"{want} seen {at}s into the {seconds:.0f}s sweep, {side_of(off)} at that "
                        f"moment, score {s:.2f}. Other objects: "
                        + (", ".join(k for k in seen if k != want) or "none"))
            return (f"{want} not seen in a {seconds:.0f}s sweep ({frames} frames). Seen: "
                    + (", ".join(f"{k} ({v[0]:.2f})" for k, v in seen.items()) or "nothing"))
        return (f"{seconds:.0f}s sweep saw: "
                + (", ".join(f"{k} (score {v[0]:.2f}, {v[2]}s in)" for k, v in seen.items())
                   or "nothing recognizable"))

    def start_guiding(self, target: str) -> str:
        cls = self.vision.set_target(target)   # COCO class, or an open-vocabulary phrase
        if not cls:
            return f"cannot guide to '{target}': the camera can't look for that"
        self.guiding = cls
        return (f"guiding to {cls} now: the band buzzes left/right, two front pulses when "
                "centered, one long buzz if lost")


# --------------------------------------------------------------------------
# Deep agent
# --------------------------------------------------------------------------

MAIN_PROMPT = f"""You help a blind person wearing a haptic band with a head camera find an \
object. The band does the fine guiding by itself once you call start_guiding; your job is \
to pick the right target and a search strategy.

Workflow:
1. Plan with write_todos first and keep it updated.
2. Ask the memory-keeper (task tool) what {MEMORY_PATH} says about the object.
3. Send the scout (task tool) to find it: look first, sweep if it isn't in view.
4. Ask the memory-keeper to record what the scout reported (object, where, when).
5. Call start_guiding with the object's name. The wearer confirms or corrects it.
If the object can't be recognized by the camera, say so and suggest a similar object it can \
see. Your final reply is spoken aloud: one or two short sentences, no markdown."""

SCOUT = {
    "name": "scout",
    "description": "Looks through the wearer's head camera. Give it the object to find; it "
                   "reports whether and where the object is (left/straight ahead/right) and "
                   "what else is in view.",
    "system_prompt": "You control a head-mounted camera. Call look() first. If the object "
                     "isn't in view, call sweep(target) once. Report in one or two lines: "
                     "found or not, direction, score, and other objects seen. Do not guess.",
}

MEMORY_KEEPER = {
    "name": "memory-keeper",
    "description": f"Keeps {MEMORY_PATH}: a table of objects seen, where and when. Ask it "
                   "what is known about an object, or give it sightings to record.",
    "system_prompt": f"You maintain {MEMORY_PATH}, a markdown table with columns object, "
                     "where, when, confidence. To answer a question, read_file it and reply "
                     "with the matching rows (or 'nothing recorded'). To record a sighting, "
                     "edit_file to update that object's row (or add one), keeping the table "
                     "format. Reply in one line.",
    "tools": [],   # built-in file tools only
}


def _make_tools(scene: SceneTools):
    from langchain_core.tools import tool

    @tool
    def look() -> str:
        """Describe the objects in the head camera's current view, with their direction."""
        return scene.look()

    @tool
    def sweep(target: str = "", seconds: float = 6.0) -> str:
        """Ask the wearer to turn their head slowly while watching for an object.

        Args:
            target: object to watch for, e.g. "banana" (empty = report everything seen)
            seconds: sweep length, 2-12 s
        """
        return scene.sweep(target, seconds)

    @tool
    def start_guiding(target: str) -> str:
        """Start the band's haptic guidance toward an object. The wearer confirms first.

        Args:
            target: the object to guide to, e.g. "banana", "cup", "cell phone"
        """
        return scene.start_guiding(target)

    return look, sweep, start_guiding


def load_model(name: Optional[str] = None, **kwargs):
    from langchain.chat_models import init_chat_model
    name = name or os.environ.get("HDB_AGENT_MODEL", DEFAULT_MODEL)
    if name.startswith("anthropic:"):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (export it or add it to .env, "
                               "which is gitignored)")
        return init_chat_model(name, api_key=key, **kwargs)
    return init_chat_model(name, **kwargs)


# --------------------------------------------------------------------------
# Hold-to-talk target picker: one fast LangChain call for the live app
# --------------------------------------------------------------------------
# The keyword matcher in app/speech.py handles "find the banana" instantly. When it
# finds no detector class ("I'm thirsty", "where did I put the thing I read with"),
# app/speech.py asks this picker. It runs only on that path, never in the cue loop.

PICK_MODEL = os.environ.get("HDB_TARGET_MODEL", "anthropic:claude-haiku-5-5")
PICK_TIMEOUT_S = 6.0

PICK_PROMPT = """A blind or low-vision person wearing a haptic band with a head camera \
held a button and said something (speech-to-text, may contain mishearings). Decide which \
physical object they want help finding, and call TargetChoice.

Prefer exactly one of these detector classes when one fits (it is faster and more \
reliable): {classes}.
If none fits, give a short concrete noun phrase (1-3 words) an open-vocabulary detector \
can look for, e.g. "keys", "headphones", "white cane".
If they are not asking to find a physical object, set target to null.
Examples: "I'm thirsty" -> bottle; "where did I leave my keys" -> keys; \
"I want to sit down" -> chair; "what time is it" -> null."""


def _target_choice_tool():
    from pydantic import BaseModel, Field

    class TargetChoice(BaseModel):
        """The object the wearer wants to find."""
        target: Optional[str] = Field(
            None, description="detector class or short noun phrase; null if no object asked for")
        reason: str = Field("", description="a few words on why")

    return TargetChoice


def pick_target(text: str, model=None) -> Optional[str]:
    """Free speech -> object to find (COCO class if one fits, else a short phrase), or
    None. Raises on model/network errors; callers fall back to keywords."""
    from langchain_core.messages import HumanMessage, SystemMessage
    if not (text or "").strip():
        return None
    tool = _target_choice_tool()
    if model is None:
        model = load_model(PICK_MODEL, timeout=PICK_TIMEOUT_S, max_retries=1)
    llm = model.bind_tools([tool], tool_choice="TargetChoice")
    msg = llm.invoke([SystemMessage(PICK_PROMPT.format(classes=", ".join(COCO_CLASSES))),
                      HumanMessage(text)])
    calls = getattr(msg, "tool_calls", None) or []
    raw = calls[0]["args"].get("target") if calls else _text(msg)
    t = " ".join(str(raw or "").lower().strip(" .\"'").split())
    if not t or t in ("null", "none", "nothing"):
        return None
    return resolve_target(t) or t


def target_picker(model=None) -> Optional[Callable[[str], Optional[str]]]:
    """The picker app/speech.py should use, or None if LangChain is off.

    HDB_TARGET_LLM=0 turns it off, =1 forces it on; by default it is on when
    langchain is installed and the model's API key is set."""
    mode = os.environ.get("HDB_TARGET_LLM", "auto").lower()
    if mode in ("0", "off", "false", "no"):
        return None
    if model is None and mode not in ("1", "on", "true", "yes"):
        if PICK_MODEL.startswith("anthropic:") and not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        if PICK_MODEL.startswith("google_genai:") and not os.environ.get("GOOGLE_API_KEY"):
            return None
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        return None
    cache = {}

    def pick(text: str) -> Optional[str]:
        if "m" not in cache:  # build the client once, on first use
            cache["m"] = model if model is not None else load_model(
                PICK_MODEL, timeout=PICK_TIMEOUT_S, max_retries=1)
        return pick_target(text, cache["m"])

    return pick


def build_agent(scene: SceneTools, model=None, scout_model=None, memory_model=None):
    from deepagents import create_deep_agent
    from langchain.agents.middleware import TodoListMiddleware
    from langgraph.checkpoint.memory import MemorySaver

    look, sweep, start_guiding = _make_tools(scene)
    scout = dict(SCOUT, tools=[look, sweep])
    keeper = dict(MEMORY_KEEPER)
    if scout_model is not None:
        scout["model"] = scout_model
    if memory_model is not None:
        keeper["model"] = memory_model
    return create_deep_agent(
        model=model if model is not None else load_model(),
        tools=[start_guiding],
        subagents=[scout, keeper],
        system_prompt=MAIN_PROMPT,
        middleware=[TodoListMiddleware()],
        interrupt_on={"start_guiding": {"allowed_decisions": ["approve", "edit", "reject"]}},
        checkpointer=MemorySaver(),
    )


@dataclass
class FindResult:
    target: Optional[str]     # what the band is guiding to now, or None
    reply: str                # final spoken reply
    todos: list
    memory: str               # /seen_objects.md after the run


def _text(msg) -> str:
    t = getattr(msg, "text", None)  # a property on current langchain-core
    return t if isinstance(t, str) else str(getattr(msg, "content", ""))


class FindAgent:
    def __init__(self, vision, say: Callable[[str], None] = print, model=None,
                 scout_model=None, memory_model=None, memory_disk: Path = MEMORY_DISK):
        self.scene = SceneTools(vision, say)
        self.memory_disk = Path(memory_disk)
        self.agent = build_agent(self.scene, model, scout_model, memory_model)
        self.on_step: Optional[Callable[[str], None]] = None
        self._logged = 0

    def _seed_files(self) -> dict:
        from deepagents.backends.utils import create_file_data
        text = self.memory_disk.read_text() if self.memory_disk.exists() else MEMORY_HEADER
        return {MEMORY_PATH: create_file_data(text)}

    def run(self, request: str, confirm: Callable[[str, str], object]) -> FindResult:
        """confirm(target, reason) -> True (approve), False (reject), or a str (guide to
        that object instead). Called when the agent wants to start guiding."""
        from langgraph.types import Command
        config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 60}
        self._logged = 0
        result = self.agent.invoke({"messages": [{"role": "user", "content": request}],
                                    "files": self._seed_files()}, config=config)
        self._log(result)
        while result.get("__interrupt__"):
            payload = result["__interrupt__"][0].value
            decisions = []
            for action in payload.get("action_requests", []):
                target = action["args"].get("target", "")
                ans = confirm(target, action.get("description", ""))
                if ans is True:
                    decisions.append({"type": "approve"})
                elif isinstance(ans, str) and ans.strip():
                    decisions.append({"type": "edit", "edited_action": {
                        "name": action["name"], "args": {"target": ans.strip()}}})
                else:
                    decisions.append({"type": "reject",
                                      "message": "The wearer said no. Ask what they want."})
            result = self.agent.invoke(Command(resume={"decisions": decisions}), config=config)
            self._log(result)
        files = result.get("files") or {}
        memory = files.get(MEMORY_PATH, {})
        memory = memory.get("content", "") if isinstance(memory, dict) else str(memory)
        if isinstance(memory, list):
            memory = "\n".join(memory)
        if memory:
            self.memory_disk.parent.mkdir(parents=True, exist_ok=True)
            self.memory_disk.write_text(memory)
        return FindResult(self.scene.guiding, _text(result["messages"][-1]),
                          list(result.get("todos") or []), memory)

    def _log(self, result) -> None:
        if not self.on_step:
            return
        msgs = result.get("messages", [])
        new, self._logged = msgs[self._logged:], len(msgs)
        for m in new:
            for call in getattr(m, "tool_calls", None) or []:
                args = call["args"]
                if call["name"] == "task":
                    self.on_step(f"-> {args.get('subagent_type')}: "
                                 f"{str(args.get('description'))[:90]}")
                elif call["name"] == "write_todos":
                    self.on_step("plan: " + " | ".join(t["content"] for t in args["todos"]))
                else:
                    self.on_step(f"{call['name']}({args})")


# --------------------------------------------------------------------------
# Self-test: scripted chat models, fake camera (no API key, no webcam)
# --------------------------------------------------------------------------

def _scripted_model(script: list):
    """A chat model that replays AIMessages in order (deepagents calls bind_tools)."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    return Scripted(messages=iter(script))


def _selftest() -> None:
    import tempfile
    import numpy as np
    from langchain_core.messages import AIMessage

    class FakeDetector:
        def detect_all(self, frame, min_score=0.3):
            # a banana left of center and a cup on the right (640 px wide frame)
            return [("banana", 100, 200, 220, 260, 0.81), ("cup", 480, 220, 560, 300, 0.66)]

    class FakeVision:
        mirror = False
        detector = FakeDetector()
        target = None

        def latest_raw(self):
            return np.zeros((480, 640, 3), np.uint8)

        def set_target(self, name):
            self.target = name
            return name

    def call(name, args, i):
        return {"name": name, "args": args, "id": f"c{i}", "type": "tool_call"}

    main = _scripted_model([
        AIMessage("", tool_calls=[call("write_todos", {"todos": [
            {"content": "check memory for banana", "status": "in_progress"},
            {"content": "scout for banana", "status": "pending"},
            {"content": "record sighting", "status": "pending"},
            {"content": "guide to banana", "status": "pending"}]}, 1)]),
        AIMessage("", tool_calls=[call("task", {"subagent_type": "memory-keeper",
                                                "description": "What do we know about banana?"}, 2)]),
        AIMessage("", tool_calls=[call("task", {"subagent_type": "scout",
                                                "description": "Find the banana."}, 3)]),
        AIMessage("", tool_calls=[call("task", {"subagent_type": "memory-keeper",
                                                "description": "Record: banana, left, now, 0.81"}, 4)]),
        AIMessage("", tool_calls=[call("start_guiding", {"target": "banana"}, 5)]),
        AIMessage("The banana is to your left. Follow the buzz."),
    ])
    row = "| banana | left | just now | 0.81 |\n"
    scout = _scripted_model([
        AIMessage("", tool_calls=[call("look", {}, 10)]),
        AIMessage("Found banana to the left (score 0.81); also a cup on the right."),
    ])
    keeper = _scripted_model([
        AIMessage("", tool_calls=[call("read_file", {"file_path": MEMORY_PATH}, 20)]),
        AIMessage("Nothing recorded for banana."),
        AIMessage("", tool_calls=[call("edit_file", {
            "file_path": MEMORY_PATH, "old_string": "|---|---|---|---|\n",
            "new_string": "|---|---|---|---|\n" + row}, 21)]),
        AIMessage("Recorded banana."),
    ])
    vision = FakeVision()
    scene_look = SceneTools(vision).look()
    assert "banana left" in scene_look and "cup far right" in scene_look, scene_look

    with tempfile.TemporaryDirectory() as d:
        disk = Path(d) / "seen_objects.md"
        fa = FindAgent(vision, model=main, scout_model=scout, memory_model=keeper,
                       memory_disk=disk)
        steps, asked = [], []
        fa.on_step = steps.append
        out = fa.run("find the banana",
                     confirm=lambda target, why: asked.append(target) or True)
        for s in steps:
            print("   ", s)
        print(f"  confirm asked for: {asked}; guiding: {out.target}; reply: {out.reply!r}")
        assert asked == ["banana"], asked
        assert out.target == "banana" and vision.target == "banana", out
        assert row in out.memory and disk.read_text() == out.memory, out.memory
        assert len(out.todos) == 4, out.todos
    print("agent_find selftest: OK")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _speaker(enabled: bool) -> Callable[[str], None]:
    def say(text: str) -> None:
        print(f"[say] {text}")
        if enabled and sys.platform == "darwin":
            subprocess.Popen(["say", text])
    return say


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find an object with a Deep Agent + the band")
    ap.add_argument("request", nargs="?", default="find the banana")
    ap.add_argument("--source", default="0", help="camera index or video file")
    ap.add_argument("--mirror", action="store_true", help="laptop camera facing you")
    ap.add_argument("--port", help="serial device for the real band (default: simulated)")
    ap.add_argument("--speak", action="store_true", help="speak prompts with macOS `say`")
    ap.add_argument("--guide-seconds", type=float, default=30,
                    help="keep guiding this long after the agent finishes")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
        return 0
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    from . import haptics
    from .guide import ObjectGuide
    from .link import Link, SerialPort, SimPort
    from .vision import Vision

    port = SerialPort(a.port) if a.port else SimPort()
    link = Link(port).start()
    link.on_sent(lambda line: print(f"[band] {line}"))
    guide = ObjectGuide(lambda cue: haptics.play(cue, link.pulse)).start()
    vision = Vision(camera=a.source, mirror=a.mirror)
    vision.on_change(guide.on_decision)
    vision.on_frame(guide.on_frame)
    vision.on_change(lambda d: print(f"[vision] {d.state} offset={d.offset}"))
    if not vision.start():
        link.close()
        return 1
    time.sleep(1.0)  # first frames
    say = _speaker(a.speak)
    try:
        fa = FindAgent(vision, say=say)
    except RuntimeError as e:
        print(f"[agent] {e}")
        vision.stop(); link.close()
        return 1
    fa.on_step = lambda s: print(f"[agent] {s}")

    def confirm(target: str, why: str):
        say(f"Guide you to the {target}? Yes, no, or name another object.")
        ans = input(f"[confirm] guide to {target!r}? [y/n/other object] ").strip()
        if ans.lower() in ("", "y", "yes"):
            return True
        if ans.lower() in ("n", "no"):
            return False
        return ans

    try:
        out = fa.run(a.request, confirm)
        say(out.reply)
        if out.target:
            print(f"[agent] band guiding to {out.target} for {a.guide_seconds:.0f}s "
                  "(Ctrl-C to stop)")
            end = time.monotonic() + a.guide_seconds
            while time.monotonic() < end:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        guide.close()
        link.close()  # sends STOP
    return 0


if __name__ == "__main__":
    sys.exit(main())
