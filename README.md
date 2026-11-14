# Haptic Direction Band (HackPrinceton)

A beanie with 4 mics, a motion sensor and a ring of 8 vibration motors that
points a deaf user at whoever is calling their name, and a blind user at the
door they asked for. The cue stays locked to the room as the head turns.
Alarms and falls are handled on the ESP32; the laptop runs keyword spotting,
Teach a Sound, vision and the Recall memory agent.

## Who owns what

Everyone works on `main`. Stay in your lane's folders; ask the owner before
touching someone else's.

| Lane | Owner | Agent on Band | Folders |
|---|---|---|---|
| Software | Shravan | HDB-Software | `app/` (except the vision/fall/notify files below), `dashboard/`, `tools/`, `tests/` |
| Hardware + firmware | hardware teammate | HDB-Firmware (Codex) | `firmware/` |
| Vision + phone | vision teammate | HDB-Vision | `app/vision.py`, `app/fall.py`, `app/notify.py`, `phone/` |
| Protocol | Shravan | HDB-Protocol | `agents/context/protocol.md` (changes need both firmware and software to agree) |
| Agents and room rules | everyone | — | `agents/` |

```
app/        laptop "brain": main loop, link, direction, heading, listen,
            haptics, recall (deepagents), vision, fall, notify
dashboard/  live page for filming (FastAPI + one WebSocket)
firmware/   ESP32 sketch: firmware/band/band.ino
phone/      camera stream and Recall chat page
tools/      calibrate.py (claps at 8 positions)
tests/      tests and fixtures (short recordings with known directions)
agents/     our Band agents: roster, room rules, setup prompts
```

The ESP32 ↔ laptop protocol, haptic patterns and pin map are in
[`agents/context/protocol.md`](agents/context/protocol.md). The 36-hour plan,
gates and cut order are in [`agents/context/timeline.md`](agents/context/timeline.md).

## Git workflow (people and agents)

1. Before starting anything: `git pull --rebase`.
2. Commit small, focused changes with a clear message.
3. `git pull --rebase && git push`. Never force-push.
4. A conflict in a file you don't own: stop and ask its owner.
5. Never commit secrets (`.env`, `agents/local.yaml`, `agents/agent_config.yaml`)
   or big files (model weights, long recordings). `.gitignore` covers these;
   download weights at runtime.

## Agents

Each of us runs our own agents on our own subscription, all in one Band room.
See [`agents/README.md`](agents/README.md). Setup prompts:
[Codex (hardware)](agents/PROMPT_FIRMWARE_CODEX.md) ·
[Claude Code (vision)](agents/PROMPT_VISION_CLAUDE.md).
