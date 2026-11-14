# Haptic Direction Band (HackPrinceton)

An assistive headband: a forward USB webcam, four microphones and eight
vibration motors. Blind or low-vision wearers say "find the banana" and buzzes
guide their head toward it; Deaf or hard-of-hearing wearers feel which side
someone called their name from. The laptop handles video, speech and the
direction estimator; the ESP32 reads buttons and mic levels and switches the
motors over USB.

## Who owns what

Each lane works on its own branch and releases to `main` when a piece works.
Stay in your lane's folders; ask the owner before touching someone else's.

| Lane | Owner | Branch | Agent on Band | Folders |
|---|---|---|---|---|
| Software / integration | Shravan | `software` | HDB-Software | `app/` (except vision/speech), `dashboard/`, `tools/`, `tests/` |
| Hardware + firmware | hardware teammate | `firmware` | HDB-Firmware (Codex) | `firmware/` |
| Vision + speech | vision teammate | `vision` | HDB-Vision | `app/vision.py`, `app/speech.py` |
| Hardware contract | Shravan | (main) | HDB-Protocol | `agents/context/protocol.md`: change only when firmware and software agree |
| Agents and room rules | everyone | (main) | — | `agents/` |

```
app/        laptop side: serial link, direction estimator, haptic cues,
            vision, speech, heading lock and Teach a Sound (later), recall (backlog)
dashboard/  motor ring, replay and USB bench (HTML dashboard, optional Python one)
firmware/   ESP32 sketch (115200 serial: PULSE/STOP in, MIC/BTN out)
tools/      calibrate.py (claps at 8 positions)
tests/      tests and fixtures (short recordings with known directions)
agents/     our Band agents: roster, room rules, setup prompts
```

The pin map, serial protocol, one-channel wiring and estimator notes are in
[`agents/context/protocol.md`](agents/context/protocol.md) (from the Nov 13
handoff). The 36-hour schedule, gates, cut order and demo are in
[`agents/context/timeline.md`](agents/context/timeline.md).

## Git workflow (people and agents)

1. Start a task: `git checkout <your-branch> && git pull --no-rebase origin main`
2. Commit small, focused changes; share any time with `git push origin <your-branch>`.
3. When it works and tests pass, release it:
   `git pull --no-rebase origin main && git push origin <your-branch>:main`
   (fast-forward; if rejected, pull again and retry). Never force-push.
4. A conflict in a file you don't own: stop and ask its owner.
5. Never commit secrets (`.env`, `agents/local.yaml`, `agents/agent_config.yaml`)
   or big files (model weights, long recordings). Download weights at runtime.

## Agents

Each of us runs our own agents on our own subscription, all in one Band room.
See [`agents/README.md`](agents/README.md). Setup prompts:
[Codex (hardware)](agents/PROMPT_FIRMWARE_CODEX.md) ·
[Claude Code (vision)](agents/PROMPT_VISION_CLAUDE.md).
