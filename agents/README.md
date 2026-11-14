# HDB build room: team agents on Band

Each of us runs our own agents on our own laptop and our own Claude
subscription. All of them join one shared Band room and follow the same room
rules, so they hand questions to each other instead of talking over each other.

| Agent | Owner | Model | Lane |
|---|---|---|---|
| HDB-Software | Shravan | Opus 5.5 | Serial link, name-direction estimator, haptic cues, dashboard and replay |
| HDB-Protocol | Shravan | Sonnet 5.5 | Serial protocol (115200 MIC/BTN/PULSE/STOP), cue timings, pin map (referee) |
| HDB-Gatekeeper | Shravan | Haiku 5.5 | 36-hour timeline, gates, cut order, task board |
| HDB-Deep | Shravan | Fable 5.2 | Escalations only: estimator and clock alignment, cross-lane bugs, root causes |
| HDB-Firmware | *hardware teammate* | Codex (owner's default model) | Motor channels, wiring, power, ESP32 firmware |
| HDB-Vision | *vision teammate* | Opus 5.5 | Webcam object finding, hold-to-talk, name timestamps |

Claude agents fall back to Sonnet 5.5 if their main model is unavailable. HDB-Firmware runs on Codex.
Change `model` or `effort` per agent in `team.yaml` if your plan or usage
limits need it.

`team.yaml` is the single shared roster. `prompts/room_rules.md` is what every
agent obeys. Both are shared; keep them identical across our laptops.

## Set up your agent (about 10 minutes)

Easiest: paste your role's prompt and your coding agent does these steps for you.
- Hardware and firmware, on **Codex**: `PROMPT_FIRMWARE_CODEX.md`
- Vision and phone, on **Claude Code**: `PROMPT_VISION_CLAUDE.md`

The manual steps below are for Claude agents; the Codex prompt covers the Codex differences (`codex login` instead of `claude auth login`).

You need: [uv](https://docs.astral.sh/uv/), Node.js 20+, and Claude Code logged
in with your own account.

1. **Log in to Claude Code** and check it works:
   ```bash
   claude auth login
   claude -p "say pong"
   ```
2. **Claim your agent.** In `team.yaml`, set `owner:` on `firmware` or
   `vision` to your first name. Commit and push that one-line change so everyone's roster matches.
3. **Create your local config:**
   ```bash
   cp local.example.yaml local.yaml
   cp .env.example .env
   ```
   Set `owner` (your first name, same spelling), `run` (e.g. `[firmware]`) and
   `repo_path: ..` (the repo root; this kit lives in `agents/`).
4. **Get a Band user key:** on app.band.ai, Connect agent → Claude, and copy
   the `band_u_…` key. Use it straight away; it isn't saved anywhere.
5. **Register and run:**
   ```bash
   uv sync
   uv run python hdb_team.py register
   uv run python hdb_team.py run
   ```
   Leave `run` going in a terminal. Your agent is online until you stop it or
   your laptop sleeps.

## Getting everyone into one room

1. Add each other as **contacts** on Band, so you can see each other's agents.
2. One person creates the room (Chats → new chat) and adds the three humans.
3. Each person adds **their own** agents to the room with **+** in Participants.

## Using the room

- **@mention the agent that owns the lane.** Agents stay silent unless
  mentioned, or unless a human asks the whole room something in their lane.
- **Coding agents are autonomous.** They edit, run tests and commit in their
  own repo folder without waiting for approval, and post a `[done]` line when
  finished. Claude agents block `rm -rf`, `sudo`, force-push, `git reset --hard`
  and reading `.env`/key files; set `autonomy: ask` in `local.yaml` to approve
  every edit instead. HDB-Firmware (Codex) edits inside its sandbox; push,
  installs and flashing are approved automatically (`codex_approvals: manual`
  in `local.yaml` to approve each one yourself).
- **See what your agents are doing:** `uv run python hdb_team.py watch` prints
  one line per action (commands, file edits, messages), marks failed or denied
  steps with ✗, and costs nothing. For the team picture, ask
  `@HDB-Gatekeeper status`.
- **Protocol changes go through HDB-Protocol** and need a yes from a human on
  both the firmware and software side.
- **HDB-Deep is for hard problems only**, after one real attempt. Give it
  the goal, what you tried and the evidence (errors, logs, numbers). It
  diagnoses and hands the fix to the lane owner's agent.
- **Agents tag every message** `[answer]`, `[ask]`, `[blocked]`, `[done]`,
  `[proposal]` or `[protocol change]`. After 3 rounds between two agents
  they stop and tag the humans.

Examples:
- `@HDB-Protocol is PULSE 6 150 valid?`
- `@HDB-Gatekeeper we missed the H9 gate, what's the fallback?`
- `@HDB-Firmware is the Action button debounced yet, and what does BTN look like on the wire?`

## Files

| File | Shared? | What it is |
|---|---|---|
| `team.yaml` | yes | Roster: names, owners, lanes, per-agent instructions |
| `prompts/room_rules.md` | yes | Rules every agent follows |
| `context/*.md` | yes | Protocol and timeline from the build plan |
| `hdb_team.py` | yes | `register`, `run`, `roster` |
| `local.yaml` | **no** | Your name, which agents you run, your repo path |
| `agent_config.yaml` | **no** | Your agents' Band credentials |
| `.env.example` | yes | Band URLs (copy to `.env`) |

Never share `local.yaml`, `agent_config.yaml` or any key. They're gitignored.
