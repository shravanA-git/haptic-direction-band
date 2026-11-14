# Setup prompt: hardware and firmware teammate (Codex)

Paste everything below the line into **Codex**, run from the folder where
you keep your projects. Fill in the two values at the top first.

---

Set up my Haptic Direction Band team agent, HDB-Firmware, on Band (app.band.ai) from our team repo https://github.com/shravanA-git/haptic-direction-band. It should run on my Codex subscription.

- My first name: <YOUR FIRST NAME>
- My Band user API key: <band_u_... from app.band.ai → Connect agent, generated just now>

Steps:

1. `git clone https://github.com/shravanA-git/haptic-direction-band` (or `git pull` if I already have it) and `cd haptic-direction-band/agents`. Read `../README.md`, `README.md`, `team.yaml`, `prompts/room_rules.md` and `hdb_team.py` before changing anything.
2. Check prerequisites: `uv` and the `codex` CLI (`npm install -g @openai/codex` if it's missing; ask me before installing). Confirm I'm logged in to Codex with my ChatGPT account: run `codex login status`, then a one-line non-interactive test such as `codex exec "reply with just: pong"`. If either fails, stop and ask me to run `codex login`, then re-check. I don't need Claude Code for this.
3. In `team.yaml`, change ONLY `owner: TBD` under the `firmware` agent to my first name. Leave `engine: codex` and everything else as is. Don't edit any other agent or `prompts/room_rules.md`; those are shared across the team.
4. `cp local.example.yaml local.yaml` and set: `owner:` my first name, `run: [firmware]`, `repo_path: ..` (the repo root). Keep `codex_approvals` at its default, `auto_decline`. Then `cp .env.example .env`.
5. `uv sync`, then register with my key passed only as an environment variable (never write it to a file):
   `BAND_USER_API_KEY=<key> uv run python hdb_team.py register`
   If Band returns 401/403, stop and ask me for a freshly generated key. Don't retry the same one. If the name is "already taken", stop and tell me; don't create a duplicate.
6. Start the agent in the background so it keeps running after you finish: `uv run python hdb_team.py run`. Wait for `Agent started: HDB-Firmware` in the log with no traceback.
7. Prove it can actually answer: download https://raw.githubusercontent.com/band-ai/add-band/main/scripts/verify_agent_reply.py to a temp file, read it, then run it from `agents/` with `BAND_USER_API_KEY=<key>`, `BAND_REST_URL=https://app.band.ai`, `VERIFY_AGENT_ID=<firmware agent_id from agent_config.yaml>`, `VERIFY_AGENT_NAME=HDB-Firmware` and `VERIFY_TIMEOUT_SECONDS=120`. Only call it done if it exits 0. If the agent reports an error, re-check `codex login status` first; an expired login is the usual cause.
8. Tell me what's running and how to stop it, and remind me to:
   - add Shravan and our third teammate as contacts on Band,
   - join the shared HDB room and add HDB-Firmware there with **+** in Participants,
   - after verification, commit ONLY my `team.yaml` owner change (`git add team.yaml && git commit -m "Claim firmware agent" && git pull --rebase && git push`) so everyone's roster matches. Make sure `local.yaml` and `agent_config.yaml` are not staged.

How my agent behaves (so I know what to expect): it can edit and run code inside my repo folder under Codex's sandbox. Flashing the ESP32, installs and anything needing network are declined automatically, and it tells me the exact command to run myself. Nobody else in the room can approve commands on my laptop.

Rules: never print, commit or share my Band key, `agent_config.yaml` or `local.yaml`. Don't touch other teammates' agents.
