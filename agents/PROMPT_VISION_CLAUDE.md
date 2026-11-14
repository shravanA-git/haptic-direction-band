# Setup prompt: vision and phone teammate (Claude Code)

Paste everything below the line into **Claude Code**, run from the folder
where you keep your projects. Fill in the two values at the top
first.

---

Set up my Haptic Direction Band team agent, HDB-Vision, on Band (app.band.ai) from our team repo https://github.com/shravanA-git/haptic-direction-band. It should run on my Claude subscription.

- My first name: <YOUR FIRST NAME>
- My Band user API key: <band_u_... from app.band.ai → Connect agent → Claude, generated just now>

Steps:

1. `git clone https://github.com/shravanA-git/haptic-direction-band` (or `git pull` if I already have it) and `cd haptic-direction-band/agents`. Read `../README.md`, `README.md`, `team.yaml`, `prompts/room_rules.md` and `hdb_team.py` before changing anything.
2. Check prerequisites: `uv`, Node.js 20+ and the `claude` CLI. Confirm Claude Code is logged in by running `claude -p "reply with just: pong"`. If it fails with an auth or 401 error, stop and ask me to run `claude auth login`, then re-check.
3. In `team.yaml`, change ONLY `owner: TBD` under the `vision` agent to my first name. Don't edit any other agent or `prompts/room_rules.md`; those are shared across the team.
4. `cp local.example.yaml local.yaml` and set: `owner:` my first name, `run: [vision]`, `repo_path: ..` (the repo root). Then `cp .env.example .env`.
5. `uv sync`, then register with my key passed only as an environment variable (never write it to a file):
   `BAND_USER_API_KEY=<key> uv run python hdb_team.py register`
   If Band returns 401/403, stop and ask me for a freshly generated key. Don't retry the same one. If the name is "already taken", stop and tell me; don't create a duplicate.
5b. Switch my clone to my lane's branch: `git fetch origin && git checkout vision` (it already exists on GitHub). My agent works on this branch and releases to `main` as the room rules describe.
6. Start the agent in the background so it keeps running after you finish: `uv run python hdb_team.py run`. Wait for `Agent started: HDB-Vision` in the log with no traceback.
7. Prove it can actually answer: fetch https://raw.githubusercontent.com/band-ai/add-band/main/scripts/verify_agent_reply.py to a temp file, read it, then run it from `agents/` with `BAND_USER_API_KEY=<key>`, `BAND_REST_URL=https://app.band.ai`, `VERIFY_AGENT_ID=<vision agent_id from agent_config.yaml>`, `VERIFY_AGENT_NAME=HDB-Vision` and `VERIFY_TIMEOUT_SECONDS=120`. Only call it done if it exits 0. If the agent reports an error, re-run `claude -p "reply with just: pong"`; an expired Claude login is the usual cause.
8. Tell me what's running and how to stop it, and remind me to:
   - add Shravan and our third teammate as contacts on Band,
   - join the shared HDB room and add HDB-Vision there with **+** in Participants,
   - after verification, commit ONLY my `team.yaml` owner change (`git add team.yaml && git commit -m "Claim vision agent" && git pull --rebase && git push`) so everyone's roster matches. Make sure `local.yaml` and `agent_config.yaml` are not staged.

How my agent behaves (so I know what to expect): it runs on Claude Opus 5.5 and works autonomously in my repo: it edits, runs tests and commits without asking. A built-in block list still refuses `rm -rf`, `sudo`, force-push, `git reset --hard` and reading `.env` or key files. If I'd rather approve every edit myself, I set `autonomy: ask` in `local.yaml` and restart it.

Rules: never print, commit or share my Band key, `agent_config.yaml` or `local.yaml`. Don't touch other teammates' agents.
