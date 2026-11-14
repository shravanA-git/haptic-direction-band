# Operator guide (Shravan's control session)

How to drive the HDB team from a Claude Code session: run prompts and tests
yourself, and hand recurring or room-facing work to the supervisor.

## What is running

| Piece | Where | Notes |
|---|---|---|
| HDB-Software, HDB-Protocol, HDB-Gatekeeper, HDB-Deep | Terminal tab running `uv run python hdb_team.py run` in `hdb/agents` | Restart: Ctrl-C in that tab, run it again. Never start a second copy (two processes on the same agent keys fight). |
| HDB-Supervisor loop | Its own Claude Code session ("HDB room supervisor loop") | Follows `agents/SUPERVISOR.md`; checks every ~20 min, posts as HDB-Supervisor. |
| HDB-Vision | Sophanit's laptop | |
| HDB-Firmware (Codex) | Jason's laptop | Hardware just starting. |

All of them share one Band room. HDB-Supervisor's `say` picks that room
automatically.

## Commands (run in `hdb/agents`)

- `uv run python hdb_team.py status 40`: last 40 actions and messages, once.
- `uv run python hdb_team.py watch`: the same feed, live (for a terminal tab).
- `uv run python hdb_team.py say "@HDB-Software [ask] ..."`: post as
  HDB-Supervisor. Tag every message; @mention only the lanes concerned.
- `uv run python hdb_team.py roster`: who owns what.

## Tests without disturbing the agents

HDB-Software and HDB-Deep edit the shared checkout `hdb/` (branch
`software`). Don't run tests, switch branches or commit there while they
work. Test the released code in a throwaway worktree:

```bash
cd /Users/shravan_anand/cowork_workspace/HackPrincetonNew/hdb
git fetch -q
git worktree add -f --detach /tmp/hdb-ctl origin/main 2>/dev/null || git -C /tmp/hdb-ctl checkout -q --detach origin/main
cd /tmp/hdb-ctl && /Users/shravan_anand/cowork_workspace/HackPrincetonNew/hdb/.venv/bin/python -m pytest -q
```

Use a different path from the supervisor's `/tmp/hdb-main`.

## Who does what

Do it in the control session:
- One-off prompts, investigations, design questions and reviews, using
  Claude skills (systematic-debugging, test-driven-development,
  verification-before-completion, code-review, writing-plans,
  dispatching-parallel-agents, and so on).
- Running tests or demos on demand, and reading code or diffs.
- Posting a specific instruction to a lane with `say`.

Hand it to the supervisor session (send it a message):
- Anything recurring: "watch the H9 gate", "check firmware every 10 min",
  "re-run tests after each release".
- Changing its focus or pace, or pausing or stopping it.

Code changes belong to the lane agents. Ask the owning agent in the room
rather than editing lane code from the control session, unless Shravan
says to.

## Limits (same as the room rules)

- Humans approve protocol changes, cuts, deletions and anything needing
  hands on hardware.
- Never read or print `.env`, `local.yaml` or `agent_config.yaml`, and never
  handle the Band user key.
- No force-push. Release to main by fast-forward
  (`git push origin <branch>:main`).
- Shravan's ChatOverflow stop hook fires after every turn. Log a no-note with
  `/Users/shravan_anand/.hermes/node/bin/node /Users/shravan_anand/.npm-global/bin/chatoverflow skip proprietary`
  unless the turn produced a genuinely reusable, project-free lesson.
