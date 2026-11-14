"""Run your share of the Haptic Direction Band team agents on Band.

    uv run python hdb_team.py register   # once: create your agents on Band
    uv run python hdb_team.py run        # start your agents (Ctrl+C to stop)
    uv run python hdb_team.py roster     # show who owns what
    uv run python hdb_team.py watch      # live feed of what your agents do (free)

Each teammate runs this on their own laptop, logged in to their own Claude
subscription (`claude auth login`), with their own Band user key.
"""

from __future__ import annotations

import asyncio
import contextlib
import getpass
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
TEAM_FILE = HERE / "team.yaml"
LOCAL_FILE = HERE / "local.yaml"
CREDS_FILE = HERE / "agent_config.yaml"
RULES_FILE = HERE / "prompts" / "room_rules.md"

logger = logging.getLogger("hdb_team")

# Blocked even in autonomous mode: irreversible, privileged or secret-touching.
DENY_RULES = [
    "Bash(rm -rf:*)",
    "Bash(rm -fr:*)",
    "Bash(sudo:*)",
    "Bash(git push --force:*)",
    "Bash(git push -f:*)",
    "Bash(git push origin --force:*)",
    "Bash(git reset --hard:*)",
    "Bash(git clean:*)",
    "Bash(git branch -D:*)",
    "Read(**/.env)",
    "Read(**/.env.*)",
    "Read(**/agent_config.yaml)",
    "Read(**/local.yaml)",
    "Read(~/.ssh/**)",
]


# ---------------------------------------------------------------- config

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        sys.exit(f"{path.name} is not a YAML mapping")
    return data


def save_yaml(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=False)
    os.replace(tmp.name, path)
    path.chmod(0o600)


def load_local() -> dict:
    local = load_yaml(LOCAL_FILE)
    if not local:
        sys.exit("No local.yaml yet. Copy local.example.yaml to local.yaml and fill it in.")
    for field in ("owner", "run"):
        if not local.get(field):
            sys.exit(f"local.yaml is missing `{field}`.")
    return local


def my_agents(team: dict, local: dict) -> dict[str, dict]:
    agents = team.get("agents", {})
    chosen = {}
    for key in local["run"]:
        if key not in agents:
            sys.exit(f"local.yaml runs `{key}`, but team.yaml has no agent with that key.")
        spec = agents[key]
        if spec.get("owner") in (None, "", "TBD"):
            sys.exit(f"Set `owner` for `{key}` in team.yaml before running it.")
        if spec["owner"].lower() != str(local["owner"]).lower():
            sys.exit(
                f"`{key}` is owned by {spec['owner']} in team.yaml, not {local['owner']}. "
                "Only run your own agents."
            )
        chosen[key] = spec
    return chosen


def band_url() -> str:
    load_dotenv(HERE / ".env")
    return os.environ.get("BAND_REST_URL", "https://app.band.ai").rstrip("/")


def user_key() -> str:
    key = os.environ.get("BAND_USER_API_KEY", "").strip()
    if not key:
        key = getpass.getpass("Band user API key (band_u_..., not saved): ").strip()
    if not key.startswith("band_u_"):
        sys.exit("That doesn't look like a Band user API key (should start with band_u_).")
    return key


# ---------------------------------------------------------------- register

def register() -> None:
    team, local = load_yaml(TEAM_FILE), load_local()
    agents = my_agents(team, local)
    creds = load_yaml(CREDS_FILE)
    base, key = band_url(), user_key()
    headers = {"X-API-Key": key, "Accept": "application/json"}

    with httpx.Client(base_url=base, headers=headers, timeout=30) as client:
        # Your Band user id: only you may /approve your coding agents' edits.
        resp = client.get("/api/v1/me/profile")
        if resp.status_code in (401, 403):
            sys.exit(f"Band rejected the user key (HTTP {resp.status_code}). Get a fresh one "
                     "from Band (Connect agent -> Claude) and paste it right away.")
        resp.raise_for_status()
        profile = resp.json().get("data", resp.json())
        if local.get("owner_user_id") != profile["id"]:
            local["owner_user_id"] = profile["id"]
            save_yaml(LOCAL_FILE, local)
        print(f"Band account: @{profile.get('handle')}")

        for agent_key, spec in agents.items():
            if creds.get(agent_key, {}).get("api_key"):
                print(f"  {spec['name']}: already registered, skipping")
                continue
            description = " ".join(spec["description"].split())
            if "owner:" not in description.lower():
                description += f" Owner: {spec['owner']}."
            body = {"agent": {"name": spec["name"], "description": description}}
            resp = client.post("/api/v1/me/agents/register", json=body)
            if resp.status_code in (401, 403):
                sys.exit(f"Band rejected the user key (HTTP {resp.status_code}); stopped "
                         "before creating more agents.")
            if resp.status_code == 422 and "taken" in resp.text:
                sys.exit(f"An agent named {spec['name']} already exists on your account. "
                         "Check Band before retrying; don't create a duplicate.")
            if resp.status_code not in (200, 201):
                sys.exit(f"Registering {spec['name']} failed (HTTP {resp.status_code}). Check "
                         "your agents on Band before retrying.")
            data = resp.json()
            inner = data.get("data", data)
            agent_id = (inner.get("agent") or {}).get("id") or inner.get("id")
            api_key = (inner.get("credentials") or {}).get("api_key") or inner.get("api_key")
            if not (agent_id and api_key):
                sys.exit(f"Band's reply for {spec['name']} had no id/key; check Band before retrying.")
            creds[agent_key] = {"agent_id": agent_id, "api_key": api_key}
            save_yaml(CREDS_FILE, creds)  # save each one before the next
            print(f"  {spec['name']}: registered ({agent_id})")


# ---------------------------------------------------------------- run

def roster_text(team: dict) -> str:
    lines = ["## Roster (who owns what)", ""]
    for spec in team["agents"].values():
        owner = spec.get("owner") or "TBD"
        lines.append(f"- **{spec['name']}** (owner: {owner}): {' '.join(spec['lane'].split())}")
    lines.append("")
    lines.append("Humans in the room are the owners above. If an agent listed here is "
                 "not in the room, ask its owner instead.")
    return "\n".join(lines)


def build_prompt(spec: dict, team: dict) -> str:
    rules = RULES_FILE.read_text().format(name=spec["name"], owner=spec["owner"])
    parts = [
        f"# Team: {team.get('team', '')}",
        rules,
        roster_text(team),
        f"## Your lane\n\n{' '.join(spec['lane'].split())}",
        f"## Your job\n\n{spec.get('prompt', '').strip()}",
    ]
    for rel in spec.get("context", []):
        parts.append((HERE / rel).read_text())
    return "\n\n".join(parts)


def build_codex_adapter(spec: dict, team: dict, local: dict):
    """Codex-backed agent (uses the owner's `codex login`)."""
    from band.adapters.codex import CodexAdapter, CodexAdapterConfig

    repo = (HERE / local.get("repo_path", "..")).resolve()
    repo.mkdir(parents=True, exist_ok=True)
    # Edits and commands run inside Codex's workspace-write sandbox (the repo
    # folder, no network). Steps that leave the sandbox (git push, installing
    # board cores/libraries, opening the USB port to flash) need an approval.
    # Default: auto_accept, so the agent works on its own; the room rules and
    # the firmware prompt still forbid force-push and deleting work. Set
    # `codex_approvals: manual` in local.yaml to approve each one in the room
    # (anyone in the room can approve; Codex can't limit it to the owner).
    approvals = local.get("codex_approvals", "auto_accept")
    if approvals not in ("auto_accept", "auto_decline", "manual"):
        sys.exit("codex_approvals in local.yaml must be auto_accept, auto_decline or manual.")
    config = CodexAdapterConfig(
        workspace_for_room=lambda _room_id: str(repo),
        model=spec.get("model"),
        reasoning_effort=spec.get("effort"),
        personality="pragmatic",
        custom_section=build_prompt(spec, team),
        include_base_instructions=True,
        sandbox="workspace-write",
        approval_policy="on-request",
        approval_mode=approvals,
        # Only manual approvals are posted to the room; auto decisions show up
        # in `uv run python hdb_team.py watch` instead of spamming the chat.
        approval_text_notifications=(approvals == "manual"),
        approval_wait_timeout_s=300,
        turn_timeout_s=900,
    )
    return CodexAdapter(config=config)


def build_adapter(spec: dict, team: dict, local: dict):
    if spec.get("engine", "claude") == "codex":
        return build_codex_adapter(spec, team, local)

    from band.adapters import ClaudeSDKAdapter, ClaudeSDKAdapterConfig
    from band.adapters.claude_sdk import (
        ClaudeApprovalOptions,
        ClaudeCLIOptions,
        ClaudePermissionMode,
    )

    kwargs: dict = {
        "model": spec.get("model"),
        # If the main model is unavailable or overloaded, keep answering on Sonnet.
        "fallback_model": spec.get("fallback_model", "claude-sonnet-5-5"),
        "custom_section": build_prompt(spec, team),
    }
    if spec.get("effort"):
        kwargs["effort"] = spec["effort"]
    if kwargs["fallback_model"] == kwargs["model"]:
        kwargs["fallback_model"] = None
    if spec.get("uses_repo"):
        repo = (HERE / local.get("repo_path", "..")).resolve()
        repo.mkdir(parents=True, exist_ok=True)
        owner_id = local.get("owner_user_id")
        if not owner_id:
            sys.exit("local.yaml has no owner_user_id; run `register` first.")
        autonomy = local.get("autonomy", "autonomous")
        if autonomy == "autonomous":
            # Edits and commands run without asking. The deny list still
            # blocks irreversible or secret-touching actions.
            kwargs.update(
                permission_mode=ClaudePermissionMode.ACCEPT_EDITS,
                approvals=ClaudeApprovalOptions(
                    mode="auto_accept", text_notifications=False, wait_timeout_s=60
                ),
            )
        elif autonomy == "ask":
            # Reads are free; edits and commands wait for the owner's /approve.
            kwargs.update(
                permission_mode=ClaudePermissionMode.DEFAULT,
                approvals=ClaudeApprovalOptions(
                    mode="manual",
                    authorized_senders=frozenset({owner_id}),
                    wait_timeout_s=300,
                ),
            )
        else:
            sys.exit("autonomy in local.yaml must be autonomous or ask.")
        kwargs.update(
            cwd=repo,
            cli=ClaudeCLIOptions(settings=json.dumps({"permissions": {"deny": DENY_RULES}})),
            turn_timeout_s=1800,
        )
    else:
        # Talk-only agents: Band tools work, file/shell tools are declined.
        kwargs.update(
            approvals=ClaudeApprovalOptions(
                mode="auto_decline", text_notifications=False, wait_timeout_s=60
            ),
            turn_timeout_s=180,
        )
    return ClaudeSDKAdapter(ClaudeSDKAdapterConfig(**kwargs))


async def run() -> None:
    from band import Agent, configure_logging

    configure_logging(logging.INFO)
    # Band logs a harmless warning every time someone's presence changes.
    logging.getLogger("band.client.streaming.client").setLevel(logging.ERROR)
    load_dotenv(HERE / ".env")
    team, local = load_yaml(TEAM_FILE), load_local()
    agents = my_agents(team, local)
    creds = load_yaml(CREDS_FILE)
    missing = [k for k in agents if not creds.get(k, {}).get("api_key")]
    if missing:
        sys.exit(f"Not registered yet: {', '.join(missing)}. Run `register` first.")

    async with contextlib.AsyncExitStack() as stack:
        running = []
        for key, spec in agents.items():
            agent = Agent.from_config(
                key, adapter=build_adapter(spec, team, local), config_path=CREDS_FILE
            )
            await stack.enter_async_context(agent)
            logger.info("%s is online (owner %s)", spec["name"], spec["owner"])
            running.append(agent.run_forever())
        logger.info("All %d agents online. Ctrl+C to stop.", len(running))
        await asyncio.gather(*running)


# --- watch: a live, plain-text feed of what your agents are doing ----------
# Free (no model calls): it reads the room history each of your agents can see
# and prints one line per action. Failed or denied steps are marked with ✗.

SKIP_TOOLS = {"ToolSearch"}
FAIL_WORDS = ("declined", "denied", "permission", "not allowed", "timed out",
              "traceback", "error:", "fatal:", "rejected", "failed")
RED, DIM, BOLD, RESET = "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def _one_line(text, limit: int = 140) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_call(name: str, args: dict) -> str | None:
    if name in SKIP_TOOLS:
        return None
    if name == "band_send_message":
        return "💬 " + _one_line(args.get("content"), 160)
    if name == "band_no_reply":
        return DIM + "· stayed quiet" + RESET
    if name == "Bash":
        return "$ " + _one_line(args.get("description") or args.get("command"))
    if name in ("Write", "Edit", "Read", "NotebookEdit"):
        path = str(args.get("file_path", ""))
        return f"{name.lower()} " + path.split("/hdb/", 1)[-1]
    if name in ("Grep", "Glob"):
        return f"{name.lower()} " + _one_line(args.get("pattern"))
    return f"{name} " + _one_line(json.dumps(args), 100)


def _format(item: dict, me: set[str]) -> str | None:
    kind = item.get("message_type")
    who = item.get("sender_name") or "?"
    content = item.get("content") or ""
    if kind == "tool_call":
        try:
            call = json.loads(content)
        except ValueError:
            return None
        return _describe_call(call.get("name", ""), call.get("args") or {})
    if kind == "tool_result":
        try:
            res = json.loads(content)
        except ValueError:
            res = {"output": content}
        out = json.dumps(res.get("output", ""))
        if res.get("is_error") or any(w in out.lower() for w in FAIL_WORDS):
            return f"{RED}✗ {res.get('name', 'tool')} failed: {_one_line(out, 160)}{RESET}"
        return None
    if kind == "text" and who not in me:
        return f"{BOLD}{who}{RESET}: " + _one_line(content, 200)
    if kind == "task" and "error" in content.lower():
        return f"{RED}✗ {_one_line(content)}{RESET}"
    return None


async def watch() -> None:
    from band.client.rest import AsyncRestClient

    team, local = load_yaml(TEAM_FILE), load_local()
    agents = my_agents(team, local)
    creds = load_yaml(CREDS_FILE)
    clients = {spec["name"]: AsyncRestClient(api_key=creds[k]["api_key"])
               for k, spec in agents.items() if creds.get(k, {}).get("api_key")}
    me = set(clients)
    backlog = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 25
    seen: set[str] = set()
    first = True
    print(f"Watching {', '.join(sorted(me))}. Ctrl+C to stop.\n")
    while True:
        rows = []
        for name, client in clients.items():
            try:
                chats = await client.agent_api_chats.list_agent_chats()
                for chat in (getattr(chats, "data", None) or [])[:3]:
                    page = 1
                    while True:
                        ctx = await client.agent_api_context.get_agent_chat_context(
                            chat_id=chat.id, page=page, page_size=100)
                        items = getattr(ctx, "data", None) or []
                        for m in items:
                            d = m.dict() if hasattr(m, "dict") else dict(m)
                            mid = str(d.get("id") or hash(json.dumps(d, default=str)))
                            if mid not in seen:
                                seen.add(mid)
                                rows.append(d)
                        if len(items) < 100 or page >= 20:
                            break
                        page += 1
            except Exception as exc:  # keep watching through network blips
                print(f"{DIM}({name}: {exc.__class__.__name__}, retrying){RESET}")
        rows.sort(key=lambda d: str(d.get("inserted_at") or d.get("created_at") or ""))
        if first:
            rows, first = rows[-backlog:], False
        for d in rows:
            line = _format(d, me)
            if not line:
                continue
            ts = d.get("inserted_at") or d.get("created_at")
            try:
                from datetime import datetime
                t = ts if hasattr(ts, "astimezone") else datetime.fromisoformat(str(ts))
                stamp = t.astimezone().strftime("%H:%M:%S")
            except (TypeError, ValueError):
                stamp = "--:--:--"
            who = d.get("sender_name") or "?"
            label = f"{who:<15}" if d.get("message_type") != "text" or who in me else " " * 15
            print(f"{DIM}{stamp}{RESET} {label} {line}", flush=True)
        await asyncio.sleep(5)


def show_roster() -> None:
    print(roster_text(load_yaml(TEAM_FILE)))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "register":
        register()
    elif command == "run":
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run())
    elif command == "watch":
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(watch())
    elif command == "roster":
        show_roster()
    else:
        print(__doc__)
