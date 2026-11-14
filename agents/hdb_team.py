"""Run your share of the Haptic Direction Band team agents on Band.

    uv run python hdb_team.py register   # once: create your agents on Band
    uv run python hdb_team.py run        # start your agents (Ctrl+C to stop)
    uv run python hdb_team.py roster     # show who owns what

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
    # folder, no network). Anything that needs to leave the sandbox asks for
    # approval; Codex approvals can't be limited to the owner, so they are
    # auto-declined unless the owner opts in with `codex_approvals: manual`.
    approvals = local.get("codex_approvals", "auto_decline")
    if approvals not in ("auto_decline", "manual"):
        sys.exit("codex_approvals in local.yaml must be auto_decline or manual.")
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


def show_roster() -> None:
    print(roster_text(load_yaml(TEAM_FILE)))


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "register":
        register()
    elif command == "run":
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run())
    elif command == "roster":
        show_roster()
    else:
        print(__doc__)
