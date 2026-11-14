# Photon 2.0 beta bridge (iMessage)

`bridge.mjs` is a child process of the laptop app (`app/notify.py`, `PhotonSender`). It sends and
receives iMessage through the Photon 2.0 beta API (`@photon-ai/api`) and talks JSON lines on
stdin/stdout. The band agent (`python -m app.main --agent`) and caregiver texts both use it.

    in : {"id": 3, "to": "+15551234567" or "name@example.com", "text": "..."}
    out: {"id": 3, "ok": true} | {"id": 3, "ok": false, "error": "..."}
    out: {"ready": true} | {"ready": false, "error": "..."}
    out: {"inbound": true, "from": "<handle>", "text": "status"}

`ok: true` means Photon accepted the message, not that it was delivered.

## Run it

    cd tools/photon && npm ci          # pinned: @photon-ai/api 0.6.0, socket.io-client 4.8.4
    # .env (gitignored, never committed) or the shell:
    #   PHOTON_PROJECT_ID=pho_prj_...     the project id
    #   PHOTON_API_KEY=pho_sk_...         a project API key
    #   WEARER_PHONE=...                  who may text the agent (+E.164 number or iMessage email)
    #   CAREGIVER_PHONE=...               same, and who gets the proactive alerts
    cd ../.. && python -m app.main --vision --agent

Tests (no network): `node --test tools/photon/bridge.test.mjs`, also run by `pytest tests/test_photon_bridge.py`.

## What the person must do (shared line)

* In the Photon dashboard: create a project API key for the beta project and make sure the shared
  iMessage line is assigned to the identity that will text the agent.
* **Text the agent first.** A shared line cannot start a conversation: until a handle has texted in,
  sends to it fail with "no conversation yet: text the agent first" (the caregiver alert then just
  logs a failure; nothing crashes). The bridge remembers each handle's conversation while it runs,
  so after a restart, text the agent once more.
* Direct chats only; text and images only; only messages from the single assigned handle arrive.
* HTTP 429: the bridge waits for `Retry-After` and retries once.

## Notes

* Receiving is POLLING: every ~2 s `listMessages({direction: 'received', createdSince, order: 'asc'})`
  (backs off on 429 using Retry-After). A fresh start begins 60 s back so old texts are not answered;
  the newest `createdAt` is saved to `tools/photon/.cursor.json` (gitignored) after each handled page.
  Messages are deduped by id. The websocket (`/v1/projects/{id}/events/ws`) stays on as an optional extra
  (a probe on the real project acked a subscribe but delivered nothing); both paths share the dedupe.
* To see why a text got no reply: the bridge logs `[photon] inbound from +1***01` to the terminal, the app
  logs `[band-agent] ignored a text from ... (not on the allowlist ...)`, and the dashboard (Settings)
  shows received / ignored / answered / failed to send / last error.
* A message is read as `status: "received"`, `sentBy.handle`, `space.id`, `receivedBy.id`, `content.text`
  (matches a masked REST probe of the real project). Websocket events are unwrapped from `event.data.message`,
  `event.message`, `event.data` or `event` by `messageOf` in `bridge-core.mjs`.
