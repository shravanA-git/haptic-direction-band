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

* Receiving uses socket.io to `/v1/projects/{id}/events/ws`; the resume cursor (`metadata.seq`) is saved
  to `tools/photon/.cursor.json` (gitignored) only after an event was handled, and replays are skipped.
* The incoming event is read from `event.data.message`, `event.message`, `event.data` or `event` (the
  message has `status: "received"`, `sentBy.handle`, `space.id`, `receivedBy.id`). Verified against the SDK
  types, not yet against a live line: if no text arrives, log one raw `delivery` event and adjust
  `messageOf` in `bridge-core.mjs`.
