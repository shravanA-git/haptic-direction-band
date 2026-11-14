// Photon 2.0 beta bridge: iMessage <-> app/notify.py (PhotonSender) over JSON lines.
// The laptop app is Python and the Photon SDK is Node, so the app runs this as a
// child process. The logic lives in bridge-core.mjs (tested with fakes).
//
//   in : {"id": 3, "to": "+15551234567", "text": "..."}
//   out: {"id": 3, "ok": true}   or   {"id": 3, "ok": false, "error": "..."}
//   out: {"ready": true}  /  {"ready": false, "error": ...}
//   out: {"inbound": true, "from": "+15551234567", "text": "status"}
//        found by polling listMessages every ~2 s (websocket events are an optional extra)
//
// Credentials only from the environment (never the repo, never printed):
//   PHOTON_PROJECT_ID   the project id (pho_prj_...)
//   PHOTON_API_KEY      a project API key (pho_sk_...)
// Setup: cd tools/photon && npm ci
// Shared line: the person must text the agent first; only the assigned iMessage
// identity can reach it. See tools/photon/README.md.
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { createBridge } from "./bridge-core.mjs";

const out = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");
const here = path.dirname(fileURLToPath(import.meta.url));

let core;
try {
  const projectId = process.env.PHOTON_PROJECT_ID;
  const key = process.env.PHOTON_API_KEY;
  if (!projectId || !key) throw new Error("PHOTON_PROJECT_ID / PHOTON_API_KEY not set");
  const { Photon } = await import("@photon-ai/api");
  const { io } = await import("socket.io-client");
  const auth = { Authorization: `Bearer ${key}` };
  const photon = new Photon({ headers: auth });
  const socket = io("https://api.photon.codes", {
    path: `/v1/projects/${encodeURIComponent(projectId)}/events/ws`,
    addTrailingSlash: false,
    transports: ["websocket"],
    extraHeaders: auth,
  });
  const log = (line) => process.stderr.write(`[photon] ${line}\n`);
  core = createBridge({ photon, socket, projectId, out, log, poll: true,
                        cursorPath: path.join(here, ".cursor.json") });
  core.wire();                 // optional websocket events (deduped against polling)
  core.startPolling();         // primary inbound path
} catch (e) {
  out({ ready: false, error: String(e && e.message ? e.message : e).replace(/pho_sk_\w+/g, "pho_sk_***") });
  process.exit(2);
}
// If Photon never answers, say so instead of hanging the Python side.
setTimeout(() => {
  if (!core.ready) { out({ ready: false, error: "no answer from Photon within 20 s" }); process.exit(2); }
}, 20000).unref();

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", core.onLine);
const stop = async () => { try { await core.drain(); } finally { process.exit(0); } };
rl.on("close", stop);
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
