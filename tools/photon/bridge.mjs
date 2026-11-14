// Caregiver texting bridge: Photon Spectrum (iMessage) <-> app/notify.py.
//
// The laptop app is Python and Spectrum is TypeScript-only, so the app runs
// this as a child process and talks JSON lines over stdin/stdout:
//   in : {"id": 3, "to": "+15551234567", "text": "..."}
//   out: {"id": 3, "ok": true}   or   {"id": 3, "ok": false, "error": "..."}
//   out: {"ready": true} once Spectrum is up, {"ready": false, "error": ...} if not
//   out: {"inbound": true, "from": "+15551234567", "text": "status"}   a text arrived
//        (band agent, app/band_agent.py; replies go back through the "in" lines above)
//
// Credentials come only from the environment (never the repo):
//   SPECTRUM_PROJECT_ID, SPECTRUM_PROJECT_SECRET  (Photon dashboard -> project Settings)
// Setup: cd tools/photon && npm install
import readline from "node:readline";
import { Spectrum } from "spectrum-ts";
import { imessage } from "spectrum-ts/providers/imessage";

const out = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");

let app;
try {
  if (!process.env.SPECTRUM_PROJECT_ID || !process.env.SPECTRUM_PROJECT_SECRET) {
    throw new Error("SPECTRUM_PROJECT_ID / SPECTRUM_PROJECT_SECRET not set");
  }
  app = await Spectrum({
    projectId: process.env.SPECTRUM_PROJECT_ID,
    projectSecret: process.env.SPECTRUM_PROJECT_SECRET,
    providers: [imessage.config()],
  });
} catch (e) {
  out({ ready: false, error: String(e && e.message ? e.message : e) });
  process.exit(2);
}
const im = imessage(app);
const spaces = new Map(); // phone -> space (reuse the conversation)
out({ ready: true });

async function send(to, text) {
  let space = spaces.get(to);
  if (!space) {
    const user = await im.user(to);
    space = await im.space.create(user);
    spaces.set(to, space);
  }
  await space.send(text);
}

// Incoming texts: handed to the laptop app as one JSON line each. Unanswered or
// failing handlers never stop the loop; the app decides who is allowed a reply.
(async () => {
  try {
    for await (const [space, message] of app.messages) {
      try {
        const c = message.content;
        if (message.direction === "outbound" || !c || c.type !== "text") continue;
        const from = (message.sender && (message.sender.id || message.sender.address)) || "";
        out({ inbound: true, from: String(from), text: String(c.text || "").slice(0, 500) });
      } catch (e) { /* skip one bad message */ }
    }
  } catch (e) {
    out({ inbound: false, error: String(e && e.message ? e.message : e) });
  }
})();

const rl = readline.createInterface({ input: process.stdin });
let queue = Promise.resolve();       // one send at a time, in order
rl.on("line", (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return out({ ok: false, error: "bad json" });
  }
  queue = queue.then(async () => {
    try {
      if (!/^\+[1-9]\d{6,14}$/.test(msg.to || "")) throw new Error("phone must be E.164, e.g. +15551234567");
      await send(msg.to, String(msg.text || "").slice(0, 1000));
      out({ id: msg.id, ok: true });
    } catch (e) {
      spaces.delete(msg.to);
      out({ id: msg.id, ok: false, error: String(e && e.message ? e.message : e) });
    }
  });
});

const stop = async () => {
  try { await queue; await app.stop(); } finally { process.exit(0); }
};
rl.on("close", stop);
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
