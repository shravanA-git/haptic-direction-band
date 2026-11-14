// Photon 2.0 (beta) bridge core: iMessage <-> app/notify.py over JSON lines.
// Separate from bridge.mjs so tests can drive it with a fake socket and client.
//
// Contract with the Python side (unchanged from the Spectrum version):
//   in : {"id": 3, "to": "+15551234567" | "name@example.com", "text": "..."}
//   out: {"id": 3, "ok": true}  or  {"id": 3, "ok": false, "error": "..."}
//   out: {"ready": true} / {"ready": false, "error": "..."}
//   out: {"inbound": true, "from": "<handle>", "text": "status"}
//
// Photon beta facts this relies on (photon.codes/docs/beta):
//   * send = photon.projects.sendMessage(...); success means ACCEPTED, not delivered
//   * receive = socket.io to /v1/projects/{id}/events/ws, 'subscribe' {startSeq},
//     'delivery' (event, metadata) with metadata.seq as the resume cursor
//   * shared line: the human texts first, direct chats only, only the assigned
//     handle reaches us, honour HTTP 429
// Secrets come from the caller (env); nothing here prints them.
import fs from "node:fs";
import { randomUUID } from "node:crypto";

export const normHandle = (h) => {
  const s = String(h || "").trim();
  if (s.includes("@")) return "mail:" + s.toLowerCase();
  const d = s.replace(/\D/g, "");
  return d ? "tel:" + d : "";
};

// The message inside a delivery event, whatever the wrapper (event.data, event.message, ...).
export function messageOf(event) {
  for (const m of [event?.data?.message, event?.message, event?.data, event]) {
    if (m && typeof m === "object" && m.content && (m.sentBy || m.space)) return m;
  }
  return null;
}

export function createBridge({ photon, socket, projectId, out, cursorPath, sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
                               uuid = randomUUID, maxWaitMs = 30000 }) {
  const convo = new Map();            // normalized handle -> {spaceId, senderId, messageId}
  let lastSeq = 0;
  try { lastSeq = Number(JSON.parse(fs.readFileSync(cursorPath, "utf8")).seq) || 0; } catch { /* first run */ }
  let queue = Promise.resolve();      // events one at a time, sends one at a time
  let ready = false;
  const saveSeq = (seq) => {
    lastSeq = seq;
    try { fs.writeFileSync(cursorPath, JSON.stringify({ seq })); } catch { /* cursor is best effort */ }
  };

  // ---- receive
  async function onDelivery(event, metadata, ack) {
    const seq = Number(metadata?.seq);
    if (Number.isFinite(seq) && seq <= lastSeq) { if (typeof ack === "function") ack(); return; }   // replay
    const m = messageOf(event);
    const c = m?.content;
    const inbound = m && (m.status === "received" || m.direction === "inbound");   // never our own sends
    if (inbound && c && c.type === "text") {
      const handle = m.sentBy?.handle || "";
      const key = normHandle(handle);
      if (key) {
        convo.set(key, { spaceId: m.space?.id, senderId: m.receivedBy?.id || m.addressedTo?.id,
                         messageId: m.id });
        out({ inbound: true, from: handle, text: String(c.text || "").slice(0, 500) });
      }
    }
    if (Number.isFinite(seq)) saveSeq(seq);       // only after the event was handled
    if (typeof ack === "function") ack();
  }

  function wire() {
    socket.on("delivery", (event, metadata, ack) => {
      queue = queue.then(() => onDelivery(event, metadata, ack)).catch(() => {});
    });
    socket.on("delivery_error", (e) => out({ inbound: false, error: String(e?.message || e?.code || e) }));
    socket.on("connect_error", (e) => {
      if (!ready) out({ ready: false, error: "cannot connect to Photon: " + String(e?.message || e) });
      else out({ inbound: false, error: "connection problem: " + String(e?.message || e) });
    });
    const subscribe = () => socket.emit("subscribe", { startSeq: lastSeq }, (res) => {
      if (res && res.ok === false) {
        if (!ready) out({ ready: false, error: "subscribe refused: " + String(res.error?.message || res.error || "unknown") });
        return;
      }
      if (!ready) { ready = true; out({ ready: true }); }
    });
    socket.on("connect", subscribe);               // also after a reconnect: resume from the cursor
    if (socket.connected) subscribe();
  }

  // ---- send
  const retryAfterMs = (err) => {
    const v = Number(err?.headers?.get?.("retry-after"));
    return Math.min(maxWaitMs, (Number.isFinite(v) && v > 0 ? v : 2) * 1000);
  };

  async function sendText(to, text) {
    const c = convo.get(normHandle(to));
    if (!c || !c.spaceId || !c.senderId) {
      return { ok: false, error: "no conversation yet: text the agent first" };
    }
    const body = { spaceId: c.spaceId, senderId: c.senderId,
                   content: { type: "text", format: "plain", text: String(text || "").slice(0, 1000) } };
    const key = uuid();
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        await photon.projects.sendMessage({ path: { id: projectId }, headers: { "idempotency-key": key }, body });
        return { ok: true };                       // accepted, not necessarily delivered
      } catch (e) {
        if (e?.status === 429 && attempt === 0) { await sleep(retryAfterMs(e)); continue; }
        const code = e?.status ? `HTTP ${e.status}` : "";
        return { ok: false, error: `${code} ${String(e?.problem?.title || e?.message || e)}`.trim().slice(0, 200) };
      }
    }
    return { ok: false, error: "rate limited (HTTP 429), try again later" };
  }

  // ---- stdin lines
  function onLine(line) {
    let msg;
    try { msg = JSON.parse(line); } catch { return out({ ok: false, error: "bad json" }); }
    queue = queue.then(async () => {
      let r;
      try { r = await sendText(msg.to, msg.text); }
      catch (e) { r = { ok: false, error: String(e?.message || e) }; }
      out({ id: msg.id, ...r });
    });
  }

  return { wire, onLine, onDelivery, sendText, convo, get lastSeq() { return lastSeq; }, get ready() { return ready; },
           drain: () => queue };
}
