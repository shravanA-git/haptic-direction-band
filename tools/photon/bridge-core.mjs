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
//   * receive, PRIMARY = poll photon.projects.listMessages({direction:'received',
//     createdSince, order:'asc'}) every ~2 s (a probe on the real project showed the
//     websocket acked a subscribe but delivered no events)
//   * receive, OPTIONAL = socket.io to /v1/projects/{id}/events/ws, 'subscribe'
//     {startSeq}, 'delivery' (event, metadata); both paths feed one handler that
//     dedupes by message id, so a message is never handled twice
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

export function createBridge({ photon, socket = null, projectId, out, cursorPath, poll = false,
                               pollEveryMs = 2000, backlogMs = 60000, now = () => Date.now(),
                               log = () => {}, sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
                               uuid = randomUUID, maxWaitMs = 30000 }) {
  const convo = new Map();            // normalized handle -> {spaceId, senderId, messageId}
  let lastSeq = 0, createdSince = null;
  try {
    const c = JSON.parse(fs.readFileSync(cursorPath, "utf8"));
    lastSeq = Number(c.seq) || 0;
    createdSince = c.createdSince || null;
  } catch { /* first run */ }
  if (poll && !createdSince) createdSince = new Date(now() - backlogMs).toISOString();   // don't answer old backlog
  let queue = Promise.resolve();      // events one at a time, sends one at a time
  let ready = false;
  const seen = new Map();             // message id -> true, newest 500 (poll and websocket share it)
  const stats = { received: 0, polls: 0, lastError: null };
  const saveCursor = () => {
    try { fs.writeFileSync(cursorPath, JSON.stringify({ seq: lastSeq, createdSince })); } catch { /* best effort */ }
  };
  const markReady = () => { if (!ready) { ready = true; out({ ready: true }); } };
  const mask = (h) => { const s = String(h); return s.includes("@") ? s[0] + "***" + s.slice(s.indexOf("@")) : s.slice(0, 2) + "***" + s.slice(-2); };

  // One message from either path. Returns true when it was an inbound text we passed on.
  function handleMessage(m) {
    const c = m?.content;
    const inbound = m && (m.status === "received" || m.direction === "inbound");   // never our own sends
    if (!inbound || !c || c.type !== "text") return false;
    if (m.id) {
      if (seen.has(m.id)) return false;            // already handled (poll vs websocket, replays)
      seen.set(m.id, true);
      if (seen.size > 500) seen.delete(seen.keys().next().value);
    }
    const handle = m.sentBy?.handle || "";
    const key = normHandle(handle);
    if (!key) return false;
    convo.set(key, { spaceId: m.space?.id, senderId: m.receivedBy?.id || m.addressedTo?.id, messageId: m.id });
    stats.received++;
    log(`inbound from ${mask(handle)}`);
    out({ inbound: true, from: handle, text: String(c.text || "").slice(0, 500) });
    return true;
  }

  // ---- receive, websocket (optional extra)
  async function onDelivery(event, metadata, ack) {
    const seq = Number(metadata?.seq);
    if (Number.isFinite(seq) && seq <= lastSeq) { if (typeof ack === "function") ack(); return; }   // replay
    handleMessage(messageOf(event));
    if (Number.isFinite(seq)) { lastSeq = seq; saveCursor(); }     // only after the event was handled
    if (typeof ack === "function") ack();
  }

  // ---- receive, polling (primary)
  const retryAfterMs = (err) => {
    const v = Number(err?.headers?.get?.("retry-after"));
    return Math.min(maxWaitMs, (Number.isFinite(v) && v > 0 ? v : 2) * 1000);
  };

  // One poll: every received message since createdSince, oldest first. Returns ms to wait before the next.
  async function pollOnce() {
    stats.polls++;
    try {
      let pageToken, newest = createdSince;
      for (let page = 0; page < 5; page++) {
        const res = await photon.projects.listMessages({ path: { id: projectId }, query: {
          createdSince, direction: "received", order: "asc", pageSize: 50, platform: "imessage",
          ...(pageToken ? { pageToken } : {}) } });
        const items = res?.items ?? res?.data?.items ?? [];
        for (const m of items) {
          handleMessage(m);
          if (m.createdAt && m.createdAt > newest) newest = m.createdAt;   // RFC 3339 sorts as text
        }
        pageToken = res?.nextPageToken ?? res?.data?.nextPageToken;
        if (!pageToken) break;
      }
      if (newest !== createdSince) { createdSince = newest; saveCursor(); }   // after the page was handled
      stats.lastError = null;
      markReady();
      return pollEveryMs;
    } catch (e) {
      stats.lastError = `${e?.status ? "HTTP " + e.status + " " : ""}${String(e?.message || e)}`.slice(0, 160);
      log(`poll error: ${stats.lastError}`);
      if (e?.status === 429) return retryAfterMs(e);                   // back off, keep going
      if (!ready) { out({ ready: false, error: "cannot read messages: " + stats.lastError }); return null; }
      return Math.min(maxWaitMs, pollEveryMs * 5);
    }
  }

  function startPolling() {
    let stopped = false;
    (async () => {
      while (!stopped) {
        const wait = await pollOnce();
        if (wait === null) return;                 // first poll failed hard: ready:false was sent
        await sleep(wait);
      }
    })();
    return () => { stopped = true; };
  }

  function wire() {
    if (!socket) return;
    socket.on("delivery", (event, metadata, ack) => {
      queue = queue.then(() => onDelivery(event, metadata, ack)).catch(() => {});
    });
    socket.on("delivery_error", (e) => log("websocket delivery_error: " + String(e?.message || e?.code || e)));
    socket.on("connect_error", (e) => {
      if (!ready && !poll) out({ ready: false, error: "cannot connect to Photon: " + String(e?.message || e) });
      else log("websocket: " + String(e?.message || e));
    });
    const subscribe = () => socket.emit("subscribe", { startSeq: lastSeq }, (res) => {
      if (res && res.ok === false) {
        if (!ready && !poll) out({ ready: false, error: "subscribe refused: " + String(res.error?.message || res.error || "unknown") });
        return;
      }
      if (!poll) markReady();                       // with polling, a successful poll decides
    });
    socket.on("connect", subscribe);               // also after a reconnect: resume from the cursor
    if (socket.connected) subscribe();
  }

  // ---- send
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

  return { wire, startPolling, pollOnce, onLine, onDelivery, handleMessage, sendText, convo, stats,
           get lastSeq() { return lastSeq; }, get createdSince() { return createdSince; },
           get ready() { return ready; }, drain: () => queue };
}
