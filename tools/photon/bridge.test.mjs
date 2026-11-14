// node --test tools/photon/bridge.test.mjs   (fake socket and fake Photon client, no network)
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createBridge, messageOf, normHandle } from "./bridge-core.mjs";

class FakeSocket {
  constructor() { this.h = {}; this.emitted = []; this.connected = false; this.subscribeAck = { ok: true }; }
  on(ev, fn) { this.h[ev] = fn; }
  emit(ev, payload, ack) { this.emitted.push([ev, payload]); if (ack) ack(this.subscribeAck); }
  connect() { this.connected = true; this.h.connect?.(); }
  deliver(event, metadata, ack) { return this.h.delivery(event, metadata, ack); }
}

let nextId = 0;
const received = (text, over = {}) => ({
  type: "message.received",
  data: { id: `msg_${++nextId}`, status: "received", platform: "imessage",
          content: { type: "text", format: "plain", text }, sentBy: { id: "usr_h", handle: "+1 (555) 000-0001" },
          receivedBy: { id: "usr_agent" }, space: { id: "spc_1" }, ...over },
});

function setup(opts = {}) {
  const lines = [], sends = [], sleeps = [], logs = [], listCalls = [];
  const photon = { projects: { listMessages: async (req) => {
    listCalls.push(req);
    const r = opts.pages ? opts.pages.shift() : { items: [] };
    if (r instanceof Error) throw r;
    return r ?? { items: [] };
  }, sendMessage: async (req) => {
    sends.push(req);
    if (opts.fail) { const e = opts.fail.shift(); if (e) throw e; }
    return { status: "accepted" };
  } } };
  const socket = new FakeSocket();
  const cursorPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "hdb-photon-")), "cursor.json");
  if (opts.cursor) fs.writeFileSync(cursorPath, JSON.stringify({ seq: opts.cursor }));
  const bridge = createBridge({ photon, socket, projectId: "pho_prj_test", out: (o) => lines.push(o), cursorPath,
                                poll: !!opts.poll, now: () => Date.parse("2026-11-14T12:00:00Z"),
                                log: (l) => logs.push(l), sleep: async (ms) => { sleeps.push(ms); },
                                uuid: () => "key-1" });
  bridge.wire();
  return { bridge, socket, lines, sends, sleeps, logs, listCalls, cursorPath };
}
const apiError = (status, retry) => Object.assign(new Error("rate"), { status, headers: new Headers(retry ? { "retry-after": String(retry) } : {}) });

test("subscribes from the saved cursor and says ready only after the ack", () => {
  const t = setup({ cursor: 41 });
  assert.deepEqual(t.lines, []);
  t.socket.connect();
  assert.deepEqual(t.socket.emitted, [["subscribe", { startSeq: 41 }]]);
  assert.deepEqual(t.lines, [{ ready: true }]);
});

test("a refused subscribe or a connect error before ready reports ready:false without secrets", () => {
  const a = setup();
  a.socket.subscribeAck = { ok: false, error: { message: "bad project" } };
  a.socket.connect();
  assert.match(a.lines[0].error, /subscribe refused: bad project/);
  const b = setup();
  b.socket.h.connect_error(new Error("websocket error"));
  assert.equal(b.lines[0].ready, false);
  assert.ok(!JSON.stringify(b.lines).includes("pho_sk"));
});

test("an inbound text becomes one inbound line, the cursor moves, a replay is ignored", async () => {
  const t = setup();
  t.socket.connect();
  await t.socket.deliver(received("status"), { seq: 5 });
  await t.bridge.drain();
  assert.deepEqual(t.lines.slice(1), [{ inbound: true, from: "+1 (555) 000-0001", text: "status" }]);
  assert.equal(JSON.parse(fs.readFileSync(t.cursorPath, "utf8")).seq, 5);
  await t.socket.deliver(received("status"), { seq: 5 });                 // replayed after a reconnect
  await t.socket.deliver(received("old"), { seq: 4 });
  await t.bridge.drain();
  assert.equal(t.lines.filter((l) => l.inbound).length, 1);
});

test("events are handled one at a time, in order", async () => {
  const t = setup();
  t.socket.connect();
  t.socket.deliver(received("one"), { seq: 1 });
  t.socket.deliver(received("two"), { seq: 2 });
  t.socket.deliver(received("three"), { seq: 3 });
  await t.bridge.drain();
  assert.deepEqual(t.lines.filter((l) => l.inbound).map((l) => l.text), ["one", "two", "three"]);
  assert.equal(t.bridge.lastSeq, 3);
});

test("non-text and our own messages are skipped, but the cursor still advances", async () => {
  const t = setup();
  t.socket.connect();
  await t.socket.deliver(received("x", { content: { type: "attachment" } }), { seq: 1 });
  await t.socket.deliver(received("echo", { status: "accepted" }), { seq: 2 });
  await t.socket.deliver({ type: "something.else" }, { seq: 3 });
  await t.bridge.drain();
  assert.equal(t.lines.filter((l) => l.inbound).length, 0);
  assert.equal(t.bridge.lastSeq, 3);
});

test("send with no conversation yet fails gracefully; after the human texts it sends", async () => {
  const t = setup();
  t.socket.connect();
  t.bridge.onLine(JSON.stringify({ id: 1, to: "+15550000001", text: "hi" }));
  await t.bridge.drain();
  assert.deepEqual(t.lines.at(-1), { id: 1, ok: false, error: "no conversation yet: text the agent first" });
  assert.equal(t.sends.length, 0);
  await t.socket.deliver(received("hello"), { seq: 1 });
  t.bridge.onLine(JSON.stringify({ id: 2, to: "+15550000001", text: "The band is connected." }));
  await t.bridge.drain();
  assert.deepEqual(t.lines.at(-1), { id: 2, ok: true });
  assert.deepEqual(t.sends[0], { path: { id: "pho_prj_test" }, headers: { "idempotency-key": "key-1" },
    body: { spaceId: "spc_1", senderId: "usr_agent", content: { type: "text", format: "plain", text: "The band is connected." } } });
});

test("emails match case-insensitively, phones by digits", async () => {
  assert.equal(normHandle("Sam@Example.COM"), normHandle("sam@example.com"));
  assert.equal(normHandle("+1 (555) 000-0001"), normHandle("15550000001"));
  assert.equal(normHandle(""), "");
  const t = setup();
  t.socket.connect();
  await t.socket.deliver(received("hi", { sentBy: { id: "u", handle: "Sam@Example.com" } }), { seq: 1 });
  t.bridge.onLine(JSON.stringify({ id: 3, to: "sam@example.com", text: "yo" }));
  await t.bridge.drain();
  assert.deepEqual(t.lines.at(-1), { id: 3, ok: true });
});

test("HTTP 429 waits for Retry-After and retries once with the same idempotency key", async () => {
  const t = setup({ fail: [apiError(429, 3)] });
  t.socket.connect();
  await t.socket.deliver(received("hi"), { seq: 1 });
  t.bridge.onLine(JSON.stringify({ id: 4, to: "+15550000001", text: "later" }));
  await t.bridge.drain();
  assert.deepEqual(t.sleeps, [3000]);
  assert.equal(t.sends.length, 2);
  assert.equal(t.sends[0].headers["idempotency-key"], t.sends[1].headers["idempotency-key"]);
  assert.deepEqual(t.lines.at(-1), { id: 4, ok: true });
});

test("other API errors come back as ok:false with the status, never a crash", async () => {
  const t = setup({ fail: [apiError(403)] });
  t.socket.connect();
  await t.socket.deliver(received("hi"), { seq: 1 });
  t.bridge.onLine(JSON.stringify({ id: 5, to: "+15550000001", text: "x" }));
  t.bridge.onLine("not json");
  await t.bridge.drain();
  assert.match(t.lines.find((l) => l.id === 5).error, /^HTTP 403/);
  assert.ok(t.lines.some((l) => l.error === "bad json"));
});

test("messageOf finds the message in the usual wrappers", () => {
  const m = received("x").data;
  for (const ev of [{ data: { message: m } }, { message: m }, { data: m }, m]) assert.equal(messageOf(ev), m);
  assert.equal(messageOf({ type: "typing" }), null);
});

// ---------------------------------------------------------------- polling (primary inbound path)
const item = (text, createdAt, over = {}) => received(text, { createdAt, ...over }).data;

test("polling: first poll starts 60 s back, asks for received/asc, emits inbound, then ready", async () => {
  const t = setup({ poll: true, pages: [{ items: [item("status", "2026-11-14T11:59:30.000Z")] }] });
  assert.equal(t.bridge.createdSince, "2026-11-14T11:59:00.000Z");            // now - 60 s: no old backlog
  const wait = await t.bridge.pollOnce();
  assert.equal(wait, 2000);
  assert.deepEqual(t.listCalls[0], { path: { id: "pho_prj_test" }, query: {
    createdSince: "2026-11-14T11:59:00.000Z", direction: "received", order: "asc", pageSize: 50, platform: "imessage" } });
  assert.deepEqual(t.lines, [{ inbound: true, from: "+1 (555) 000-0001", text: "status" }, { ready: true }]);
  assert.equal(t.bridge.createdSince, "2026-11-14T11:59:30.000Z");
  assert.equal(JSON.parse(fs.readFileSync(t.cursorPath, "utf8")).createdSince, "2026-11-14T11:59:30.000Z");
  assert.deepEqual(t.logs, ["inbound from +1***01"]);                          // masked
});

test("polling: createdSince is inclusive, so the newest message comes back and is not handled twice", async () => {
  const same = item("status", "2026-11-14T11:59:30.000Z");
  const t = setup({ poll: true, pages: [{ items: [same] }, { items: [same] }, { items: [same, item("more", "2026-11-14T11:59:40.000Z")] }] });
  await t.bridge.pollOnce(); await t.bridge.pollOnce(); await t.bridge.pollOnce();
  assert.deepEqual(t.lines.filter((l) => l.inbound).map((l) => l.text), ["status", "more"]);
  assert.equal(t.listCalls[1].query.createdSince, "2026-11-14T11:59:30.000Z");
  assert.equal(t.bridge.stats.received, 2);
});

test("polling and websocket share one dedupe: a message is never handled twice", async () => {
  const m = item("hello", "2026-11-14T11:59:30.000Z");
  const t = setup({ poll: true, pages: [{ items: [m] }] });
  t.socket.connect();
  await t.socket.deliver({ data: { message: m } }, { seq: 9 });
  await t.bridge.drain();
  await t.bridge.pollOnce();
  assert.equal(t.lines.filter((l) => l.inbound).length, 1);
  assert.equal(t.bridge.lastSeq, 9);
});

test("polling follows pages and ignores our own sent messages", async () => {
  const sent = item("reply", "2026-11-14T11:59:20.000Z", { status: "delivered" });
  const t = setup({ poll: true, pages: [
    { items: [sent, item("a", "2026-11-14T11:59:21.000Z")], nextPageToken: "p2" },
    { items: [item("b", "2026-11-14T11:59:22.000Z")] }] });
  await t.bridge.pollOnce();
  assert.deepEqual(t.lines.filter((l) => l.inbound).map((l) => l.text), ["a", "b"]);
  assert.equal(t.listCalls[1].query.pageToken, "p2");
  assert.equal(t.bridge.createdSince, "2026-11-14T11:59:22.000Z");
});

test("polling resumes from the saved createdSince instead of going back 60 s", () => {
  const t = setup({ poll: true });
  fs.writeFileSync(t.cursorPath, JSON.stringify({ seq: 0, createdSince: "2026-11-14T11:30:00.000Z" }));
  const again = createBridge({ photon: {}, projectId: "p", out: () => {}, cursorPath: t.cursorPath, poll: true,
                               now: () => Date.parse("2026-11-14T12:00:00Z") });
  assert.equal(again.createdSince, "2026-11-14T11:30:00.000Z");
});

test("polling: 429 backs off for Retry-After and keeps going; a hard error before ready says ready:false", async () => {
  const a = setup({ poll: true, pages: [{ items: [] }, apiError(429, 7), { items: [] }] });
  await a.bridge.pollOnce();
  assert.equal(await a.bridge.pollOnce(), 7000);
  assert.equal(await a.bridge.pollOnce(), 2000);
  assert.deepEqual(a.lines, [{ ready: true }]);
  const b = setup({ poll: true, pages: [apiError(401)] });
  assert.equal(await b.bridge.pollOnce(), null);
  assert.match(b.lines[0].error, /cannot read messages: HTTP 401/);
  assert.ok(!JSON.stringify(b.lines).includes("pho_sk"));
  const c = setup({ poll: true, pages: [{ items: [] }, apiError(500)] });          // after ready: keep trying
  await c.bridge.pollOnce();
  assert.equal(await c.bridge.pollOnce(), 10000);
  assert.deepEqual(c.lines, [{ ready: true }]);
  assert.match(c.bridge.stats.lastError, /HTTP 500/);
});

test("a polled message lets us reply to that handle", async () => {
  const t = setup({ poll: true, pages: [{ items: [item("hi", "2026-11-14T11:59:30.000Z")] }] });
  await t.bridge.pollOnce();
  t.bridge.onLine(JSON.stringify({ id: 8, to: "+15550000001", text: "Hello." }));
  await t.bridge.drain();
  assert.deepEqual(t.lines.at(-1), { id: 8, ok: true });
  assert.equal(t.sends[0].body.spaceId, "spc_1");
});
