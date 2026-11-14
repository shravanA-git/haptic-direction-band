"""Dashboard server over a real WebSocket, with the simulated band."""
import asyncio
import json

from aiohttp.test_utils import TestClient, TestServer

from app.link import Link, SimPort
from app.server import App


async def _collect(ws, pred, timeout=3.0):
    async def go():
        while True:
            m = json.loads((await ws.receive()).data)
            if pred(m):
                return m
    return await asyncio.wait_for(go(), timeout)


def test_ws_pulse_stop_and_replay():
    async def run():
        port = SimPort()
        link = Link(port).start()
        app = App(link)
        async with TestClient(TestServer(app.make_app())) as client:
            r = await client.get("/")
            assert r.status == 200 and "Motor ring" in await r.text()
            ws = await client.ws_connect("/ws")
            await _collect(ws, lambda m: m["type"] == "status")
            await _collect(ws, lambda m: m["type"] == "mic" and not m["replay"])

            await ws.send_json({"cmd": "pulse", "motor": 0, "ms": 150})
            m = await _collect(ws, lambda m: m["type"] == "pulse")
            assert m == {"type": "pulse", "motor": 0, "ms": 150, "dry": False}
            await ws.send_json({"cmd": "stop"})
            await _collect(ws, lambda m: m["type"] == "stop")
            assert port.sent == ["PULSE 0 150", "STOP"]

            await ws.send_json({"cmd": "replay", "name": "fixtures/back_left.json", "speed": 20})
            est = await _collect(ws, lambda m: m["type"] == "estimate")
            assert est["motor"] == 5
            dry = await _collect(ws, lambda m: m["type"] == "pulse")
            assert dry == {"type": "pulse", "motor": 5, "ms": 200, "dry": True}
            assert port.sent == ["PULSE 0 150", "STOP"]  # replay never drove motors

            await ws.send_json({"cmd": "replay", "name": "../../etc/passwd"})
            m = await _collect(ws, lambda m: m["type"] == "log")
            assert m["text"].startswith("error")
            await ws.close()
        link.close()

    asyncio.run(run())


class FakeVision:
    def __init__(self):
        self.cb, self.paused, self.target = None, False, None

    def on_change(self, cb):
        self.cb = cb

    def set_target(self, name):
        self.target = name
        return name

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


def test_vision_decisions_drive_band_and_action_pauses():
    from types import SimpleNamespace as D

    from app import haptics
    from app.guide import ObjectGuide

    async def run():
        port = SimPort()
        link = Link(port).start()
        app = App(link)
        fv = FakeVision()
        guide = ObjectGuide(lambda cue: haptics.play(cue, link.pulse).join())
        app.attach_vision(fv, guide)
        async with TestClient(TestServer(app.make_app())) as client:
            ws = await client.ws_connect("/ws")
            await ws.send_json({"cmd": "target", "name": "banana"})
            m = await _collect(ws, lambda m: m["type"] == "log")
            assert fv.target == "banana" and "banana" in m["text"]
            fv.cb(D(state="left", target="banana"))
            st = await _collect(ws, lambda m: m["type"] == "state")
            assert st["object"] == "banana: left"
            fv.cb(D(state="centered", target="banana"))
            port.inject("BTN ACTION DOWN")
            await _collect(ws, lambda m: m["type"] == "btn")
            assert fv.paused and guide.paused
            fv.cb(D(state="lost", target="banana"))   # suppressed while held
            await ws.close()
        link.close()
        assert port.sent[:3] == ["PULSE 6 200", "PULSE 0 100", "PULSE 0 100"]
        assert "PULSE 0 600" not in port.sent

    asyncio.run(run())
