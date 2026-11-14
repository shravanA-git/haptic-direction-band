"""Dashboard server over a real WebSocket, with the simulated band."""
import asyncio
import time
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
        assert sorted(port.sent[:2]) == ["PULSE 2 120", "PULSE 6 120"]   # typed target: "got it"
        assert port.sent[2:5] == ["PULSE 6 200", "PULSE 0 100", "PULSE 0 100"]
        assert "PULSE 0 600" not in port.sent

    asyncio.run(run())


def test_stop_cancels_pending_cue_steps():
    from app import haptics
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    app.handle({"cmd": "cue", "name": "centered"})   # two front pulses, 200 ms apart
    time.sleep(0.05)
    app.handle({"cmd": "stop"})
    time.sleep(0.35)
    assert port.sent == ["PULSE 0 100", "STOP"]      # second pulse never sent
    haptics.play(haptics.centered(), link.pulse).join()  # new cues still work
    assert port.sent[-2:] == ["PULSE 0 100", "PULSE 0 100"]
    link.close()


def test_bench_sequences():
    port = SimPort()
    link = Link(port).start()
    app = App(link)
    app.handle({"cmd": "bench", "name": "gate"})
    time.sleep(2.8)
    assert port.sent == ["PULSE 0 150", "PULSE 0 1000", "STOP", "PULSE 0 300"]
    app.handle({"cmd": "bench", "name": "cardinals"})
    time.sleep(2.3)
    assert port.sent[4:] == ["PULSE 0 250", "PULSE 2 250", "PULSE 4 250", "PULSE 6 250"]
    h = app.status()["health"]
    assert h["state"] == "streaming" and h["mic"] > 0
    link.close()


class FakeSpeech:
    def __init__(self):
        self.on_result, self.presses = None, []

    def on_button(self, down):
        self.presses.append(down)


def test_action_drives_speech_and_spoken_target_reaches_vision():
    from types import SimpleNamespace as R

    from app import haptics
    from app.guide import ObjectGuide

    async def run():
        port = SimPort()
        link = Link(port).start()
        app = App(link)
        fv, sp = FakeVision(), FakeSpeech()
        app.attach_vision(fv, ObjectGuide(lambda cue: haptics.play(cue, link.pulse)))
        app.attach_speech(sp)
        async with TestClient(TestServer(app.make_app())) as client:
            ws = await client.ws_connect("/ws")
            port.inject("BTN ACTION DOWN")               # board button
            await _collect(ws, lambda m: m["type"] == "btn" and m["down"])
            await ws.send_json({"cmd": "action", "down": False})   # dashboard button
            await _collect(ws, lambda m: m["type"] == "btn" and not m["down"])
            assert sp.presses == [True, False] and not fv.paused

            sp.on_result(R(text="Find my phone.", target="cell phone", error=None))
            st = await _collect(ws, lambda m: m["type"] == "state")
            assert st["heard"] == "Find my phone." and fv.target == "cell phone"

            sp.on_result(R(text="Hello there.", target=None, error=None))
            m = await _collect(ws, lambda m: m["type"] == "log" and "no known object" in m["text"])
            assert fv.target == "cell phone"             # typed/previous target kept
            sp.on_result(R(text="", target=None, error="too short"))
            await _collect(ws, lambda m: m["type"] == "log" and "too short" in m["text"])
            await ws.close()
        link.close()

    asyncio.run(run())


def test_video_streams_mjpeg_of_annotated_frames():
    import numpy as np
    import pytest
    pytest.importorskip("cv2")

    class Cam(FakeVision):
        def __init__(self):
            super().__init__()
            self.frame = np.zeros((48, 64, 3), np.uint8)

        def latest_frame(self):
            return self.frame

    async def run():
        link = Link(SimPort()).start()
        app = App(link)
        async with TestClient(TestServer(app.make_app())) as client:
            r = await client.get("/video")
            assert r.status == 503                      # no vision attached
            r = await client.get("/")
            assert 'id="cam"' in await r.text()
            from app import haptics
            from app.guide import ObjectGuide
            app.attach_vision(Cam(), ObjectGuide(lambda cue: haptics.play(cue, link.pulse)))
            r = await client.get("/video")
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
            chunk = b""
            while chunk.count(b"\xff\xd9") < 1:
                chunk += await asyncio.wait_for(r.content.read(4096), 3)
            assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg") and b"\xff\xd8" in chunk
            r.close()
        link.close()

    asyncio.run(run())
