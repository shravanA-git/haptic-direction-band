"""4-motor band: logical 0..7 collapse onto the motors that exist (haptics.adapt)."""
import asyncio

import pytest

from app import guide as G
from app import haptics
from app.link import Link, SimPort

FOUR = (0, 2, 4, 6)


@pytest.fixture(autouse=True)
def reset():
    yield
    haptics.set_motors(None)


def test_all_eight_is_unchanged():
    cue = haptics.directional(1) + haptics.centered()
    assert haptics.adapt(cue) == cue and haptics.ACTIVE is None
    haptics.set_motors(range(8))
    assert haptics.ACTIVE is None


def test_cardinals_pass_through_diagonals_become_two_neighbours_back_to_back():
    assert haptics.adapt(haptics.directional(2), FOUR) == [(0, 2, 200)]
    ne = haptics.adapt(haptics.directional(1), FOUR)                  # between front 0 and right 2
    assert [m for _, m, _ in ne] == [0, 2] and ne[0][2] == ne[1][2] == haptics.PAIR_MAX_MS
    assert ne[1][0] == ne[0][0] + ne[0][2] + haptics.SEQ_GAP_MS       # sequenced, never overlapping
    assert [m for _, m, _ in haptics.adapt(haptics.directional(7), FOUR)] == [6, 0]
    assert [m for _, m, _ in haptics.adapt(haptics.directional(3), FOUR)] == [2, 4]
    assert [m for _, m, _ in haptics.adapt(haptics.directional(5), FOUR)] == [4, 6]


def test_simultaneous_steps_are_sequenced_one_motor_at_a_time():
    ack = haptics.adapt(haptics.target_ack(), FOUR)                   # both sides at once -> 6 then 2
    assert [m for _, m, _ in ack] == [6, 2] and ack[1][0] >= ack[0][0] + ack[0][2]
    for name, steps in [("sound", haptics.taught_sound()), ("unknown", haptics.target_unknown()),
                        ("walk", haptics.walk()), *[(c, haptics.taught_sound(c)) for c in haptics.SOUND_PATTERNS]]:
        out = haptics.adapt(steps, FOUR)
        assert {m for _, m, _ in out} <= set(FOUR), name
        assert all(b[0] >= a[0] + a[2] for a, b in zip(out, out[1:])), name   # no overlap at all


def test_hot_cold_uses_left_centre_right_with_four_motors():
    haptics.set_motors(FOUR)
    assert G.motor_for("left", -0.3) == 6 and G.motor_for("left", -0.8) == 6      # was 7 / 6
    assert G.motor_for("right", 0.3) == 2 and G.motor_for("right", 0.8) == 2      # was 1 / 2
    assert G.motor_for("centered", 0.0) == 0 and G.motor_for("left", None) == 6
    haptics.set_motors(None)
    assert G.motor_for("left", -0.3) == 7 and G.motor_for("right", 0.3) == 1


def test_nearest_active_breaks_ties_toward_the_objects_side():
    assert haptics.nearest_active(7, -0.2, FOUR) == 6 and haptics.nearest_active(7, 0.2, FOUR) == 0
    assert haptics.nearest_active(1, 0.2, FOUR) == 2 and haptics.nearest_active(2, 0.0, FOUR) == 2
    assert haptics.nearest_active(1, 0.0, (0, 4)) in (0, 4)                       # a 2-motor band still works


def test_parse_and_validate():
    assert haptics.parse_motors("0, 2,4,6") == FOUR and haptics.parse_motors("") is None
    with pytest.raises(ValueError):
        haptics.set_motors([0, 9])
    with pytest.raises(ValueError):
        haptics.set_motors([])


def test_play_sends_only_existing_motors_and_status_lists_them():
    from aiohttp.test_utils import TestClient, TestServer

    from app.server import App
    from tests.test_server import _collect

    async def run():
        port = SimPort()
        link = Link(port).start()
        app = App(link)
        haptics.set_motors(FOUR)
        async with TestClient(TestServer(app.make_app())) as client:
            ws = await client.ws_connect("/ws")
            st = await _collect(ws, lambda m: m["type"] == "status")
            assert st["motors"] == [0, 2, 4, 6]
            await ws.send_json({"cmd": "pulse", "motor": 1, "ms": 150})        # tap a diagonal
            await asyncio.sleep(0.4)
            sent = [l for l in port.sent if l.startswith("PULSE")]
            assert [l.split()[1] for l in sent] == ["0", "2"]
            r = await client.get("/")
            assert "showMotors" in await r.text()
        link.close()

    asyncio.run(run())
