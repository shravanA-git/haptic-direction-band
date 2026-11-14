"""Bench tool against the simulated band: gate, mapping, link health."""
import time

from app.link import Link, SimPort, load_motor_map, save_motor_map
from tools import bench


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_check_link_reports_streaming_sim():
    link = Link(SimPort()).start()
    out = []
    assert bench.check_link(link, 0.5, out.append)
    assert link.health()["state"] == "streaming"
    assert any("MIC every 50" in s for s in out)
    link.close()


def test_check_link_warns_when_no_mic_frames():
    port = SimPort(frames=[])            # early firmware: pulses only
    link = Link(port).start()
    port.inject("ets Jun  8 2016 rst:0x1 (POWERON_RESET)")
    out = []
    assert not bench.check_link(link, 0.3, out.append)
    assert link.health()["state"] == "lines"
    assert any("no MIC frames" in s for s in out)
    link.close()


def test_gate_sends_pulse_stop_finite_sequence():
    port = SimPort()
    link = Link(port).start()
    assert bench.run_gate(link, scripted(["y", "y", "y"]), lambda s: None)
    assert port.sent == ["PULSE 0 150", "PULSE 0 1000", "STOP", "PULSE 0 300"]
    assert not bench.run_gate(link, scripted(["y", "n", "y"]), lambda s: None)
    link.close()


def wiring_ask(port, wiring):
    """Simulated wearer: reports where the last pulsed channel is mounted."""
    def ask(prompt):
        ch = int([l for l in port.sent if l.startswith("PULSE")][-1].split()[1])
        p = wiring[ch]
        return "n" if p is None else bench.POS[p]
    return ask


def test_map_identity_passes():
    port = SimPort()
    link = Link(port).start()
    felt = bench.run_map(link, wiring_ask(port, list(range(8))), out=lambda s: None, pulse_ms=10)
    assert felt == {k: k for k in range(8)}
    assert bench.summarize_map(felt, lambda s: None) is None
    link.close()


def test_map_detects_swap_and_remap_fixes_it(tmp_path):
    wiring = [0, 1, 2, 5, 4, 3, 6, 7]          # channels 3 and 5 swapped on the band
    port = SimPort()
    link = Link(port).start()
    felt = bench.run_map(link, wiring_ask(port, wiring), out=lambda s: None, pulse_ms=10)
    m = bench.summarize_map(felt, lambda s: None)
    assert m == [0, 1, 2, 5, 4, 3, 6, 7]
    link.close()

    path = save_motor_map(m, tmp_path / "motor_map.json")
    port2 = SimPort()
    link2 = Link(port2, motor_map=load_motor_map(path)).start()
    seen = []
    link2.on_sent(seen.append)
    link2.pulse(3, 200)                         # logical back-right
    assert port2.sent[-1] == "PULSE 5 200"      # goes to the channel mounted back-right
    assert seen[-1] == "PULSE 3 200"            # dashboard still shows back-right
    link2.close()


def test_map_dead_channel_and_cardinal_subset():
    wiring = [0, None, 2, 3, 4, 5, 6, 7]
    port = SimPort()
    link = Link(port).start()
    felt = bench.run_map(link, wiring_ask(port, wiring), channels=[0, 2, 4, 6],
                         out=lambda s: None, pulse_ms=10)
    assert felt == {0: 0, 2: 2, 4: 4, 6: 6}     # 4-cardinal fallback passes
    felt = bench.run_map(link, wiring_ask(port, wiring), out=lambda s: None, pulse_ms=10)
    out = []
    assert bench.summarize_map(felt, out.append) is None
    assert any("no buzz on channel(s) [1]" in s for s in out)
    link.close()


def test_parse_position():
    assert bench.parse_position("FR") == 1
    assert bench.parse_position("back-left") == 5
    assert bench.parse_position("6") == 6
    assert bench.parse_position("n") is None
    assert bench.parse_position("a") == "again"
    assert bench.parse_position("up") == "?"


def test_cli_gate_in_sim(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "LOG", tmp_path / "bench.log")
    rc = bench.main(["--sim", "--gate", "--seconds", "0.3"], ask=scripted(["y", "y", "y"]))
    assert rc == 0
    assert "H3 PASS" in (tmp_path / "bench.log").read_text()
