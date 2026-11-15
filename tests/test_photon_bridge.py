"""The Photon beta bridge's own tests (node --test, fake socket and client, no network)."""
import shutil
import subprocess
from pathlib import Path

import pytest

PHOTON = Path(__file__).resolve().parent.parent / "tools" / "photon"


@pytest.mark.skipif(not shutil.which("node") or not (PHOTON / "node_modules" / "socket.io-client").exists(),
                    reason="node or tools/photon dependencies missing (cd tools/photon && npm ci)")
def test_bridge_node_tests_pass():
    r = subprocess.run(["node", "--test", "bridge.test.mjs"], cwd=PHOTON, capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-500:]


def test_photon_sender_needs_the_beta_env_and_package(monkeypatch):
    from app.notify import PhotonSender
    for k in ("PHOTON_PROJECT_ID", "PHOTON_API_KEY", "SPECTRUM_PROJECT_ID", "SPECTRUM_PROJECT_SECRET"):
        monkeypatch.delenv(k, raising=False)
    assert PhotonSender.available() is False
    monkeypatch.setenv("SPECTRUM_PROJECT_ID", "old")
    monkeypatch.setenv("SPECTRUM_PROJECT_SECRET", "old")
    assert PhotonSender.available() is False                      # the stable-SDK env no longer counts
    monkeypatch.setenv("PHOTON_PROJECT_ID", "pho_prj_x")
    monkeypatch.setenv("PHOTON_API_KEY", "pho_sk_x")
    assert PhotonSender.available() == (PHOTON / "node_modules" / "@photon-ai" / "api").exists()


def test_photon_sender_gets_inbound_and_sends_through_a_bridge_process(tmp_path):
    """A stand-in bridge that follows the JSON-lines contract: ready, one inbound text, ok replies."""
    import sys
    import threading

    from app.notify import PhotonSender
    fake = tmp_path / "fake_bridge.py"
    fake.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "print(json.dumps({'inbound': True, 'from': 'Sam@Example.com', 'text': 'status'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    m = json.loads(line)\n"
        "    r = {'id': m['id'], 'ok': True} if m['to'] == 'sam@example.com' else "
        "{'id': m['id'], 'ok': False, 'error': 'no conversation yet: text the agent first'}\n"
        "    print(json.dumps(r), flush=True)\n")
    got, evt = [], threading.Event()
    s = PhotonSender(node=sys.executable, bridge=fake, timeout_s=5)
    s.on_message(lambda frm, text: (got.append((frm, text)), evt.set()))
    assert evt.wait(5) and got == [("Sam@Example.com", "status")]
    assert s.send("sam@example.com", "The band is connected.") == (True, None)
    ok, err = s.send("+15550009999", "hello?")
    assert not ok and "text the agent first" in err                  # the caregiver alert degrades, no crash
    s.close()


def test_photon_sender_counts_received_and_sent(tmp_path):
    import sys
    import threading

    from app.notify import PhotonSender
    fake = tmp_path / "fake_bridge2.py"
    fake.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "print(json.dumps({'inbound': True, 'from': '+15550000001', 'text': 'a'}), flush=True)\n"
        "print(json.dumps({'inbound': True, 'from': '+15550000001', 'text': 'b'}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    m = json.loads(line)\n"
        "    ok = m['text'] != 'bad'\n"
        "    print(json.dumps({'id': m['id'], 'ok': ok, **({} if ok else {'error': 'HTTP 403 nope'})}), flush=True)\n")
    n, evt = [], threading.Event()
    s = PhotonSender(node=sys.executable, bridge=fake, timeout_s=5)
    s.on_message(lambda f, t: (n.append(t), len(n) == 2 and evt.set()))
    assert evt.wait(5)
    s.send("+15550000001", "fine")
    s.send("+15550000001", "bad")
    assert s.stats() == {"received": 2, "sent_ok": 1, "sent_failed": 1, "last_error": "HTTP 403 nope"}
    s.close()


def test_photon_sender_starts_when_inbound_backlog_precedes_ready(tmp_path):
    """The bridge's first poll prints unseen inbound texts before its ready line (a new caregiver's
    first text, a backlog). That used to read as 'bridge failed to start' and kill the bridge on
    every launch, so no caregiver text could ever be sent."""
    import sys
    import threading

    from app.notify import PhotonSender
    fake = tmp_path / "fake_bridge3.py"
    fake.write_text(
        "import json, sys\n"
        "print(json.dumps({'inbound': True, 'from': '+15550000093', 'text': 'hi'}), flush=True)\n"
        "print(json.dumps({'inbound': True, 'from': '+15550000064', 'text': 'status'}), flush=True)\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    m = json.loads(line)\n"
        "    print(json.dumps({'id': m['id'], 'ok': True}), flush=True)\n")
    got, evt = [], threading.Event()
    s = PhotonSender(node=sys.executable, bridge=fake, timeout_s=5)
    s.on_message(lambda frm, text: (got.append((frm, text)), len(got) == 2 and evt.set()))
    assert evt.wait(5) and got == [("+15550000093", "hi"), ("+15550000064", "status")]
    assert s.send("+15550000093", "caregiver alert") == (True, None)
    assert s.error is None and s.stats()["received"] == 2
    s.close()


def test_photon_sender_reports_a_bridge_that_exits_before_ready(tmp_path):
    import sys

    from app.notify import PhotonSender
    fake = tmp_path / "fake_bridge4.py"
    fake.write_text("import sys\nsys.exit(1)\n")
    s = PhotonSender(node=sys.executable, bridge=fake, timeout_s=2)
    ok, err = s.send("+15550000093", "x")
    assert not ok and "exited before it was ready" in err
