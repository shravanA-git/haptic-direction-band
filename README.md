# PulsePath (Haptic Direction Band)

**An assistive headband that turns vision and sound into touch.** A webcam, four microphones, four vibration
motors and a touch pad on a hat, connected over USB to a laptop that does the heavy thinking.

- **Blind or low-vision:** hold the pad and say *"find the cup"*. The band buzzes the side the object is on, and
  the buzzing speeds up as you get closer ("hot and cold"). Tap the pad when you've found it and it stops.
- **Deaf or hard of hearing:** when someone calls your name, the band buzzes from the direction they called.
  Teach it any sound (a doorbell, a kettle) and it buzzes a pattern for it, then toward where it came from.
- **Caregiver in the loop:** a caregiver gets texts when the wearer is called, finds something, or is stuck, and
  can text the band's number ("status", "what did I miss today", "find my keys", "is she ok") and get a short plain
  answer.

HackPrinceton 2026, **Healthcare** track. Built in 36 hours by Shravan (software and integration), Sophanit (vision
and speech) and Jason (hardware and firmware), with a team of AI agents on Band working alongside us.

## How it works

```
 hat: 4 mics (MAX4466) + touch pad + 4 motors        laptop (Python)
 ┌───────────────────────────────┐   USB serial   ┌───────────────────────────────────────────────┐
 │ ESP32  band_link.ino          │ ─ MIC f r b l ─▶│ link.py       reconnects itself on USB drops  │
 │  MIC every 25 ms, raw levels  │ ─ BTN ACTION ──▶│ direction.py  where did the sound come from?  │
 │  BTN ACTION DOWN/UP (pad)     │ ◀─ PULSE m ms ──│ haptics.py    cues, <=2 motors, <=1 s pulses  │
 │  PULSE / STOP, motors off at  │ ◀─ STOP ────────│ vision.py + guide.py   hot/cold find          │
 │  boot, forced release at 12 s │                 │ speech.py     hold-to-talk + name spotting    │
 └───────────────────────────────┘                 │ listen.py     teach-a-sound                   │
        laptop webcam ───────────────────────────▶ │ notify.py     caregiver texts (Photon)        │
        laptop microphone ───────────────────────▶ │ band_agent.py LangChain deep agent over text  │
                                                   │ server.py     dashboard (localhost + phone)   │
                                                   └───────────────────────────────────────────────┘
```

| Feature | How | Where |
|---|---|---|
| Find an object | YOLO11s (COCO) plus YOLOE open-vocabulary for anything else; the object's offset and size drive a hot/cold pulse (1.2 s apart when cold to 0.18 s when hot), relative to its starting size | `app/vision.py`, `app/guide.py` |
| Hold-to-talk | faster-whisper (`tiny.en`) on the laptop mic; keywords first, a LangChain call only for "I'm thirsty" style requests | `app/speech.py`, `app/agent_find.py` |
| Name direction | faster-whisper spots the name with word timestamps; the four hat mics give the direction over that window (spike-only robust estimator, learned per-wearer centroids) | `app/direction.py`, `tools/calibrate.py` |
| Teach a sound | acoustic signature templates; a taught sound gets a category sweep, then a pulse toward it | `app/listen.py` |
| Caregiver texts, Band Agent | Photon 2.0 iMessage bridge; a LangChain **deepagents** agent with tools over the same status and alert code, and an offline keyword fallback | `app/notify.py`, `app/band_agent.py`, `tools/photon/` |
| Dashboard | motor ring, live mic levels, test pulses, STOP, replay, settings, Teach a Sound, phone "Add to Home Screen" | `dashboard/`, `app/server.py` |
| Safety | motors off at boot, every pulse finite, STOP always wins, max 2 motors at once, firmware forces a release after 12 s, a USB drop stops the motors and the app reconnects | `firmware/band_link/`, `app/link.py` |

### How we used LangChain

LangChain does the fuzzy thinking and plain Python does the safety-critical buzzing, so a slow model can never delay
a buzz. (1) **Target picker** (`app/agent_find.py`): "I'm thirsty" becomes *bottle* through a structured-output call
(`TargetChoice`), only when keyword matching fails. (2) **Band Agent** (`app/band_agent.py`): a LangChain Deep Agent
(`deepagents` on LangGraph) answers the wearer's and the caregiver's texts with seven tools that wrap existing code.
It answers only the configured numbers and has no access to haptics, so a text can never fire a motor. A plug-in hook
(`app/costar_tools.py`, or `HDB_AGENT_PLUGINS`) adds tools without editing the file. (3) `python -m app.agent_find`
is a stand-alone Deep Agent for finding objects (planning, a camera "scout" and a "memory-keeper" sub-agent, and
human confirmation before guiding).

## Run it

Hardware: ESP32 (classic WROOM), four MAX4466 mics (front 32, right 35, back 34, left 33), four motors through
2N2222 drivers (front GPIO26, right 23, back 18, left 16), a capacitive touch module (IO on GPIO27 or 4), and a 2xAA
motor pack with a common ground. Flash [`firmware/band_link/band_link.ino`](firmware/band_link/band_link.ino) from the
Arduino IDE (ESP32 Dev Module, 115200).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-agent.txt
cd tools/photon && npm ci && cd ../..                    # only needed for texting
cp .env.example .env                                      # then fill it in; .env is never committed
.venv/bin/python -m app.main --port /dev/cu.usbserial-XXXX --vision --agent --lan --sound --motors 0,2,4,6 --name Max
# no hardware at all:   .venv/bin/python -m app.main --sim --name Max          dashboard: http://localhost:8765
```

On macOS, allow **Microphone and Camera** for your terminal app, or the mics return silence.

A clean relaunch, a one-command end-to-end check, and a fix table for every problem we hit are in
[`docs/final_test/RELAUNCH_AND_TEST.md`](docs/final_test/RELAUNCH_AND_TEST.md).

## Tests

```bash
.venv/bin/python -m pytest -q                      # 300+ tests, no hardware needed (simulated band, fake models)
node --test tools/photon/bridge.test.mjs           # Photon bridge
.venv/bin/python docs/final_test/final_check.py    # against a running app: link, names, sounds, hold-to-talk
```

## Honest limits

- Direction from the hat microphones is a coarse eight-way estimate, only as good as the mounting and calibration
  (`tools/calibrate.py`, `tools/learn_centroids.py`). A speaker on a desk is not a person wearing the hat.
- With the camera running, a name takes about 3 s from the end of the name to the buzz (about 1 s without the camera).
- The hat microphones hear loudness, not words: speech recognition uses the laptop microphone.
- Photon's shared iMessage line only texts numbers that texted the agent first.

## Repository map

`app/` laptop software, `dashboard/` web UI, `firmware/` ESP32, `tools/` calibration, bench and eval, `tests/`
tests and recorded fixtures, `agents/` the Band agents' roster and room rules, `docs/` handoffs and the test
runbook. Keys and phone numbers live only in `.env` (gitignored); see `.env.example`.

MIT licensed. Names, numbers and keys in tests are placeholders.
