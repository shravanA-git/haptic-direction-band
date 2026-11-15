# Relaunch the app and run every test (works with no AI)

Haptic Direction Band: ESP32 hat (4 mics, 4 motors, touch pad) + laptop app (camera, whisper, dashboard).
Everything below is copy-paste. Written Nov 15, 2026, ~04:00. App folder: `HackPrincetonNew/hdb`.

## 0. Prompt for a fresh Claude Code session (paste this as the first message)

> You are helping me test the Haptic Direction Band hackathon prototype. Read
> `docs/final_test/RELAUNCH_AND_TEST.md` first and follow it.
> Rules: never read or print `.env`, `local.yaml` or `agent_config.yaml`; never print phone numbers or keys;
> do not force-push, stash, reset or switch branches in `hdb`; only one program may own the USB serial port, so
> stop anything holding it before launching; do not start the agent runner unless I ask. Start with section 1
> (clean slate), then 3 (launch), then 4 (run `final_check.py`) and tell me every FAIL/WARN with the fix from
> section 7. Ask me before sending any real text message (`--caregiver`) or buzzing motors (`--pulse`).

## 1. Clean slate (stop anything that would interfere)

```bash
# who holds the serial port, who holds the dashboard port
lsof /dev/cu.usbserial-0001; lsof -nP -iTCP:8765 -sTCP:LISTEN
# stop leftover app / bridge / test processes (safe: only project processes)
pkill -f "app.main" ; pkill -f "bridge.mjs" ; pkill -f "app.speech --listen" ; sleep 2
lsof /dev/cu.usbserial-0001 || echo PORT_FREE
```
Also: quit the **Arduino IDE** completely (its serial helper holds the port) and close Serial Monitor/Plotter.
Hang on? A Ctrl-C'd app can stay alive holding the port: `kill -TERM <pid>` from `lsof`, then `kill -KILL` if needed.
Optional: pause the AI agents (they edit the code and run camera/speaker tests): `pkill -f "hdb_team.py run"`.

## 2. Hardware checklist

- ESP32 on a **data** USB cable, plugged straight into the Mac. Strain-relief the cable (lifting the hat dropped USB before).
- Firmware = `band_link/band_link.ino` flashed from the Arduino IDE (ESP32 Dev Module, 115200). After flashing: close Serial Monitor, quit the IDE.
- Motor pins: laptop motor 0 front = GPIO26, 2 right = GPIO23, 4 back = GPIO18, 6 left = GPIO16. Touch pad IO on GPIO27 or GPIO4 (3V3 + GND).
- Motor battery pack OFF until the link checks pass, ON for motor tests. If a buzz resets the board, the battery is sagging.
- Webcam plugged in. macOS permissions (System Settings > Privacy & Security): **Microphone and Camera for the app that runs your terminal**
  (Claude app / Terminal). Without Microphone the app hears pure silence and names/sounds never fire.

## 3. Launch (one terminal tab)

```bash
cd <path-to-repo>
# FULL demo: band + camera + names + sounds + band agent + phone dashboard
env -u CAREGIVER_PHONE -u WEARER_PHONE -u HDB_NAME .venv/bin/python -m app.main --port /dev/cu.usbserial-0001 --vision --agent --lan --sound --motors 0,2,4,6 --name Max
```
Why `env -u`: the app never overrides variables that are already set, so a stale `CAREGIVER_PHONE` in the terminal beats the one in `.env`.

Faster names (camera off, ~1-2 s): remove `--vision`. No band at all: replace `--port ...` with `--sim`.
Startup lines that must appear: `4-motor band`, `mic calibration from app/calibration.json: gains [1.388, 1.376, 1.0, 1.708]`,
`listening for name(s): Max`, `serial: /dev/cu.usbserial-0001`, `dashboard: http://localhost:8765`, and NOT `vision OFF` / `name listener off`.
Dashboard: http://localhost:8765 (phone: the `?k=` URL printed at startup, same Wi-Fi).

## 4. Automated checks (needs the app running; uses laptop speakers; quiet room)

```bash
cd <path-to-repo>
.venv/bin/python -I docs/final_test/final_check.py                 # link, settings, name x3, taught sound, hold-to-talk
.venv/bin/python -I docs/final_test/final_check.py --touch         # + touch pad window (touch it when told)
.venv/bin/python -I docs/final_test/final_check.py --pulse         # + buzz motors front, right, back, left (battery ON)
.venv/bin/python -I docs/final_test/final_check.py --caregiver     # + ONE real test text to the caregiver
```
Pass criteria: link `streaming` at ~25 ms; names heard >=2/3 with median latency < 3000 ms; taught sound matched and buzzes
(sweep, then a ~450 ms pulse toward the sound); `hold-to-talk 'find the cup'` sets target cup; touch events down+up; caregiver `ok`.
**The room must be quiet.** The checks play speech and tones through the speakers and the mic hears everything: a person talking,
typing, or an agent running a speaker test makes names late/missed, teaches the wrong sound (the app teaches from the LOUDEST event in
its 2.5 s window) and makes "find the cup" hear someone else's words. Reference run in a quiet room, camera on, 03:55:
names 3/3 heard, median 3021 ms (was 10928 ms before the clock fix), 2/3 directional; hold-to-talk "Find a cup." -> target cup.
Noisy run (someone talking): names 1/3 at ~5 s, taught-sound WARN, hold-to-talk heard other speech. Rerun when quiet before judging.
Offline unit tests (no hardware): `cd hdb && .venv/bin/python -m pytest -q` (315 passed at 03:50).

Other tools: `python -m tools.bench --check` (serial link health), `python -m tools.bench --map --motors 0,2,4,6` (say where you felt each motor),
`python -m tools.preflight` (camera, detector, ports).

## 5. Manual checks (30 s each, on http://localhost:8765)

1. **Touch pad**: hold, say "find the cup", release. Dashboard: heard "Find the cup." -> target cup -> camera searching.
2. **Hot/cold**: cup left/right -> that side's motor; centered -> front; closer -> faster pulses; remove it -> one lost buzz.
3. **Name**: say "Max" 3 times, 2 s apart, from the hat's front/right/back/left. Buzz on the matching side (head mounted).
4. **Taught sound**: dashboard Teach a sound, make the sound once, label it, replay it: sweep + direction pulse.
5. **Caregiver**: switch caregiver texts on; trigger a find; phone gets the update.
6. **STOP**: dashboard STOP / Esc stops all motors; unplug USB = motors off (firmware).

## 6. State of the code

Everything below is on `main`: the 5 s windowed audio-clock offset for names and taught sounds (fixes names always buzzing front and the ~10 s
latency), taught-sound cue = category sweep then a 450 ms pulse toward the sound (`HDB_SOUND_STYLE=direction` for one pulse), the Photon bridge
ready-line race fix (caregiver texts send), tap-on-pad = "found it" (stops the search), empty recordings say so without a buzz.
`firmware/band_link/band_link.ino` is the firmware that matches this app (flash it from the Arduino IDE).
Queued but not done at the time of writing: vision-vs-names latency, a touch hold-to-talk rewrite (names keep running while held),
caregiver failsafe/activity texts.

## 7. Known problems and the fix

| Symptom | Cause | Fix |
|---|---|---|
| `Could not exclusively lock port` / Resource busy | another program owns the serial port (IDE helper, old app) | section 1; `lsof /dev/cu.usbserial-0001` |
| `address already in use` on 8765 | old app still running | `lsof -nP -iTCP:8765 -sTCP:LISTEN`, kill that pid |
| Band `silent`, 0 MIC lines | board running an old/test sketch, or stuck after a USB drop | press EN, re-upload `band_link.ino`; raw check: read the port with pyserial for `MIC` lines |
| Band drops when the hat moves | USB cable/socket | strain relief, short braided cable, plug straight in; app reconnects by itself after ~2 s silence |
| Names never buzz, no errors | **no microphone permission** (mic returns zeros) | grant Microphone to the terminal host app, restart the app |
| Names late (~10 s) and always front, reason `no frames in window` | audio clock drift after dropped samples | needs the windowed-offset fix in `app/speech.py` (section 6) |
| Names slow only with `--vision` | camera + models starve the CPU | run names without `--vision`; fix is queued with the agent |
| Name alerts show but nothing buzzes | `name_alerts` setting off | dashboard toggle, or `final_check.py` settings line |
| Touch pad does nothing | firmware without the pad, or IO not on GPIO27/4 | reflash `band_link.ino`; check the module LED and 3V3 |
| Caregiver text goes to the old number | stale `CAREGIVER_PHONE` in the terminal env | launch with `env -u CAREGIVER_PHONE ...` (section 3) |
| `bridge failed to start` on caregiver test | Photon bridge prints inbound texts before ready | needs the notify.py fix (section 6) |
| `no conversation yet: text the agent first` | Photon shared line: a number must text the agent first | have the caregiver text the agent once |
| A voice says "hey Shravan" | an agent's latency test using the speakers | pause agents (section 1) |
| Camera `not authorized` | camera permission for the terminal host app | System Settings > Privacy & Security > Camera |
| Taught sound buzzes several motors | by design: the category sweep, then the direction pulse | `HDB_SOUND_STYLE=direction` for a single pulse |

## 8. Resume the AI agents (optional)

```bash
cd <path-to-repo>/agents && uv run python hdb_team.py run
```
They edit `hdb` while running: do not relaunch the band app mid-edit, and keep them off during a final run.
Never read or print `.env`, `local.yaml`, `agent_config.yaml`. The repo `shravanA-git/haptic-direction-band` is PRIVATE and must be made public
before the 8 AM Sunday deadline (scan git history for secrets first). Devpost: Healthcare track, 2-minute video, Discord names.
