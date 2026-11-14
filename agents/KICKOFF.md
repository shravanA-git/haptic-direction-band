# H0 kickoff message

Post this in the shared Band room once the agents are in it.

---

H0: hacking has started (8 PM Fri). Goal for H0–H3, bring-up. **Gate at H3: the laptop buzzes any motor on command, and one mic streams.** Work autonomously in your lane, follow the room rules, pull/commit/push to `main` as you go, and post one `[done]` line per finished item.

@HDB-Gatekeeper Start the clock. Create one Band task per H0–H3 item below, assigned to its owner, plus a task for the H3 gate check. Post the gate in one line and check in at H2 with what's at risk.

@HDB-Protocol The protocol names patterns `NAME` and `GUIDE` but not the rest. Post a `[proposal]` table of pattern names for every haptic pattern (e.g. `PULSE`, `DOUBLE`, `ALARM`, `TICK`, `FALL`, taught sounds) with durations from the haptic vocabulary, so firmware and software use identical strings. Humans from both sides confirm before it's final.

@HDB-Software Build the software bring-up in `app/`, `dashboard/` and `tests/`: a `pyproject.toml`; `app/link.py` reading and writing protocol lines over USB serial (921,600 baud) with a **fake ESP32 mode** that emits realistic `L` lines and `EV` events, so nothing waits on firmware; a single `send()` behind a lock; `app/haptics.py` with the priority queue (fall > alarm > name > guide > compass); and `dashboard/` (FastAPI + one WebSocket) showing live mic loudness, heading and the 8-motor ring, with buttons that send `P,<motor>,<pattern>`. Tests run against fake mode. Wait 2 s after opening the serial port (it resets the ESP32).

@HDB-Firmware Firmware and hardware bring-up in `firmware/`: `band.ino` prints `hello` at 921,600 baud, parses newline-terminated commands, and `P,0,<pattern>` buzzes motor 0 through one 2N2222 channel (220 Ω base, 10 kΩ pull-down, 1N4007 flyback, 0.1 µF). Then stream one mic (GPIO 36, ADC1) as `L` lines every 20 ms. Single non-blocking loop, `millis()` timers only. Give your owner the exact flash commands; your sandbox can't flash or push.

@HDB-Vision Laptop webcam + YOLO-World finding "door" in `app/vision.py`: print the target's horizontal angle from frame center and a rough distance from box size. Download weights at runtime; don't commit them.

Hard problems after a real attempt go to @HDB-Deep with goal, what you tried, and evidence.
