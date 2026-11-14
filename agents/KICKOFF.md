# H0 kickoff message

Post this in the shared Band room once the agents are in it.

---

H0: hacking has started. Spec is the Nov 13 handoff: `agents/context/protocol.md` (pins, 115200 serial: `MIC`/`BTN` in, `PULSE`/`STOP` out, one-channel wiring) and `agents/context/timeline.md`. Goal for H0–H3, bring-up. **Gate at H3: the laptop pulses motor 0 on command and `STOP` works.** Work autonomously in your lane on your own branch, release to `main` when something works, and post one `[done]` line per finished item.

@HDB-Gatekeeper Start the clock. Create one Band task per H0–H3 item below, assigned to its owner, plus a task for the H3 gate check. Post the gate in one line and check in at H2 with what's at risk.

@HDB-Protocol Post a one-screen `[answer]` summary of the serial contract and the haptic cue timings (directional 200 ms, generic attention 300 ms on the front motor, two front pulses when centered, lost cue, 150 ms test pulse), so everyone builds against identical strings. Flag anything the handoff leaves undefined (intensity buttons, IMU lines, the exact lost-cue and two-pulse timings) as a `[proposal]` for the humans to settle.

@HDB-Software On branch `software`: `app/link.py` for USB serial at 115200 (one owner of the port, parses `MIC` and `BTN`, sends `PULSE <motor> <ms>` and `STOP`, plus a simulated port that replays recorded frames); a motor-ring dashboard built from scratch (a local web page served by the Python app over a WebSocket: 8-motor ring, live mic levels, per-motor test pulse, STOP); trace recording and replay with two hand-made fixtures (back-left → motor 5, unclear → generic attention); tests against the simulator. Nothing exists yet: start from an empty repo.

@HDB-Firmware On branch `firmware`: wire motor 0 on GPIO 18 exactly per the one-channel diagram (check C/B/E on the 2N2222 datasheet, diode stripe to motor +, 100 nF across the motor, 100 µF on the rail, common ground). Firmware at 115200 that boots with all motors off, handles `PULSE <motor> <ms>` with a finite pulse and `STOP`, and stops on a stale command or lost link. Give your owner the exact flash commands and the bench test: short pulse, `STOP`, transistor stays cool, then repeat starts. Post sample serial lines when it works.

@HDB-Vision On branch `vision`: `app/vision.py` with the front webcam, OpenCV and an object detector that finds the demo target (a banana) from a typed target name, tracks it across frames and reports left / right / centered (with a dead band) / lost. Expose that as a small Python interface HDB-Software can call. Download weights at runtime; don't commit them.

Hard problems after a real attempt go to @HDB-Deep with goal, what you tried, and evidence.
