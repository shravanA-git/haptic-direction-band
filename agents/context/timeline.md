# Schedule, priorities and demo

Clock: H0 = 8 PM Friday Nov 13. Hard submission deadline H36 = 8 AM Sunday
Nov 15 (submit by 7:30 AM). Judging starts 9:30 AM Sunday. Features, build
order and demo follow the Nov 13 team handoff; it replaces the Nov 9 plan
where they differ. Each phase ends in a gate; if it fails, take the fallback
and move on.

## Feature priorities

1. **Find an object** (first live demo). Hold Action, say "find the banana",
   release. The webcam tracks the target; left/right buzzes guide the head
   turn, two front pulses when centered, a distinct cue when lost. Typed
   target as a fallback.
2. **Hear your name** (second live demo). A teammate says the configured name
   once; the band estimates direction from the 4 mics and buzzes that side;
   unclear direction gets a generic attention buzz.
3. **Heading lock** (after basic name alerts; needs a stable IMU). The buzz
   follows the saved room direction as the wearer turns; two front pulses on
   alignment.
4. **Teach a Sound** (planned): one taught sound with its own pattern; add
   direction once recognition is reliable in noise.
5. **Recall and caregiver texting** (backlog, separate from the core demo):
   alert log, a chatbot with two tools first (look up recent alerts, change an
   alert setting); later Photon texting and LangChain Deep Agents.
6. Optional experiments: forward range sensor, nod acknowledgment, simulated
   fall (impact plus stillness with a cancel window), phone companion.

## Phases and gates

1. **H0–H3 (Fri 8–11 PM), bring-up.** Hardware: motor 0 on GPIO 18 wired and
   checked. Firmware: finite `PULSE` and `STOP` bench-tested at 115200. Vision
   and speech: webcam + detector with a typed target. Software: serial link,
   motor ring dashboard, replay of the two built-in traces.
   **Gate H3:** the laptop pulses motor 0 on command and `STOP` works.
2. **H3–H9 (Fri 11 PM–Sat 5 AM), full band and inputs.** Hardware: label
   channels, test cardinal positions, expand to 8 motors. Firmware: stream
   four `MIC` levels, debounce Action (`BTN ACTION DOWN/UP`). Vision: left /
   right / centered / lost cues from the bounding box. Software: validate the
   physical motor mapping, direction estimator on replays.
   **Gate H9:** all 8 motors map correctly from the dashboard; a 4-mic stream
   is visible. **Fallback:** 4 cardinal motors.
3. **H9–H15 (Sat 5–11 AM), both demos end to end.** Vision and speech:
   hold-to-talk target selection; name recognition with word timestamps.
   Software: clock alignment and live name alerts from speech timestamps.
   **Gate H15:** "find the banana" works end to end (left/right, centered,
   lost); a called name gives the right side or a generic alert 4/5 times.
   **Fallback:** coarse positions only (front/back/left/right).
4. **H15–H23 (Sat 11 AM–7 PM), sleep in shifts.** Heading lock if the IMU is
   stable; one taught sound; record quiet and noisy trials.
   **Gate H23:** both demos survive 5 runs in a noisy room.
5. **H23–H27 (Sat 7–11 PM), integration and polish.** Dashboard polish,
   intensity buttons, Recall backlog only if everything above is solid.
6. **H27–H30 (Sat 11 PM–Sun 2 AM), freeze and measure.** No new features.
   Record correct directions, wrong directions, generic alerts, false alerts
   and latency for the pitch.
7. **H30–H33 (Sun 2–5 AM), video and Devpost.** Short video of a working run;
   Devpost with everyone's Discord username. Presenter sleeps.
8. **H33–H36 (Sun 5–8 AM), rehearse and submit.** Rehearse pause, `STOP` and
   unplug behavior; submit by 7:30 AM.

**Cut order (first to go):** Recall/caregiver texting, Teach a Sound, heading
lock, range sensor, 8 → 4 motors. **Never cut:** find an object, hear your
name, `STOP`/safety behavior.

## Live demo (2 minutes, stationary and supervised)

1. **Object first.** Say "find the banana". Move the target left and right;
   show the motor ring; two front pulses on alignment. Remove it to show the
   lost state. Keep a typed target and a short video ready.
2. **Attention second.** A teammate calls the configured name from a known
   position. Show the cue, then heading lock if stable. If direction only
   works for coarse positions, use those positions.
3. Present it as a prototype for orientation cues.

## Tracks

Main: Best Healthcare Hack. Optional/special tracks from the earlier plan
(Hardware, Physical AI, Amazon, LangChain deepagents, Photon) depend on the
backlog items; confirm sponsor criteria with organizers before pitching them.
