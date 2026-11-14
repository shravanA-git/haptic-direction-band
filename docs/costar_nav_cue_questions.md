# --nav (CoStar "visit before you go"): cue questions for HDB-Protocol

For Sophanit to forward. All values below are PROVISIONAL in `app/nav.py`;
nothing here changes the serial protocol (laptop-sequenced `PULSE` lines only).

Current behaviour:

| Event | Cue now |
|---|---|
| Route accepted | `target_ack` (motors 2 + 6, 120 ms together) |
| Next step | 200 ms on the motor toward the next scan point, relative to where the tour faces |
| Step repeat | Same pulse again on reaching a new scan point, when the needed motor changes (the wearer turned), and every 4 s if nothing changes; never two step cues within 0.8 s |
| Arrived | `centered` (two front pulses), then guidance stops |
| Unknown room / no route | `target_unknown` (three back pulses) |
| Name or sound alert | Guidance stays silent for the alert pattern + 0.4 s, then repeats the step |
| Action held | Guidance paused |

Questions:

1. **Arrival.** Is two front pulses right for "you're there", or should arrival
   differ from find-an-object's "centered" (same pattern today)? Proposal: keep
   it the same, since both mean "it's in front of you / you've reached it".
2. **Step pulse while walking.** Is 200 ms enough to feel while moving, or should
   steps use the 300 ms generic length (or two 150 ms pulses)?
3. **Repeat rate.** Every 4 s with no change, or only on change? Too often may
   numb the wearer; too rarely may leave them unsure.
4. **Turn vs. go.** Should a large turn (motors 3–5, behind) get a different cue
   from "go this way" (e.g. a double pulse on that side), so the wearer turns
   before walking?
5. **Distance.** Should the cue carry distance (e.g. faster repeats when close),
   or stay direction only?
6. **Route accepted / can't.** Reuse hold-to-talk's `ack` and `unknown`, or give
   navigation its own?
7. **Priority.** Name and sound alerts pre-empt guidance (hold, then repeat the
   step). Should find-an-object and obstacle avoidance also pre-empt it, or are
   they mutually exclusive (today: starting one stops the other)?
8. **Intensity.** Nav cues follow the global intensity setting. Should nav have
   its own mute toggle on the dashboard like name/find/sound?
