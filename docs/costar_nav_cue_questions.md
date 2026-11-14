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

## Route playback timing (route mode, no live tour position)

Current behaviour (`app/nav.py`, all PROVISIONAL):

| Event | Now |
|---|---|
| Leg length | max(2.5 s, metres / speed); speed slow 0.6 / normal 0.8 / demo 1.5 m/s |
| Leg start (first leg) | 200 ms on the leg's motor, leg spoken ("ahead 12 metres") |
| 1.5 s before a leg ends | spoken only: "in 1 metre, turn right" (no buzz) |
| Leg end = decision point | 200 ms on the NEW leg's motor (the turn cue), new leg spoken |
| During a leg > 4 s | `PULSE 0 80` every 4 s = "still straight" (stops 1.5 s before the turn) |
| Last leg ends | two front pulses + "You've arrived at the kitchen" |
| Name/sound alert, Action held | the route clock freezes; nothing fires or is spoken until it ends |

Questions:

9. **Keep-going pulse.** Is `PULSE 0 80` acceptable as a cue? It isn't in the
   cue table. It must not be confused with the generic attention cue (300 ms
   front) or with "centered" (two front pulses). Alternatives: no keep-going
   pulse at all, or a very light one on both front diagonals (1 + 7).
10. **Pre-turn warning.** Spoken only today. Should it also buzz, e.g. a weak
    pulse on the side of the coming turn, so a wearer who can't hear the laptop
    still gets warned?
11. **Turn timing.** The turn pulse fires when the leg ends at the assumed
    walking speed. Is 0.8 m/s right for a blind or low-vision walker indoors,
    and should the turn cue come slightly early (e.g. 0.5 s) to cover reaction
    time?
12. **Big turns.** Legs that start behind (motors 3–5) are spoken as "turn
    around". Should they use a different pattern from a plain direction pulse?
