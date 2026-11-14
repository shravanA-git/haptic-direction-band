# 36-hour timeline (from the HackPrinceton build plan, Nov 9 2026)

Tracks in parallel: Software (Shravan), Firmware, Hardware, Vision and video.
H0 = 8 PM Friday Nov 13. Hard submission deadline H36 = 8 AM Sunday Nov 15
(submit by 7:30 AM). Judging starts 9:30 AM Sunday. Each phase ends in a gate;
if it fails, take the fallback and move on.

1. **H0–H3 (Fri 8–11 PM), bring-up.** Software: repo, frame parser with fake
   data, dashboard showing the motor ring. Firmware: serial frames, buzz a
   motor on command, read one mic. Hardware: 8-channel driver board, one motor
   on batteries, mics marked on the beanie. Vision: webcam + YOLO-World finds
   "door". **Gate:** the laptop buzzes any motor; one mic streams.
2. **H3–H9 (Fri 11 PM–Sat 5 AM), direction.** Software: direction v1,
   calibration script, heading lock, recorded fixtures. Firmware: all 4 mics at
   16 kHz, BNO055 and button frames. Hardware: 8 motors on the band, mics and
   BNO055 on the beanie, strain relief. Vision: target angle and distance, hot
   and cold mapping. **Gate H9:** in a quiet room, claps from 8 directions
   light the right motor or its neighbour 8/8. **Fallback:** 4 directions.
3. **H9–H15 (Sat 5–11 AM), both heroes end to end.** Software: Vosk grammar,
   Who's Talking to Me loop with heading lock. Firmware: patterns stored on the
   ESP32, link stress test. Hardware: can 3 people tell 8 positions apart? Move
   motors if not. Vision: Where Am I? on the band. **Gate H15:** teammate in
   noise-cancelling headphones turns to the right caller 4/5; blindfolded
   teammate reaches the door from 3 m.
4. **H15–H23 (Sat 11 AM–7 PM), sleep in shifts.** Two sleep H15–H19, two sleep
   H19–H23. Software: Teach a Sound, event timeline, then Recall's agent and
   tools. Vision: fall check with nod cancel; phone page for camera and Recall
   chat; caregiver messages via Photon (Twilio fallback). Firmware/hardware:
   Wi-Fi link and audio stream, power bank mount; direction v2 with Software
   if v1 missed H9. **Gate H23:** doorbell taught in under 30 s recognised 4/5,
   and Recall answers "who called me in the last 5 minutes?" correctly and
   points the band at them.
5. **H23–H27 (Sat 7–11 PM), integration.** Button modes, priority queue,
   dashboard polish, switch demo to Wi-Fi, Conversation Compass if time.
6. **H27–H30 (Sat 11 PM–Sun 2 AM), freeze and measure.** No new features. 5
   people, 16 calls each from 8 directions with and without the band; time to
   the door blindfolded; Teach a Sound hit rate. These numbers go in the pitch.
7. **H30–H33 (Sun 2–5 AM), video and Devpost.** Film the optional video in a
   quiet room; write Devpost with everyone's Discord username. Presenter sleeps.
8. **H33–H36 (Sun 5–8 AM), rehearse and submit.** Run the live demo 5 times on
   people who haven't worn the beanie; reset under a minute; submit by 7:30 AM.

**Cut order (first to go first):** route mode, Conversation Compass, Photon
(keep Twilio), sound alerts, distance sensor (use box size), direction v2,
Wi-Fi (demo on USB cable). **Never cut:** Who's Talking to Me, heading lock,
Teach a Sound, Recall.

## Tracks

Main: Best Healthcare Hack. Special/optional: Best Hardware Hack, Best Physical
AI Hack ($500), Amazon Best Practical AI Innovation, LangChain best use of
deepagents, Photon Messaging Agent. Each special track adds a separate judge
visit, so the demo runs 6+ times Sunday. Track swaps for the last 40 s:
Physical AI/Hardware: Wi-Fi off, smoke alarm, beanie still buzzes. Amazon: ask
Recall to stop buzzing for "hey". LangChain: show the deepagent plan and tool
calls. Photon: teammate texts the caregiver number, answer from the timeline.

## Live demo script (2 minutes, judge wears the beanie)

1. 0:00–0:15 problem; type the judge's name on the dashboard.
2. 0:15–0:40 Who's Talking to Me: teammate behind-left says the name.
3. 0:40–1:05 Teach a Sound: doorbell 3 times, then hidden behind the judge.
4. 1:05–1:25 Where Am I?: blindfolded, select "chair", guided 2–3 m.
5. 1:25–1:50 Recall: "who tried to get their attention while blindfolded?"
   then "point me to them".
6. 1:50–2:00 close with accuracy numbers.
