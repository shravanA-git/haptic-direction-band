# Hardware contract: pins, serial protocol, wiring

Source of truth: the team handoff of Nov 13, 2026. It replaces the Nov 9 build
plan's protocol and pin map. Agree in the room before changing pins or message
formats (see HDB-Protocol).

## System split

- **Laptop:** video (USB webcam), speech (hold-to-talk, name recognition), the
  direction estimator, haptic decisions, dashboard.
- **ESP32 (HiLetgo ESP-WROOM-32):** reads 3 buttons and 4 mic levels, switches
  motors on laptop command over USB. Two AAs power the motors.

## Serial protocol: USB, 115200 baud, one text line per message

| Direction | Line | Meaning |
|---|---|---|
| ESP32 → laptop | `MIC <ms> <front> <right> <back> <left>` | Monotonic ESP32 milliseconds and four nonnegative, calibrated loudness values. One frame every 20–50 ms |
| ESP32 → laptop | `BTN ACTION DOWN` / `BTN ACTION UP` | Action button pressed / released (debounced) |
| Laptop → ESP32 | `PULSE <motor> <duration_ms>` | Run one motor for a finite time. Example: `PULSE 6 150` = left motor, 150 ms |
| Laptop → ESP32 | `STOP` | All motors off now |

Firmware safety rules: initialize every motor off; every pulse is finite;
stop all motors on `STOP`, on pause, on stale commands, and if the USB link is
lost. Exactly one program owns the serial port at a time: close the Arduino
Serial Monitor before the dashboard or laptop app opens it.

Not yet assigned in the protocol: intensity buttons (14, 4) and any IMU data.
Propose line formats through HDB-Protocol before using them.

## Haptic cues (current)

| Cue | Pattern |
|---|---|
| Directional (object left/right, name direction) | 200 ms pulse on the motor for that direction |
| Generic attention (direction unclear, weak or stale evidence) | 300 ms on the front motor (0) |
| Centered / aligned | Two front pulses |
| Lost target | One distinct cue, then no directional pulses until reacquired |
| Test pulse | 150 ms |

Measure mic levels in the quiet gaps between motor pulses.

## Pin map (classic ESP-WROOM-32; confirm against your board's labels)

| Motor | Position | GPIO | Motor | Position | GPIO |
|---|---|---|---|---|---|
| 0 | Front | 18 | 4 | Back | 26 |
| 1 | Front-right | 19 | 5 | Back-left | 27 |
| 2 | Right | 23 | 6 | Left | 16 |
| 3 | Back-right | 25 | 7 | Front-left | 17 |

| Peripheral | Pins and notes |
|---|---|
| 4 × MAX4466 mics | Front 32, right 33, back 34, left 35 (all ADC1). Power from ESP32 3.3 V, common ground |
| 3 buttons | Action 13, intensity down 14, intensity up 4. To GND with internal pull-ups and debounce |
| Optional IMU | I²C SDA 21, SCL 22. GY-521/MPU-6050 pictured; BNO055 was proposed. Confirm breakout and power |
| Optional range | One forward HC-SR04 on 5 V, shared ground. TRIG 4 only if intensity-up moves to the laptop. ECHO to GPIO 39 through a divider: 10 kΩ Echo→input, 15 kΩ input→GND |

## One motor channel (build motor 0 on GPIO 18 first)

```
AA pack + → rocker switch → switched motor + rail
Motor + → switched rail        Motor − → transistor C
GPIO18 → 220 Ω → transistor B
Transistor B → 10 kΩ → GND     Transistor E → GND
Diode stripe → motor +         Diode other end → motor −
104 ceramic (100 nF) across motor + and motor −
100 µF shared: + → switched rail, − (stripe) → GND
AA pack − → common GND ← ESP32 GND
```

Per channel: coin motor (1.5–3 V), 2N2222-family NPN (e.g. onsemi P2N2222A,
TO-92; check the datasheet pinout for your exact marking), 220 Ω base resistor
(red-red-brown), 10 kΩ pull-down (brown-black-orange), 1N4007/1N4001 flyback
(stripe = cathode, to motor +), 100 nF ceramic ("104"). Shared: 2 × AA in
series with a rocker in the positive lead, 100 µF electrolytic (470 µF is a
bench alternative).

Power and safety: tie pack − to ESP32 GND; keep the battery + rail separate
from ESP32 USB and 3.3 V. Before power, check diode and electrolytic polarity,
pack voltage, ground continuity and shorts. First test: short pulse, `STOP`,
confirm the transistor stays cool, then repeat starts, one motor at a time.
Keep motor wiring away from analog mic leads; watch for split breadboard rails.

Mounting: camera above the forehead, four mic openings exposed outside the
band, eight motors against the inside. Label both ends of every lead,
insulate joints, add strain relief. The driver board can stay on the table for
the first demo.

## Name-direction estimator (laptop)

1. The speech recognizer gives the name's start and end (e.g. 2.10–2.55 s →
   450 ms of band mic frames).
2. Align clocks: buffer ~5 s of `MIC` frames; timestamp audio capture and
   serial receipt on the laptop's monotonic clock and estimate the ESP32
   offset. Check alignment with a sharp test sound.
3. Remove DC offset, match gains across channels. Per short window, compute
   (right − left) and (front − back) normalized by total level; combine usable
   vectors across the phrase; pick the nearest motor direction. Weak,
   conflicting or stale evidence → generic attention pulse.

## Replay format (HTML dashboard)

```json
{"frames":[{"ms":0,"levels":[20,18,90,101]},
           {"ms":50,"levels":[19,20,91,100]}],
 "alerts":[{"startMs":0,"endMs":100}]}
```

Levels are front/right/back/left. The back-left sample should light motor 5;
the unclear sample should give the generic attention cue. The HTML dashboard's
integration point is `headband.alert(startMs, endMs)`. The optional Python
dashboard replays JSONL rows with `t_ms` and `levels`.
