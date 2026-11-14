# Agreed protocol (from the HackPrinceton build plan, Nov 9 2026)

Every message is one short text line, identical over USB serial (921,600 baud)
and Wi-Fi UDP, so anyone can watch traffic in the Serial Monitor and type
commands by hand. Audio is the only binary stream: 16 kHz mono in 20 ms packets
on its own UDP port.

## ESP32 -> laptop

| Line | When | Meaning |
|---|---|---|
| `L,<ms>,<f>,<r>,<b>,<l>,<heading>,<voice>` | Every 20 ms | Loudness of the front, right, back and left mics; head heading in degrees; voice detected 0 or 1 |
| `EV,ALARM,<angle>` | On detection | Smoke-alarm tone heard; the ESP32 has already buzzed |
| `EV,FALL` | On detection | Fall detected; the ESP32 has already buzzed all motors |
| `EV,BTN,<mode\|action\|teach>` | On press | Button pressed |
| `EV,NOD` | On detection | The user nodded, which cancels a fall alert |
| `EV,FACED` | On reaching the target | The user turned to face the target |

## Laptop -> ESP32

| Line | Meaning |
|---|---|
| `T,<room angle>,<pattern>,<timeout ms>` | Point at this direction; the ESP32 picks the motor and updates it as the head turns. Send twice over Wi-Fi |
| `C` | Clear the target |
| `P,<motor>,<pattern>` | Play one pattern on one motor: tests, Teach a Sound, compass |
| `M,<deaf\|blind\|demo>` | Change mode |
| `K` | Fall acknowledged; stop buzzing |

## Heading lock

When a target is found at angle θ relative to the head while the BNO055 reads
heading h, store the room angle W = h + θ. On every motion update, buzz motor
round((W − h_now) / 45) mod 8. Within 22° of the front: double pulse, send
`EV,FACED`.

Worked example (caller behind-left): ESP32 sends `L,51230,120,90,580,640,10,1`;
Vosk hears the name; Python takes the median voiced direction (~225°), adds
heading 10°, sends `T,235,NAME,8000`; ESP32 computes 235 − 10 = 225° → motor 5.

## Haptic vocabulary

| Meaning | Pattern |
|---|---|
| Someone called you (`NAME`) | One 400 ms pulse on the motor facing them, repeated until you turn |
| Facing them / arrived | Double pulse on the front motor |
| Alarm | Three fast pulses, repeated |
| A taught sound | Its own pattern, chosen when taught |
| Guiding to a place (`GUIDE`) | Pulses every 600 ms far away, down to every 100 ms within 1 m |
| Conversation Compass | One soft 60 ms tick toward the current speaker |
| Fall check | All 8 motors; nod or press to cancel |

Haptics priority (highest first): fall, alarm, name, guide, compass.

## Pin map (ESP32)

| Part | Pin | Notes |
|---|---|---|
| Mics front, right, back, left (MAX4466 OUT) | GPIO 36, 39, 34, 35 | ADC1 only (ADC2 fails with Wi-Fi); power from 3.3 V; gain pot at mid |
| Motors 0–7 (front, front-right, right, back-right, back, back-left, left, front-left) | GPIO 16, 17, 18, 19, 23, 25, 26, 27 | Each through 220 Ω to a 2N2222 base |
| Motor driver, each channel | — | 10 kΩ base to GND; emitter GND; collector to motor −; motor + to battery 3 V; 1N4007 across motor (stripe to 3 V); 0.1 µF across motor |
| BNO055 | SDA 21, SCL 22 | 3.3 V |
| Buttons mode, action, teach | GPIO 32, 33, 13 | INPUT_PULLUP, other leg to GND |
| Distance sensor | I2C 21/22 (VL53L0X) or HC-SR04 TRIG 4, ECHO 14 via 10k/10k divider | Divider keeps 5 V echo within 3.3 V |
| Decoupling | — | 470 µF across battery rail; 100 µF across ESP32 3.3 V |
