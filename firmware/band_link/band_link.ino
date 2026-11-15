// Band <-> laptop link firmware: streams the 4 hat mics continuously, reports the Action button, drives motors.
// Matches app/link.py exactly (115200, one text line per message):
//   ESP32 -> laptop  MIC <ms> <front> <right> <back> <left>   every 25 ms, RAW levels (max-min of the 12-bit ADC)
//                    BTN ACTION DOWN | BTN ACTION UP          hold-to-talk: touch pad (HIGH while held) OR button (GPIO13 to GND),
//                                                             debounced; forced UP after 12 s so a stuck pad can never lock recording on
//   laptop -> ESP32  PULSE <motor 0-7> <ms 1-1000>            finite pulse, at most 2 motors at once
//                    STOP                                     all motors off now
//
// The levels are RAW on purpose: the laptop divides by the gains in app/calibration.local.json.
// Do NOT also divide here (your test sketch's GAIN_RATIO), or the correction is applied twice.
// Needs no 't' trigger and prints nothing else after boot. Keep Serial Monitor CLOSED while the app runs.
// Test the first time with the motor battery pack DISCONNECTED.

#include <Arduino.h>

constexpr int MIC_COUNT = 4;
constexpr int MIC_PINS[MIC_COUNT] = {32, 35, 34, 33};      // front, right, back, left (tested hat wiring)
constexpr uint32_t WINDOW_US = 25000;

constexpr int BUTTON_PIN = 13;                              // Action button: to GND, internal pull-up (optional)
constexpr int TOUCH_PINS[2] = {27, 4};                      // capacitive touch module IO, HIGH while touched: wired to 27 (as tested) or 4; either works
constexpr uint32_t DEBOUNCE_MS = 30;                        // touch/press must hold this long to report DOWN
constexpr uint32_t RELEASE_DEBOUNCE_MS = 150;               // release must hold this long to report UP: merges 0.1 s finger/pad flicker into one hold
constexpr uint32_t ACTION_MAX_MS = 12000;                   // safety: report UP if held longer; ignore the pad until it is released

constexpr int MOTOR_COUNT = 8;
// Laptop motor IDs: 0 front, 2 right, 4 back, 6 left (odd IDs = diagonals, unused on the 4-motor band).
// Built band (Directional_Haptics_Integration_Handoff): FRONT on GPIO26, RIGHT 23, BACK 18, LEFT 16.
// So ID 0 -> GPIO26 and ID 4 -> GPIO18; protocol.md's old map had them the other way round (front/back would be swapped).
constexpr int MOTOR_PINS[MOTOR_COUNT] = {26, 19, 23, 25, 18, 21, 16, 17};   // motor 5 moved 27 -> 21 so GPIO27 can be the touch input
constexpr uint32_t MAX_PULSE_MS = 1000;                     // matches app/link.py MAX_PULSE_MS
constexpr int MAX_AT_ONCE = 2;                              // power-rail safety

bool motorOn[MOTOR_COUNT] = {};
uint32_t motorOffAt[MOTOR_COUNT] = {};

char lineBuf[40];
int lineLen = 0;

uint32_t windowStart = 0;
int lowV[MIC_COUNT], highV[MIC_COUNT];

bool buttonStable = false;       // true = pressed (touch pad OR button)
bool buttonRaw = false;
bool actionLatched = false;      // forced UP after ACTION_MAX_MS: wait for a real release
uint32_t buttonChangedAt = 0;
uint32_t actionDownAt = 0;

void motorsAllOff() {
  for (int m = 0; m < MOTOR_COUNT; ++m) {
    digitalWrite(MOTOR_PINS[m], LOW);
    motorOn[m] = false;
  }
}

int motorsActive() {
  int n = 0;
  for (int m = 0; m < MOTOR_COUNT; ++m) n += motorOn[m] ? 1 : 0;
  return n;
}

void handleLine(const char* line) {
  if (strcmp(line, "STOP") == 0) {
    motorsAllOff();
    return;
  }
  int motor = -1;
  unsigned long ms = 0;
  if (sscanf(line, "PULSE %d %lu", &motor, &ms) != 2) return;
  if (motor < 0 || motor >= MOTOR_COUNT || ms < 1 || ms > MAX_PULSE_MS) return;
  if (!motorOn[motor] && motorsActive() >= MAX_AT_ONCE) return;   // ignore a third motor
  motorOn[motor] = true;
  motorOffAt[motor] = millis() + ms;
  digitalWrite(MOTOR_PINS[motor], HIGH);
}

void readCommands() {
  for (int budget = 64; budget > 0 && Serial.available(); --budget) {
    const int c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      if (lineLen > 0) handleLine(lineBuf);
      lineLen = 0;
    } else if (lineLen < (int)sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = static_cast<char>(c);
    } else {
      lineLen = 0;   // overlong line: drop it
    }
  }
}

void updateMotors() {
  const uint32_t now = millis();
  for (int m = 0; m < MOTOR_COUNT; ++m) {
    if (motorOn[m] && static_cast<int32_t>(now - motorOffAt[m]) >= 0) {
      digitalWrite(MOTOR_PINS[m], LOW);
      motorOn[m] = false;
    }
  }
}

void resetWindow() {
  for (int i = 0; i < MIC_COUNT; ++i) { lowV[i] = 4095; highV[i] = 0; }
  windowStart = micros();
}

void sampleMics() {
  for (int i = 0; i < MIC_COUNT; ++i) {
    const int v = analogRead(MIC_PINS[i]);
    if (v < lowV[i]) lowV[i] = v;
    if (v > highV[i]) highV[i] = v;
  }
  if (micros() - windowStart >= WINDOW_US) {
    Serial.printf("MIC %lu %d %d %d %d\n", (unsigned long)millis(),
                  highV[0] - lowV[0], highV[1] - lowV[1], highV[2] - lowV[2], highV[3] - lowV[3]);
    resetWindow();
  }
}

bool actionPressed() {
  return digitalRead(BUTTON_PIN) == LOW || digitalRead(TOUCH_PINS[0]) == HIGH || digitalRead(TOUCH_PINS[1]) == HIGH;
}

void updateButton() {
  bool raw = actionPressed();
  const uint32_t now = millis();
  if (actionLatched) {                                   // a stuck/overlong hold: ignore until released
    if (!raw) { actionLatched = false; buttonRaw = false; buttonChangedAt = now; }
    return;
  }
  if (raw != buttonRaw) { buttonRaw = raw; buttonChangedAt = now; }
  if (raw != buttonStable && now - buttonChangedAt >= (raw ? DEBOUNCE_MS : RELEASE_DEBOUNCE_MS)) {
    buttonStable = raw;
    if (raw) actionDownAt = now;
    Serial.println(buttonStable ? "BTN ACTION DOWN" : "BTN ACTION UP");
  }
  if (buttonStable && now - actionDownAt >= ACTION_MAX_MS) {
    buttonStable = false;                                // force the release the laptop is waiting for
    actionLatched = true;
    Serial.println("BTN ACTION UP");
  }
}

void setup() {
  for (int m = 0; m < MOTOR_COUNT; ++m) {        // motors off first, before anything else
    digitalWrite(MOTOR_PINS[m], LOW);
    pinMode(MOTOR_PINS[m], OUTPUT);
  }
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  for (int i = 0; i < 2; ++i) pinMode(TOUCH_PINS[i], INPUT_PULLDOWN);   // floating or unplugged reads LOW: never a phantom press
  analogReadResolution(12);
  Serial.begin(115200);
  resetWindow();
}

void loop() {
  readCommands();
  updateMotors();
  sampleMics();
  updateButton();
}
