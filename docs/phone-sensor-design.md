# Phone as the sensor (Phase 2, design only, not built)

Goal: if the band's camera or mics fail, the wearer's phone supplies video and
audio to the same laptop pipeline (find an object, name alerts, crash warning).
Phase 1 (on main) already makes the phone a controller and a backup buzzer over
`python -m app.main --lan`.

## Constraint: https

`getUserMedia` (camera/mic) only works in a secure context: `https://` or
`localhost`. A phone opening `http://10.x.x.x:8765` gets no camera. Options:

| Option | How | Cost |
|---|---|---|
| Tunnel (recommended) | `cloudflared tunnel --url http://localhost:8765` (or ngrok) gives an https URL | Needs internet at the venue; adds 50-150 ms; a new tool on the laptop |
| Self-signed cert | aiohttp `ssl_context` with a cert generated at startup | Phone shows a scary warning each time; iOS must trust the cert manually |
| mkcert local CA | install the CA profile on the phone once | Most robust offline, but a setup step per phone |

Pick the tunnel for the demo, mkcert if the venue has no internet.

## Data path

1. The page asks for `getUserMedia({video: {facingMode: "environment", width: 640},
   audio: {sampleRate: 16000, channelCount: 1}})`.
2. Video: draw to a canvas at ~8 fps, `canvas.toBlob("image/jpeg", 0.7)`, send as a
   binary WebSocket message on a new endpoint `/ws/phone-cam` (one JPEG per
   message, ~30-50 KB). Laptop: decode with OpenCV and feed `Vision` through a
   frame source that replaces `cv2.VideoCapture` (Vision already accepts a file
   path; add a queue-backed source).
3. Audio: an `AudioWorklet` posts 100 ms Float32 blocks; send as binary on
   `/ws/phone-mic` with the phone's `performance.now()` timestamp. Laptop: feed
   `NameListener.feed(samples, t_first)` after mapping the phone clock to the
   laptop's monotonic clock (one ping/pong offset estimate per second, the same
   idea as the ESP32 clock alignment).
4. Direction: the phone has one mic, so name direction falls back to the generic
   cue; find-an-object and the crash warning still give sides from the camera.

## Protocol changes this needs (HDB-Protocol sign-off first)

- Two new WebSocket endpoints (binary). The existing `/ws` JSON protocol is unchanged.
- A `{"cmd":"source","camera":"band|phone","mic":"band|phone"}` command so the
  dashboard can switch inputs, and status fields reporting which source is live.

## Risks

- Phone screen must stay on (Wake Lock API on Android; iOS needs the page in front).
- Bandwidth on venue Wi-Fi: 8 fps x 40 KB = ~2.5 Mbit/s, fine on a hotspot.
- Latency: tunnel + JPEG encode adds ~150-250 ms to find cues; acceptable for find, borderline for the crash warning.
