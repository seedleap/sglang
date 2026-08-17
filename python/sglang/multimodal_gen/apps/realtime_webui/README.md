# SGLang Diffusion Realtime WebUI

Standalone browser demo for `/v1/realtime_video/generate`.

Open `index.html` directly in a browser, point it at an SGLang Diffusion server,
and generate. The app sends msgpack init / event messages and renders lossless
raw RGB frame batches on a canvas.

The first version is intentionally static: no npm install, no build step, and no
server-side dependencies. Presets are UI-side templates for prompt, LingBot
example images, album artwork references, and session parameters. The default
preset preloads a reference image so the demo can be tested without a file
upload.

By default, `Continuous session` is enabled for long-running camera control.
Keyboard and pointer controls send state transitions instead of scripted preset
actions. The telemetry `Chunk wait` measures request-to-chunk arrival time, not
client-side RGB decode time. Continuous playback adapts to the measured chunk
production rate so the canvas does not play a chunk at target FPS and then sit
on the last frame while waiting for the next chunk.

The interface shape follows camera-control-first video playgrounds such as
Reactor LingBot: reference image, scene prompt, enhancement, clip controls,
move/look camera controls, recordings history, and model telemetry.

The dual-model root page is presented as **World Studio**: visitors choose a
world or upload a first frame, provide the initial world description, and
compare Zing with LingBot2 side by side. Model and transport parameters remain
in hidden compatibility controls and use deployment defaults. During a live
world, a separate text/voice composer sends prompt updates to both models, and
the compact WASD/IJKL keycaps retain the existing shared camera-action path.
Only presets with packaged first-frame images are shown in the world list;
metadata-only test cases are deliberately excluded from the visitor UI.

For the isolated live dual-model preview, run `./run_live_dual.sh` and open
`http://127.0.0.1:18083/`. Override `REALTIME_DUAL_GATEWAY_HTTP` and
`REALTIME_DUAL_GATEWAY_WS` when the deployment owner rotates the gateway URL.
The local page connects its video WebSockets directly to that gateway so frame
traffic does not take an additional Python relay hop.
The World Studio presentation defaults both players to low-latency playback,
discarding stale backlog so camera and prompt updates remain responsive.
World drafts may also include optional rules. Up to nine skill instructions and
one probability-based goal are rewritten concurrently against the initial world
description before the realtime sessions connect. Prepared skills appear above
the movement controls and can be sent without another LLM round trip by clicking
or pressing 1-9. The goal is evaluated once after each successful user or skill
prompt, can complete only once per session, and shows its achievement notice five
seconds after the prepared goal prompt is sent.
The showcase session lifetime is provided by the hidden
`sessionMaxLifetimeSeconds` runtime setting; the live dual-model launcher uses
90 seconds, and both model workloads must use the same hard lifetime.

## I2V and T2V

Deployments opt in to the mode selector through runtime config:

```json
{
  "generationModes": ["i2v", "t2v"],
  "defaultGenerationMode": "i2v",
  "t2vFrameStep": 4,
  "t2vDefaultNumFrames": 121
}
```

I2V sends `generation_mode: "i2v"` and requires `first_frame`. T2V sends
`generation_mode: "t2v"` and omits `first_frame`. The WebUI defaults T2V to a
continuous session by omitting both `num_frames` and `max_chunks`; it runs until
the user presses Stop. Uncheck Continuous to request a finite output horizon.
For finite MinWM T2V, `num_frames` must equal `1 + N * 4`; the adapter derives
the exact chunk count, so the WebUI does not send `max_chunks`. The
`mode=i2v|t2v` query parameter can select an enabled mode.
