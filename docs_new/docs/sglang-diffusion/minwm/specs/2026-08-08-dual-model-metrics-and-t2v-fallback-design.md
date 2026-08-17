# Dual-model metrics and T2V fallback

## Goal

Keep the existing shared Realtime Studio controls while making stream state truthful for each model. I2V continues to compare MinWM and LingBot2; T2V runs MinWM alone because the deployed LingBot2 backend requires a reference image.

## Behavior

- Shared header: connection state, recording, preview scale, fullscreen, output size, and the shared camera action.
- Per-model footer: chunk, source/render rate, buffer/queue, frames/bytes, decode time, and display lag.
- I2V: connect both backends and broadcast identical prompt/action events.
- T2V: connect MinWM only. Clear LingBot2's previous frame and show an unavailable state; LingBot2 failure must not abort MinWM.
- Size starts at `1280x704`. Mode changes, server model hydration, and preset selection preserve the current user-entered size.
- Remove the generic capability summary row while retaining LingBot presets and History.

## Deployment

The change is confined to the WebUI/controller served by the Gateway. Build a new Gateway image and roll only the Gateway deployment; do not restart GPU model workers.
