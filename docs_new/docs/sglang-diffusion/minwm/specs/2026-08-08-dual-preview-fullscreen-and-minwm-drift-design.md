# Dual Preview Fullscreen and MinWM Drift Design

## Scope

This change adds a fullscreen comparison mode to the existing dual-model realtime
WebUI and corrects the MinWM request defaults to match the deployed checkpoint.
It does not add image filters or alter model outputs to hide long-horizon quality
degradation.

## Fullscreen Interaction

- Add one icon-only fullscreen button to the realtime stage top bar.
- Use the browser Fullscreen API on the complete realtime stage, not only the
  canvas grid. Fullscreen therefore keeps the MinWM and LingBot2 labels, both
  video canvases, shared camera controls, timeline status, and telemetry visible.
- Update the button label and icon when `fullscreenchange` fires so browser-driven
  exits, including Escape, are reflected correctly.
- Keep a 1:1 horizontal comparison on normal desktop fullscreen viewports. Stack
  the players vertically when the fullscreen viewport is too narrow for readable
  side-by-side playback.
- If the Fullscreen API is unavailable or rejects the request, leave the current
  layout unchanged and add a concise history entry instead of breaking playback.

## MinWM Request Defaults

The deployed MinWM artifact declares `sink_size=8` and
`sliding_window_num_frames=32`, and the server starts with the same values. The
dual WebUI currently falls back to the HTML defaults `9/18` and sends them on
every init request, overriding the server configuration. The deployment UI config
will explicitly set `sinkSize=8` and `windowFrames=32`.

LingBot2 keeps its backend-specific request transformation. The shared controls
remain visible because they are part of the existing WebUI contract.

## MinWM Fog Diagnosis

The following observations define the current root-cause boundary:

1. MinWM is clear in the first chunk and degrades during the same session while
   LingBot2 remains clear. A static CSS overlay, canvas color conversion, or WebP
   quality setting cannot explain that progression.
2. Running with the checkpoint-native `8/32` values does not remove long-horizon
   green/cyan drift, so the UI override is a correctness bug but not the primary
   visual cause.
3. Removing `mist` and `humid daylight colors` from the prompt delays and reduces
   early haze, but long sessions still develop color drift and block artifacts.
4. The production `taew2_2.pth` decoder was exercised for 200 chunks with a fixed
   latent sequence. RGB means and variance stabilized after startup and remained
   unchanged, which rules out autonomous TAEHV state drift for a stable input.

The remaining primary source is MinWM denoiser latent drift during open-ended
autoregressive generation. The durable fix belongs in checkpoint training,
long-horizon alignment, or a separately designed re-anchoring strategy. This UI
change will not apply post-processing that masks the problem.

## Tests

- Contract test: the stage contains an accessible fullscreen toggle and retains
  exactly two model players plus shared controls.
- Unit/DOM test: entering, exiting, Escape-driven exit, unsupported API, and
  rejected fullscreen requests keep button state and history consistent.
- Deployment contract: the dual WebUI runtime config contains MinWM `8/32`
  defaults.
- Browser verification: enter fullscreen, verify both canvases and shared controls
  are visible, exit with Escape, start a dual I2V session, and confirm both players
  continue rendering.

## Rollout

Build and publish the existing application image, update the dual deployment image
reference and UI configuration, roll only the stateless gateway pods, and verify
the public endpoint. GPU denoiser, LingBot2, and VAE workers remain running and
are not restarted for this frontend/configuration-only rollout.
