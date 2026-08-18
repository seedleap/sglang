const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const appJs = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");

const replayHtmlBuilder = appJs.slice(
  appJs.indexOf("function buildReplayHtml"),
  appJs.indexOf("function formatReplayMs"),
);

assert.match(
  appJs,
  /function drawRecordingStageFrame\(/,
  "recording should compose the full stage instead of capturing only the viewport canvas",
);
assert.match(
  appJs,
  /capture_scope:\s*"stage"/,
  "recording metadata should describe that the stage was captured",
);
assert.match(
  appJs,
  /configuredNumber\("gameplayRecordingWidth", 1920\)/,
  "stage recording should default to a higher-resolution 1080p canvas",
);
assert.match(
  appJs,
  /recordingCtx\.imageSmoothingQuality = "high"/,
  "stage recording should use high-quality canvas resampling",
);
assert.match(
  appJs,
  /video_bits_per_second: recordingVideoBitrate/,
  "recording metadata should expose the high-quality bitrate target",
);
assert.match(
  appJs,
  /timing:\s*"wall_clock"/,
  "gameplay recordings should preserve real user timing independently of model FPS",
);
assert.match(
  appJs,
  /function startRecordingFramePump\(\)/,
  "recording should use a lightweight frame pump",
);
assert.match(
  appJs,
  /await track\.encoder\.flush\(\);\s*if \(!track\.samples\.length\)/,
  "short recordings should flush the encoder before checking for output samples",
);
assert.match(
  appJs,
  /RECORDING_MAX_ENCODER_QUEUE_SIZE/,
  "recording backpressure should drop capture frames instead of stalling realtime play",
);
assert.doesNotMatch(
  appJs,
  /recordDecodedFrameBatch\(decodedFrames\)/,
  "decoded model batches must not drive recording time",
);
assert.match(
  appJs,
  /function drawRecordingPromptComposer\(\)/,
  "downloaded gameplay should include the prompt composer overlay",
);
assert.match(
  appJs,
  /function drawRecordingPromptComposer\(\)[\s\S]{0,180}?const x = 450;[\s\S]{0,100}?const width = 700;/,
  "the recorded prompt composer should stay compact and leave the world visible",
);
assert.match(
  appJs,
  /runtime_prompt_input/,
  "recording should retain the user's real prompt input timeline",
);
assert.match(
  appJs,
  /runtime_prompt_submitted/,
  "recording should distinguish submit time from model send time",
);
assert.match(
  appJs,
  /runtime_prompt_sent/,
  "recording should mark the exact model send event",
);
assert.match(
  appJs,
  /stopWorldExperienceTiming\(\{ recordingReason: "session_timeout" \}\)/,
  "session timeout should finalize the downloadable gameplay recording",
);
assert.match(
  appJs,
  /setRecordingDownloads\(outputs\)/,
  "both finalized gameplay videos should be exposed through one download control",
);
assert.doesNotMatch(
  appJs,
  /getDisplayMedia|display-media-webm|key: "screen"/,
  "recording should not use browser screen capture or single-screen artifacts",
);
assert.match(
  appJs,
  /key: "comparison"[\s\S]{0,300}?key: "zing"/,
  "recording should encode comparison and Zing-only tracks",
);
assert.match(
  appJs,
  /fileName: `\$\{baseFileName\}-\$\{track\.key\}\.\$\{extension\}`/,
  "the two synchronized tracks should receive distinct downloadable file names",
);
assert.match(
  appJs,
  /function downloadGameplayRecordings[\s\S]*?for \(const item of recordingDownloads\)[\s\S]*?link\.click\(\)/,
  "one download action should synchronously trigger both video downloads",
);
assert.match(
  appJs,
  /function markWorldExperienceReady\(modelKey\)[\s\S]*?source: "first_visible_frame"/,
  "recording must be gated by the first visible world frame",
);
assert.match(
  appJs,
  /const RECORDING_READY_TOAST_MS = 5000;/,
  "the recording-ready toast should stay visible for five seconds",
);
assert.match(
  appJs,
  /\["session_timeout", "session_closed", "primary_disconnected"\][\s\S]{0,220}?showRecordingReadyToast\(\)/,
  "world-ending recording paths should show the download reminder toast",
);
assert.match(
  replayHtmlBuilder,
  /class="replay-stage"/,
  "exported replay index should render a stage-style video area",
);
assert.doesNotMatch(
  replayHtmlBuilder,
  /data-replay-action=/,
  "exported replay should not render a second row of camera controls below the recorded stage",
);
assert.doesNotMatch(
  replayHtmlBuilder,
  /replay-stage-controls/,
  "exported replay should rely on the recorded stage controls instead of duplicating them",
);
assert.match(
  replayHtmlBuilder,
  /function syncReplayControls/,
  "replay index should update button highlights as the video plays",
);
assert.match(
  replayHtmlBuilder,
  /camera_actions_sent/,
  "replay index should use camera action events to reconstruct active input state",
);
assert.match(
  replayHtmlBuilder,
  /id="replayInspector"/,
  "replay index should expose a hover inspector near the recorded video",
);
assert.match(
  replayHtmlBuilder,
  /\.replay-inspector\s*\{\s*position:\s*fixed;/,
  "replay inspector should float near the cursor instead of being pinned over the video stage",
);
assert.match(
  replayHtmlBuilder,
  /function positionReplayInspector\(event\)/,
  "replay inspector should be positioned from the pointer coordinates",
);
assert.match(
  replayHtmlBuilder,
  /event\.clientX\s*\+\s*REPLAY_INSPECTOR_OFFSET_PX/,
  "replay inspector should stay to the lower-right of the cursor",
);
assert.match(
  replayHtmlBuilder,
  /event\.clientY\s*\+\s*REPLAY_INSPECTOR_OFFSET_PX/,
  "replay inspector should stay below the cursor",
);
assert.doesNotMatch(
  replayHtmlBuilder,
  /event\.clientX\s*-\s*width/,
  "replay inspector should not flip to the left side of the cursor",
);
assert.doesNotMatch(
  replayHtmlBuilder,
  /event\.clientY\s*-\s*height/,
  "replay inspector should not flip above the cursor",
);
assert.match(
  replayHtmlBuilder,
  /positionReplayInspector\(event\);\s*inspectReplayAt\(replayClientMsFromPointer\(event\)\)/,
  "replay inspector should move before refreshing hover context on mouse move",
);
assert.match(
  replayHtmlBuilder,
  /function replayClientMsFromPointer/,
  "replay index should map pointer position over the video to recording time",
);
assert.match(
  replayHtmlBuilder,
  /function inspectReplayAt/,
  "replay index should show prompt, image, and event context for the hovered time",
);
assert.match(
  replayHtmlBuilder,
  /function sglangActionsForEventId/,
  "replay index should derive the SGLang-sampled actions from server chunk event ids",
);
assert.match(
  replayHtmlBuilder,
  /Prompt at cursor/,
  "replay inspector should include the prompt active at the hovered point",
);
assert.match(
  replayHtmlBuilder,
  /Reference image/,
  "replay inspector should include reference image metadata",
);
assert.match(
  replayHtmlBuilder,
  /function replayReferenceImageSrc/,
  "replay index should resolve reference images from either embedded data or URLs",
);
assert.match(
  replayHtmlBuilder,
  /referenceImage\?\.data_url\s*\|\|\s*referenceImage\?\.url/,
  "preset reference URLs should render instead of a black placeholder when data_url is omitted",
);
assert.match(
  replayHtmlBuilder,
  /inspectorImage\.src = referenceSrc/,
  "replay inspector should use the same resolved reference image source as the sidebar",
);

console.log("recording replay export ok");
