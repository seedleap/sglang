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
  replayHtmlBuilder,
  /class="replay-stage"/,
  "exported replay index should render a stage-style video area",
);
assert.match(
  replayHtmlBuilder,
  /data-replay-action="w"/,
  "replay index should include visible movement controls",
);
assert.match(
  replayHtmlBuilder,
  /data-replay-action="l"/,
  "replay index should include visible look controls",
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

console.log("recording replay export ok");
