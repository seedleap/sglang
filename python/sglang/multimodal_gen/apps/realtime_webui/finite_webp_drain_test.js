const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  FiniteWebpDrainController,
  finitePlaybackIntegrity,
  playbackIsDrained,
  shouldDrainFiniteWebpClose,
} = require("./finite_webp_drain.js");

const idleActivity = () => ({
  pendingHeader: false,
  decodeInProgress: false,
  pendingDecodeBatches: 0,
  decodeQueueLength: 0,
  queuedDecodeFrames: 0,
  queuedDecodeBytes: 0,
  playbackQueueFrames: 0,
  renderActivity: 0,
});

assert.equal(playbackIsDrained(idleActivity()), true);
for (const [field, value] of Object.entries({
  pendingHeader: true,
  decodeInProgress: true,
  pendingDecodeBatches: 1,
  decodeQueueLength: 1,
  queuedDecodeFrames: 1,
  queuedDecodeBytes: 1,
  playbackQueueFrames: 1,
  renderActivity: 1,
})) {
  assert.equal(
    playbackIsDrained({ ...idleActivity(), [field]: value }),
    false,
    `${field} must keep finite playback in draining`,
  );
}

const normalFiniteClose = {
  closeCode: 1000,
  finiteSession: true,
  opened: true,
  expectedClose: false,
  clearQueueOnClose: false,
  serverError: "",
  transportError: false,
  decodeErrors: 0,
  sessionLifetimeClose: false,
  pendingHeader: false,
};
assert.equal(shouldDrainFiniteWebpClose(normalFiniteClose), true);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, closeCode: 1001 }), true);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, closeCode: 1011 }), false);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, finiteSession: false }), false);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, expectedClose: true }), false);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, serverError: "failed" }), false);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, transportError: true }), false);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, decodeErrors: 1 }), false);
assert.equal(shouldDrainFiniteWebpClose({ ...normalFiniteClose, pendingHeader: true }), false);
assert.deepEqual(
  finitePlaybackIntegrity({ decodeErrors: 0, decodedFrames: 121, renderedFrames: 121 }),
  { complete: true, reason: "" },
);
assert.equal(
  finitePlaybackIntegrity({ decodeErrors: 0, decodedFrames: 0, renderedFrames: 0 }).complete,
  false,
  "a zero-output 1000 close must fail closed",
);
assert.equal(
  finitePlaybackIntegrity({ decodeErrors: 1, decodedFrames: 121, renderedFrames: 121 }).complete,
  false,
  "decode errors must not be reported as a completed finite generation",
);
assert.equal(
  finitePlaybackIntegrity({ decodeErrors: 0, decodedFrames: 121, renderedFrames: 120 }).complete,
  false,
  "a missing rendered tail frame must fail closed",
);

// App-level lifecycle simulation: a physical 1000 close happens while one
// asynchronous batch is decoding. Re-entry stays fenced until all decoded
// frames have actually passed through the render loop and the recorder samples
// the final canvas.
const app = {
  connectDisabled: false,
  status: "Live",
  modelState: "live",
  recordingActive: true,
  resetCount: 0,
  renderedFrames: 0,
  notice: "",
  activity: {
    ...idleActivity(),
    decodeInProgress: true,
    pendingDecodeBatches: 1,
    decodeQueueLength: 1,
    queuedDecodeFrames: 3,
    queuedDecodeBytes: 30,
  },
  log: [],
};
const controller = new FiniteWebpDrainController({
  onBegin: () => {
    app.connectDisabled = true;
    app.modelState = "draining";
    app.status = "Finishing playback";
    app.log.push("draining");
  },
  onComplete: () => {
    if (app.recordingActive) app.log.push("capture final recording frame");
    app.recordingActive = false;
    app.log.push("stop timing and recording");
    app.modelState = "closed";
    app.status = "Closed";
    app.log.push("closed");
    app.connectDisabled = false;
    app.log.push("enable reconnect");
    app.notice = "finite playback complete";
  },
});
function connect() {
  if (controller.isActive()) {
    app.connectDisabled = true;
    return false;
  }
  app.resetCount += 1;
  return true;
}
function setStreamingStatus() {
  if (controller.isActive(7)) {
    app.status = "Finishing playback";
    return false;
  }
  app.status = "Live";
  return true;
}
function checkDrain() {
  return controller.completeIfDrained(7, app.activity);
}

controller.begin({ epoch: 7, closeCode: 1000, closeReason: "complete" });
assert.equal(app.connectDisabled, true, "1000 close must not enable reconnect during tail decode");
assert.equal(app.modelState, "draining");
assert.equal(app.status, "Finishing playback");
assert.equal(app.recordingActive, true, "recording must remain active while the tail drains");
assert.equal(connect(), false, "reconnect must not reset/clear a draining finite queue");
assert.equal(app.resetCount, 0);

// The async decoder completes after onclose. Its old `Live` update must be
// suppressed, and the newly enqueued frames keep the drain open.
app.activity = { ...idleActivity(), playbackQueueFrames: 3 };
assert.equal(setStreamingStatus(), false);
assert.equal(app.status, "Finishing playback", "post-close decode must not revive Live");
assert.equal(checkDrain(), false);

for (let remaining = 2; remaining >= 0; remaining -= 1) {
  app.activity.playbackQueueFrames = remaining;
  app.activity.renderActivity = 1;
  app.renderedFrames += 1;
  assert.equal(checkDrain(), false, "completion cannot run inside an active render");
  app.activity.renderActivity = 0;
  assert.equal(checkDrain(), remaining === 0);
}

assert.equal(app.renderedFrames, 3, "every tail frame must render after close 1000");
assert.equal(app.recordingActive, false);
assert.equal(app.modelState, "closed");
assert.equal(app.status, "Closed");
assert.equal(app.connectDisabled, false);
assert.equal(app.notice, "finite playback complete");
assert.deepEqual(app.log.slice(-4), [
  "capture final recording frame",
  "stop timing and recording",
  "closed",
  "enable reconnect",
]);
assert.equal(connect(), true);
assert.equal(app.resetCount, 1);

const appJs = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const indexHtml = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");
const onClose = appJs.slice(
  appJs.indexOf("socket.onclose = (event) =>"),
  appJs.indexOf("socket.onerror = () =>", appJs.indexOf("socket.onclose = (event) =>")),
);
const decodeDone = appJs.slice(
  appJs.indexOf("async function decodeAndEnqueueFrameBatch"),
  appJs.indexOf("function recordFrameBatchReceived"),
);
const drainFinalize = appJs.slice(
  appJs.indexOf("function finalizePrimaryWebpDrain"),
  appJs.indexOf("function enqueueDecodeBatch"),
);

assert.ok(
  indexHtml.indexOf("finite_webp_drain.js") < indexHtml.indexOf("app.js"),
  "the drain controller must load before the app",
);
assert.match(onClose, /shouldDrainFiniteWebpClose/);
assert.match(onClose, /if \(shouldDrain\)[\s\S]*?beginPrimaryWebpDrain[\s\S]*?return;/);
assert.ok(
  onClose.indexOf("beginPrimaryWebpDrain") < onClose.indexOf("discardPrimaryWebpPlayback"),
  "normal finite close must branch into drain before immediate cleanup",
);
assert.match(onClose, /discardPrimaryWebpPlayback\(closeText, \{ clearFrames: true \}\)/);
assert.match(onClose, /setStatus\("Socket closed", "error"\)/);
assert.match(decodeDone, /setPrimaryWebpStreamingStatus\(\)/);
assert.doesNotMatch(appJs, /setStatus\("Updating", "live"\)/);
assert.match(appJs, /setPrimaryWebpStreamingStatus\("Updating"\)/);
assert.match(
  appJs,
  /function sendEvent\(kind, payload, historyText = null\) \{\s*if \(primaryWebpDrainActive\(\)\)[\s\S]*?reason: "finite playback draining"[\s\S]*?return null;/,
  "prompt and camera activity must stop when finite tail draining begins",
);
assert.match(
  appJs,
  /async function connect\(\) \{\s*if \(primaryWebpDrainActive\(\)\)[\s\S]*?return;/,
  "connect must fail closed while finite tail playback is draining",
);
assert.match(
  appJs,
  /primaryRenderActivity = Math\.max\(0, primaryRenderActivity - 1\);\s*maybeFinalizePrimaryWebpDrain/,
  "drain completion must run only after the render activity ends",
);
assert.match(drainFinalize, /if \(recordingActive\) captureRecordingFrame\(\)/);
assert.match(drainFinalize, /recordingReason: "generation_complete"/);
assert.match(
  drainFinalize,
  /captureRecordingFrame\(\)[\s\S]*?stopWorldExperienceTiming[\s\S]*?setStatus\("Closed"\)[\s\S]*?connectBtn"\)\.disabled = false/,
  "the last frame must be captured before recording stops and reconnect is enabled",
);
assert.doesNotMatch(
  appJs.slice(
    appJs.indexOf("function beginPrimaryWebpDrain"),
    appJs.indexOf("function maybeFinalizePrimaryWebpDrain"),
  ),
  /stopWorldExperienceTiming|stopRecording|连接已中断/,
  "drain start must not stop recording or show an interruption notice",
);
assert.match(appJs, /finiteWebpDrain: primaryWebpDrainController\.snapshot\(\)/);
assert.match(appJs, /lastPrimarySocketCloseCode/);

console.log("finite WebP app drain lifecycle ok");
