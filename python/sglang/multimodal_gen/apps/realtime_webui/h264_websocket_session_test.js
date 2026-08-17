const assert = require("node:assert/strict");
const { H264WebSocketSession } = require("./h264_websocket_session.js");

class FakeWebSocket {}
FakeWebSocket.OPEN = 1;

class FakeMediaSource {
  static isTypeSupported(type) {
    return type === 'video/mp4; codecs="avc1.4D401F"';
  }
}

const video = {
  readyState: 0,
  videoWidth: 0,
  videoHeight: 0,
  addEventListener() {},
  cancelVideoFrameCallback() {},
};
const stats = [];
const session = new H264WebSocketSession({
  video,
  WebSocketImpl: FakeWebSocket,
  MediaSourceImpl: FakeMediaSource,
  onStats: (value) => stats.push(value),
});
const sent = [];
session.socket = {
  readyState: 1,
  bufferedAmount: 17,
  send: (value) => sent.push(JSON.parse(value)),
};

assert.equal(session.bufferedAmount, 17);
assert.equal(session.sendEvent({
  type: "event",
  kind: "camera_actions",
  event_id: 9,
  client_sent_epoch_ms: 1234,
  payload: { transitions: [] },
}), true);
assert.equal(sent[0].event_id, 9);
assert.equal(session.controlSentEpochByEvent.get(9), 1234);
assert.equal(FakeMediaSource.isTypeSupported('video/mp4; codecs="avc1.4D401F"'), true);

session.mediaFps = 24;
session._handleMetadata({
  type: "media_batch",
  num_frames: 1,
  repeated_frame: false,
  startup_dropped_frames: 8,
});
session._handleMetadata({ type: "media_batch", num_frames: 1, repeated_frame: true });
assert.equal(stats.at(-1).sourceFps, 24);
assert.equal(stats.at(-1).serverFps, 1);
assert.equal(stats.at(-1).deliveryFps, 2);
assert.equal(stats.at(-1).startupDroppedFrames, 0);
assert.equal(stats.at(-2).startupDroppedFrames, 8);

session._handleMetadata({
  type: "media_batch",
  first_frame_index: 7,
  num_frames: 1,
  bridge_encoder_feed_ms: 0,
});
session._handleMetadata({
  type: "media_encode_timing",
  first_frame_index: 7,
  bridge_encoded_epoch_ms: Date.now(),
  bridge_encoder_feed_ms: 6.5,
});
assert.equal(stats.at(-1).lastBridgeEncoderFeedMs, 6.5);
assert.equal(
  session.mediaBatches.find((item) => item.sourceFrameIndex === 7).bridgeEncoderFeedMs,
  6.5,
);

session._handleMetadata({
  type: "media_payload",
  sequence: 3,
  num_bytes: 4,
  server_sent_epoch_ms: Date.now() - 20,
});
session._queueMedia(new Uint8Array([1, 2, 3, 4]));
assert.equal(session.pendingPayloadTimings.length, 0);
assert.equal(session.appendQueue.length, 1);
assert.ok(stats.at(-1).lastWebSocketDownlinkMs >= 0);
session.sourceBuffer = {
  updating: false,
  appendBuffer(data) { this.lastData = data; },
};
session._appendNext();
assert.deepEqual(Array.from(session.sourceBuffer.lastData), [1, 2, 3, 4]);
session.activeAppendItem.receivedAtMs -= 12;
session.activeAppendItem.appendStartedAtMs -= 5;
session._handleAppendEnd();
assert.ok(stats.at(-1).lastMseQueueMs >= 7);
assert.ok(stats.at(-1).lastMseAppendMs >= 5);

function playbackPolicyHarness({ mode = "live", currentTime = 0, end = 2, hidden = false } = {}) {
  const listeners = new Map();
  const documentListeners = new Map();
  const video = {
    readyState: 2,
    videoWidth: 96,
    videoHeight: 64,
    currentTime,
    playbackRate: 1,
    hidden: true,
    playCalls: 0,
    pauseCalls: 0,
    addEventListener(name, handler) {
      const current = listeners.get(name) || [];
      current.push(handler);
      listeners.set(name, current);
    },
    play() {
      this.playCalls += 1;
      return Promise.resolve();
    },
    pause() { this.pauseCalls += 1; },
    cancelVideoFrameCallback() {},
  };
  const documentRef = {
    hidden,
    addEventListener(name, handler) { documentListeners.set(name, handler); },
    emitVisibilityChange() { documentListeners.get("visibilitychange")?.(); },
  };
  const sourceBuffer = {
    updating: false,
    removed: [],
    buffered: {
      length: 1,
      start: () => 0,
      end: () => end,
    },
    remove(start, removeEnd) { this.removed.push([start, removeEnd]); },
  };
  const playbackStats = [];
  const playbackSession = new H264WebSocketSession({
    video,
    mode,
    documentRef,
    WebSocketImpl: FakeWebSocket,
    MediaSourceImpl: FakeMediaSource,
    onStats: (value) => playbackStats.push(value),
  });
  playbackSession.sourceBuffer = sourceBuffer;
  playbackSession.state = "connecting";
  playbackSession._startStats = () => {};
  playbackSession._startPresentedFrameMonitor = () => {};
  return {
    documentRef,
    session: playbackSession,
    sourceBuffer,
    stats: playbackStats,
    video,
    setEnd(value) { end = value; },
  };
}

function testPlaybackPolicies() {
  const live = playbackPolicyHarness({ mode: "live", currentTime: 8, end: 8.8 });
  live.session._maintainLiveEdge();
  assert.equal(live.video.currentTime, 8.72, "default live mode must retain live-edge seeking");
  assert.deepEqual(live.sourceBuffer.removed, [[0, 5]], "live mode must retain history cleanup");
  assert.equal(live.video.playbackRate, 1);

  const timeline = playbackPolicyHarness({ mode: "timeline", currentTime: 0, end: 2 });
  timeline.session._maintainLiveEdge();
  assert.equal(timeline.video.currentTime, 0, "timeline mode must never seek over buffered media");
  assert.deepEqual(timeline.sourceBuffer.removed, []);
  assert.equal(timeline.video.playbackRate, 1);

  const smooth = playbackPolicyHarness({ mode: "smooth_timeline", end: 0.3 });
  smooth.session._maintainLiveEdge();
  assert.equal(smooth.video.playCalls, 0, "smooth timeline must hold for its startup buffer");
  assert.equal(smooth.video.pauseCalls, 1);
  assert.equal(smooth.session.playbackStarted, false);
  assert.equal(smooth.stats.at(-1).playbackBuffering, true);
  smooth.setEnd(2);
  smooth.session._maintainLiveEdge();
  assert.equal(smooth.session.playbackStarted, true);
  assert.equal(smooth.video.playCalls, 1);
  assert.equal(smooth.video.currentTime, 0, "smooth timeline must catch up without seeking");
  assert.ok(smooth.video.playbackRate > 1 && smooth.video.playbackRate <= 1.1);
  assert.equal(smooth.stats.at(-1).playbackTargetLagMs, 650);
  assert.equal(smooth.stats.at(-1).playbackMaxLagMs, 1800);

  const finite = playbackPolicyHarness({ mode: "live", currentTime: 8, end: 12 });
  finite.session.finiteRequest = true;
  finite.video.playbackRate = 1.1;
  finite.session._maintainLiveEdge();
  assert.equal(finite.video.currentTime, 8, "finite live playback must preserve its full tail");
  assert.deepEqual(finite.sourceBuffer.removed, []);
  assert.equal(finite.video.playbackRate, 1);

  const finiteSmooth = playbackPolicyHarness({ mode: "smooth_timeline", end: 0.3 });
  finiteSmooth.session.finiteRequest = true;
  finiteSmooth.session._maintainLiveEdge();
  assert.equal(
    finiteSmooth.video.playCalls,
    0,
    "finite smooth playback must still use startup buffering while generation is active",
  );
  finiteSmooth.session.state = "draining";
  finiteSmooth.session._maintainLiveEdge();
  assert.equal(
    finiteSmooth.video.playCalls,
    1,
    "a short finite tail must start immediately once draining begins",
  );
  assert.equal(finiteSmooth.video.currentTime, 0);
  assert.equal(finiteSmooth.video.playbackRate, 1);

  const draining = playbackPolicyHarness({ mode: "smooth_timeline", currentTime: 8, end: 12 });
  draining.session.playbackStarted = true;
  draining.session.state = "draining";
  draining.video.playbackRate = 1.1;
  draining.session._maintainLiveEdge();
  assert.equal(draining.video.currentTime, 8, "draining playback must never seek to live edge");
  assert.deepEqual(draining.sourceBuffer.removed, []);
  assert.equal(draining.video.playbackRate, 1);

  const background = playbackPolicyHarness({ mode: "smooth_timeline", end: 3, hidden: true });
  background.session.playbackStarted = true;
  background.video.playbackRate = 1.08;
  background.session._maintainLiveEdge();
  assert.equal(background.video.playCalls, 0, "hidden pages must not run playback correction work");
  assert.equal(background.video.playbackRate, 1);
  assert.equal(background.stats.at(-1).playbackPolicySuspended, true);
  background.documentRef.hidden = false;
  background.documentRef.emitVisibilityChange();
  assert.equal(background.video.playCalls, 1);
  assert.ok(background.video.playbackRate > 1 && background.video.playbackRate <= 1.1);

  const configured = playbackPolicyHarness();
  configured.session.sourceBuffer = null;
  const configuredSnapshot = configured.session.configure({
    mode: "smooth_timeline",
    targetFps: 48,
    smoothTimelinePlaybackRateMax: 1.22,
  });
  assert.equal(configuredSnapshot.mode, "smooth_timeline");
  assert.equal(configured.session.targetFps, 48);
  assert.equal(configured.session.smoothTimelinePlaybackRateMax, 1.22);
}

testPlaybackPolicies();

class CloseAwareWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static instances = [];

  constructor() {
    this.readyState = 0;
    this.bufferedAmount = 0;
    this.sent = [];
    CloseAwareWebSocket.instances.push(this);
  }

  open() {
    this.readyState = CloseAwareWebSocket.OPEN;
    this.onopen?.();
  }

  receive(value) {
    this.onmessage?.({ data: value });
  }

  send(value) {
    this.sent.push(value);
  }

  serverClose(code, reason = "") {
    this.readyState = CloseAwareWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }

  close(code = 1000, reason = "") {
    this.serverClose(code, reason);
  }
}

class DrainSourceBuffer {
  constructor() {
    this.updating = false;
    this.appended = [];
    this.buffered = {
      length: 1,
      start: () => 0,
      end: () => 1,
    };
    this.listeners = new Map();
    this.removed = [];
  }

  addEventListener(name, handler) {
    this.listeners.set(name, handler);
  }

  appendBuffer(value) {
    this.appended.push(value);
    this.updating = true;
  }

  finishAppend() {
    this.updating = false;
    this.listeners.get("updateend")?.();
  }

  remove(start, end) { this.removed.push([start, end]); }
}

class DrainMediaSource {
  static instances = [];

  static isTypeSupported(type) {
    return type === 'video/mp4; codecs="avc1.4D401F"';
  }

  constructor() {
    this.readyState = "open";
    this.endOfStreamCalls = 0;
    this.sourceBuffer = new DrainSourceBuffer();
    DrainMediaSource.instances.push(this);
  }

  addEventListener() {}
  removeEventListener() {}

  addSourceBuffer() {
    return this.sourceBuffer;
  }

  endOfStream() {
    this.endOfStreamCalls += 1;
    this.readyState = "ended";
  }
}

function drainVideo({ playError = null } = {}) {
  const listeners = new Map();
  return {
    readyState: 2,
    videoWidth: 96,
    videoHeight: 64,
    currentTime: 0,
    ended: false,
    hidden: true,
    addEventListener(name, handler) {
      const current = listeners.get(name) || [];
      current.push(handler);
      listeners.set(name, current);
    },
    emit(name) {
      for (const handler of listeners.get(name) || []) handler();
    },
    play() { return playError ? Promise.reject(playError) : Promise.resolve(); },
    pause() {},
    removeAttribute() {},
    load() {},
    cancelVideoFrameCallback() {},
  };
}

async function connectDrainSession({
  onState,
  onError,
  markPlayable = true,
  playError = null,
  sessionOptions = {},
  init = {},
}) {
  const video = drainVideo({ playError });
  const drainingSession = new H264WebSocketSession({
    video,
    WebSocketImpl: CloseAwareWebSocket,
    MediaSourceImpl: DrainMediaSource,
    onState,
    onError,
    ...sessionOptions,
  });
  const connected = drainingSession.connect({
    first_frame: "data:image/png;base64,AA==",
    fps: 24,
    ...init,
  });
  await new Promise((resolve) => setImmediate(resolve));
  const socket = CloseAwareWebSocket.instances.at(-1);
  socket.open();
  socket.receive(JSON.stringify({ type: "status", state: "connected" }));
  await connected;
  if (markPlayable) video.emit("loadeddata");
  return {
    video,
    session: drainingSession,
    socket,
    mediaSource: DrainMediaSource.instances.at(-1),
  };
}

async function testGracefulCloseDrainsMseWithoutError() {
  const originalCreateObjectUrl = URL.createObjectURL;
  const originalRevokeObjectUrl = URL.revokeObjectURL;
  URL.createObjectURL = () => "blob:h264-drain-test";
  URL.revokeObjectURL = () => {};
  try {
    const errors = [];
    const states = [];
    const { video, session: drainingSession, socket, mediaSource } = (
      await connectDrainSession({
        onState: (state) => states.push(state),
        onError: (error) => errors.push(error),
      })
    );
    socket.receive(new Uint8Array([0, 0, 0, 1]).buffer);
    assert.equal(mediaSource.sourceBuffer.updating, true);
    socket.receive(JSON.stringify({
      type: "stream_complete",
      chunk_index: 0,
      num_frames: 2,
      encoded_frames: 2,
    }));
    assert.equal(drainingSession.state, "draining");
    assert.equal(mediaSource.endOfStreamCalls, 0);
    socket.serverClose(1000, "generation complete");
    assert.equal(drainingSession.state, "draining");
    assert.equal(errors.length, 0);
    assert.equal(video.hidden, false);
    assert.equal(mediaSource.endOfStreamCalls, 0);

    mediaSource.sourceBuffer.finishAppend();
    assert.equal(mediaSource.endOfStreamCalls, 1);

    video.ended = true;
    video.emit("ended");
    assert.equal(drainingSession.state, "closed");
    assert.equal(states.includes("error"), false);
    assert.equal(states.at(-2), "draining");
    assert.equal(states.at(-1), "closed");

    const closeOnlyErrors = [];
    const closeOnly = await connectDrainSession({
      onState: () => {},
      onError: (error) => closeOnlyErrors.push(error),
    });
    closeOnly.socket.serverClose(1001, "server going away");
    assert.equal(closeOnly.session.state, "draining");
    assert.equal(closeOnlyErrors.length, 0);
    closeOnly.video.ended = true;
    closeOnly.video.emit("ended");
    assert.equal(closeOnly.session.state, "closed");

    const terminalFirstStates = [];
    const terminalFirst = await connectDrainSession({
      onState: (state) => terminalFirstStates.push(state),
      onError: (error) => errors.push(error),
      markPlayable: false,
      init: { generation_mode: "i2v", max_chunks: 1 },
    });
    assert.equal(terminalFirst.session.finiteRequest, true);
    terminalFirst.socket.receive(JSON.stringify({
      type: "stream_complete",
      encoded_frames: 1,
    }));
    assert.equal(terminalFirst.session.state, "draining");
    terminalFirst.video.emit("loadeddata");
    assert.equal(terminalFirst.session.state, "draining");
    assert.equal(terminalFirst.video.hidden, false);
    assert.equal(terminalFirstStates.at(-1), "draining");
    terminalFirst.video.ended = true;
    terminalFirst.video.emit("ended");
    assert.equal(terminalFirst.session.state, "closed");

    const removeUpdate = await connectDrainSession({
      onState: () => {},
      onError: (error) => errors.push(error),
    });
    removeUpdate.session.state = "draining";
    removeUpdate.session.activeAppendItem = null;
    removeUpdate.mediaSource.sourceBuffer.updating = true;
    removeUpdate.mediaSource.sourceBuffer.buffered.length = 0;
    assert.equal(removeUpdate.mediaSource.endOfStreamCalls, 0);
    removeUpdate.mediaSource.sourceBuffer.finishAppend();
    assert.equal(removeUpdate.mediaSource.endOfStreamCalls, 1);
    assert.equal(removeUpdate.session.state, "closed");

    const timeupdate = await connectDrainSession({
      onState: () => {},
      onError: (error) => errors.push(error),
    });
    timeupdate.socket.receive(JSON.stringify({ type: "stream_complete", encoded_frames: 1 }));
    assert.equal(timeupdate.session.state, "draining");
    timeupdate.video.currentTime = 0.995;
    timeupdate.video.emit("timeupdate");
    assert.equal(timeupdate.video.ended, false);
    assert.equal(timeupdate.session.state, "closed");

    const rejectionStates = [];
    const rejectedPlay = await connectDrainSession({
      onState: (state, details) => rejectionStates.push({ state, details }),
      onError: (error) => errors.push(error),
      playError: new Error("autoplay blocked"),
      sessionOptions: {
        drainPlaybackGraceMs: 5,
        drainPlaybackMaxWaitMs: 20,
      },
    });
    rejectedPlay.socket.receive(JSON.stringify({
      type: "stream_complete",
      encoded_frames: 1,
    }));
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(rejectedPlay.session.state, "draining");
    assert.equal(rejectionStates.at(-1).details.playbackError, "autoplay blocked");
    await new Promise((resolve) => setTimeout(resolve, 80));
    assert.equal(rejectedPlay.session.state, "closed");
    assert.equal(rejectionStates.at(-1).details.drainTimedOut, true);
    assert.ok(rejectionStates.at(-1).details.remainingPlaybackMs > 0);
    assert.equal(errors.length, 0);

    const disconnectErrors = [];
    const disconnectStates = [];
    const disconnected = await connectDrainSession({
      onState: (state) => disconnectStates.push(state),
      onError: (error) => disconnectErrors.push(error),
    });
    disconnected.socket.serverClose(1006, "network lost");
    assert.equal(disconnected.session.state, "error");
    assert.equal(disconnectStates.at(-1), "error");
    assert.equal(disconnectErrors.length, 1);
    assert.match(disconnectErrors[0].message, /1006/);
    assert.equal(disconnected.session.statsTimer, 0, "disconnect must stop telemetry polling");
    assert.equal(disconnected.session.frameCallback, 0, "disconnect must stop frame callbacks");
    await disconnected.session.close("test cleanup", { emitState: false });
  } finally {
    URL.createObjectURL = originalCreateObjectUrl;
    URL.revokeObjectURL = originalRevokeObjectUrl;
  }
}

testGracefulCloseDrainsMseWithoutError()
  .then(() => console.log("h264_websocket_session_test: ok"))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
