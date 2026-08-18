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

session._queueMetric("client_render_wait", 4.5, {
  codec: "h264",
  scope: "frame",
});
session._flushClientMetrics();
assert.equal(sent[0].type, "client_metric");
assert.equal(sent[0].stage, "client_render_wait");
assert.equal(sent[0].codec, "h264");
sent.length = 0;

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

console.log("h264_websocket_session_test: ok");
