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

console.log("h264_websocket_session_test: ok");
