const assert = require("node:assert/strict");

global.location = { protocol: "http:", host: "webui.example.test" };
global.RTCRtpReceiver = {
  getCapabilities: () => ({
    codecs: [
      { mimeType: "video/VP8" },
      { mimeType: "video/H264", sdpFmtpLine: "profile-level-id=42e01f" },
    ],
  }),
};

class FakeEventTarget {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(name, handler) {
    const handlers = this.listeners.get(name) || [];
    handlers.push(handler);
    this.listeners.set(name, handlers);
  }

  removeEventListener(name, handler) {
    this.listeners.set(
      name,
      (this.listeners.get(name) || []).filter((item) => item !== handler),
    );
  }

  emit(name, event = {}) {
    for (const handler of this.listeners.get(name) || []) handler(event);
    this[`on${name}`]?.(event);
  }
}

class FakeVideo extends FakeEventTarget {
  constructor() {
    super();
    this.hidden = true;
    this.readyState = 0;
    this.videoWidth = 0;
    this.videoHeight = 0;
    this.srcObject = null;
    this.nextVideoFrameCallback = 1;
    this.videoFrameCallbacks = new Map();
  }

  async play() {
    this.readyState = 4;
    this.videoWidth = 832;
    this.videoHeight = 480;
    queueMicrotask(() => this.emit("playing"));
  }

  pause() {}

  requestVideoFrameCallback(callback) {
    const id = this.nextVideoFrameCallback++;
    this.videoFrameCallbacks.set(id, callback);
    return id;
  }

  cancelVideoFrameCallback(id) {
    this.videoFrameCallbacks.delete(id);
  }

  present(metadata) {
    const [id, callback] = this.videoFrameCallbacks.entries().next().value || [];
    if (!callback) return;
    this.videoFrameCallbacks.delete(id);
    callback(100, metadata);
  }
}

const controlSockets = [];
class FakeWebSocket extends FakeEventTarget {
  constructor(url) {
    super();
    this.url = url;
    this.readyState = 0;
    this.bufferedAmount = 0;
    this.sent = [];
    controlSockets.push(this);
    queueMicrotask(() => {
      this.readyState = 1;
      this.emit("open");
    });
  }

  send(payload) {
    this.sent.push(payload);
  }

  close(code = 1000) {
    this.readyState = 3;
    this.emit("close", { code });
  }
}

const peers = [];
class FakePeerConnection extends FakeEventTarget {
  constructor() {
    super();
    this.iceGatheringState = "complete";
    this.connectionState = "new";
    this.localDescription = null;
    this.codecPreferences = [];
    this.receiver = {
      jitterBufferTarget: null,
      playoutDelayHint: null,
    };
    this.closed = false;
    peers.push(this);
  }

  addTransceiver() {
    return {
      receiver: this.receiver,
      setCodecPreferences: (codecs) => {
        this.codecPreferences = codecs;
      },
    };
  }

  async createOffer() {
    return { type: "offer", sdp: "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n" };
  }

  async setLocalDescription(offer) {
    this.localDescription = offer;
  }

  async setRemoteDescription(answer) {
    this.remoteDescription = answer;
    this.connectionState = "connected";
    queueMicrotask(() => this.ontrack?.({
      streams: [{ id: "stream-a" }],
      track: { id: "track-a", kind: "video" },
    }));
  }

  async getStats() {
    return new Map([["video", {
      type: "inbound-rtp",
      kind: "video",
      framesDecoded: 32,
      framesDropped: 1,
      bytesReceived: 200000,
      jitter: 0.004,
      jitterBufferDelay: 0.2,
      jitterBufferTargetDelay: 16,
      jitterBufferEmittedCount: 32,
    }]]);
  }

  close() {
    this.closed = true;
    this.connectionState = "closed";
  }
}

function response({ status = 200, json, text = "", headers = {} }) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => json,
    text: async () => text,
    headers: { get: (name) => headers[name.toLowerCase()] || null },
  };
}

const requests = [];
async function fakeFetch(url, options = {}) {
  requests.push({ url, options });
  if (url === "./api/webrtc/sessions" && options.method === "POST") {
    return response({
      status: 201,
      json: {
        id: "session-a",
        state: "generating",
        whep_url: "http://media.example.test/zing-session-a/whep",
      },
    });
  }
  if (url === "./api/webrtc/sessions/session-a" && !options.method) {
    return response({
      json: {
        id: "session-a",
        state: "streaming",
        frames: 16,
        width: 832,
        height: 480,
        codec: "h264",
        bitrate_kbps: 3500,
        average_source_mbps: 27.5,
        whep_url: "http://media.example.test/zing-session-a/whep",
      },
    });
  }
  if (url === "http://media.example.test/zing-session-a/whep") {
    return response({
      status: 201,
      text: "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 H264/90000\r\n",
      headers: { location: "/zing-session-a/whep/session-resource" },
    });
  }
  if (url === "./api/webrtc/sessions/session-a" && options.method === "DELETE") {
    return response({ json: { stopped: true } });
  }
  throw new Error(`unexpected fetch ${options.method || "GET"} ${url}`);
}

const { PrimaryWebRTCSession } = require("./primary_webrtc_session.js");

(async () => {
  const video = new FakeVideo();
  const canvas = { hidden: false };
  const states = [];
  const stats = [];
  const presentedFrames = [];
  let playable = null;
  const session = new PrimaryWebRTCSession({
    video,
    canvas,
    fetchImpl: fakeFetch,
    WebSocketImpl: FakeWebSocket,
    RTCPeerConnectionImpl: FakePeerConnection,
    mediaPollIntervalMs: 1,
    startupTimeoutMs: 1000,
    controlKeepaliveMs: 5,
    controlReconnectBaseMs: 1,
    mediaDisconnectGraceMs: 5,
    mediaReconnectBaseMs: 1,
    playoutDelayMs: 500,
    onState: (state) => states.push(state),
    onPlayable: (details) => { playable = details; },
    onPresentedFrame: (details) => presentedFrames.push(details),
    onStats: (snapshot) => stats.push(snapshot),
  });

  await session.connect({ type: "init", first_frame: "data:image/png;base64,AA==" });
  assert.equal(session.connected, true);
  assert.equal(video.hidden, false);
  assert.equal(canvas.hidden, true);
  assert.equal(playable.codec, "h264");
  assert.deepEqual(states.slice(0, 2), ["connecting", "live"]);
  assert.equal(peers[0].codecPreferences[0].mimeType, "video/H264");
  assert.equal(peers[0].receiver.jitterBufferTarget, 500);
  assert.equal(peers[0].receiver.playoutDelayHint, null);
  assert.match(peers[0].remoteDescription.sdp, /H264\/90000/);
  assert.equal(
    controlSockets[0].url,
    "ws://webui.example.test/api/webrtc/sessions/session-a/control",
  );
  session._queueMediaBatch({
    type: "media_batch",
    chunk_index: 1,
    event_id: 5,
    first_frame_index: 48,
    num_frames: 1,
    bridge_encoded_epoch_ms: Date.now() - 25,
  });
  video.present({
    mediaTime: 0,
    presentedFrames: 1,
    width: 832,
    height: 480,
  });
  assert.equal(presentedFrames.length, 1);
  assert.equal(presentedFrames[0].chunkIndex, 1);
  assert.equal(presentedFrames[0].eventId, 5);
  assert.equal(presentedFrames[0].sourceFrameIndex, 48);
  assert.ok(presentedFrames[0].bridgeEncodedEpochMs > 0);
  assert.ok(stats.some((snapshot) => snapshot.lastPresentedMediaEventId === 5));
  assert.ok(stats.some((snapshot) => snapshot.lastPresentedTransportMs >= 0));
  assert.ok(stats.some((snapshot) => snapshot.lastPresentedAfterMetadataMs >= 0));

  assert.equal(session.sendEvent({ type: "event", kind: "camera_actions" }), true);
  assert.deepEqual(JSON.parse(controlSockets[0].sent[0]), {
    type: "event",
    kind: "camera_actions",
  });
  controlSockets[0].emit("message", { data: JSON.stringify({
    type: "control_ack",
    kind: "camera_actions",
    event_id: 9,
    client_sent_epoch_ms: Date.now() - 25,
    bridge_received_epoch_ms: Date.now() - 5,
    bridge_forward_ms: 0.4,
    minimum_event_id: 9,
  }) });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(stats.some((snapshot) => snapshot.framesDecoded === 32));
  assert.ok(stats.some((snapshot) => snapshot.jitterBufferTargetMs === 500));
  assert.ok(stats.some((snapshot) => (
    snapshot.lastControlEventId === 9
    && snapshot.controlBridgeRoundTripMs >= 20
    && snapshot.controlBridgeForwardMs === 0.4
  )));
  const mediaEventSentAt = Date.now() - 40;
  assert.equal(session.sendEvent({
    type: "event",
    kind: "camera_actions",
    event_id: 9,
    client_sent_epoch_ms: mediaEventSentAt,
  }), true);
  controlSockets[0].emit("message", { data: JSON.stringify({
    type: "media_batch",
    chunk_index: 2,
    event_id: 9,
    first_frame_index: 32,
    num_frames: 16,
    bridge_received_epoch_ms: Date.now() - 10,
  }) });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(stats.some((snapshot) => (
    snapshot.lastMediaEventId === 9
    && snapshot.mediaControlToBatchMs >= 30
  )));

  controlSockets[0].close(1002);
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(controlSockets.length, 2);
  assert.equal(session.connected, true);
  assert.equal(session.sendEvent({ type: "event", kind: "camera_actions", event_id: 2 }), true);
  assert.deepEqual(JSON.parse(controlSockets[1].sent.at(-1)), {
    type: "event",
    kind: "camera_actions",
    event_id: 2,
  });

  peers[0].connectionState = "disconnected";
  peers[0].onconnectionstatechange();
  assert.equal(states.at(-1), "connecting");
  peers[0].connectionState = "connected";
  peers[0].onconnectionstatechange();
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(peers.length, 1, "transient disconnect should recover without renegotiation");
  assert.equal(states.at(-1), "live");

  peers[0].connectionState = "failed";
  peers[0].onconnectionstatechange();
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(peers.length, 2, "failed media should negotiate a fresh WHEP peer");
  assert.equal(session.connected, true);
  assert.equal(states.at(-1), "live");

  await session.close("test complete");
  assert.equal(video.srcObject, null);
  assert.equal(video.hidden, true);
  assert.equal(canvas.hidden, false);
  assert.ok(peers[0].closed);
  assert.ok(requests.some(({ url, options }) => (
    url === "./api/webrtc/sessions/session-a" && options.method === "DELETE"
  )));

  let pendingReadResolve = null;
  let delivered = false;
  let readerCanceled = false;
  const managedFrame = {
    timestamp: 0,
    displayWidth: 1280,
    displayHeight: 704,
    closed: false,
    close() { this.closed = true; },
  };
  class FakeMediaStreamTrackProcessor {
    constructor({ track }) {
      assert.equal(track.id, "track-a");
      this.readable = {
        getReader: () => ({
          read: async () => {
            if (!delivered) {
              delivered = true;
              return { done: false, value: managedFrame };
            }
            return new Promise((resolve) => { pendingReadResolve = resolve; });
          },
          cancel: async () => {
            readerCanceled = true;
            pendingReadResolve?.({ done: true });
          },
        }),
      };
    }
  }

  const managedVideo = new FakeVideo();
  const managedCanvas = { hidden: true };
  const managedFrames = [];
  let managedPlayable = null;
  const managedSession = new PrimaryWebRTCSession({
    video: managedVideo,
    canvas: managedCanvas,
    managedPlayback: true,
    MediaStreamTrackProcessorImpl: FakeMediaStreamTrackProcessor,
    fetchImpl: fakeFetch,
    WebSocketImpl: FakeWebSocket,
    RTCPeerConnectionImpl: FakePeerConnection,
    mediaPollIntervalMs: 1,
    startupTimeoutMs: 1000,
    controlKeepaliveMs: 0,
    playoutDelayMs: 750,
    onFrame: (frame) => managedFrames.push(frame),
    onPlayable: (details) => { managedPlayable = details; },
  });
  managedSession._queueMediaBatch({
    type: "media_batch",
    chunk_index: 7,
    event_id: 3,
    first_frame_index: 0,
    num_frames: 16,
  });
  await managedSession.connect({ type: "init", fps: 24, first_frame: "data:image/png;base64,AA==" });
  assert.equal(managedSession.connected, true);
  assert.equal(managedVideo.hidden, true);
  assert.equal(managedCanvas.hidden, false);
  assert.equal(managedFrames.length, 1);
  assert.equal(managedFrames[0].width, 1280);
  assert.equal(managedFrames[0].height, 704);
  assert.equal(managedPlayable.managedPlayback, true);
  managedSession._queueMediaBatch({
    type: "media_batch",
    chunk_index: 7,
    event_id: 3,
    first_frame_index: 1,
    num_frames: 1,
    frame_batch_index: 15,
    num_frame_batches: 16,
    is_final_frame_batch: true,
  });
  const mappedFrame = managedSession._metadataForFrame({ timestamp: 41667 }, 1);
  assert.equal(mappedFrame.chunkIndex, 7);
  assert.equal(mappedFrame.eventId, 3);
  assert.equal(mappedFrame.frameBatchIndex, 15);
  assert.equal(mappedFrame.numFrameBatches, 16);
  assert.equal(mappedFrame.isFinalFrameBatch, true);
  managedSession._queueMediaBatch({
    type: "media_batch",
    chunk_index: 8,
    event_id: 4,
    first_frame_index: 2,
    num_frames: 1,
    frame_batch_index: 2,
    num_frame_batches: 0,
    is_final_frame_batch: false,
  });
  const partialFrame = managedSession._metadataForFrame({ timestamp: 83333 }, 2);
  assert.equal(partialFrame.chunkIndex, 8);
  assert.equal(partialFrame.eventId, 4);
  assert.equal(partialFrame.numFrameBatches, 1_000_000);
  assert.equal(partialFrame.isFinalFrameBatch, false);
  managedSession._finalizeMediaChunk({
    type: "media_chunk_complete",
    chunk_index: 8,
    event_id: 4,
    num_frames: 3,
  });
  const completedFrame = managedSession._metadataForFrame({ timestamp: 83333 }, 2);
  assert.equal(completedFrame.numFrameBatches, 3);
  assert.equal(completedFrame.isFinalFrameBatch, true);

  const nativeMappingSession = new PrimaryWebRTCSession({
    video: new FakeVideo(),
    canvas: { hidden: false },
    onPresentedFrame: () => {},
  });
  for (let frameIndex = 900; frameIndex <= 904; frameIndex += 1) {
    nativeMappingSession._queueMediaBatch({
      type: "media_batch",
      chunk_index: 56,
      event_id: frameIndex < 903 ? 20 : 21,
      first_frame_index: frameIndex,
      num_frames: 1,
      bridge_encoded_epoch_ms: 10_000 + (frameIndex - 900) * (1000 / 24),
    });
  }
  const nativeFirst = nativeMappingSession._metadataForFrame({ timestamp: 5_000_000 }, 0);
  assert.equal(nativeFirst.sourceFrameIndex, 902);
  assert.equal(nativeFirst.eventId, 20);
  const nativeAfterRtpGap = nativeMappingSession._metadataForFrame({ timestamp: 5_084_000 }, 1);
  assert.equal(nativeAfterRtpGap.sourceFrameIndex, 904);
  assert.equal(nativeAfterRtpGap.eventId, 21);
  const managedCreate = requests.filter(({ url, options }) => (
    url === "./api/webrtc/sessions" && options.method === "POST"
  )).at(-1);
  assert.equal(JSON.parse(managedCreate.options.body).managed_playback, true);
  managedFrame.close();
  await managedSession.close("managed test complete");
  assert.equal(readerCanceled, true);
  console.log("primary_webrtc_session_test: ok");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
