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
  }

  async play() {
    this.readyState = 4;
    this.videoWidth = 832;
    this.videoHeight = 480;
    queueMicrotask(() => this.emit("playing"));
  }

  pause() {}
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
    this.closed = false;
    peers.push(this);
  }

  addTransceiver() {
    return {
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
    queueMicrotask(() => this.ontrack?.({ streams: [{ id: "stream-a" }] }));
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
  let playable = null;
  const session = new PrimaryWebRTCSession({
    video,
    canvas,
    fetchImpl: fakeFetch,
    WebSocketImpl: FakeWebSocket,
    RTCPeerConnectionImpl: FakePeerConnection,
    mediaPollIntervalMs: 1,
    startupTimeoutMs: 1000,
    onState: (state) => states.push(state),
    onPlayable: (details) => { playable = details; },
    onStats: (snapshot) => stats.push(snapshot),
  });

  await session.connect({ type: "init", first_frame: "data:image/png;base64,AA==" });
  assert.equal(session.connected, true);
  assert.equal(video.hidden, false);
  assert.equal(canvas.hidden, true);
  assert.equal(playable.codec, "h264");
  assert.deepEqual(states.slice(0, 2), ["connecting", "live"]);
  assert.equal(peers[0].codecPreferences[0].mimeType, "video/H264");
  assert.match(peers[0].remoteDescription.sdp, /H264\/90000/);
  assert.equal(
    controlSockets[0].url,
    "ws://webui.example.test/api/webrtc/sessions/session-a/control",
  );

  assert.equal(session.sendEvent({ type: "event", kind: "camera_actions" }), true);
  assert.deepEqual(JSON.parse(controlSockets[0].sent[0]), {
    type: "event",
    kind: "camera_actions",
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.ok(stats.some((snapshot) => snapshot.framesDecoded === 32));

  await session.close("test complete");
  assert.equal(video.srcObject, null);
  assert.equal(video.hidden, true);
  assert.equal(canvas.hidden, false);
  assert.ok(peers[0].closed);
  assert.ok(requests.some(({ url, options }) => (
    url === "./api/webrtc/sessions/session-a" && options.method === "DELETE"
  )));
  console.log("primary_webrtc_session_test: ok");
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
