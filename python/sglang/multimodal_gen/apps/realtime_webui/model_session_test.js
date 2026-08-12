const assert = require("assert");

const { RealtimeModelSession } = require("./model_session.js");

class FakeSocket {
  static OPEN = 1;
  static CLOSED = 3;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    FakeSocket.instances.push(this);
  }

  open() {
    this.readyState = FakeSocket.OPEN;
    this.onopen?.();
  }

  message(data) {
    this.onmessage?.({ data });
  }

  send(data) {
    this.sent.push(data);
  }

  close(code = 1000, reason = "") {
    this.readyState = FakeSocket.CLOSED;
    this.onclose?.({ code, reason });
  }
}

class FakePlaybackController {
  constructor() {
    this.queue = [];
    this.inputEvents = [];
  }

  enqueueDecodedFrames(_header, frames) {
    this.queue.push(...frames);
    return { droppedFrames: [], snapshot: this.snapshot() };
  }

  render() {
    if (!this.queue.length) return { action: "idle", droppedFrames: [] };
    return { action: "draw", frame: this.queue.shift(), droppedFrames: [] };
  }

  snapshot() {
    return { queueFrames: this.queue.length, sourceFps: 24, targetFps: 24 };
  }

  reset() {
    this.queue = [];
  }

  noteInputEvent(eventId, sentAt, options) {
    this.inputEvents.push({ eventId, sentAt, options });
  }
}

function fakeCanvas() {
  const draws = [];
  return {
    width: 1280,
    height: 704,
    draws,
    getContext: () => ({
      drawImage: (...args) => draws.push(args),
      putImageData: (...args) => draws.push(args),
      fillRect: (...args) => draws.push(args),
      imageSmoothingEnabled: true,
      imageSmoothingQuality: "medium",
    }),
    setAttribute() {},
  };
}

async function flush() {
  await new Promise((resolve) => setImmediate(resolve));
}

async function main() {
  FakeSocket.instances = [];
  const frames = [];
  const canvas = fakeCanvas();
  const scheduled = [];
  const states = [];
  const decodedHeaders = [];
  const session = new RealtimeModelSession({
    key: "lingbot2",
    canvas,
    pack: (value) => value,
    unpack: (value) => value,
    WebSocketCtor: FakeSocket,
    PlaybackController: FakePlaybackController,
    decodeBatch: async (header) => {
      decodedHeaders.push(header);
      return [{
        image: { width: 640, height: 360, close() {} },
        chunk: header.chunk_index,
        receivedAt: 100,
        decodeMs: 2,
      }];
    },
    requestFrame: (callback) => scheduled.push(callback),
    now: () => 125,
    onState: (state) => states.push(state),
    onFrame: (frame) => frames.push(frame),
  });

  const connecting = session.connect({ type: "init", trace_id: "trace:lingbot2" }, "/lingbot2");
  const socket = FakeSocket.instances[0];
  socket.open();
  await connecting;
  assert.equal(socket.sent[0].type, "init");
  assert.ok(states.includes("live"));

  session.sendEvent({ type: "event", kind: "prompt", payload: "new", event_id: 7 });
  assert.equal(socket.sent[1].event_id, 7);
  assert.equal(socket.sent[1].trace_id, "trace:lingbot2");
  assert.equal(
    session.playback.inputEvents.at(-1).options.cutoverMode,
    "prompt",
    "prompt updates should cut over old-prompt playback without changing action playback",
  );
  assert.equal(session.snapshot().lastSentEventId, 7);
  assert.equal(
    session.snapshot().lastAppliedEventId,
    0,
    "sending an event must not be reported as model-applied before an output chunk confirms it",
  );

  session.sendEvent({
    type: "event",
    kind: "camera_actions",
    payload: { transitions: [{ actions: ["w"] }] },
    event_id: 8,
  });
  assert.equal(
    session.playback.inputEvents.at(-1).options.cutoverMode,
    "motion",
    "an active LingBot2 camera action should cut over stale playback frames",
  );
  session.sendEvent({
    type: "event",
    kind: "camera_actions",
    payload: { transitions: [{ actions: [] }] },
    event_id: 9,
  });
  assert.equal(
    session.playback.inputEvents.at(-1).options.cutoverMode,
    "settle",
    "a released LingBot2 camera state may settle buffered frames",
  );

  socket.message({
    type: "frame_batch",
    chunk_index: 3,
    event_id: 7,
    num_frames: 1,
    content_type: "image/webp",
    payload: new Uint8Array([1, 2, 3]),
  });
  await flush();
  assert.equal(session.snapshot().lastAppliedEventId, 7);
  assert.equal(session.snapshot().queueFrames, 1);
  assert.equal(canvas.draws.length, 0, "receiving one model does not render synchronously");
  scheduled.shift()(130);
  assert.equal(canvas.draws.length, 1);
  assert.equal(frames[0].chunk, 3);
  assert.equal(session.snapshot().renderFps, 1);

  socket.message({
    type: "frame_batch_header",
    chunk_index: 4,
    event_id: 7,
    num_frames: 1,
    content_type: "image/webp",
  });
  socket.message(new Uint8Array([82, 73, 70, 70]).buffer);
  await flush();
  assert.equal(decodedHeaders.at(-1).chunk_index, 4,
    "split frame_batch_header payloads must be decoded as media, not MsgPack");

  const reconnecting = session.connect({ type: "init", trace_id: "trace:lingbot2:next" }, "/lingbot2");
  const replacementSocket = FakeSocket.instances.at(-1);
  assert.equal(session.snapshot().renderFps, 0, "a new request must not inherit the prior render rate");
  replacementSocket.open();
  await reconnecting;

  const otherCanvas = fakeCanvas();
  const other = new RealtimeModelSession({
    key: "minwm",
    canvas: otherCanvas,
    pack: (value) => value,
    unpack: (value) => value,
    WebSocketCtor: FakeSocket,
    PlaybackController: FakePlaybackController,
    decodeBatch: async () => [],
    requestFrame: () => {},
  });
  assert.equal(other.snapshot().queueFrames, 0, "model playback queues remain independent");

  const stableCanvas = fakeCanvas();
  const stableStates = [];
  const stableSession = new RealtimeModelSession({
    key: "lingbot2-stable-start",
    canvas: stableCanvas,
    startupMinChunk: 1,
    pack: (value) => value,
    unpack: (value) => value,
    WebSocketCtor: FakeSocket,
    PlaybackController: FakePlaybackController,
    decodeBatch: async (header) => [{
      image: { width: 640, height: 360, close() {} },
      chunk: header.chunk_index,
      receivedAt: 100,
      decodeMs: 2,
    }],
    requestFrame: () => {},
    now: () => 150,
    onState: (state) => stableStates.push(state),
  });
  const stableConnecting = stableSession.connect(
    { type: "init", trace_id: "trace:stable" },
    "/lingbot2",
  );
  const stableSocket = FakeSocket.instances.at(-1);
  stableSocket.open();
  await stableConnecting;
  assert.equal(
    stableStates.includes("live"),
    false,
    "a gated LingBot2 session must retain the prior canvas until a stable chunk arrives",
  );
  stableSocket.message({
    type: "frame_batch",
    chunk_index: 0,
    event_id: 0,
    num_frames: 1,
    content_type: "image/webp",
    payload: new Uint8Array([1]),
  });
  await flush();
  stableSession.render(160);
  assert.equal(stableCanvas.draws.length, 0, "the unstable startup chunk must not flash");
  assert.equal(stableStates.includes("live"), false);

  stableSocket.message({
    type: "frame_batch",
    chunk_index: 1,
    event_id: 0,
    num_frames: 1,
    content_type: "image/webp",
    payload: new Uint8Array([2]),
  });
  await flush();
  stableSession.render(170);
  assert.equal(stableCanvas.draws.length, 1);
  assert.equal(stableStates.includes("live"), true);
  stableSession.close("test complete", { notify: false });

  session.setUnavailable("T2V unavailable");
  assert.equal(replacementSocket.readyState, FakeSocket.CLOSED);
  assert.ok(states.includes("unavailable"), "disabled model should expose an unavailable state");
  console.log("realtime model session ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
