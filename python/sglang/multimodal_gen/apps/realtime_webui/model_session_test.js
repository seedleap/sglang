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

  noteInputEvent() {}
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

  socket.message({
    type: "frame_batch",
    chunk_index: 3,
    event_id: 7,
    num_frames: 1,
    content_type: "image/webp",
    payload: new Uint8Array([1, 2, 3]),
  });
  await flush();
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

  session.setUnavailable("T2V unavailable");
  assert.equal(replacementSocket.readyState, FakeSocket.CLOSED);
  assert.ok(states.includes("unavailable"), "disabled model should expose an unavailable state");
  console.log("realtime model session ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
