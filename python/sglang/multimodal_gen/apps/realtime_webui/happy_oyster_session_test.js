const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
const server = fs.readFileSync(path.join(root, "server.py"), "utf8");
const sdk = fs.readFileSync(path.join(root, "happy_oyster_sdk.js"), "utf8");
const {
  HappyOysterSession,
  translateHappyOysterCameraActions,
} = require("./happy_oyster_session.js");

for (const slot of ["modelSlot0", "modelSlot1", "modelSlot2"]) {
  assert.match(html, new RegExp(`id="${slot}"`), `${slot} should expose a model picker`);
}
assert.match(html, /id="happyOysterPlayer"/);
assert.match(html, /id="happyoysterViewport"[^>]*autoplay[^>]*playsinline/);
assert.match(html, /happy_oyster_sdk\.js/);
assert.match(html, /happy_oyster_sdk\.js\?v=happyoyster-sdk-0\.1\.0-ticket-only/);
assert.match(html, /happy_oyster_session\.js/);
assert.match(html, /happy_oyster_session\.js\?v=happyoyster-session-v5/);
assert.match(html, /dual_model_controller\.js\?v=dual-model-v6/);
assert.match(html, /app\.js\?v=world-studio-control-telemetry-v1/);
assert.match(html, /styles\.css\?v=world-studio-h264-rules-v5/);
assert.match(app, /happyoyster:\s*happyOysterSession/);
assert.match(app, /enabled:\s*\(init\) => modelSelected\("happyoyster"\)/);
assert.match(server, /\/api\/happyoyster\/prepare/);
assert.match(server, /HAPPYOYSTER_API_KEY/);
assert.doesNotMatch(server, /SGLANG_REALTIME_UI_CONFIG.*HAPPYOYSTER_API_KEY/);
assert.match(server, /base_url=_happyoyster_token_base_url\(\)/);
assert.match(server, /async def _generated_world_image[\s\S]*?return web\.FileResponse/);
assert.match(
  sdk,
  /enterTravel,\{ticket:r\}/,
  "enter-travel must use the current ticket-only Open API contract",
);
assert.doesNotMatch(
  sdk,
  /enterTravel,\{accessToken:r,ticket:r\}/,
  "the deprecated accessToken field makes current enter-travel reject the request",
);

assert.deepStrictEqual(
  translateHappyOysterCameraActions({ transitions: [{ actions: ["w", "a", "j"] }] }),
  { translation: "Front_Left", rotation: "Mouse_Left", interaction: "None" },
);
assert.deepStrictEqual(
  translateHappyOysterCameraActions({ transitions: [{ actions: ["d", "k", "space"] }] }),
  { translation: "Right", rotation: "Mouse_Down", interaction: "Jump" },
);
assert.deepStrictEqual(
  translateHappyOysterCameraActions({ transitions: [{ actions: [] }] }),
  { translation: "None", rotation: "None", interaction: "None" },
);

function jsonResponse(payload, ok = true, status = 200) {
  return { ok, status, json: async () => payload };
}

async function testSessionLifecycle() {
  const calls = [];
  const states = [];
  const commands = [];
  let ended = false;
  const fetchImpl = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/config")) return jsonResponse({ enabled: true });
    if (url.endsWith("/worlds/resolve")) return jsonResponse({ status: "missing" });
    if (url.endsWith("/share-image")) return jsonResponse({ url: "https://example.test/frame.png" });
    if (url.endsWith("/worlds")) return jsonResponse({ encryptedWorldId: "world-test" });
    if (url.includes("/build-status")) return jsonResponse({ status: "ready" });
    if (url.endsWith("/prepare")) {
      return jsonResponse({
        apiBaseUrl: "https://example.test/api/v2/apps/happyoyster-1.0",
        token: "temporary-token",
        ticket: "one-time-ticket",
      });
    }
    throw new Error(`unexpected URL ${url}`);
  };
  const travel = {
    on: () => () => {},
    onError: () => () => {},
    start: async () => {},
    end: async () => { ended = true; },
    sendCommand: async (command) => { commands.push(command); },
  };
  global.HappyOysterSDK = {
    HappyOysterEngine: class {
      createTravel({ ticket, videoElement }) {
        assert.strictEqual(ticket, "one-time-ticket");
        assert.ok(videoElement);
        return travel;
      }
    },
  };
  const video = {
    muted: false,
    defaultMuted: false,
    pause() {},
    removeAttribute() {},
    load() {},
  };
  const session = new HappyOysterSession({
    video,
    fetchImpl,
    onState: (state) => states.push(state),
  });
  await session.prepare({
    prompt: "A safe test world",
    firstFrame: new Uint8Array([1, 2, 3]),
    firstFrameMimeType: "image/png",
    presetKey: "safe-test-world",
  });
  assert.strictEqual(calls.filter(({ url }) => url.endsWith("/worlds")).length, 1);
  assert.strictEqual(session.connected, false, "prepare should not start RTC playback");
  await session.connect();
  assert.strictEqual(session.connected, true);
  assert.strictEqual(video.muted, true);
  assert.strictEqual(session.sendEvent({
    kind: "camera_actions",
    payload: { transitions: [{ actions: ["w", "j"] }] },
  }), true);
  await session.pendingCommand;
  assert.deepStrictEqual(commands, [{
    translation: "Front",
    rotation: "Mouse_Left",
    interaction: "None",
  }]);
  assert.strictEqual(session.sendEvent({ kind: "prompt", payload: "turn left" }), false);
  await session.close();
  assert.strictEqual(ended, true);
  assert.ok(states.includes("ready"));
  assert.ok(states.includes("live"));
}

async function testPrebuiltWorldSkipsUploadAndCreation() {
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    calls.push({ url, options });
    if (url.endsWith("/config")) return jsonResponse({ enabled: true });
    if (url.endsWith("/worlds/resolve")) {
      return jsonResponse({
        status: "ready",
        source: "prebuilt",
        encryptedWorldId: "world-prebuilt",
      });
    }
    if (url.endsWith("/prepare")) {
      assert.deepStrictEqual(JSON.parse(options.body), { encryptedWorldId: "world-prebuilt" });
      return jsonResponse({
        apiBaseUrl: "https://example.test/api/v2/apps/happyoyster-1.0",
        token: "temporary-token",
        ticket: "one-time-ticket",
      });
    }
    throw new Error(`unexpected URL ${url}`);
  };
  const session = new HappyOysterSession({ video: null, fetchImpl });
  await session.prepare({
    prompt: "Prebuilt world",
    firstFrame: new Uint8Array([1, 2, 3]),
    presetKey: "prebuilt-world",
  });
  assert.strictEqual(calls.some(({ url }) => url.endsWith("/share-image")), false);
  assert.strictEqual(calls.some(({ url }) => url.endsWith("/worlds")), false);
  assert.strictEqual(session.prepared.encryptedWorldId, undefined);
}

async function testDefaultFetchKeepsGlobalReceiver() {
  const originalFetch = global.fetch;
  let calls = 0;
  global.fetch = async function (url, options = {}) {
    assert.strictEqual(this, global, "default fetch must retain its global receiver");
    calls += 1;
    assert.strictEqual(url, "./api/happyoyster/config");
    assert.deepStrictEqual(options, { cache: "no-store" });
    return jsonResponse({ enabled: true });
  };
  try {
    const session = new HappyOysterSession({ video: null });
    assert.deepStrictEqual(await session.configured(), { enabled: true });
    assert.strictEqual(calls, 1);
  } finally {
    global.fetch = originalFetch;
  }
}

Promise.all([
  testSessionLifecycle(),
  testPrebuiltWorldSkipsUploadAndCreation(),
  testDefaultFetchKeepsGlobalReceiver(),
])
  .then(() => console.log("HappyOyster SBS contract checks passed"))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
