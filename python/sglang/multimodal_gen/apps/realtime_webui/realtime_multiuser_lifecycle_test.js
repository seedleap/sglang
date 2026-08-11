const assert = require("assert");
const fs = require("fs");
const path = require("path");

const appJs = fs.readFileSync(path.join(__dirname, "app.js"), "utf8");
const indexHtml = fs.readFileSync(path.join(__dirname, "index.html"), "utf8");

assert.match(
  appJs,
  /url\.searchParams\.set\("user_id", browserUserId\)/,
  "webui should carry a stable browser identity for per-user admission",
);
assert.match(
  appJs,
  /UI_CONFIG\.singleExperienceUserIds\?\.\[key\]/,
  "showcase mode should use fixed per-backend admission identities",
);
assert.match(
  appJs,
  /url\.searchParams\.set\("user_id", backendUserId\)/,
  "dual-model websocket URLs should not share one coordinator user fence",
);
assert.match(
  appJs,
  /当前正有人体验，请等待45s/,
  "a second showcase visitor should receive the product-facing busy message",
);
assert.match(
  appJs,
  /dualModelController\.close\("showcase session is occupied"\)/,
  "partial dual-model admission must be released when the showcase is occupied",
);
assert.match(
  appJs,
  /const CONTROL_HELD_STATE_HEARTBEAT_MS = 100;/,
  "held actions should be refreshed every 100ms",
);
assert.match(
  appJs,
  /actions: Array\.from\(this\.activeActions\)\.sort\(\)/,
  "each held-key refresh should send the complete key state",
);
assert.match(
  appJs,
  /dualModelController\.sendEvent\("heartbeat"/,
  "idle connected clients should keep both model leases alive explicitly",
);
assert.match(
  appJs,
  /const SESSION_MAX_LIFETIME_MS = 45_000;/,
  "browser safety guard should match the 45 second server lifetime",
);
assert.match(
  appJs,
  /连接已断开，请重新连接/,
  "expired sessions should give the user a clear reconnect message",
);
assert.match(
  indexHtml,
  /id="sessionNotice"[^>]*role="alert"/,
  "session expiry should be exposed as an accessible visible notice",
);

console.log("realtime multi-user lifecycle ok");
