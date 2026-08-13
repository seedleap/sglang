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
  /当前正有人体验，请等待\$\{SESSION_MAX_LIFETIME_SECONDS\}s/,
  "the product-facing busy message should follow the configured session lifetime",
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
  /configuredNumber\("sessionMaxLifetimeSeconds", 60\)/,
  "browser safety guard should use a 60 second fallback",
);
assert.match(
  appJs,
  /const SESSION_MAX_LIFETIME_MS = SESSION_MAX_LIFETIME_SECONDS \* 1000;/,
  "browser safety guard should use the configured session lifetime",
);
assert.match(indexHtml, /id="sessionCountdownText">01:00<\/b>/);
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
assert.match(
  indexHtml,
  /id="sessionCountdown"[^>]*role="timer"[^>]*hidden/,
  "the remaining play time should be exposed as a hidden timer until Generate succeeds",
);
assert.match(
  appJs,
  /function markSessionPlayable[\s\S]*?sessionLifetimeGuard\.start\(\);\s*startSessionCountdown\(\);/,
  "the countdown should start only when a selected model renders a playable frame",
);
assert.doesNotMatch(
  appJs,
  /const connectionReport = await dualModelController\.connect\(init\);[\s\S]{0,2200}sessionLifetimeGuard\.start\(\)/,
  "opening model sockets must not consume playable session time",
);
assert.match(
  appJs,
  /window\.setInterval\(updateSessionCountdown, 1000\)/,
  "the countdown should refresh once per second",
);
assert.match(
  appJs,
  /function closeSession[\s\S]*?sessionLifetimeGuard\.cancel\(\);\s*stopSessionCountdown\(\);/,
  "closing a session should stop and hide the countdown",
);

console.log("realtime multi-user lifecycle ok");
