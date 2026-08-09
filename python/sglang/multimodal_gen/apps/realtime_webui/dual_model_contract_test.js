const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");

assert.equal(
  (html.match(/class="model-player"/g) || []).length,
  2,
  "comparison UI should render exactly two model players",
);
assert.match(html, /id="minwmViewport"/, "left player should expose a MinWM canvas");
assert.match(html, /id="lingbot2Viewport"/, "right player should expose a LingBot2 canvas");
assert.match(html, /class="model-player" data-model-key="minwm"[\s\S]*?<strong>MinWM<\/strong>/);
assert.match(html, /class="model-player" data-model-key="lingbot2"[\s\S]*?<strong>LingBot2<\/strong>/);
assert.ok(
  html.indexOf('data-model-key="minwm"') < html.indexOf('data-model-key="lingbot2"'),
  "MinWM should remain on the left of LingBot2",
);
assert.equal((html.match(/id="connectBtn"/g) || []).length, 1, "Generate remains shared");
assert.equal((html.match(/class="stage-controls"/g) || []).length, 1, "camera controls remain shared");
assert.equal((html.match(/id="firstFrame"/g) || []).length, 1, "reference picker remains shared");
assert.equal((html.match(/id="prompt"/g) || []).length, 1, "prompt remains shared");
assert.equal((html.match(/id="fullscreenBtn"/g) || []).length, 1, "comparison fullscreen remains shared");
assert.equal((html.match(/class="model-player-telemetry"/g) || []).length, 2, "each player needs its own telemetry");
assert.match(html, /id="minwmDisplayLagText"/, "MinWM should expose independent display lag");
assert.match(html, /id="lingbot2DisplayLagText"/, "LingBot2 should expose independent display lag");
assert.doesNotMatch(html, /class="stage-telemetry"/, "shared stream telemetry is misleading");
assert.doesNotMatch(html, /class="spec-grid"/, "generic LingBot capability cards should be removed");
assert.match(
  html,
  /id="fullscreenBtn"[\s\S]*?aria-label="Enter fullscreen comparison"/,
  "fullscreen control should be accessible without visible text",
);
assert.doesNotMatch(html, /SP2|CUDA Graph|4 GPU profile/, "hardware profile should not be visible");
assert.match(css, /\.model-player-grid\s*\{/);
assert.match(css, /grid-template-columns:\s*repeat\(2,/);
assert.match(css, /@media[^}]*max-width[\s\S]*\.model-player-grid\s*\{[\s\S]*grid-template-columns:\s*1fr/);
assert.match(css, /\.stage\s*\{[\s\S]*container-type:\s*inline-size/);
assert.match(css, /@container[^}]*max-width:\s*1180px[\s\S]*\.topbar\s*\{[\s\S]*flex-wrap:\s*wrap/);
assert.match(css, /\.stage:fullscreen\s*\{/);
assert.match(css, /\.stage:fullscreen\s*\{[\s\S]*?height:\s*100vh/);
assert.match(html, /model_session\.js\?v=dual-model-v3/);
assert.match(html, /dual_model_controller\.js\?v=dual-model-v3/);
assert.match(html, /app\.js\?v=realtime-production-gateway-v19/);
assert.match(html, /fullscreen_controller\.js\?v=dual-fullscreen-v1/);

const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
const server = fs.readFileSync(path.join(root, "server.py"), "utf8");
assert.match(app, /const connectionReport = await dualModelController\.connect\(init\)/);
assert.match(app, /const delivery = dualModelController\.sendEvent\(kind, payload\)/);
assert.match(app, /formatModelDelivery\(delivery\.sent\)/);
assert.match(app, /trackPendingModelEvent\(delivery/);
assert.doesNotMatch(app, /lingbot2Session\.close\("MinWM peer failed"\)/);
assert.doesNotMatch(app, /lingbot2Session\.close\("MinWM receive failed"\)/);
assert.doesNotMatch(app, /lingbot2Session\.close\("MinWM server error"\)/);
const lingbotErrorHandler = app.slice(
  app.indexOf('addHistory(`LingBot2 session failed'),
  app.indexOf("const primarySessionAdapter"),
);
assert.doesNotMatch(
  lingbotErrorHandler,
  /abortCurrentSession|lingbot2Session\.close/,
  "LingBot2 failures must remain local to the LingBot2 player",
);
assert.match(app, /function abortCurrentSession[\s\S]{0,220}resetControls = true/);
assert.match(app, /if \(resetControls\) controlStateController\?\.reset/);
assert.match(app, /backendWebSocketUrl\("minwm"/);
assert.match(app, /backendWebSocketUrl\("lingbot2"/);
assert.match(app, /realtime_interactive_event_grace_ms:\s*250/);
assert.doesNotMatch(
  app,
  /if \(!ws \|\| ws\.readyState !== WebSocket\.OPEN\) return;\s*dualModelController\.sendEvent\("heartbeat"/,
  "LingBot2 heartbeat delivery must not depend on the MinWM socket",
);
assert.doesNotMatch(
  app,
  /if \(ws && ws\.bufferedAmount > CONTROL_BUFFERED_AMOUNT_LIMIT\)/,
  "one model's socket pressure must not alter the shared control stream",
);
assert.match(app, /enabled:\s*\(init\)\s*=>\s*init\.generation_mode\s*!==\s*"t2v"/);
assert.match(app, /function drawRecordingComparisonPreview\(/);
assert.match(app, /createFullscreenController/);
assert.match(server, /BACKEND_ENV_PREFIXES = \{/);
assert.match(server, /"minwm": "MINWM"/);
assert.match(server, /"lingbot2": "LINGBOT2"/);

console.log("dual model DOM contract ok");
