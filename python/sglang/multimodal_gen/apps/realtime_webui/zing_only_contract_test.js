const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

function read(name) {
  return fs.readFileSync(path.join(__dirname, name), "utf8");
}

const html = read("index.html");
const css = read("styles.css");
const app = read("app.js");
const server = read("server.py");
const readme = read("README.md");

const runtimeConfigIndex = html.indexOf('<script src="./runtime-config.js"></script>');
const experienceModeIndex = html.indexOf("realtime_experience_mode.js?v=zing-only-v1");
const stylesheetIndex = html.indexOf(
  "styles.css?v=world-studio-zing-only-rife3-finite-transport-v2",
);
const appIndex = html.indexOf(
  "app.js?v=world-studio-zing-only-rife3-finite-transport-v2",
);
assert.ok(
  runtimeConfigIndex >= 0
    && experienceModeIndex > runtimeConfigIndex
    && stylesheetIndex > experienceModeIndex
    && appIndex > stylesheetIndex,
  "runtime config and Zing-only mode must load before styles and app startup",
);

for (const marker of [
  'class="model-slot-config" aria-label="选择对比模型" data-zing-only-hide',
  'data-model-key="lingbot2" data-session-state="idle" data-zing-only-hide',
  'data-model-key="happyoyster" data-session-state="idle" data-zing-only-hide',
  'aria-label="LingBot2 parameters" data-zing-only-hide',
]) {
  assert.ok(html.includes(marker), `missing Zing-only hide marker: ${marker}`);
}
assert.match(html, /data-zing-only-copy="Zing 实时世界"/);
assert.match(html, /data-zing-only-copy="AI 改写后发送至 Zing"/);
assert.match(html, /data-zing-only-copy="下载 Zing 录像"/);
assert.match(html, /data-zing-only-aria-label="Enter fullscreen Zing world"/);

assert.match(
  css,
  /data-realtime-experience="zing-only"[^\n]*data-zing-only-hide[\s\S]{0,220}?display:\s*none\s*!important/,
  "Zing-only content must be hidden before app initialization",
);
assert.match(
  css,
  /data-realtime-experience="zing-only"[^\n]*\.model-player-grid\s*\{\s*grid-template-columns:\s*minmax\(0, 1fr\)/,
  "the Zing player should occupy the full stage",
);

assert.match(app, /const ZING_ONLY = EXPERIENCE_MODE\.isZingOnly\(UI_CONFIG\)/);
assert.match(app, /let activeModelSlotCount = ZING_ONLY \? 1 : 2/);
assert.match(app, /return EXPERIENCE_MODE\.selectedModelKeys\(UI_CONFIG, keys\)/);
assert.match(
  app,
  /sessions:\s*\{[\s\S]{0,140}?minwm:\s*primarySessionAdapter,[\s\S]{0,140}?\.\.\.\(ZING_ONLY \? \{\} : \{/,
  "Zing-only sessions must exclude LingBot2 and HappyOyster",
);
assert.match(
  app,
  /backends:\s*\{[\s\S]{0,500}?\.\.\.\(ZING_ONLY \? \{\} : \{\s*lingbot2:/,
  "Zing-only backend topology must not resolve secondary endpoints",
);
assert.match(app, /const lingbot2H264Session = H264_WEBSOCKET_ENABLED\s*&& !ZING_ONLY/);
assert.match(app, /function canReconnectLingbot2\(\) \{\s*return \(\s*!ZING_ONLY/);
assert.match(app, /if \(!ZING_ONLY\) lingbot2Session\.close\("Zing primary session closed"\)/);
assert.match(app, /if \(!ZING_ONLY\) \{\s*for \(let slotIndex = 0;/);
assert.match(
  app,
  /for \(const key of ZING_ONLY \? \["minwm"\] : \["minwm", "lingbot2"\]\)/,
  "secondary control setup must be skipped in Zing-only mode",
);

assert.match(app, /const variants = EXPERIENCE_MODE\.recordingVariants\(UI_CONFIG\)/);
assert.match(app, /const expectedDownloads = ZING_ONLY \? 1 : 2/);
assert.match(app, /capture_scope: ZING_ONLY \? "zing" : "stage"/);
assert.match(app, /tracksByKey\[primary\.key\]/);
assert.match(app, /videos\[primary\.key\]/);

assert.match(server, /zing_only = config\.get\("zingOnly", False\)/);
assert.match(server, /REALTIME_UI_CONFIG_JSON\.zingOnly must be a boolean/);
assert.match(server, /config\["zingOnly"\] = zing_only/);
assert.match(readme, /Omitting `zingOnly`, or setting it to the\s+JSON boolean `false`, preserves/);
assert.match(readme, /Neither\s+`LINGBOT2_UPSTREAM_HTTP`\/`LINGBOT2_UPSTREAM_WS` nor HappyOyster credentials or\s+production endpoints are required/);

console.log("Zing-only WebUI contract ok");
