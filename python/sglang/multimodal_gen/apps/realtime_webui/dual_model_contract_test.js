const assert = require("assert");
const fs = require("fs");
const path = require("path");

const root = __dirname;
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");

assert.equal(
  (html.match(/class="model-player"/g) || []).length,
  3,
  "comparison UI should render two default players plus one optional SBS player",
);
assert.match(html, /id="minwmViewport"/, "left player should expose a Zing canvas");
assert.match(html, /id="lingbot2Viewport"/, "right player should expose a LingBot2 canvas");
assert.match(html, /class="model-player" data-model-key="minwm"[\s\S]*?<strong>Zing<\/strong>/);
assert.match(html, /class="model-player" data-model-key="lingbot2"[\s\S]*?<strong>LingBot2<\/strong>/);
assert.ok(
  html.indexOf('data-model-key="minwm"') < html.indexOf('data-model-key="lingbot2"'),
  "Zing should remain on the left of LingBot2",
);
assert.equal((html.match(/id="connectBtn"/g) || []).length, 1, "Generate remains shared");
assert.equal((html.match(/class="stage-controls"/g) || []).length, 1, "camera controls remain shared");
assert.equal((html.match(/id="firstFrame"/g) || []).length, 1, "reference picker remains shared");
assert.equal((html.match(/id="prompt"/g) || []).length, 1, "prompt remains shared");
assert.equal((html.match(/id="fullscreenBtn"/g) || []).length, 1, "comparison fullscreen remains shared");
assert.equal((html.match(/class="model-player-telemetry"/g) || []).length, 3, "each player needs its own telemetry");
assert.equal((html.match(/class="model-parameter-panel"/g) || []).length, 2, "each player needs independent parameters");
assert.match(
  html,
  /<div class="model-parameters-grid"[^>]*\shidden(?:\s|>)/,
  "model parameters should stay in the DOM for request defaults but remain hidden from users",
);
assert.match(html, /id="size" value="1280x704"/, "MinWM should keep the 720p default");
assert.match(html, /id="lingbot2Size" value="832x480"/, "LingBot2 should use its native default size");
assert.match(html, /id="fps" type="number" value="24"/, "Zing should keep its 24 FPS default");
assert.match(html, /id="lingbot2Fps" type="number" value="16"/, "LingBot2 should use its official 16 FPS default");
assert.match(html, /id="lingbot2SinkSize" type="number" value="9"/, "LingBot2 should match the official sink size");
assert.match(html, /id="lingbot2WindowFrames" type="number" value="18"/, "LingBot2 should match the official attention window");
assert.match(app, /const DEFAULT_LINGBOT2_TARGET_FPS\s*=\s*configuredModelNumber\("lingbot2", "targetFps", 16\)/);
assert.match(app, /const DEFAULT_LINGBOT2_SINK_SIZE\s*=\s*configuredModelNumber\("lingbot2", "sinkSize", 9\)/);
assert.match(app, /const DEFAULT_LINGBOT2_WINDOW_FRAMES\s*=\s*configuredModelNumber\("lingbot2", "windowFrames", 18\)/);
assert.match(
  app,
  /startupTimeoutMs:\s*configuredModelNumber\("lingbot2", "startupTimeoutMs", 60000\)/,
  "LingBot2 should wait for its measured cold first-frame latency instead of the generic 12s timeout",
);
assert.match(
  app,
  /if \(message\.type === "chunk_telemetry"\) \{[\s\S]*?chunkTelemetry: \{ \.\.\.message \}[\s\S]*?return;/,
  "Zing must consume chunk telemetry as control data instead of treating it as a legacy frame header",
);
assert.match(
  app,
  /message\.type === "frame_batch_header" \|\| \(!message\.type && message\.content_type\)/,
  "only an explicit or structurally valid legacy frame header may arm the binary payload path",
);
assert.doesNotMatch(
  app,
  /\n\s*pendingHeader = message;\n\s*if \(pendingHeader && !renderedPreviewFrames\)/,
  "unknown control messages must not become Zing frame headers",
);
assert.match(
  app,
  /this\.enqueueTransition\(\{ immediate: active \}\)/,
  "key presses should bypass the transition batching delay",
);
assert.match(
  app,
  /if \(immediate\) this\.flush\(\);\s*else this\.scheduleFlush\(\);/,
  "key releases should retain the transition batching window",
);
assert.match(
  app,
  /const oppositeAction = CONTROL_OPPOSITE_ACTIONS\.get\(action\);\s*if \(oppositeAction\) this\.activeActions\.delete\(oppositeAction\);/,
  "mutually exclusive controls should use the most recently pressed direction",
);
assert.match(
  app,
  /scheduleMinwmReconnect\(error\.message \|\| "stream failed"\)/,
  "Zing should recover an interrupted H.264 session automatically",
);
assert.doesNotMatch(
  app,
  /for \(const key of \["minwm", "lingbot2"\]\) \{\s*modelControl\(key, "fps"\)\.value = UI_CONFIG\.targetFps/s,
  "selecting a shared preset must not overwrite LingBot2's independent official FPS",
);
assert.doesNotMatch(html, />MinWM</, "the former MinWM product name should not remain visible");
const comparisonIndex = html.indexOf('class="model-player-grid"');
const cameraControlsIndex = html.indexOf('class="stage-controls"');
const parameterGridIndex = html.indexOf('class="model-parameters-grid"');
assert.ok(
  comparisonIndex < cameraControlsIndex && cameraControlsIndex < parameterGridIndex,
  "shared camera controls should sit below both videos and above both parameter panels",
);
assert.match(html, /id="minwmDisplayLagText"/, "MinWM should expose independent display lag");
assert.match(html, /id="lingbot2DisplayLagText"/, "LingBot2 should expose independent display lag");
assert.match(html, /<span>FPS<b id="minwmRateText">-<\/b><\/span>/, "MinWM should show its own FPS");
assert.match(html, /<span>FPS<b id="lingbot2RateText">-<\/b><\/span>/, "LingBot2 should show its own FPS");
assert.match(html, /<span>FPS<b id="minwmPerfFps">-<\/b><\/span>/, "MinWM should show stage FPS");
assert.match(html, /<span>FPS<b id="lingbot2PerfFps">-<\/b><\/span>/, "LingBot2 should show stage FPS");
assert.match(html, /下行带宽<b id="minwmPerfData">-<\/b>/, "Zing should show measured downlink bandwidth");
assert.match(html, /H\.264 前队列<b id="minwmPerfH264Queue">-<\/b>/, "Zing should show encoder input queue latency");
assert.match(html, /FFmpeg 写入<b id="minwmPerfH264Feed">-<\/b>/, "Zing should show FFmpeg feed latency");
assert.match(html, /WS 下行<b id="minwmPerfDownlink">-<\/b>/, "Zing should show wire downlink latency separately");
assert.match(html, /MSE 队列<b id="minwmPerfMseQueue">-<\/b>/, "Zing should show browser append queue latency");
assert.match(html, /MSE 追加<b id="minwmPerfMseAppend">-<\/b>/, "Zing should show SourceBuffer append latency");
assert.match(html, /播放缓冲<b id="minwmPerfPlaybackBuffer">-<\/b>/, "Zing should show playback lead");
assert.match(html, /H\.264 前队列<b id="lingbot2PerfH264Queue">-<\/b>/, "LingBot2 should show encoder input queue latency");
assert.match(html, /FFmpeg 写入<b id="lingbot2PerfH264Feed">-<\/b>/, "LingBot2 should show FFmpeg feed latency");
assert.match(html, /WS 下行<b id="lingbot2PerfDownlink">-<\/b>/, "LingBot2 should show wire downlink latency separately");
assert.match(html, /MSE 队列<b id="lingbot2PerfMseQueue">-<\/b>/, "LingBot2 should show browser append queue latency");
assert.match(html, /MSE 追加<b id="lingbot2PerfMseAppend">-<\/b>/, "LingBot2 should show SourceBuffer append latency");
assert.match(html, /播放缓冲<b id="lingbot2PerfPlaybackBuffer">-<\/b>/, "LingBot2 should show playback lead");
assert.doesNotMatch(html, /H\.264\/下行/, "H.264 and downlink metrics must not share one field");
assert.match(app, /\$\(`\$\{key\}PerfH264Queue`\)\.textContent/);
assert.match(app, /\$\(`\$\{key\}PerfH264Feed`\)\.textContent/);
assert.match(app, /\$\(`\$\{key\}PerfDownlink`\)\.textContent/);
assert.match(app, /\$\(`\$\{key\}PerfMseQueue`\)\.textContent/);
assert.match(app, /\$\(`\$\{key\}PerfMseAppend`\)\.textContent/);
assert.match(app, /\$\(`\$\{key\}PerfPlaybackBuffer`\)\.textContent/);
assert.match(app, /activeH264Models\.has\("minwm"\)/, "H.264 stats should not be overwritten by WebP playback stats");
assert.match(
  app,
  /renderProtocolPerformance\("minwm", \{[\s\S]*?bytes,[\s\S]*?transport: "webp"/,
  "Zing WebP telemetry should receive measured bytes and an explicit transport",
);
assert.match(
  app,
  /receiveMbps: Math\.max\([\s\S]*?bytes - primaryNetworkSample\.bytes/,
  "Zing should calculate rolling WebP receive bandwidth",
);
assert.match(
  app,
  /server_sent_epoch_ms[\s\S]*?lastDownlinkMs: Math\.max/,
  "Zing should derive WebSocket downlink latency from frame timestamps",
);
assert.match(
  app,
  /primaryControlSentEpochByEvent[\s\S]*?lastControlToVideoMs: Math\.max/,
  "Zing should measure the first rendered frame after each control event",
);
assert.match(
  app,
  /const isH264 = stats\.transport === "h264"[\s\S]*?: "不适用"/,
  "WebP sessions should mark H.264/MSE-only metrics as not applicable",
);
assert.match(
  app,
  /function protocolMetricText\(value\) \{[\s\S]*?value === null \? "-" : performanceMs\(value\)/,
  "measured zero latency must render as 0 ms rather than missing data",
);
assert.match(
  app,
  /"h264StartupDropFrames",[\s\S]*?key === "lingbot2" \? 8 : 0/,
  "LingBot2 should hide eight startup transition frames without changing Zing",
);
for (const id of [
  "size", "fps", "numFrames", "seed", "steps", "guidance", "sinkSize",
  "windowFrames", "transportFormat", "transportQuality", "playbackMode",
  "superResolution", "upscalingScale", "upscalingModel", "frameInterpolation", "continuous",
]) {
  assert.equal((html.match(new RegExp(`id="${id}"`, "g")) || []).length, 1, `Zing control ${id} should exist once`);
  const lingbotId = `lingbot2${id[0].toUpperCase()}${id.slice(1)}`;
  assert.equal((html.match(new RegExp(`id="${lingbotId}"`, "g")) || []).length, 1, `LingBot2 control ${lingbotId} should exist once`);
}
const sidebarHtml = html.slice(html.indexOf('<section class="panel controls"'), html.indexOf('</section>', html.indexOf('<section class="panel controls"')));
assert.match(sidebarHtml, /id="presetList"/, "shared presets should live in the left sidebar");
assert.doesNotMatch(sidebarHtml, /id="size"|id="lingbot2Size"/, "model parameters should not remain in the shared sidebar");
assert.doesNotMatch(html, /class="stage-telemetry"/, "shared stream telemetry is misleading");
assert.doesNotMatch(html, /class="spec-grid"/, "generic LingBot capability cards should be removed");
assert.match(
  html,
  /id="fullscreenBtn"[\s\S]*?aria-label="Enter fullscreen comparison"/,
  "fullscreen control should be accessible without visible text",
);
assert.doesNotMatch(html, /SP2|CUDA Graph|4 GPU profile/, "hardware profile should not be visible");
assert.match(css, /\.model-player-grid\s*\{/);
assert.match(
  css,
  /\.model-parameters-grid\[hidden\]\s*\{[\s\S]*?display:\s*none/,
  "author styles must not override the hidden parameter panel",
);
assert.match(css, /grid-template-columns:\s*repeat\(2,/);
assert.match(css, /\.model-player-grid\.is-single\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/);
assert.match(app, /Array\.isArray\(UI_CONFIG\.modelSlots\)/);
assert.match(app, /if \(CONFIGURED_MODEL_SLOTS\.length\) return \[\.\.\.CONFIGURED_MODEL_SLOTS\]/);
assert.match(app, /slotConfig\.hidden = MODEL_SLOTS_LOCKED/);
assert.match(css, /@media[^}]*max-width[\s\S]*\.model-player-grid\s*\{[\s\S]*grid-template-columns:\s*1fr/);
assert.match(css, /\.stage\s*\{[\s\S]*container-type:\s*inline-size/);
assert.match(css, /@container[^}]*max-width:\s*1180px[\s\S]*\.topbar\s*\{[\s\S]*flex-wrap:\s*wrap/);
assert.match(css, /\.stage:fullscreen\s*\{/);
assert.match(css, /\.stage:fullscreen\s*\{[\s\S]*?height:\s*100vh/);
assert.match(
  app,
  /const previewFrame = document\.querySelector\("\.stage"\)/,
  "fullscreen must target the complete comparison stage so fullscreen-only rules apply",
);
assert.match(
  css,
  /\.stage:fullscreen \.model-parameters-grid,[\s\S]*?\.stage:fullscreen \.session-notice\s*\{[\s\S]*?display:\s*none/,
  "fullscreen must hide model parameters and notices while preserving both videos",
);
assert.match(
  css,
  /\.world-studio \.stage:fullscreen\s*\{[\s\S]*?grid-template-rows:\s*minmax\(0, 1fr\) auto/,
  "fullscreen should reserve the flexible row for video and one compact row for prompt input",
);
for (const selector of ["model-slot-config", "stage-controls", "prompt-update-heading", "prompt-helper", "prompt-log-panel"]) {
  assert.match(
    css,
    new RegExp(`\\.world-studio \\.stage:fullscreen \\.${selector}`),
    `fullscreen should hide ${selector}`,
  );
}
assert.match(html, /playback_controller\.js\?v=realtime-playback-v34/);
assert.match(html, /model_session\.js\?v=dual-h264-telemetry-v1/);
assert.match(html, /dual_model_controller\.js\?v=dual-model-v6/);
assert.match(html, /h264_websocket_session\.js\?v=h264-stage-timing-v1/);
assert.match(html, /prompt_rewrite_controller\.js\?v=prompt-rewrite-v3/);
assert.match(html, /world_rules_controller\.js\?v=world-rules-v4/);
assert.match(html, /styles\.css\?v=world-studio-h264-rules-v7/);
assert.match(html, /app\.js\?v=world-studio-transport-telemetry-v5/);
assert.match(html, /id="minwmH264Viewport"/);
assert.match(html, /id="lingbot2H264Viewport"/);
assert.match(html, /id="minwmPerfScheduler"/);
assert.match(html, /id="lingbot2PerfScheduler"/);
assert.doesNotMatch(
  app,
  /window\.location\.hostname === "localhost"/,
  "localhost previews should use the same-origin dual-backend proxy",
);
assert.match(html, /fullscreen_controller\.js\?v=fullscreen-focus-v1/);
assert.doesNotMatch(
  html,
  /assets\/presets\/lingbot_testset_20_20260810\/presets\.js\?v=20260810/,
  "metadata-only presets without first-frame images should stay out of the visitor UI",
);
assert.match(html, /id="runtimePrompt"/, "runtime prompt updates should use a dedicated composer");
assert.match(html, /<details id="promptLogPanel" class="prompt-log-panel">/, "prompt log should be collapsed by default");
assert.doesNotMatch(html, /<details id="promptLogPanel"[^>]*\sopen/, "prompt log must not start expanded");
assert.match(html, /id="promptLogList"/, "prompt log should render every sent prompt");
assert.match(html, /id="clearWorldBtn"/, "world drafts should support one-click clearing");
assert.match(html, /id="enhanceBtn"[^>]*class="complete-world"/, "world drafts should expose AI completion");
assert.match(html, /id="firstFrameState"/, "first-frame completeness should be visible");
assert.match(html, /id="referenceDropZone"/, "first-frame picker should expose a drag-and-drop target");
assert.match(html, /点击或拖入 PNG、JPG、WebP/, "first-frame picker should advertise drag and drop");
assert.match(html, /id="worldDescriptionState"/, "description completeness should be visible");
assert.match(html, /<details id="worldRulesPanel" class="world-rules-panel">/, "world rules should be optional and collapsed by default");
assert.doesNotMatch(html, /<details id="worldRulesPanel"[^>]*\sopen/, "world rules must not occupy sidebar space until expanded");
assert.match(html, /id="addSkillRuleBtn"/, "world rules should support multiple skills");
assert.match(html, /id="addGoalRuleBtn"/, "world rules should support multiple goals");
assert.match(html, /id="goalRuleList"/, "goals should render as a dynamic list");
assert.match(html, /目标<\/b><small>最多 9 个<\/small>/, "goals should have the same maximum as skills");
assert.doesNotMatch(html, /id="goalMinPlaySeconds"|id="goalProbability"|id="goalRuleInput"/, "goals should no longer be a singleton static form");
assert.match(app, /minPlay\.max = String\(MAX_GOAL_MIN_PLAY_SECONDS\)/, "goals should configure a bounded minimum play duration");
assert.match(app, /probability\.max = "1"/, "goal probability should be constrained to 0-1");
assert.match(app, /function addGoalRule\(goal = \{\}/, "goals should use the same add/remove pattern as skills");
assert.doesNotMatch(html, /id="goalName"|id="goalPrompt"/, "a goal must not require separate name and prompt fields");
assert.match(html, /id="runtimeSkillBar"[^>]*hidden/, "prepared skills should render above movement controls only when active");
assert.match(html, /id="runtimeSkillHint"[^>]*>[^<]*共享 10s CD/, "skill controls should disclose the shared cooldown");
assert.match(html, /id="goalAchievementToast"[^>]*hidden/, "goal completion should have an accessible popup");
assert.match(app, /function clearWorldDraft\(\)/);
assert.match(app, /async function completeWorldDraft\(\)/);
assert.match(app, /function setWorldCompletionBusy\(pending, completingFromImage = false\)/);
assert.match(app, /function setupFirstFrameDropZone\(\)/);
assert.match(app, /async function prepareWorldRulesForEntry\(description\)/);
assert.match(app, /fetch\("\.\/api\/world-rule\/complete"/);
assert.match(app, /worldRulesController\.activate\(preparedWorldRules\)/);
assert.match(app, /worldRulesController\?\.startSession\(\)/, "goal timing should begin on the first visible world frame");
assert.match(app, /achievementDelayMs:\s*5000/);
assert.match(app, /skillCooldownMs:\s*10000/);
assert.match(app, /skillCooldownRemainingMs/, "all skill controls should observe the shared cooldown");
assert.doesNotMatch(app, /noteUserPromptSuccess/, "user prompts must not roll timed world goals");
assert.match(app, /rule === "goal_time_probability"/);
assert.match(app, /function keyboardSkill\(event\)/);
assert.match(app, /function appendPromptLog\(prompt, metadata = \{\}\)/);
assert.match(app, /metadata\.trigger === "rule" \|\| metadata\.phase === "restore"/);
assert.match(app, /rule === "one_time_timeout_restore"/);
assert.match(
  app,
  /if \(eventId\) \{[\s\S]*?appendPromptLog\(prompt, metadata\);[\s\S]*?markRecordingPromptSent\(prompt, metadata, eventId\);/,
);
assert.match(app, /dropZone\.addEventListener\("drop"/);
assert.match(app, /selectedReferenceBytes = new Uint8Array\(await file\.arrayBuffer\(\)\)/);
assert.match(app, /classList\.toggle\("is-loading", pending\)/);
assert.doesNotMatch(app, /function enhancePrompt\(\)/, "legacy local prompt suffix must not bypass world completion");
assert.doesNotMatch(app, /\$\("enhanceBtn"\)\.onclick = enhancePrompt/);
assert.match(
  app,
  /\$\("connectBtn"\)\.disabled = worldCompletionPending/,
  "incomplete world drafts should keep the button clickable so users get an explicit reason",
);
assert.match(
  app,
  /if \(!hasWorldDescription\(\) \|\| !hasFirstFrame\(\)\) \{[\s\S]*?Complete world first[\s\S]*?world draft incomplete/,
  "entering a world must require both a first frame and a world description",
);
assert.match(html, /id="voicePromptBtn"/, "runtime prompt composer should expose voice input");
assert.match(html, /id="recordBtn"[^>]*class="gameplay-record-button"/, "gameplay recording must be visible");
assert.match(html, /id="recordDownloadBtn"[^>]*class="gameplay-download-button"/, "finished gameplay must be downloadable");
assert.match(html, /下载两份录像/, "one action should download comparison and Zing-only videos");
assert.match(html, /id="recordingReadyToast"[^>]*role="status"[^>]*hidden/, "finished worlds should announce downloadable recordings");
assert.match(html, /data-action="w"[^>]*>W<\/button>/, "movement controls should use compact keycaps");
assert.match(html, /data-action="i"[^>]*>↑<\/button>/, "look-up should use an arrow keycap");
assert.match(html, /data-action="j"[^>]*>←<\/button>/, "look-left should use an arrow keycap");
assert.match(html, /data-action="k"[^>]*>↓<\/button>/, "look-down should use an arrow keycap");
assert.match(html, /data-action="l"[^>]*>→<\/button>/, "look-right should use an arrow keycap");
assert.match(html, /id="enhanceBtnLabel">补全世界<\/b>/, "world completion should use the requested label");
assert.match(app, /globalThis\.indexedDB\.open\(CUSTOM_WORLD_DB_NAME, CUSTOM_WORLD_DB_VERSION\)/);
assert.match(app, /keyPath: "fingerprint"/, "custom worlds should deduplicate by image and description");
assert.match(app, /return \[\.\.\.presets, \.\.\.customWorldPresets\]/, "custom worlds should follow built-ins");
assert.match(app, /await writeStoredCustomWorld\(record\)/, "custom worlds should persist in browser storage");
assert.match(
  app,
  /const connectionReport = await dualModelController\.connect\(init\);[\s\S]*?rememberEnteredWorld\(/,
  "custom worlds should only be saved after a model connection succeeds",
);
assert.match(app, /function sendRuntimePromptUpdate\(\)/);
assert.match(
  app,
  /runtimePromptRewritePending = true;[\s\S]{0,120}?input\.blur\(\);\s*canvas\.focus\(\{ preventScroll: true \}\);/,
  "sending a runtime prompt should immediately return keyboard control to the world",
);
assert.match(
  app,
  /catch \(error\) \{[\s\S]*?input\.focus\(\{ preventScroll: true \}\);[\s\S]*?\} finally \{[\s\S]*?sendPromptBtn[\s\S]*?\}/,
  "only a failed rewrite should return focus to the prompt input",
);
assert.doesNotMatch(
  app,
  /finally \{[\s\S]{0,180}?input\.focus/,
  "a successful rewrite must not focus the prompt input",
);
assert.match(app, /window\.SpeechRecognition \|\| window\.webkitSpeechRecognition/);
assert.match(app, /recognition\.lang = "zh-CN"/);
assert.match(app, /if \(!window\.isSecureContext\)/);
assert.match(app, /secureBaseUrl/);
assert.match(app, /function h264WebSocketEndpoint\(key\)/);
assert.match(app, /UI_CONFIG\.h264WebSocketBaseUrl/);
assert.match(app, /endpoint: h264WebSocketEndpoint\(key\)/);
assert.match(app, /async function connectH264SessionWithRetry\(key, h264Session, init\)/);
assert.match(app, /H264_CONNECT_MAX_ATTEMPTS/);
assert.doesNotMatch(app, /自动回退 WebP/);
assert.match(
  app,
  /if \(H264_WEBSOCKET_REQUESTED\) \{[\s\S]*?await connectH264SessionWithRetry\("minwm", minwmH264Session, init\);[\s\S]*?return;[\s\S]*?\}[\s\S]*?return openPrimarySession\(init, url\);/,
  "Zing should retry H.264 and only use WebP when H.264 is explicitly disabled",
);
assert.match(
  app,
  /if \(H264_WEBSOCKET_REQUESTED\) \{[\s\S]*?await connectH264SessionWithRetry\(key, h264Session, init\);[\s\S]*?return;[\s\S]*?\}[\s\S]*?return fallbackSession\.connect\(init, url\);/,
  "secondary models should retry H.264 and must not automatically fall back to WebP",
);
assert.match(app, /"not-allowed": "麦克风未授权"/);
assert.match(app, /status\.textContent = next \? "正在聆听" : idleStatus/);
assert.match(
  app,
  /button\.addEventListener\("pointerdown", \(event\) => \{[\s\S]*?event\.preventDefault\(\);[\s\S]*?focusInputAtEnd\(\);/,
  "pressing the voice button should preserve textarea focus",
);
assert.match(
  app,
  /button\.onclick = \(\) => \{\s*focusInputAtEnd\(\);/,
  "starting speech recognition should focus the prompt at the insertion point",
);
assert.doesNotMatch(
  app,
  /recognition\.onend = \(\) => \{[\s\S]{0,120}?input\.focus/,
  "recognition ending must not steal focus back after the user moves to world controls",
);

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
assert.match(app, /realtime_interactive_event_grace_ms:\s*1800/);
assert.match(
  app,
  /kind === "prompt"[\s\S]{0,40}\? "prompt"/,
  "MinWM prompt playback should cut over frames from the previous prompt",
);
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
assert.match(app, /enabled:\s*\(init\) => modelSelected\("lingbot2"\) && init\.generation_mode !== "t2v"/);
assert.match(app, /function drawRecordingComparisonPreview\(/);
assert.match(app, /createFullscreenController/);
const placeholderIndex = app.indexOf("await drawInitialReferencePlaceholders(firstFrame);");
const connectIndex = app.indexOf("dualModelController.connect(init)", placeholderIndex);
const activateRulesIndex = app.indexOf("worldRulesController.activate(preparedWorldRules)", placeholderIndex);
assert.ok(
  placeholderIndex >= 0 && connectIndex > placeholderIndex,
  "I2V should retain the selected reference while both models prepare their first generated frame",
);
assert.ok(
  activateRulesIndex > placeholderIndex && activateRulesIndex < connectIndex,
  "prepared skill controls should mount before waiting for every comparison backend to connect",
);
const visiblePlaceholderIndex = app.indexOf("drawVisibleReferencePlaceholders();");
const firstFrameReadIndex = app.indexOf("enteredFirstFrame = await readFirstFrame()", visiblePlaceholderIndex);
assert.ok(
  visiblePlaceholderIndex >= 0 && firstFrameReadIndex > visiblePlaceholderIndex,
  "Generate should paint the already-visible reference before waiting for its request bytes",
);
assert.match(server, /BACKEND_ENV_PREFIXES = \{/);
assert.match(server, /"minwm": "MINWM"/);
assert.match(server, /"lingbot2": "LINGBOT2"/);

console.log("dual model DOM contract ok");
